import argparse
import gc
import io
import pickle
import warnings
import subprocess
import re
import math
import torch
from Bio import PDB
from biotite.structure.io import pdb
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from model.modeling_progen import ProGenForCausalLM
from transformers import AutoModel, AutoTokenizer
import esm
import biotite.structure.io as bsio
from datasets import load_dataset
import wandb
import random
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TORCH_HOME"] = "your_path"
warnings.filterwarnings("ignore")


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_emb(ids,seq_path,structure_path):
    processed_samples = {
        "seq_emb": [],
        "struct_emb": [],
        'receptor_mask': []
    }
    for i in ids:
        pkl = os.path.join(seq_path, f"{i}.pkl")
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        tmp = data['emb'].squeeze(0)
        len_current, _ = tmp.size()
        mask = torch.ones(len_current)
        processed_samples["seq_emb"].append(tmp)
        processed_samples["receptor_mask"].append(mask)
        file_path = os.path.join(structure_path, f"{i}.pkl")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        processed_samples["struct_emb"].append(data['mpnn_emb'][0])
    seq_emb = pad_sequence(processed_samples['seq_emb'], batch_first=True).to('cuda')
    receptor_mask = pad_sequence(processed_samples['receptor_mask'], batch_first=True).to('cuda')
    struct_emb = pad_sequence(processed_samples['struct_emb'], batch_first=True).to('cuda')
    return seq_emb,struct_emb,receptor_mask



class CustomEnv:
    def __init__(self,temperature=1.0):
        self.device = 'cuda'
        self.esmfold_model = esm.pretrained.esmfold_v1().eval().cuda()
        self.start = None
        self.len = 0
        self.temperature = temperature

    def reset(self,tokenizer,num):
        self.start = torch.tensor(tokenizer.encode('<|bos|>1')).unsqueeze(0).repeat(num, 1).to(self.device)
        self.len = 0
    def get_reward(self,seqs,output_pdb_filename="tmp/tmp_output.pdb"):
        rewards=[]
        for i,seq in enumerate(seqs):
            with torch.no_grad():
                pdb_string = self.esmfold_model.infer_pdb(seq[1:])
            with open(output_pdb_filename, "w") as f:
                f.write(pdb_string)
            pdb_io = io.StringIO(pdb_string)
            pdb_file = pdb.PDBFile.read(pdb_io)
            structure = pdb.get_structure(pdb_file, extra_fields=["b_factor"])
            rwd = torch.tensor(structure.b_factor.mean())
            rewards.append(rwd)
        mean_rewards = sum(rewards) / len(rewards)
        return mean_rewards

    def get_loss(self,model,batch,seq_path,structure_path):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        seq_emb, struct_emb, receptor_mask = get_emb(batch['id'], seq_path, structure_path)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            seq_emb=seq_emb,
            struct_emb=struct_emb,
            receptor_mask=receptor_mask,
            labels=input_ids
        )
        loss = outputs.loss
        return loss

    def step(self,model,tokenizer,batch,seq_path,structure_path):
        if(self.len==0):
            num=8
            self.len+=8
        else:
            num=6
            self.len+=6
        model.train()
        seq_emb, struct_emb, receptor_mask = get_emb(batch['id'], seq_path, structure_path)
        logprobs = []
        while num:
            res = model(
                input_ids=self.start,
                seq_emb=seq_emb,
                struct_emb=struct_emb,
                receptor_mask=receptor_mask
            )
            logits = res.logits
            next_token_logits = logits[:, -1, :]
            next_token_logits[:, 0:5] = float('-inf')
            next_token_logits[:, 30:32] = float('-inf')
            next_token_probs = F.softmax(next_token_logits / self.temperature, dim=-1)
            next_token_id = torch.multinomial(next_token_probs, 1)
            if (next_token_id < 5).any():
                replacement = torch.argmax(next_token_probs, dim=-1).unsqueeze(-1)
                mask = next_token_id < 5  # 生成布尔掩码
                next_token_id[mask] = replacement[mask]  # 只替换满足条件的元素
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            selected_log_probs = log_probs.gather(1, next_token_id)
            logprobs.append(selected_log_probs)
            self.start = torch.cat((self.start, next_token_id), dim=-1)
            num-=1
        decoded_texts = tokenizer.batch_decode(self.start, skip_special_tokens=True)
        reward = self.get_reward(decoded_texts)
        log_prob = torch.stack(logprobs, dim=1)
        return reward,log_prob

def data_prepare(csv_path,tokenizer,batch_size):
    data_files = {
        "train": csv_path
    }
    dataset = load_dataset('csv', data_files=data_files)
    dataset = dataset['train'].shuffle(seed = 1)
    def preprocess_function(samples):
        processed_samples = {
            "input_ids": [],
            "attention_mask": [],
            'id':[]
        }
        for i, peptide, receptor in zip(samples['id'], samples['seq1'], samples['seq2']):
            processed_samples['id'].append(i)
            tokenized_input = tokenizer(f"<|bos|>1{peptide}2<|eos|>")
            processed_samples["input_ids"].append(tokenized_input["input_ids"])
            processed_samples["attention_mask"].append(tokenized_input["attention_mask"])
        max_len = max(len(lst) for lst in processed_samples['input_ids'])
        inputs_ids = [lst + [0] * (max_len - len(lst)) for lst in processed_samples['input_ids']]
        attention_mask = [lst + [0] * (max_len - len(lst)) for lst in processed_samples['attention_mask']]
        inputs_ids = torch.tensor(inputs_ids)
        attention_mask = torch.tensor(attention_mask)
        res = {
            'input_ids': inputs_ids,
            'attention_mask': attention_mask,
            'id':processed_samples['id'],
        }
        return res
    total_size = len(dataset)
    batches = []
    for i in range(0, total_size, batch_size):
        batch = dataset.select(range(i, min(i + batch_size, total_size)))  # 切分
        batch = preprocess_function(batch)
        batches.append(batch)
    return batches

def train(model_path,batch_size, n_steps,structure_path,seq_path,train_csv_path,temperature,gamma,output_path,checkpoint_steps,learning_rate,min_learning_rate,cosine_cycle_steps,alpha,beta,device = 'cuda'):
    model = ProGenForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    params_to_optimize = []
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in  ['struct', 'lm_head', 'cross', 'ln_2',
                'mlp_f', 'ln_f', 'ln_3']):
            params_to_optimize.append(param)
    data_batches =  data_prepare(train_csv_path,tokenizer,batch_size)
    env = CustomEnv(temperature)
    optimizer = Adam(params_to_optimize, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_cycle_steps, eta_min=min_learning_rate)
    print("Model initialized, starting training...")
    bmax = len(data_batches)
    idx = 0
    for i in tqdm(range(n_steps)):
        if(idx==bmax):
            idx=0
        batch = data_batches[idx]
        idx+=1
        num = batch['input_ids'].size(0)
        env.reset(tokenizer,num)
        rewards = []
        log_probs = []
        while env.len<20:
            reward,prob = env.step(model,tokenizer,batch,seq_path,structure_path)
            rewards.append(reward)
            log_probs.append(prob)
        loss = env.get_loss(model,batch,seq_path,structure_path)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        returns = [r.view(1, 1) for r in returns]
        RL_losses = [a * b for a, b in zip(returns, log_probs)]
        for j, item in enumerate(RL_losses):
            RL_losses[j] = torch.sum(item)
        RL_losses = -torch.mean(torch.stack(RL_losses))
        loss =  (RL_losses*alpha + loss*beta)
        if (i%checkpoint_steps==0 and i!=0):
            folder_path = os.path.join(output_path, f"FoldBack_{i}")
            os.makedirs(folder_path, exist_ok=True)
            model.save_pretrained(folder_path)
            tokenizer.save_pretrained(folder_path, pretty=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


if __name__ == "__main__":
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', action='store', dest='n_steps', type=int,
                        default=801,help='Total number of tuning steps.')
    parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                        default=6,help='Batch size for training.')
    parser.add_argument('--checkpoint-steps', action='store', dest='checkpoint_steps', type=int,
                        default=200,help='Number of steps between saving checkpoints.')
    parser.add_argument('--learning-rate', action='store', dest='learning_rate', type=float,
                        default=5e-4,help='Initial learning rate.')
    parser.add_argument('--min-learning-rate', action='store', dest='min_learning_rate', type=float,
                        default=1e-5,help='Minimum learning rate for cosine annealing.')
    parser.add_argument('--cosine-cycle-steps', action='store', dest='cosine_cycle_steps', type=int,
                        default=800,help='Number of steps for one cosine annealing cycle.')
    parser.add_argument('--model', action='store', dest='model_path',
                        default='../FoldPep/res_model/FoldPep_15000',help='Path to the pretrained model checkpoint.')
    parser.add_argument('--struct', action='store', dest='structure_path',
                        default='../datasets/receptor_structure',help='Path to receptor structure data.')
    parser.add_argument('--seq', action='store', dest='seq_path',
                        default='../datasets/receptor_seq',help='Path to receptor sequence data.')
    parser.add_argument('--train', action='store', dest='train_csv_path',
                        default='../datasets/train.csv',help='Path to the training dataset (CSV file).')
    parser.add_argument('--temperature', action='store', dest='temperature',type=float,
                        default=1.0,help="Temperature between 0.2 and 1.0.")
    parser.add_argument('--alpha', action='store', dest='alpha',type=float,
                        default=0.5,help="Scaling factor 1")
    parser.add_argument('--beta', action='store', dest='beta',type=float,
                        default=2.0,help="Scaling factor 2")
    parser.add_argument('--gamma', action='store', dest='gamma',type=float,
                        default=0.8,help="RL Scaling factor")
    parser.add_argument('--output', action='store', dest='output_path',
                        default='res_model',help='Directory to save trained models and outputs.')
    arg_dict = vars(parser.parse_args())
    train(**arg_dict)