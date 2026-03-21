import argparse
import gc
import pickle
import warnings
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from tqdm import tqdm
from model.modeling_progen import ProGenForCausalLM
from transformers import AutoModel, AutoTokenizer
import os
from datasets import load_dataset

warnings.filterwarnings("ignore")
os.environ["WANDB_MODE"]="offline"


device = 'cuda'
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loss(model, batch):
    model.train()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    seq_emb = batch['seq_emb'].to(device)
    struct_emb = batch['struct_emb'].to(device)
    receptor_mask = batch['receptor_mask'].to(device)
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

def data_prepare(structure_path,seq_path,train_csv_path,tokenizer,batch_size):
    data_files = {
        "train": train_csv_path,
    }
    dataset = load_dataset('csv', data_files=data_files)
    dataset = dataset['train']
    def preprocess_function(samples):
        processed_samples = {
            "input_ids": [],
            "attention_mask": [],
            "seq_emb": [],
            "struct_emb": [],
            'receptor_mask': []
        }
        for i, peptide, receptor in zip(samples['id'], samples['seq1'], samples['seq2']):

            tokenized_input = tokenizer(f"<|bos|>1{peptide}2<|eos|>")
            processed_samples["input_ids"].append(tokenized_input["input_ids"])
            processed_samples["attention_mask"].append(tokenized_input["attention_mask"])

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

        max_len = max(len(lst) for lst in processed_samples['input_ids'])
        inputs_ids = [lst + [0] * (max_len - len(lst)) for lst in processed_samples['input_ids']]
        attention_mask = [lst + [0] * (max_len - len(lst)) for lst in processed_samples['attention_mask']]
        inputs_ids = torch.tensor(inputs_ids)
        attention_mask = torch.tensor(attention_mask)
        seq_emb = pad_sequence(processed_samples['seq_emb'], batch_first=True)
        receptor_mask = pad_sequence(processed_samples['receptor_mask'], batch_first=True)
        struct_emb = pad_sequence(processed_samples['struct_emb'], batch_first=True)
        res = {
            'input_ids': inputs_ids,
            'attention_mask': attention_mask,
            'seq_emb': seq_emb,
            'receptor_mask': receptor_mask,
            'struct_emb': struct_emb
        }
        return res
    total_size = len(dataset)
    batches = []
    for i in range(0, total_size, batch_size):
        batch = dataset.select(range(i, min(i + batch_size, total_size)))
        batch = preprocess_function(batch)
        batches.append(batch)

    return batches

def train(model_path,batch_size, n_steps,structure_path,seq_path,train_csv_path,learning_rate,min_learning_rate,cosine_cycle_steps,checkpoint_steps,output_path):
    device = 'cuda'
    model = ProGenForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    params_to_optimize = []
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in
               ['struct', 'lm_head', 'cross', 'ln_2',
                'mlp_f', 'ln_f', 'ln_3']):
            params_to_optimize.append(param)
    data_batches =  data_prepare(structure_path,seq_path,train_csv_path,tokenizer,batch_size)
    optimizer = AdamW(params_to_optimize, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_cycle_steps, eta_min=min_learning_rate)
    print("Starting training...")
    bmax = len(data_batches)
    idx = 0
    for i in tqdm(range(n_steps)):
        if(idx==bmax):
            idx=0
        batch = data_batches[idx]
        idx+=1
        loss = get_loss(model,batch)
        if (i!=0 and (i % checkpoint_steps==0)):
            folder_path = os.path.join(output_path, f"FoldPep_{i}")
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
                        default=30001,help='Total number of training steps.')
    parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                        default=32,help='Batch size for training.')
    parser.add_argument('--checkpoint-steps', action='store', dest='checkpoint_steps', type=int,
                        default=5000,help='Number of steps between saving checkpoints.')
    parser.add_argument('--learning-rate', action='store', dest='learning_rate', type=float,
                        default=1e-3,help='Initial learning rate.')
    parser.add_argument('--min-learning-rate', action='store', dest='min_learning_rate', type=float,
                        default=5e-5,help='Minimum learning rate for cosine annealing.')
    parser.add_argument('--cosine-cycle-steps', action='store', dest='cosine_cycle_steps', type=int,
                        default=30000,help='Number of steps for one cosine annealing cycle.')
    parser.add_argument('--model', action='store', dest='model_path',
                        default='../FoldPep_progen_ckpt',help='Path to the pretrained model checkpoint.')
    parser.add_argument('--struct', action='store', dest='structure_path',
                        default='../datasets/receptor_structure',help='Path to receptor structure data.')
    parser.add_argument('--seq', action='store', dest='seq_path',
                        default='../datasets/receptor_seq',help='Path to receptor sequence data.')
    parser.add_argument('--train', action='store', dest='train_csv_path',
                        default='../datasets/train.csv',help='Path to the training dataset (CSV file).')
    parser.add_argument('--output', action='store', dest='output_path',
                        default='res_model',help='Directory to save trained models and outputs.')
    arg_dict = vars(parser.parse_args())
    train(**arg_dict)
