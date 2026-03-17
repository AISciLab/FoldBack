import pickle
import warnings
import random
import argparse
import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from transformers import AutoTokenizer
import torch.nn.functional as F
from model.modeling_progen import ProGenForCausalLM
import os
warnings.filterwarnings("ignore")
from transformers.utils import logging
logging.set_verbosity_error()

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def data_prepare(r_seq_path,r_structure_path):
    with open(r_seq_path, "rb") as f:
        data = pickle.load(f)
    seq_emb = data['emb']

    _,len_current, _ = seq_emb.size()
    mask = torch.ones(len_current).unsqueeze(0)

    with open(r_structure_path, "rb") as f:
        data = pickle.load(f)
    structure_emb = data['mpnn_emb']
    return mask, seq_emb, structure_emb

def save_fasta(seqs,file_path):
    records=[]
    for i, seq in enumerate(seqs):
        record = SeqRecord(
            Seq(seq),
            id=f"seq_{i + 1}",
            description=""
        )
        records.append(record)

    SeqIO.write(records, file_path, "fasta")

def sample(model,tokenizer,seqlen,seq_emb,struct_emb,receptor_mask,temperature,device='cuda'):
    start = torch.tensor(tokenizer.encode('<|bos|>1')).unsqueeze(0).to(device)
    for i in range(seqlen):
        res_pdb = model(
            input_ids=start,
            seq_emb=seq_emb,
            struct_emb=struct_emb,
            receptor_mask=receptor_mask
        )
        logits = res_pdb.logits
        next_token_logits = logits[:, -1, :]
        next_token_logits[:, 0:5] = float('-inf')
        next_token_logits[:, 30:32] = float('-inf')
        next_token_probs = F.softmax(next_token_logits / temperature, dim=-1)
        next_token_id = torch.multinomial(next_token_probs, 1)
        if(next_token_id < 5):
            next_token_id = torch.argmax(next_token_probs, dim=-1).unsqueeze(-1)
        start = torch.cat((start, next_token_id), dim=-1)
    decoded_texts = tokenizer.batch_decode(start, skip_special_tokens=True)
    seq = decoded_texts[0][1:]
    return seq


def main(args):
    device = 'cuda'
    set_seed()
    model = ProGenForCausalLM.from_pretrained(args.ckpt_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, trust_remote_code=True)
    receptor_mask,seq_emb,struct_emb = data_prepare(args.receptor_seq_path,args.receptor_structure_path)
    receptor_mask = receptor_mask.to(device)
    seq_emb = seq_emb.to(device)
    struct_emb = struct_emb.to(device)
    seqs = []
    seen = set()
    seed_i = 1
    pbar = tqdm(total=args.num)
    while len(seqs) < args.num:
        set_seed(seed_i)
        seq = sample(
            model,
            tokenizer,
            args.length,
            seq_emb,
            struct_emb,
            receptor_mask,
            args.temperature,
            device
        )

        if seq not in seen:
            seen.add(seq)
            seqs.append(seq)
            pbar.update(1)
        seed_i += 1

    pbar.close()
    save_fasta(seqs,args.out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", type=str, default="example/output/peptide.fasta",help="Path to the output FASTA file where the generated protein sequence will be saved.")
    parser.add_argument("--ckpt-path", type=str, default='checkpoint',help="Path to the model weights file.")
    parser.add_argument("--receptor-seq-path", type=str, default='example/receptor_seq_emb.pkl',help="Path to the target sequence characterization file.")
    parser.add_argument("--receptor-structure-path", type=str, default='example/receptor_structure_emb.pkl',help="Path to the target structure characterization file.")
    parser.add_argument("--length", type=int, default=15,help="Length of the peptide sequence to be generated.")
    parser.add_argument("--temperature", type=float, default=1.0,help="Temperature between 0.2 and 1.0.")
    parser.add_argument("--num", type=int, default=10,help="Num of the peptide sequence to be generated.")
    args = parser.parse_args()
    if not (0.2 <= args.temperature <= 1.0):
        parser.error("temperature must be between 0.2 and 1.0")
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main(args)
