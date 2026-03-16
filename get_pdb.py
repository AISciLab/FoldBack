import torch
import esm
import os
import random
import numpy as np
import argparse
from Bio import SeqIO

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def work(model,sequence,target):
    with torch.no_grad():
        pdb_string = model.infer_pdb(sequence)  # 获取 PDB 格式的字符串
    with open(target, "w") as f:
        f.write(pdb_string)

def main(args):
    set_seed()
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    for record in SeqIO.parse(args.fasta_path, "fasta"):
        seq_id = record.id
        sequence = str(record.seq)
        pdb_path = os.path.join(args.out_path, f'{seq_id}.pdb')
        work(model,sequence,pdb_path)
    print("work down!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", type=str, default="example/output/pdb_dir",help="Path to the output PDB dir.")
    parser.add_argument("--fasta-path", type=str, default="example/output/peptide.fasta",help="Path to the input FASTA file.")
    args = parser.parse_args()
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main(args)