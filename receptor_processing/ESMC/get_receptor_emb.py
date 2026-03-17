import argparse
import random

from Bio import SeqIO
import csv
import numpy as np
import pickle

from tqdm import tqdm

from esm.models.esmc import ESMC
from esm.sdk.api import *
import os

os.environ["INFRA_PROVIDER"] = "True"
device = torch.device("cuda:0")
model = ESMC.from_pretrained("esmc_600m", device=device)


all_amino_acid_number = {'A': 5, 'C': 23, 'D': 13, 'E': 9, 'F': 18,
                                 'G': 6, 'H': 21, 'I': 12, 'K': 15, 'L': 4,
                                 'M': 20, 'N': 17, 'P': 14, 'Q': 16, 'R': 10,
                                 'S': 8, 'T': 11, 'V': 7, 'W': 22, 'Y': 19,
                                 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28}


def esm_encoder_seq(seq, pad_len):
    s = [all_amino_acid_number[x] for x in seq]
    while len(s) < pad_len:
        s.append(1)
    s.insert(0, 0)
    s.append(2)
    return torch.tensor(s)


def get_esm_embedding(seq):
    protein_tensor = ESMProteinTensor(sequence=esm_encoder_seq(seq, len(seq)).to(device))
    logits_output = model.logits(
        protein_tensor,  # 必须是 ESMProteinTensor 类型！
        config=LogitsConfig(sequence=True, return_embeddings=True)
    )

    esm_embedding = logits_output.embeddings
    assert isinstance(esm_embedding, torch.Tensor)
    return esm_embedding


def main(args):
    for record in SeqIO.parse(args.fasta_path, "fasta"):
        seq_id = record.id
        sequence = str(record.seq)
        print(seq_id)
        emb = get_esm_embedding(sequence)
        data = {
            'emb': emb[:, 1:-1, :]
        }
        out_path = os.path.join(args.out_path, f'{seq_id}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
    print("work down!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta-path", type=str, default="../../example/receptor.fasta",help="Path to the fasta file which seq features will be extracted.")
    parser.add_argument("--out-path", type=str, default="../../example",help="Path to the output PKL file where the extracted features will be saved.")
    args = parser.parse_args()
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main(args)
