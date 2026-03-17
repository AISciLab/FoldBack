# FoldBack: Protein Folding Code-guided Specific Peptide Design through 3D Dynamic Feedback
Peptides often exert therapeutic effects over diseases by binding to target proteins. Protein language model (PLM) is a promising perspective for peptide drug design. However, most of PLMs are difficult to consider the specificity of peptide to given target proteins. On the other hand, PLMs often produce peptides with unrealistic 3D molecular structures. Therefore, this work proposes FoldBack that is a unified framework to design peptides with target specificity and practical structure. To be specific, the sequence and structure of targets are treated as context of peptide generation process. Based on this intuition, we developed multimodal Adapter tuning to enable PLMs to generate target-aware peptides. In addition, 3D structure of peptides is evaluated and dynamically fed back to the multimodal Adapter by reinforcement learning technologies, thus preventing the formation of unusual structures. Finally, a systematic assessment pipeline is established, which integrates the peptide-target affinity, structure quality, physicochemical properties, diversity and novelty analysis. Results suggest that FoldBack achieves SOTA performance in various evaluation criteria comparing with baselines. Overall, FoldBack provides an effective framework for targeted peptide drug discovery.

## 1.Environment Setup

For the FoldBack model, please refer to the dependencies specified in the  [environment.yml](./environment.yml) file.  
Since ESMC has higher Python version requirements, an additional environment is needed. Please ensure you use a newer Python version (e.g., Python 3.12 or above) to properly obtain sequence embeddings.

Installation of ESMFold can be found in the official documentation.

The FoldBack model weights can be downloaded from [this link](https://drive.google.com/file/d/1oHXzoVKJnTpBWgJrORAO-Z-j-FsEfivq/view?usp=drive_link).

# 2.Example Workflow

**First**, structural features of the receptor are extracted using `get_receptor_emb.py` from the `ProteinMPNN` module in the `receptor_processing` directory. For example:

```bash
python get_receptor_emb.py \
  --pdb-path ../../example/receptor.pdb \
  --out-path ../../example/receptor_structure_emb.pkl
```

Then, sequence features of the receptor are extracted using `get_receptor_emb.py` from the `ESMC` module located in the `receptor_processing` directory.For example:

```bash
python get_receptor_emb.py \
  --fasta-path ../../example/receptor.fasta \
  --out-path ../../example
```

These representations are subsequently used to guide the generation of compatible peptide sequences.

You can view the full list of command-line arguments and their descriptions for any Python script by running it with the `--help` flag. For example:

```text
usage: get_receptor_emb.py [-h] [--fasta-path FASTA_PATH] [--out-path OUT_PATH]

options:
  -h, --help            show this help message and exit
  --fasta-path FASTA_PATH
                        Path to the fasta file which seq features will be extracted.
  --out-path OUT_PATH   Path to the output PKL file where the extracted features will be saved.
```

**Second**, you can generate peptide sequences conditioned on a precomputed receptor feature file (`.pkl`). 
You are free to specify the peptide length and the number of peptide  sequences .For example:
```bash
python sample.py \
  --out-path example/output/peptide.fasta \
  --ckpt-path checkpoint \
  --receptor-seq-path example/receptor_seq_emb.pkl \
  --receptor-structure-path example/receptor_structure_emb.pkl \
  --length 15 \
  --temperature 1.0 \
  --num 10
```
This will display the following help message:
```text
usage: sample.py [-h] [--out-path OUT_PATH] [--ckpt-path CKPT_PATH] [--receptor-seq-path RECEPTOR_SEQ_PATH] [--receptor-structure-path RECEPTOR_STRUCTURE_PATH] [--length LENGTH]
                 [--temperature TEMPERATURE] [--num NUM]

options:
  -h, --help            show this help message and exit
  --out-path OUT_PATH   Path to the output FASTA file where the generated protein sequence will be saved.
  --ckpt-path CKPT_PATH
                        Path to the model weights file.
  --receptor-seq-path RECEPTOR_SEQ_PATH
                        Path to the target sequence characterization file.
  --receptor-structure-path RECEPTOR_STRUCTURE_PATH
                        Path to the target structure characterization file.
  --length LENGTH       Length of the peptide sequence to be generated.
  --temperature TEMPERATURE
                        Temperature between 0.2 and 1.0.
  --num NUM             Num of the peptide sequence to be generated.
```

**Third**, The generated sequences, stored in FASTA format, can be further processed using ESMFold to predict their corresponding 3D structures. 
The resulting PDB files can then be utilized for downstream analyses, such as binding affinity evaluation.For example:
```bash
python get_pdb.py \
  --out-path example/output/pdb_dir \
  --fasta-path example/output/peptide.fasta
```
This will display the following help message:
```text
usage: get_pdb.py [-h] [--out-path OUT_PATH] [--fasta-path FASTA_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --out-path OUT_PATH   Path to the output PDB dir.
  --fasta-path FASTA_PATH
                        Path to the input FASTA file.
```