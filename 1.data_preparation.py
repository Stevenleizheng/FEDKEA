import os
import sys
import esm
import argparse
import torch
from Bio import SeqIO

def argument():
    parser = argparse.ArgumentParser(description='Enzyme Commission Predicting')
    parser.add_argument('-i', '--fasta_data', type=str, help='fasta file path')
    parser.add_argument('-d', '--data_path', type=str, default=os.path.join(os.getcwd(), 'data/'), help='data file path')
    args = parser.parse_args()
    fasta_path = args.fasta_data
    data_path = args.data_path
    return fasta_path, data_path

def format_transfer(fasta_path, data_path):
    """
    Convert data from fasta format to accession sequence format.
    Here is a example:
    Accession
    ADMFGHGILOPAAAMMM
    """
    data = []
    record_id = []
    records = SeqIO.parse(fasta_path, "fasta")
    for record in records:
        record_id.append(record.id)
        data.append((record.id, record.seq))
    os.makedirs(data_path, exist_ok = True)  
    with open(os.path.join(data_path, "accession.txt"), 'w') as fw:
        for i, _ in data:
            fw.write(str(i) + '\n')
    with open(os.path.join(data_path, "raw_data.txt"), 'w') as fw:
        for i, s in data:
            fw.write(str(i) + '\n')
            fw.write(str(s) + '\n')
    return data

def token(data, data_path):
    """Tokenize sequences using the ESM2_650M model."""
    alphabet = esm.pretrained.esm2_t33_650M_UR50D()[1]
    batch_converter = alphabet.get_batch_converter()
    batch_tokens = batch_converter(data)[2][:, :1024]
    torch.save(batch_tokens.clone(),os.path.join(data_path, "token.pt"))
    return batch_tokens

def main():
    """Main program running！"""
    fasta_path, data_path = argument()
    data = format_transfer(fasta_path, data_path)
    token(data, data_path)


if __name__ =="__main__":
    main()