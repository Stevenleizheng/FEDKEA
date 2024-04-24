import os
import esm
import argparse
import pandas as pd
from Bio import SeqIO

def argument():
    parser = argparse.ArgumentParser(description='Enzyme Commission Predicting')
    parser.add_argument('-i', '--fasta_data', type=str, help='fasta file path')
    args = parser.parse_args()
    fasta_path = args.fasta_data
    return fasta_path

def format_transfer(fasta_path):
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
    os.makedirs('./data', exist_ok = True)
    with open('./data/accession.txt', 'w') as fw:
        for i, _ in data:
            fw.write(str(i) + '\n')
    with open('./data/raw_data.txt', 'w') as fw:
        for i, s in data:
            fw.write(str(i) + '\n')
            fw.write(str(s) + '\n')
    return data

def token(data):
    """Tokenize sequences using the ESM2_650M model."""
    alphabet = esm.pretrained.esm2_t33_650M_UR50D()[1]
    batch_converter = alphabet.get_batch_converter()
    batch_tokens = pd.DataFrame(batch_converter(data)[2]).iloc[:, :1024]
    batch_tokens.to_csv(f"data/token.txt", sep='\t', header=False, index=False)
    return batch_tokens

def main():
    """Main program running！"""
    fasta_path = argument()
    data = format_transfer(fasta_path)
    token(data)


if __name__ =="__main__":
    main()