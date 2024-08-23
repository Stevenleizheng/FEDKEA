import os
import pandas as pd 
import argparse
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq

def argument():
    """import parameters"""
    parser = argparse.ArgumentParser(description='Delete *')
    parser.add_argument('-i', '--fasta_data', type=str, help="the name of file,(eg. 'project')")
    parser.add_argument('-d', '--data_path', type=str, default=os.path.join(os.getcwd(), 'data/'), help='data file path')
    args = parser.parse_args()
    fasta_path = args.fasta_data
    data_path = args.data_path
    return fasta_path, data_path

def mod_sequences(fasta_path, data_path): 
    """Delete *"""
    filename, ext = os.path.splitext(os.path.basename(fasta_path))
    new_filename = filename + "_mod" + ext
    os.makedirs(data_path, exist_ok = True)  
    with open(fasta_path, 'r') as infile, open(os.path.join(data_path, new_filename), 'w') as outfile:
        for record in SeqIO.parse(infile, 'fasta'):
            sequence = str(record.seq)
            if not sequence:
                print(f"Warning: Found an empty sequence for record {record.id}. Skipping...")
                continue
            if sequence.endswith('*'):
                sequence = sequence[:-1]
            record.seq = Seq(sequence)
            SeqIO.write(record, outfile, 'fasta')

def main():
    fasta_path, data_path = argument()
    mod_sequences(fasta_path, data_path)

if __name__ =="__main__":
    main()
