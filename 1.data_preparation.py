import os
import sys
import esm
import argparse
import torch
from Bio import SeqIO

def argument():
    """
    Parse command line arguments for fasta file path and data folder path.
    
    Returns:
    fasta_path (str): Path to the fasta file.
    data_path (str): Path to the data folder.
    """
    parser = argparse.ArgumentParser(description='Enzyme Commission Predicting')
    parser.add_argument('-i', '--fasta_data', type=str, help='fasta file path')
    parser.add_argument('-d', '--data_path', type=str, default=os.path.join(os.getcwd(), 'data/'), help='data file path')
    args = parser.parse_args()
    fasta_path = args.fasta_data
    data_path = args.data_path
    return fasta_path, data_path

def format_transfer(fasta_path, data_path):
    """
    Convert FASTA format to structured data files

    Args:
        fasta_path (str): Path to input FASTA file
        data_path (str): Output directory for processed data
        
    Returns:
        list: List of tuples containing (accession_id, protein_sequence)
        
    Raises:
        SystemExit: If input FASTA file is empty
    """
    data = []
    record_id = []
    # Parse FASTA file and validate content
    records = SeqIO.parse(fasta_path, "fasta")
    if not records:
        print("The fasta file is empty. Please check the file.")
        sys.exit(1)
    # Process each protein sequence record
    for record in records:
        sequence = str(record.seq)
        # Remove trailing stop codon '*'
        if sequence.endswith('*'):
            sequence = sequence[:-1]
        record_id.append(record.id)
        data.append((str(record.id), sequence))
    # Create output directory if not exists
    os.makedirs(data_path, exist_ok=True)  
    # Write accession IDs list file
    with open(os.path.join(data_path, "accession.txt"), 'w') as fw:
        for i, _ in data:
            fw.write(str(i) + '\n')        
    # Write raw data file with IDs and sequences
    with open(os.path.join(data_path, "raw_data.txt"), 'w') as fw:
        for i, s in data:
            fw.write(str(i) + '\n')
            fw.write(str(s) + '\n')
    return data

def token(data, data_path):
    """Tokenize protein sequences using ESM-2 650M model
    
    Args:
        data (list): List of tuples containing (accession_id, protein_sequence)
        data_path (str): Output directory for token tensor
        
    Returns:
        torch.Tensor: Token tensor of shape [batch_size, 1024]
        
    Note:
        - Uses ESM2_t33_650M_UR50D model's alphabet for tokenization
        - Truncates sequences longer than 1024 tokens
        - Saves tensor to {data_path}/token.pt
    """
    # Load ESM model's alphabet and batch converter
    alphabet = esm.pretrained.esm2_t33_650M_UR50D()[1]  
    batch_converter = alphabet.get_batch_converter()
    # Convert sequences to tokens and truncate to first 1024 tokens
    batch_tokens = batch_converter(data)[2][:, :1024]    
    # Save token tensor to disk
    torch.save(batch_tokens.clone(), os.path.join(data_path, "token.pt"))
    return batch_tokens

def main():
    """Orchestrate the complete data preparation pipeline
    
    Execution Flow:
    1. Retrieve input paths using command line arguments
    2. Convert FASTA file to structured text format
    3. Generate sequence tokens for deep learning
    
    Calls three core components:
    - argument(): Command line interface configuration
    - format_transfer(): Data format conversion
    - token(): Neural network tokenization
    """
    # Parse input arguments (FASTA path and output directory)
    fasta_path, data_path = argument()
    # Process raw FASTA into structured data files
    data = format_transfer(fasta_path, data_path)
    # Generate ESM-2 token embeddings for model input
    batch_tokens = token(data, data_path)


if __name__ =="__main__":
    main()
