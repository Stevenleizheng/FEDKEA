import os
import sys
import argparse
import subprocess
from utilis import Timer

def argument():
    parser = argparse.ArgumentParser(description='Binary task prediction')
    parser.add_argument('-i', '--fasta_data', type=str, help='fasta file path')
    parser.add_argument('-g', '--gpu', type=str, default=None, help="the number of GPU,(eg. '1')")
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size,(eg. 32)')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='threshold,(eg. 0.5)')
    parser.add_argument('-o', '--output', type=str, default=f'{os.getcwd()}', help='output,(eg. result)')
    args = parser.parse_args()
    fasta_path = args.fasta_data
    strgpu = args.gpu
    batch_size = args.batch_size
    threshold = args.threshold
    output = args.output
    return fasta_path, strgpu, batch_size, threshold, output

def main():
    fasta_path, strgpu, batch_size, threshold, output = argument()
    cmds = [['python', f'{sys.path[0]}/1.data_preparation.py', '-i', f'{fasta_path}'],
            ['python', f'{sys.path[0]}/2.binary_prediction.py', '-g', f'{strgpu}', '-b', f'{batch_size}', '-t', f'{threshold}'],
            ['python', f'{sys.path[0]}/3.enzyme_prediction_prompt.py', '-g', f'{strgpu}', '-b', f'{batch_size}'],
            ['python', f'{sys.path[0]}/4.enzyme_prediction.py', '-g', f'{strgpu}', '-b', f'{batch_size}'],
            ['python', f'{sys.path[0]}/6.EC_name.py'],
            ['rm', '-rf', f'{sys.path[0]}/data/'],
            ['mv', f'{sys.path[0]}/result/', f'{output}']]
    sens =['Data preparation(token) has been done.',
          'Binary prediction has been done.',
          'Enzyme prediction prompts have been prepared.',
          'Enzyme commission prediction has been done.',
          'EC number and name conversion has been completed.',
          'Wait.',
          'All finished.']
    for cmd, sen in zip(cmds,sens):    
        timer = Timer()
        subprocess.run(cmd)
        print(sen)
        timer.stop()
        timer.sum()

if __name__ =="__main__":
    main()