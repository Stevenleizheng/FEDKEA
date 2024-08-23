import os
import sys
import argparse
import subprocess
from utilis import Timer

def argument():
    parser = argparse.ArgumentParser(description='Binary task prediction')
    parser.add_argument('-i', '--fasta_data', type=str, help='fasta file path')
    parser.add_argument('-g', '--gpu', type=str, default='None', help="the number of GPU,(eg. '1')")
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size,(eg. 32)')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='threshold,(eg. 0.5)')
    parser.add_argument('-d', '--data_path', type=str, default=os.path.join(os.getcwd(), 'data/'), help='data file path')
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.getcwd(), 'result/'), help='output,(eg. result)')
    parser.add_argument('-a', '--asterisk', type=int, default=0, help='Check if the sequence ends with an asterisk (*), return 1 if it does, return 0 if it doesn\'t.')
    args = parser.parse_args()
    fasta_path = args.fasta_data
    strgpu = args.gpu
    batch_size = args.batch_size
    threshold = args.threshold
    data_path = args.data_path
    output = args.output
    asterisk = args.asterisk
    return fasta_path, strgpu, batch_size, threshold, data_path, output, asterisk

def main():
    fasta_path, strgpu, batch_size, threshold, data_path, output, asterisk = argument()
    if asterisk == 0:
        cmds = [
                ['python', f'{sys.path[0]}/1.data_preparation.py', '-i', f'{fasta_path}', '-d', f'{data_path}'],
                ['python', f'{sys.path[0]}/2.binary_prediction.py', '-g', f'{strgpu}', '-b', f'{batch_size}', '-t', f'{threshold}', '-d', f'{data_path}', '-o', f'{output}'],
                ['python', f'{sys.path[0]}/3.enzyme_prediction_prompt.py', '-g', f'{strgpu}', '-b', f'{batch_size}', '-d', f'{data_path}', '-o', f'{output}'],
                ['python', f'{sys.path[0]}/4.enzyme_prediction.py', '-g', f'{strgpu}', '-b', f'{batch_size}', '-d', f'{data_path}', '-o', f'{output}'],
                ['python', f'{sys.path[0]}/5.EC_name.py', '-o', f'{output}'],
        ]
        sens =[
            'Data preparation(token) has been done.',
            'Binary prediction has been done.',
            'Enzyme prediction prompts have been prepared.',
            'Enzyme commission prediction has been done.',
            'EC number and name conversion has been completed.',
            ]
        for cmd, sen in zip(cmds,sens):    
            timer = Timer()
            subprocess.run(cmd)
            print(sen)
            timer.stop()
            timer.sum()
        print('All finished.')
    else:
        filename, ext = os.path.splitext(os.path.basename(fasta_path))
        new_filename = filename + "_mod" + ext
        cmds = [
        ['python', f'{sys.path[0]}/0.deal_xing.py', '-i', f'{fasta_path}', '-d', f'{data_path}'],
        ['python', f'{sys.path[0]}/1.data_preparation.py', '-i', os.path.join(data_path, new_filename), '-d', f'{data_path}'],
        ['python', f'{sys.path[0]}/2.binary_prediction.py', '-g', f'{strgpu}', '-b', f'{batch_size}', '-t', f'{threshold}', '-d', f'{data_path}', '-o', f'{output}'],
        ['python', f'{sys.path[0]}/3.enzyme_prediction_prompt.py', '-g', f'{strgpu}', '-b', f'{batch_size}', '-d', f'{data_path}', '-o', f'{output}'],
        ['python', f'{sys.path[0]}/4.enzyme_prediction.py', '-g', f'{strgpu}', '-b', f'{batch_size}', '-d', f'{data_path}', '-o', f'{output}'],
        ['python', f'{sys.path[0]}/5.EC_name.py', '-o', f'{output}'],
        ]
        sens =[
            'Delete *.',
            'Data preparation(token) has been done.',
            'Binary prediction has been done.',
            'Enzyme prediction prompts have been prepared.',
            'Enzyme commission prediction has been done.',
            'EC number and name conversion has been completed.',
            ]
        for cmd, sen in zip(cmds,sens):    
            timer = Timer()
            subprocess.run(cmd)
            print(sen)
            timer.stop()
            timer.sum()
        print('All finished.')

if __name__ =="__main__":
    main()