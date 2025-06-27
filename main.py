import os
import sys
import argparse
import subprocess
from utilis import Timer

def argument():
    parser = argparse.ArgumentParser(description='Enzyme Commission Prediction Pipeline')
    parser.add_argument('-i', '--fasta_data', type=str, help='fasta file path')
    parser.add_argument('-g', '--gpu', type=str, default='None', help="the number of GPU,(eg. '1')")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size,(eg. 1)')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='threshold,(eg. 0.5)')
    parser.add_argument('-d', '--data_path', type=str, default=os.path.join(os.getcwd(), 'data/'), help='data file path')
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.getcwd(), 'result/'), help='output,(eg. result)')
    parser.add_argument('-r', '--running_mode', type=int, default=1, help='running_mode: 1 for both enzyme identification and function prediction, 2 for function prediction only')
    args = parser.parse_args()
    fasta_path = args.fasta_data
    strgpu = args.gpu
    batch_size = args.batch_size
    threshold = args.threshold
    data_path = args.data_path
    output = args.output
    running_mode = args.running_mode
    return fasta_path, strgpu, batch_size, threshold, data_path, output, running_mode

def main():
    fasta_path, strgpu, batch_size, threshold, data_path, output, running_mode = argument()
    if running_mode == 1:
        cmds = [
                ['python', f'{sys.path[0]}/1.data_preparation.py', '-i', f'{fasta_path}', '-d', f'{data_path}'],
                ['python', f'{sys.path[0]}/2.binary_prediction.py', '-g', f'{strgpu}', '-b', f'{batch_size}', '-t', f'{threshold}', '-d', f'{data_path}', '-o', f'{output}'],
                ['python', f'{sys.path[0]}/3.enzyme_prediction_prompt.py', '-g', f'{strgpu}', '-b', f'{batch_size}', '-d', f'{data_path}', '-o', f'{output}', '-r', f'{running_mode}'],
                ['python', f'{sys.path[0]}/4.enzyme_prediction.py', '-g', f'{strgpu}', '-b', f'{batch_size}', '-d', f'{data_path}', '-o', f'{output}'],
                ['python', f'{sys.path[0]}/5.EC_name.py', '-o', f'{output}'],
                ['python', f'{sys.path[0]}/6.confidence_values.py', '-o', f'{output}'],
        ]
        sens =[
            'Data preparation(token) has been done.',
            'Binary prediction has been done.',
            'Enzyme prediction prompts have been prepared.',
            'Enzyme commission prediction has been done.',
            'EC number and name conversion has been completed.',
            'Confidence values assignment has been completed.',
            ]
        for cmd, sen in zip(cmds,sens):    
            timer = Timer()
            result = subprocess.run(cmd)
            if result.returncode != 0:
                sys.exit(1)
            print(sen)
            timer.stop()
            timer.sum()
        print('All finished.')
    elif running_mode == 2:
        cmds = [
        ['python', f'{sys.path[0]}/1.data_preparation.py', '-i', f'{fasta_path}', '-d', f'{data_path}'],
        ['python', f'{sys.path[0]}/3.enzyme_prediction_prompt.py', '-g', f'{strgpu}', '-b', f'{batch_size}', '-d', f'{data_path}', '-o', f'{output}', '-r', f'{running_mode}'],
        ['python', f'{sys.path[0]}/4.enzyme_prediction.py', '-g', f'{strgpu}', '-b', f'{batch_size}', '-d', f'{data_path}', '-o', f'{output}'],
        ['python', f'{sys.path[0]}/5.EC_name.py', '-o', f'{output}'],
        ['python', f'{sys.path[0]}/6.confidence_values.py', '-o', f'{output}'],
        ]
        sens =[
            'Data preparation(token) has been done.',
            'Enzyme prediction prompts have been prepared.',
            'Enzyme commission prediction has been done.',
            'EC number and name conversion has been completed.',
            'Confidence values assignment has been completed.',
            ]
        for cmd, sen in zip(cmds,sens):    
            timer = Timer()
            result = subprocess.run(cmd)
            if result.returncode != 0:
                sys.exit(1)
            print(sen)
            timer.stop()
            timer.sum()
        print('All finished.')

if __name__ =="__main__":
    main()