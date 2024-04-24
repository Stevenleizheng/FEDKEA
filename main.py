import argparse
import subprocess
from utilis import Timer

def argument():
    parser = argparse.ArgumentParser(description='Binary task prediction')
    parser.add_argument('-i', '--fasta_data', type=str, help='fasta file path')
    parser.add_argument('-g', '--gpu', type=str, default=None, help="the number of GPU,(eg. '1')")
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size,(eg. 32)')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='threshold,(eg. 0.5)')
    args = parser.parse_args()
    fasta_path = args.fasta_data
    strgpu = args.gpu
    batch_size = args.batch_size
    threshold = args.threshold
    return fasta_path, strgpu, batch_size, threshold

def main():
    fasta_path, strgpu, batch_size, threshold = argument()
    cmds = [['python', '1.data_preparation.py', '-i', f'{fasta_path}'],
            ['python', '2.binary_prediction.py', '-g', f'{strgpu}', '-b', f'{batch_size}', '-t', f'{threshold}'],
            ['python', '3.enzyme_prediction_prompt.py', '-g', f'{strgpu}', '-b', f'{batch_size}'],
            ['python', '4.enzyme_prediction.py', '-g', f'{strgpu}', '-b', f'{batch_size}']]
    sens =['Data preparation(token) is done.',
          'Binary prediction is done.',
          'Enzyme prediction prompts are prepared.',
          'Enzyme commission prediction is done.']
    for cmd, sen in zip(cmds,sens):    
        timer = Timer()
        subprocess.run(cmd)
        print(sen)
        timer.stop()
        timer.sum()

if __name__ =="__main__":
    main()