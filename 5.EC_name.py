import os
import sys
import pandas as pd 
import argparse

def argument():
    """Configure and parse command-line arguments for EC number annotation.
    
    Returns:
        str: Path to output directory where results will be saved
    
    Features:
        - Default output directory: 'result/' in current working directory
        - Maintains consistent output handling with other pipeline components
    """
    parser = argparse.ArgumentParser(description='transfer EC number into description')
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.getcwd(), 'result/'), help='output,(eg. result)')
    args = parser.parse_args()
    output = args.output
    return output

def main(): 
    """Annotate enzyme prediction results with official EC nomenclature
    
    Workflow:
    1. Load EC number-to-name mappings from reference data
    2. Merge enzyme predictions with taxonomic classifications
    3. Export final annotated results
    """ 
    output = argument()
    data = pd.read_csv(f'{sys.path[0]}/EC_name/EC-Name.csv')
    accepted_name = {i:n for i,n in zip(list(data['ec_num']),list(data['accepted_name']))}
    other_names = {i:n for i,n in zip(list(data['ec_num']),list(data['other_names']))}
    result = pd.read_csv(os.path.join(output, 'enzyme_result_with_probability.csv'))
    result['accepted_name'] = [accepted_name[i] if i in accepted_name.keys() else 'None' for i in list(result['FinalResult'])]
    result['other_names'] = [other_names[i] if i in other_names.keys() else 'None' for i in list(result['FinalResult'])]
    result.to_csv(os.path.join(output, 'enzyme_result_with_probability.csv'),index=None)

if __name__ == '__main__':
    main()