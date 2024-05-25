import sys
import pandas as pd 

def main():
    data = pd.read_csv(f'{sys.path[0]}/EC_name/EC-Name.csv')
    accepted_name = {i:n for i,n in zip(list(data['ec_num']),list(data['accepted_name']))}
    other_names = {i:n for i,n in zip(list(data['ec_num']),list(data['other_names']))}
    result = pd.read_csv(f'{sys.path[0]}/result/enzyme_result.csv')
    result['accepted_name'] = [accepted_name[i] if i in accepted_name.keys() else 'None' for i in list(result['FinalResult'])]
    result['other_names'] = [other_names[i] if i in other_names.keys() else 'None' for i in list(result['FinalResult'])]
    result.to_csv(f'{sys.path[0]}/result/enzyme_result.csv',index=None)

if __name__ == '__main__':
    main()