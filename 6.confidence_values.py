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
    """Assign confidence levels based on hierarchical probability thresholds
    
    Workflow:
    1. Load enzyme prediction results
    2. Apply multi-level threshold checks:
       - First level: EC class probability
       - Second level: EC subclass probability
       - Third level: EC sub-subclass probability
       - Fourth level: Final EC number probability
    3. Categorize confidence levels (high/medium/low)
    4. Save annotated results with confidence ratings
    
    Threshold Structure:
    - first_threshold: Minimum probabilities for each EC class (1-7)
    - [n]_threshold: Level-specific thresholds for each EC hierarchy combination
    """
    output = argument()
    lines = []
    with open(os.path.join(output, 'enzyme_result_with_probability.csv'), "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            lines.append(line)
    # High confidence identification
    lines = lines[1:]
    names = []
    first_threshold = {1:0.4865, 2:0.0000, 3:0.5687, 4:0.4969, 5:0.0000, 6:0.0000, 7:0.6095}
    for line in lines:
        data = line.split(",")
        if int(data[1]) == 1:
            if float(data[2]) > first_threshold[1]:
                one_threshold = {1:0.7721, 2:0.0000, 3:0.7063, 4:0.0000, 5:0.5148, 6:0.8467, 7:0.0000, 8:0.5861, 10:0.0000, 11:0.6811, 13:0.0000, 14:0.0000, 17:0.0000, 18:0.4413, 0:0.8898}
                if int(data[3]) == 1:
                    if float(data[4]) > one_threshold[1]:
                        names.append(data[0])
                elif int(data[3]) == 2:
                    if float(data[4]) > one_threshold[2]:
                        names.append(data[0])
                elif int(data[3]) == 3:
                    if float(data[4]) > one_threshold[3]:
                        names.append(data[0])
                elif int(data[3]) == 4:
                    if float(data[4]) > one_threshold[4]:
                        names.append(data[0])
                elif int(data[3]) == 5:
                    if float(data[4]) > one_threshold[5]:
                        names.append(data[0])
                elif int(data[3]) == 6:
                    if float(data[4]) > one_threshold[6]:
                        names.append(data[0])
                elif int(data[3]) == 7:
                    if float(data[4]) > one_threshold[7]:
                        names.append(data[0])
                elif int(data[3]) == 8:
                    if float(data[4]) > one_threshold[8]:
                        names.append(data[0])
                elif int(data[3]) == 10:
                    if float(data[4]) > one_threshold[10]:
                        names.append(data[0])
                elif int(data[3]) == 11:
                    if float(data[4]) > one_threshold[11]:
                        names.append(data[0])
                elif int(data[3]) == 13:
                    if float(data[4]) > one_threshold[13]:
                        names.append(data[0])
                elif int(data[3]) == 14:
                    if float(data[4]) > one_threshold[14]:
                        one_fourteen_threshold = {14:0.6138, 11:0.7652, 0:0.5028}
                        if int(data[5]) == 14:
                            if float(data[6]) > one_fourteen_threshold[14]:
                                names.append(data[0])
                        elif int(data[5]) == 11:
                            if float(data[6]) > one_fourteen_threshold[11]:
                                names.append(data[0])
                        elif int(data[5]) == 0: 
                            if float(data[6]) > one_fourteen_threshold[0]:
                                names.append(data[0])
                elif int(data[3]) == 17:
                    if float(data[4]) > one_threshold[17]:
                        one_seventeen_threshold = {7:0.0000, 1:0.4887, 0:0.0000}
                        if int(data[5]) == 7:
                            if float(data[6]) > one_seventeen_threshold[7]:
                                names.append(data[0])
                        elif int(data[5]) == 1:
                            if float(data[6]) > one_seventeen_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 0:
                            if float(data[6]) > one_seventeen_threshold[0]:
                                names.append(data[0])
                elif int(data[3]) == 18:
                    if float(data[4]) > one_threshold[18]:
                        names.append(data[0])
                elif int(data[3]) == 0:
                    if float(data[4]) > one_threshold[0]:
                        names.append(data[0])
        elif int(data[1]) == 2:
            if float(data[2]) > first_threshold[2]:
                two_threshold = {1:0.0000, 2:0.0000, 3:0.0000, 4:0.0000, 5:0.0000, 6:0.6337, 7:0.0000, 8:0.0000, 0:0.0000}
                if int(data[3]) == 1:
                    if float(data[4]) > two_threshold[1]:
                        two_one_threshold = {1:0.0000, 2:0.0000, 3:0.0000, 0:0.0000}
                        if int(data[5]) == 1:
                            if float(data[6]) > two_one_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 2:
                            if float(data[6]) > two_one_threshold[2]:
                                names.append(data[0])
                        elif int(data[5]) == 3:
                            if float(data[6]) > two_one_threshold[3]:
                                names.append(data[0])
                        elif int(data[5]) == 0:
                            if float(data[6]) > two_one_threshold[0]:
                                names.append(data[0])
                elif int(data[3]) == 2:
                    if float(data[4]) > two_threshold[2]:
                        names.append(data[0])
                elif int(data[3]) == 3:
                    if float(data[4]) > two_threshold[3]:
                        two_three_threshold = {1:0.0000, 2:0.0000, 3:0.0000}
                        if int(data[5]) == 1:
                            if float(data[6]) > two_three_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 2:
                            if float(data[6]) > two_three_threshold[2]:
                                names.append(data[0])
                        elif int(data[5]) == 3:
                            if float(data[6]) > two_three_threshold[3]:
                                names.append(data[0])
                elif int(data[3]) == 4:
                    if float(data[4]) > two_threshold[4]:
                        two_four_threshold = {2:0.0000, 1:0.0000, 99:0.0000, 0:0.3303}
                        if int(data[5]) == 2:
                            if float(data[6]) > two_four_threshold[2]:
                                names.append(data[0])
                        elif int(data[5]) == 1:
                            if float(data[6]) > two_four_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 99:
                            if float(data[6]) > two_four_threshold[99]:
                                names.append(data[0])
                        elif int(data[5]) == 0:
                            if float(data[6]) > two_four_threshold[0]:
                                names.append(data[0])
                elif int(data[3]) == 5:
                    if float(data[4]) > two_threshold[5]:
                        names.append(data[0])
                elif int(data[3]) == 6:
                    if float(data[4]) > two_threshold[6]:
                        names.append(data[0])
                elif int(data[3]) == 7:
                    if float(data[4]) > two_threshold[7]:
                        two_seven_threshold = {7:0.0000, 1:0.0000, 11:0.9225, 4:0.0000, 2:0.0000, 8:0.0000, 10:0.0000, 0:0.8909}
                        if int(data[5]) == 7:
                            if float(data[6]) > two_seven_threshold[7]:
                                names.append(data[0])
                        elif int(data[5]) == 1:
                            if float(data[6]) > two_seven_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 11:
                            if float(data[6]) > two_seven_threshold[11]:
                                names.append(data[0])
                        elif int(data[5]) == 4:
                            if float(data[6]) > two_seven_threshold[4]:
                                names.append(data[0])
                        elif int(data[5]) == 2:
                            if float(data[6]) > two_seven_threshold[2]:
                                names.append(data[0])
                        elif int(data[5]) == 8:
                            if float(data[6]) > two_seven_threshold[8]:
                                names.append(data[0])
                        elif int(data[5]) == 10:
                            if float(data[6]) > two_seven_threshold[10]:
                                names.append(data[0])
                        elif int(data[5]) == 0:
                            if float(data[6]) > two_seven_threshold[0]:
                                names.append(data[0])
                elif int(data[3]) == 8:
                    if float(data[4]) > two_threshold[8]:
                        two_eight_threshold = {1:0.0000, 4:0.0000, 0:0.0000}
                        if int(data[5]) == 1:
                            if float(data[6]) > two_eight_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 4:
                            if float(data[6]) > two_eight_threshold[4]:
                                names.append(data[0])
                        elif int(data[5]) == 0:
                            if float(data[6]) > two_eight_threshold[0]:
                                names.append(data[0])
                elif int(data[3]) == 0:
                    if float(data[4]) > two_threshold[0]:
                        names.append(data[0])
        elif int(data[1]) == 3:
            if float(data[2]) > first_threshold[3]:
                three_threshold = {1:0.0000, 2:0.8230, 4:0.5678, 5:0.5155, 6:0.3029, 0:0.5693}    
                if int(data[3]) == 1:
                    if float(data[4]) > three_threshold[1]:
                        three_one_threshold = {1:0.6409, 3:0.5077, 26:0.0000, 21:0.8990, 11:0.0000, 2:0.0000, 4:0.6378, 0:0.0000}
                        if int(data[5]) == 1:
                            if float(data[6]) > three_one_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 3:
                            if float(data[6]) > three_one_threshold[3]:
                                names.append(data[0])
                        elif int(data[5]) == 26:
                            if float(data[6]) > three_one_threshold[26]:
                                names.append(data[0])
                        elif int(data[5]) == 21:
                            if float(data[6]) > three_one_threshold[21]:
                                names.append(data[0])
                        elif int(data[5]) == 11:
                            if float(data[6]) > three_one_threshold[11]:
                                names.append(data[0])
                        elif int(data[5]) == 2:
                            if float(data[6]) > three_one_threshold[2]:
                                names.append(data[0])
                        elif int(data[5]) == 4:
                            if float(data[6]) > three_one_threshold[4]:
                                names.append(data[0])
                        elif int(data[5]) == 0:
                            if float(data[6]) > three_one_threshold[0]:
                                names.append(data[0])
                elif int(data[3]) == 2:
                    if float(data[4]) > three_threshold[2]:
                        three_two_threshold = {1:0.0000, 2:0.0000}
                        if int(data[5]) == 1:
                            if float(data[6]) > three_two_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 2:
                            if float(data[6]) > three_two_threshold[2]:
                                names.append(data[0])
                elif int(data[3]) == 4:
                    if float(data[4]) > three_threshold[4]:
                        three_four_threshold = {21:0.0000, 24:0.4465, 11:0.0000, 23:0.0000, 22:0.4990, 19:0.0000, 25:0.0000, 0:0.4953}
                        if int(data[5]) == 21:
                            if float(data[6]) > three_four_threshold[21]:
                                names.append(data[0])
                        elif int(data[5]) == 24:
                            if float(data[6]) > three_four_threshold[24]:
                                names.append(data[0])
                        elif int(data[5]) == 11:
                            if float(data[6]) > three_four_threshold[11]:
                                names.append(data[0])
                        elif int(data[5]) == 23:
                            if float(data[6]) > three_four_threshold[23]:
                                names.append(data[0])
                        elif int(data[5]) == 22:
                            if float(data[6]) > three_four_threshold[22]:
                                names.append(data[0])
                        elif int(data[5]) == 19:
                            if float(data[6]) > three_four_threshold[19]:
                                names.append(data[0])
                        elif int(data[5]) == 25:
                            if float(data[6]) > three_four_threshold[25]:
                                names.append(data[0])
                        elif int(data[5]) == 0:
                            if float(data[6]) > three_four_threshold[0]:
                                names.append(data[0])
                elif int(data[3]) == 5:
                    if float(data[4]) > three_threshold[5]:
                        three_five_threshold = {1:0.0000, 4:0.0000, 2:0.0000, 0:0.0000}
                        if int(data[5]) == 1:
                            if float(data[6]) > three_five_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 4:
                            if float(data[6]) > three_five_threshold[4]:
                                names.append(data[0])
                        elif int(data[5]) == 2:
                            if float(data[6]) > three_five_threshold[2]:
                                names.append(data[0])
                        elif int(data[5]) == 0:
                            if float(data[6]) > three_five_threshold[0]:
                                names.append(data[0])
                elif int(data[3]) == 6:
                    if float(data[4]) > three_threshold[6]:
                        three_six_threshold = {1:0.0000, 4:0.0000, 5:0.0000, 0:0.0000}
                        if int(data[5]) == 1:
                            if float(data[6]) > three_six_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 4:
                            if float(data[6]) > three_six_threshold[4]:
                                names.append(data[0])
                        elif int(data[5]) == 5:
                            if float(data[6]) > three_six_threshold[5]:
                                names.append(data[0])
                        elif int(data[5]) == 0:
                            if float(data[6]) > three_six_threshold[0]:
                                names.append(data[0])
                elif int(data[3]) == 0:
                    if float(data[4]) > three_threshold[0]:
                        names.append(data[0])
        elif int(data[1]) == 4:
            if float(data[2]) > first_threshold[4]:
                four_threshold = {1:0.4900, 2:0.0000, 3:0.0000, 6:0.0000, 0:0.7093}
                if int(data[3]) == 1:
                    if float(data[4]) > four_threshold[1]:
                        four_one_threshold = {1:0.4024, 99:0.0000, 2:0.7526, 3:0.4553}
                        if int(data[5]) == 1:
                            if float(data[6]) > four_one_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 99:
                            if float(data[6]) > four_one_threshold[99]:
                                names.append(data[0])
                        elif int(data[5]) == 2:
                            if float(data[6]) > four_one_threshold[2]:
                                names.append(data[0])
                        elif int(data[5]) == 3:
                            if float(data[6]) > four_one_threshold[3]:
                                names.append(data[0])
                elif int(data[3]) == 2:
                    if float(data[4]) > four_threshold[2]:
                        four_two_threshold = {1:0.0000, 3:0.0000, 0:0.0000}
                        if int(data[5]) == 1:
                            if float(data[6]) > four_two_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 3:
                            if float(data[6]) > four_two_threshold[3]:
                                names.append(data[0])
                        elif int(data[5]) == 0:
                            if float(data[6]) > four_two_threshold[0]:
                                names.append(data[0])
                elif int(data[3]) == 3:
                    if float(data[4]) > four_threshold[3]:
                        four_three_threshold = {2:0.0000, 3:0.0000, 1:0.5355, 0:0.4192}
                        if int(data[5]) == 2:
                            if float(data[6]) > four_three_threshold[2]:
                                names.append(data[0])
                        elif int(data[5]) == 3:
                            if float(data[6]) > four_three_threshold[3]:
                                names.append(data[0])
                        elif int(data[5]) == 1:
                            if float(data[6]) > four_three_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 0:
                            if float(data[6]) > four_three_threshold[0]:
                                names.append(data[0])
                elif int(data[3]) == 6:
                    if float(data[4]) > four_threshold[6]:
                        names.append(data[0])
                elif int(data[3]) == 0:
                    if float(data[4]) > four_threshold[0]:
                        names.append(data[0])
        elif int(data[1]) == 5:
            if float(data[2]) > first_threshold[5]:
                five_threshold = {1:0.5436, 2:0.0000, 3:0.0000, 4:0.0000, 6:0.0000, 0:0.3496}
                if int(data[3]) == 1:
                    if float(data[4]) > five_threshold[1]:
                        five_one_threshold = {1:0.0000, 3:0.5241, 0:0.0000}
                        if int(data[5]) == 1:
                            if float(data[6]) > five_one_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 3:
                            if float(data[6]) > five_one_threshold[3]:
                                names.append(data[0])
                        elif int(data[5]) == 0:
                            if float(data[6]) > five_one_threshold[0]:
                                names.append(data[0])
                elif int(data[3]) == 2:
                    if float(data[4]) > five_threshold[2]:
                        names.append(data[0])
                elif int(data[3]) == 3:
                    if float(data[4]) > five_threshold[3]:
                        names.append(data[0])
                elif int(data[3]) == 4:
                    if float(data[4]) > five_threshold[4]:
                        five_four_threshold = {99:0.0000, 2:0.0000, 3:0.0000, 0:0.0000}
                        if int(data[5]) == 99:
                            if float(data[6]) > five_four_threshold[99]:
                                names.append(data[0])
                        elif int(data[5]) == 2:
                            if float(data[6]) > five_four_threshold[2]:
                                names.append(data[0])
                        elif int(data[5]) == 3:
                            if float(data[6]) > five_four_threshold[3]:
                                names.append(data[0])
                        elif int(data[5]) == 0:
                            if float(data[6]) > five_four_threshold[0]:
                                names.append(data[0])
                elif int(data[3]) == 6:
                    if float(data[4]) > five_threshold[6]:
                        five_six_threshold = {1:0.0000, 2:0.0000}
                        if int(data[5]) == 1:
                            if float(data[6]) > five_six_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 2:
                            if float(data[6]) > five_six_threshold[2]:
                                names.append(data[0])
                elif int(data[3]) == 0:
                    if float(data[4]) > five_threshold[0]:
                        names.append(data[0])
        elif int(data[1]) == 6:
            if float(data[2]) > first_threshold[6]:
                six_threshold = {1:0.0000, 2:0.5186, 3:0.0000, 5:0.0000, 0:0.0000}
                if int(data[3]) == 1:
                    if float(data[4]) > six_threshold[1]:
                        names.append(data[0])
                elif int(data[3]) == 2:
                    if float(data[4]) > six_threshold[2]:
                        names.append(data[0])
                elif int(data[3]) == 3:
                    if float(data[4]) > six_threshold[3]:
                        six_three_threshold = {2:0.0000, 4:0.0000, 5:0.0000, 1:0.7897, 3:0.0000}
                        if int(data[5]) == 2:
                            if float(data[6]) > six_three_threshold[2]:
                                names.append(data[0])
                        elif int(data[5]) == 4:
                            if float(data[6]) > six_three_threshold[4]:
                                names.append(data[0])
                        elif int(data[5]) == 5:
                            if float(data[6]) > six_three_threshold[5]:
                                names.append(data[0])
                        elif int(data[5]) == 1:
                            if float(data[6]) > six_three_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 3:
                            if float(data[6]) > six_three_threshold[3]:
                                names.append(data[0])
                elif int(data[3]) == 5:
                    if float(data[4]) > six_threshold[5]:
                        names.append(data[0])
                elif int(data[3]) == 0:
                    if float(data[4]) > six_threshold[0]:
                        names.append(data[0])
        elif int(data[1]) == 7:
            if float(data[2]) > first_threshold[7]:
                seven_threshold = {1:0.0000, 2:0.7867, 3:0.3606, 4:4630, 6:0.8820, 0:0.6714}
                if int(data[3]) == 1:
                    if float(data[4]) > seven_threshold[1]:
                        seven_one_threshold = {1:0.0000, 2:0.0000, 0:0.0000}
                        if int(data[5]) == 1:
                            if float(data[6]) > seven_one_threshold[1]:
                                names.append(data[0])
                        elif int(data[5]) == 2:
                            if float(data[6]) > seven_one_threshold[2]:
                                names.append(data[0])
                        elif int(data[5]) == 0:
                            if float(data[6]) > seven_one_threshold[0]:
                                names.append(data[0])
                elif int(data[3]) == 2:
                    if float(data[4]) > seven_threshold[2]:
                        names.append(data[0])
                elif int(data[3]) == 3:
                    if float(data[4]) > seven_threshold[3]:
                        names.append(data[0])
                elif int(data[3]) == 4:
                    if float(data[4]) > seven_threshold[4]:
                        names.append(data[0])
                elif int(data[3]) == 6:
                    if float(data[4]) > seven_threshold[6]:
                        names.append(data[0])
                elif int(data[3]) == 0:
                    if float(data[4]) > seven_threshold[0]:
                        names.append(data[0])
    # Low confidence identification
    low_confidence_names = []
    for line in lines:
        data = line.split(",")
        if int(data[1]) == 1:
            if float(data[2]) <= first_threshold[1]:
                low_confidence_names.append(data[0])
        elif int(data[1]) == 2:
            if float(data[2]) <= first_threshold[2]:
                low_confidence_names.append(data[0])
        elif int(data[1]) == 3:
            if float(data[2]) <= first_threshold[3]:
                low_confidence_names.append(data[0])
        elif int(data[1]) == 4:
            if float(data[2]) <= first_threshold[4]:
                low_confidence_names.append(data[0])
        elif int(data[1]) == 5:
            if float(data[2]) <= first_threshold[5]:
                low_confidence_names.append(data[0])
        elif int(data[1]) == 6:
            if float(data[2]) <= first_threshold[6]:
                low_confidence_names.append(data[0])
        elif int(data[1]) == 7:
            if float(data[2]) <= first_threshold[7]:
                low_confidence_names.append(data[0])
    # Annotate and save results
    result = pd.read_csv(os.path.join(output, 'enzyme_result_with_probability.csv'))
    result['confidence'] = 'Medium'
    result.loc[result['Accession'].isin(names), 'confidence'] = 'High'
    result.loc[result['Accession'].isin(low_confidence_names), 'confidence'] = 'Low'
    # Format final output columns
    result = result[['Accession', 'FinalResult', 'confidence', 'accepted_name', 'other_names']]
    result.to_csv(os.path.join(output, 'enzyme_result_with_confidence.csv'), index=False)

if __name__ == '__main__':
    main()