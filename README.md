# FEDKEA: Enzyme Function Prediction using Fine-tuning ESM2 and Distance-based KNN

FEDKEA has been developed into a user-friendly web server, freely accessible at https://www.viromelab.com/software/fedkea.

## Description
A deep learning model for the enzyme annotation of proteins.
Two levels of annotation:
(1) Enzyme identification: enzyme or non-enzyme
(2) EC number prediction if one protein is an enzyme
## Steps to predict
### Step 1: Prepare the environment
(1) Download the FEDKEA software from github

``git clone https://github.com/Stevenleizheng/FEDKEA.git``

(2) Go to the directory of FEDKEA, for example:

``cd FEDKEA`` 

(3) Create a new conda environment, for example:

``conda create -n fedkea python=3.9``

(4) Enter the conda environment

``conda activate fedkea``

(5) Install the following software

a. pytorch:
If you want to use the CPU version, please go to https://pytorch.org/get-started and get the pip install command or run ``pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu``

If you want to use the GPU version, please go to https://pytorch.org/get-started and get the pip install command according to your device and demand.

b. fair-esm: ``pip install fair-esm``

c. pandas: ``conda install -c bioconda pandas`` or ``pip install pandas``

d. biopython: ``conda install -c bioconda biopython``

e. scikit-learn: ``pip install scikit-learn``

### Step 2: Download the trained model
(1) Download the model (The working path is still 'FEDKEA'). The model parameter file is saved at: https://zenodo.org/records/14868763. The file size is 5.6 GB.

``wget -c https://zenodo.org/records/14868763/files/model_param.tar.gz``

or

``wget -c https://zenodo.org/records/14868763/files/model_param.tar.gz?download=1``

(2) Unpack the file

``tar -xzvf model_param.tar.gz``

### (Optional) Step 3: Test the software
Run this command (a test prediction with 2877 proteins) to see whether the software has installed correctly using CPU.

``python main.py -i Testset/data/UniProt_202505_IsEnzyme.fasta -b 32``

If the software is installed correctly and completely, this step will finish in less than 60 minutes (might be longer if your device is too low) without any errors. The results of the test prediction will be saved in the result folder.

Run this command (a test prediction with 2877 proteins) to see whether the software has installed correctly using GPU (e.g. NVIDIA GeForce RTX 3090).

``python main.py -i Testset/data/UniProt_202505_IsEnzyme.fasta -g 0 -b 32``

If the software is installed correctly and completely, this step will finish in less than 15 minutes (might be longer if your device is too old) without any errors. This time estimate does not include the duration for downloading ESM2 parameters. The results of the test prediction will be saved in the result folder.

### Step 4: Prediction
#### (1) Preparations
Your proteins in a fasta file (path: xxx.fa).
A directory to save the output files (path: xxx/).
If you want to use GPU(s), please prepare the IDs of the GPU(s) you want to use, for example, a single-GPU machine, here it is prepared to be 0; multi-GPU machine using only one GPU, here it is prepared as x (x is the GPU ID used); multi-GPU machine using multiple GPUs, here it is prepared as x1,x2,... (x1,x2,... are the GPU IDs you want to use).

#### (2) Prediction

CPU: ``python main.py -i xxx.fa ``

single GPU machine: ``python main.py -i xxx.fa -g '0'``

multi GPU machine, using one GPU: ``python main.py -i xxx.fa -g 'x'``

multi GPU machine, using multi GPUs: ``python main.py -i xxx.fa -g 'x1,x2,...'``

-o determines the output directory, -g determines the IDs of GPUs you want to use (not given -g, will use CPU)

If you want to change the batch size (default is 2), please use -b, please note that the batch size cannot be negative and should not be smaller than the number of GPUs used.

If you want to change the threshold of the binary task (default is 0.5), please use -t. You can set the number between 0 and 1.
Example commands:

Predict proteins in 'example.fasta', save the results to 'result/', and the batch size is 64. The intermediate process data is saved in the 'data/' directory.

CPU: ``python main.py -i example.fa  -b 64 -d data/ -o result/``

single GPU machine: ``python main.py -i example.fa -g '0' -b 64 -d data/ -o result/``

multi GPU machine, using one GPU (ID:2): ``python main.py -i example.fa -g '2' -b 64 -d data/ -o result/`` 
 
multi GPU machine, using eight GPUs (ID:0-7): ``python main.py -i example.fa -g '0,1,2,3,4,5,6,7' -b 16 -d data/ -o result/`` 

The descriptions for the result files are in the 'binary_result.txt' and 'enzyme_result.csv' files of the output directory.

Additionally, our tool provides two parameters: one is -r, which controls the running mode. If -r is 1, it means enzyme identification and EC number prediction, and if -r is 2, it means only EC number prediction, not enzyme identification.

The other parameter is -t, which controls the threshold for binary classification models (ranging from 0 to 1, the default is 0.5). A higher value increases the confidence in the selected enzymes, while a lower value allows for the detection of more enzymes, but may also result in a higher rate of false positives. If the sequences are known to be enzymes, the parameter -t can be set to 0.

### Step 5: Result interpretation
The results will be saved in the output directory you specified. The results include: binary_result.txt, enzyme_result_with_probability.csv and enzyme_result_with_confidence.csv. 
Here's what each file contains:
- **binary_result.txt**: Contains binary classification results (enzyme/non-enzyme) with corresponding probabilities
- **enzyme_result_with_probability.csv**: Records EC number predictions with hierarchical probabilities. When the KNN model is used at any prediction level, this will directly output the complete EC number with probabilities set to 1, indicating the nearest EC neighbor's result is selected as the final prediction.
- **enzyme_result_with_confidence.csv**: Provides predicted EC numbers with confidence levels:
  - *High confidence*: When all hierarchical prediction probabilities exceed the threshold
  - *Low confidence*: When the first-level prediction probability falls below the threshold
  - *Medium confidence*: All other cases between high and low confidence thresholds
