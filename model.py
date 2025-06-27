import esm
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnzymeBinary(nn.Module):
    """Binary classification model for enzyme/non-enzyme prediction.
    
    Architecture:
    1. ESM-2 (650M parameters) as base model
    2. Five-layer MLP head with dimension reduction
    3. Final classification layer for binary decision
    """
    def __init__(self, n_class):
        """Initialize model with frozen ESM-2 and trainable classifier head.
        
        Args:
            n_class: Number of output classes (2 for binary classification)
        """
        super(EnzymeBinary, self).__init__()
        # Load pretrained ESM-2 model (33 layers, 650M params)
        self.esm = esm.pretrained.esm2_t33_650M_UR50D()[0]
        # Freeze all ESM parameters
        for name, param in self.named_parameters():
            param.requires_grad =False
        # Initialize classifier layers
        self.fc1 = nn.Linear(1280, 960)
        self.fc2 = nn.Linear(960, 480)
        self.fc3 = nn.Linear(480, 120)
        self.fc4 = nn.Linear(120, 30)
        self.fc5 = nn.Linear(30, n_class)
    def forward(self, batch_tokens):
        """Forward pass for enzyme classification.
        
        Args:
            batch_tokens: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Logits tensor of shape (batch_size, n_class)
            
        Processing Steps:
            1. Extract ESM embeddings from penultimate layer
            2. Mask padding tokens and average valid residues
            3. Pass through MLP with ReLU activations
        """
        x = self.esm(batch_tokens, repr_layers=[33], return_contacts=False)["representations"][33]
        batch_tokens = batch_tokens.unsqueeze(-1)
        x = x.masked_fill(batch_tokens==2, 0)
        x = x.masked_fill(batch_tokens==1, 0)[:, 1:, :]
        num = torch.sum(batch_tokens>2, axis=1)
        x = x.sum(axis=1) / num
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)       
        return x

class EsmEnzymeFirst(nn.Module):
    """First-level EC number classifier predicting main enzyme classes (1-7).
    
    Architecture:
    1. Frozen ESM-2 (650M params) as feature extractor
    2. Batch-normalized MLP with dimension reduction
    3. Final classification layer for 7 EC classes
    """
    def __init__(self, n_class=7): 
        """Initialize model with frozen ESM-2 and trainable classifier head.
        
        Args:
            n_class: Output classes (7 for EC first-level prediction)
        """
        super(EsmEnzymeFirst, self).__init__()
        # Load pretrained ESM-2 model
        self.esm = esm.pretrained.esm2_t33_650M_UR50D()[0]
        # Freeze ESM parameters
        for name, param in self.named_parameters():
            param.requires_grad =False
        # Initialize BN-enhanced classifier
        self.fc1 = nn.Linear(1280, 960)
        self.bn1 = nn.BatchNorm1d(960)
        self.fc2 = nn.Linear(960, 480)
        self.bn2 = nn.BatchNorm1d(480)
        self.fc3 = nn.Linear(480, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc4 = nn.Linear(120, n_class)
    def forward(self, batch_tokens):
        """Process input tokens through ESM and classifier.
        
        Args:
            batch_tokens: Input tensor (batch_size, seq_len)
            
        Returns:
            List containing:
            [0]: Pooled residue embeddings (batch_size, 1280)
            [1]: Final class logits (batch_size, 7)
            
        Processing:
            1. Extract ESM embeddings from layer 33
            2. Mask special tokens and average valid residues
            3. Process through BN-MLP layers
        """
        # Get ESM embeddings
        x = self.esm(batch_tokens, repr_layers=[33], return_contacts=False)["representations"][33]
        # Mask padding/special tokens and average
        batch_tokens = batch_tokens.unsqueeze(-1)
        x = x.masked_fill(batch_tokens<=2, 0) # Mask tokens 0,1,2
        num = torch.sum(batch_tokens>2, axis=1)
        x = x.sum(axis=1) / num # Average pooling       
        res = []
        res.append(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  
        res.append(x)
        return res

class EnzymeGeneral(nn.Module):
    """Batch-normalized MLP for hierarchical enzyme classification.
    
    Architecture:
    - Four-layer MLP with dimension reduction
    - Batch normalization and ReLU activation
    - Final linear projection for class logits
    
    Typical Use:
    Processes pooled protein embeddings for EC subclass prediction
    """
    def __init__(self, n_class):
        """Initialize BN-MLP classifier.
        
        Args:
            n_class: Number of target classes for final layer
        """
        super(EnzymeGeneral, self).__init__()
        self.fc1 = nn.Linear(1280, 960)
        self.bn1 = nn.BatchNorm1d(960)
        self.fc2 = nn.Linear(960, 480)
        self.bn2 = nn.BatchNorm1d(480)
        self.fc3 = nn.Linear(480, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc4 = nn.Linear(120, n_class)
    def forward(self, x):
        """Process input features through BN-MLP.
        
        Args:
            x: Input tensor of shape (batch_size, 1280)
            
        Returns:
            Logits tensor of shape (batch_size, n_class)
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)        
        return x