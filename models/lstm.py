import torch
import torch.nn as nn
import torch.nn.functional as F

class fMRI_LSTM(nn.Module):
    """
    A two-layer LSTM model to process fMRI time-series data.
    """
    def __init__(self, input_features):
        super(fMRI_LSTM, self).__init__()
        
        # Layer 1: LSTM that returns sequences for the next layer
        self.lstm1 = nn.LSTM(
            input_size=input_features,
            hidden_size=int(input_features / 4),
            num_layers=1,
            batch_first=True  # Expects input shape (batch, seq, feature)
        )
        # ✅ ADD LAYER NORM 1
        self.ln1 = nn.LayerNorm(int(input_features / 4))
        
        # Layer 2: LSTM that returns only the output of the last time step
        self.lstm2 = nn.LSTM(
            input_size=int(input_features / 4),
            hidden_size=int(input_features / 16),
            num_layers=1,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(int(input_features / 16))
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(in_features=int(input_features / 16), out_features=128)
        self.relu = nn.ReLU(inplace=False)
        self.classifier = nn.Linear(in_features=128, out_features=4)


    def forward(self, x):
        """
        Defines the forward pass of the model.
        Args:
            x (torch.Tensor): The input fMRI data with shape 
                              (batch_size, sequence_length, input_features).
        """
        # Pass through the first LSTM layer
        # lstm1_out shape: (batch_size, sequence_length, hidden_size_1)
        lstm1_out, _ = self.lstm1(x)
        lstm1_out_norm = self.ln1(lstm1_out)
        
        # Pass the output sequence to the second LSTM layer
        # We only need the hidden state of the final time step
        lstm2_out, _ = self.lstm2(lstm1_out_norm)
        lstm2_out_norm = self.ln2(lstm2_out)

        last_time_step_out = lstm2_out_norm[:, -1, :]  # Shape: (batch_size, hidden_size_2)
        
        # Apply dropout, fully connected layer, and activation
        x = self.dropout(last_time_step_out)
        x = self.fc(x)
        x = self.relu(x)
        logits = self.classifier(x)
        
        
        return logits, x