import torch.nn as nn
import torch
class SingleClassWrapper(nn.Module):
    """Extracts one class from multi-class model output"""
    def __init__(self, model, class_id):
        super().__init__()
        self.model = model
        self.class_id = class_id
    
    def forward(self, x, test_svm=False):
        y_pred, alpha, beta, gamma, fSCT = self.model(x, test_svm=test_svm)
        if self.class_id == 'obj':
            y_pred = y_pred[:, 1:5]
        elif self.class_id == 'subj':
            y_pred_1 = y_pred[:, 0:1]
            y_pred_2 = y_pred[:, 5:15]
            y_pred = torch.cat((y_pred_1, y_pred_2), dim=1)
        else:
            self.class_id = int(self.class_id)
            y_pred = y_pred[:, self.class_id:self.class_id+1]  # Extract one class
        return y_pred, alpha, beta, gamma, fSCT
    
    def __getattr__(self, name):
        """Proxy attribute access to wrapped model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            # If attribute not found in wrapper, get it from wrapped model
            return getattr(self.model, name)