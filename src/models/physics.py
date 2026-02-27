import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedModule(nn.Module):
    """
    Physics-Informed Module to enforce Tmax >= Tmin.
    Instead of predicting Tmax and Tmin directly, we predict:
    - T_avg (Average Temperature)
    - DTR (Diurnal Temperature Range) -> forced to be non-negative
    
    Then:
    Tmax = T_avg + DTR / 2
    Tmin = T_avg - DTR / 2
    """
    def __init__(self, input_dim):
        super(PhysicsInformedModule, self).__init__()
        # Input dim is usually the output of the MPT (bias corrections or latent features)
        # If input is bias correction features, we map them to T_avg correction and DTR correction
        self.fc_tavg = nn.Linear(input_dim, 1)
        self.fc_dtr = nn.Linear(input_dim, 1)
        self.fc_precip = nn.Linear(input_dim, 1) # Precipitation bias/correction

    def forward(self, x):
        """
        x: Latent representation from MPT
        """
        t_avg_correction = self.fc_tavg(x)
        dtr_correction = self.fc_dtr(x)
        
        # Enforce DTR >= 0 using Softplus or ReLU. 
        # Since this is a correction to DTR, we might want to apply it to the base DTR.
        # However, if this module outputs the final values (or offsets), we need to ensure consistency.
        # Let's assume this module outputs the *correction* terms. 
        # To strictly enforce Tmax >= Tmin on the *final* output, we usually predicting the *values* themselves 
        # or we ensure the base_DTR + dtr_correction >= 0.
        # For now, let's assume we are predicting the corrected T_avg and DTR directly for simplicity of this module's logic,
        # or we rely on the loss function to guide it. 
        # BUT the requirement says "Enforce consistency... via coupled equations".
        # So we will output DTR_pred and force it to be positive.
        
        dtr_pred = F.softplus(dtr_correction) # Ensure positive range
        precip_pred = F.softplus(self.fc_precip(x)) # Precipitation cannot be negative

        return t_avg_correction, dtr_pred, precip_pred

    def get_tmax_tmin(self, t_avg, dtr):
        tmax = t_avg + dtr / 2.0
        tmin = t_avg - dtr / 2.0
        return tmax, tmin
