"""
models_partner_conditioned.py — Partner-conditioned peasant policy models.

This file extends the original DouZero model architecture with a
partner encoder that conditions landlord_up's policy on a summary
of landlord_down's recent behavior.

Architecture:
    FarmerPartnerConditionedModel:
        - LSTM encoder for action history: (162) → 128
        - Partner encoder MLP: (6) → 32
        - Concatenation: [lstm_128 | state_484 | partner_32] = 644
        - 6-layer value MLP: 644 → 512 → ... → 1

The partner encoder takes 6 behavioral features computed from
the partner's in-game actions (see partner_features.py):
    1. Pass frequency
    2. Mean cards per play
    3. Landlord blocking rate
    4. Cards remaining (normalized)
    5. Rounds since last play
    6. Initiative ratio
"""

import numpy as np
import torch
from torch import nn

from .models import LandlordLstmModel, FarmerLstmModel


# ================================================================
# Partner Encoder
# ================================================================

class PartnerEncoder(nn.Module):
    """
    Small MLP that encodes partner behavioral features into
    a dense embedding for policy conditioning.

    Input: 6-dim behavioral feature vector
    Output: 32-dim partner embedding
    """
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, partner_features):
        """
        Args:
            partner_features: [batch, 6] tensor of behavioral features
        Returns:
            [batch, 32] partner embedding
        """
        return self.net(partner_features)


# ================================================================
# Partner-Conditioned Farmer Model
# ================================================================

PARTNER_EMBED_DIM = 32
PARTNER_FEATURE_DIM = 6


class FarmerPartnerConditionedModel(nn.Module):
    """
    Modified FarmerLstmModel that conditions on partner behavior.

    The key difference from FarmerLstmModel:
    - Adds a PartnerEncoder that processes 6 partner behavioral features
    - Concatenates the 32-dim partner embedding with lstm_out and state
    - First dense layer input: 484 + 128 + 32 = 644 (was 484 + 128 = 612)
    - All other layers remain identical

    When partner_features is None, falls back to a zero embedding
    (equivalent to the original model with a constant offset).
    """
    def __init__(self):
        super().__init__()
        # Same LSTM as original FarmerLstmModel
        self.lstm = nn.LSTM(162, 128, batch_first=True)

        # NEW: Partner encoder
        self.partner_encoder = PartnerEncoder(
            input_dim=PARTNER_FEATURE_DIM,
            hidden_dim=32,
            output_dim=PARTNER_EMBED_DIM,
        )

        # Modified dense layers: input dim is now 484 + 128 + 32 = 644
        self.dense1 = nn.Linear(484 + 128 + PARTNER_EMBED_DIM, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, partner_features=None,
                return_value=False, flags=None):
        """
        Forward pass with optional partner conditioning.

        Args:
            z: [batch, 5, 162] historical action sequence for LSTM
            x: [batch, 484+54] state features + action candidate
            partner_features: [batch, 6] partner behavioral features,
                              or None (uses zeros)
            return_value: if True, return Q-values; else return action
            flags: training flags (for exploration epsilon)

        Returns:
            dict with 'values' or 'action'
        """
        # LSTM encoding of action history
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]  # Take last hidden state

        # Partner encoding
        if partner_features is not None:
            partner_embed = self.partner_encoder(partner_features)
        else:
            # Fallback: zero embedding (graceful degradation)
            batch_size = z.size(0)
            device = z.device
            partner_embed = torch.zeros(
                batch_size, PARTNER_EMBED_DIM, device=device
            )

        # Concatenate: [lstm_128 | state_484 | action_54 | partner_32]
        # Note: x already contains state+action concatenated by the caller
        x = torch.cat([lstm_out, x, partner_embed], dim=-1)

        # 6-layer value MLP (same structure as original)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)

        if return_value:
            return dict(values=x)
        else:
            if (flags is not None and flags.exp_epsilon > 0
                    and np.random.rand() < flags.exp_epsilon):
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x, dim=0)[0]
            return dict(action=action)


# ================================================================
# Model wrapper with partner-conditioned landlord_up
# ================================================================

class ModelPartnerConditioned:
    """
    Wrapper for the three DouZero models, with partner-conditioned
    landlord_up.

    - landlord: LandlordLstmModel (unchanged)
    - landlord_up: FarmerPartnerConditionedModel (NEW)
    - landlord_down: FarmerLstmModel (unchanged)

    The forward() method routes partner_features only to landlord_up.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device_str = 'cuda:' + str(device)
        else:
            device_str = 'cpu'
        dev = torch.device(device_str)

        self.models['landlord'] = LandlordLstmModel().to(dev)
        self.models['landlord_up'] = FarmerPartnerConditionedModel().to(dev)
        self.models['landlord_down'] = FarmerLstmModel().to(dev)

    def forward(self, position, z, x, training=False, flags=None,
                partner_features=None):
        """
        Forward pass through the appropriate position model.

        partner_features is only used when position == 'landlord_up'.
        """
        model = self.models[position]
        if position == 'landlord_up':
            return model.forward(z, x, partner_features=partner_features,
                                 return_value=training, flags=flags)
        else:
            return model.forward(z, x, return_value=training, flags=flags)

    def share_memory(self):
        for model in self.models.values():
            model.share_memory()

    def eval(self):
        for model in self.models.values():
            model.eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
