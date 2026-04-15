import torch
import numpy as np

from douzero.env.env import get_obs

def _load_model(position, model_path):
    from douzero.dmc.models import model_dict

    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')

    # Auto-detect partner-conditioned model by checking for the
    # partner_encoder keys or a dense1.weight with 644 columns
    is_partner_conditioned = False
    if any('partner_encoder' in k for k in pretrained.keys()):
        is_partner_conditioned = True
    elif 'dense1.weight' in pretrained and pretrained['dense1.weight'].shape[1] == 644:
        is_partner_conditioned = True

    if is_partner_conditioned:
        from douzero.dmc.models_partner_conditioned import FarmerPartnerConditionedModel
        model = FarmerPartnerConditionedModel()
    else:
        model = model_dict[position]()

    model_state_dict = model.state_dict()
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

class DeepAgent:

    def __init__(self, position, model_path):
        self.model = _load_model(position, model_path)

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = get_obs(infoset) 

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        # Partner-conditioned models accept partner_features=None gracefully
        # (they use a zero embedding as fallback during evaluation)
        y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]

        return best_action
