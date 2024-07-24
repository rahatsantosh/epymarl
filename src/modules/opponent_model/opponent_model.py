from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def calculate_multi_label_accuracy(predictions, targets, threshold=0.5):
    # Ensure both predictions and targets are tensors
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    
    # Check if both predictions and targets have the same shape
    assert predictions.shape == targets.shape, "Shapes of predictions and targets must match"

    # Apply the threshold to get predicted labels
    predicted_labels = (predictions >= threshold).float()
    
    # Calculate the number of correct predictions for each label
    correct_predictions = (predicted_labels == targets).float().sum(dim=1)
    
    # Calculate the accuracy per instance
    accuracy_per_instance = correct_predictions / targets.size(1)
    
    # Calculate the overall accuracy
    accuracy = accuracy_per_instance.mean().item()
    
    return accuracy

class OpponentDataset(Dataset):
    def __init__(self, args, batch, t):
        self.args = args
        self.ego_obs = self._build_inputs(batch, t)

        opp_obs_shape = list(batch['obs'].shape)
        opp_obs_shape[-1] *= (args.n_agents - 1)
        self.observations = torch.empty(*opp_obs_shape, dtype=batch['obs'].dtype)
        for i in range(args.n_agents):
            self.observations[:,:,i] = torch.cat((batch['obs'][:,:,:i], batch['obs'][:,:,i+1:]), dim=2).view(self.observations[:,:,i].shape)

        opp_act_shape = list(batch['actions'].shape)
        opp_act_shape[-1] *= (args.n_agents - 1)
        self.actions = torch.empty(*opp_act_shape, dtype=batch['actions'].dtype)
        for i in range(args.n_agents):
            self.actions[:,:,i] = torch.cat((batch['actions'][:,:,:i], batch['actions'][:,:,i+1:]), dim=2).view(self.actions[:,:,i].shape)
        
        self.ego_obs = torch.flatten(self.ego_obs, end_dim=-2)
        self.observations = torch.flatten(self.observations, end_dim=-2)
        self.actions = torch.flatten(self.actions, end_dim=-2)

    def __len__(self):
        return self.ego_obs.shape[0]

    def __getitem__(self, idx):
        return self.ego_obs[idx], self.observations[idx], self.actions[idx]#, self.rewards[idx]
    
    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(torch.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(torch.eye(self.args.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = torch.cat([x.reshape(bs*self.args.n_agents, -1) for x in inputs], dim=1)
        return inputs



class OpponentModel(nn.Module):
    def __init__(self, scheme, args):
        super(OpponentModel, self).__init__()
        self.args = args
        self.action_dim = scheme["actions_onehot"]["vshape"][0]
        input_shape = self._get_input_shape(scheme, args)
        reconstruction_dims_obs, reconstruction_dims_act = self._get_reconstruction_dims(scheme)
        
        self.encode = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, args.latent_dims),
        )
        self.decode = nn.Sequential(
            nn.Linear(args.latent_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.decode_obs_head = nn.Sequential(nn.Linear(64, int(reconstruction_dims_obs)))
        self.decode_act_head = nn.Sequential(nn.Linear(64, int(reconstruction_dims_act)))

        self.criterion = nn.HuberLoss()
        self.criterion_action = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr_opponent_modelling)
        
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        obs_head = self.decode_obs_head(decoded)
        act_head = self.decode_act_head(decoded)
        return obs_head, act_head
    
    def encoder(self, x):
        return self.encode(x)
    
    def batch_encoder(self, batch, t):
        x = self._build_inputs(batch, t)
        x_shape = list(x.shape)
        return self.encode(x.view(-1, x_shape[-1])).view(x_shape[0], x_shape[1], x_shape[2], -1)
    
    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        episode_size = batch["obs"].shape[1]
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(torch.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(torch.eye(self.args.n_agents, device=batch.device).unsqueeze(0).expand(bs, episode_size, -1, -1))

        inputs = torch.cat(inputs, dim=-1)
        return inputs
    
    def _get_input_shape(self, scheme, args):
        input_shape = scheme["obs"]["vshape"]
        if args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if args.obs_agent_id:
            input_shape += self.args.n_agents

        return input_shape
    
    def _get_reconstruction_dims(self, scheme):
        reconstruction_dim_obs = scheme["obs"]["vshape"] * (self.args.n_agents-1) # Observations
        reconstruction_dim_act = scheme["actions_onehot"]["vshape"][0] * (self.args.n_agents-1) # Actions
        return reconstruction_dim_obs, reconstruction_dim_act
    
    def learn(self, batch, logger, t_env, t, log_stats_t):
        dataset = OpponentDataset(self.args, batch, t)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size_opponent_modelling, shuffle=True)
        # Training loop
        for _ in range(self.args.opponent_model_epochs):
            for ego_obs, opp_obs, opp_acts in dataloader:
                self.optimizer.zero_grad()
                reconstructions_obs, reconstructions_act = self.forward(ego_obs)
                loss_obs, loss_act = 0.0, 0.0
                loss = 0.0
                if self.args.opponent_model_decode_observations:
                    loss_obs = self.criterion(reconstructions_obs, opp_obs)
                if self.args.opponent_model_decode_observations:
                    target_actions = torch.zeros(opp_acts.shape[0], opp_acts.shape[1] * self.action_dim).to(self.args.device)
                    for i in range(opp_acts.shape[1]):
                        target_actions.scatter_(1, opp_acts[:, i].unsqueeze(1) + i * self.action_dim, 1)
                    loss_act += self.criterion_action(reconstructions_act, target_actions)
                    accuracy = calculate_multi_label_accuracy(reconstructions_act, target_actions, self.args.opponent_action_threshold)
                loss = loss_obs + loss_act
                loss.backward()
                self.optimizer.step()

        if t_env - log_stats_t >= self.args.learner_log_interval:
            if self.args.opponent_model_decode_observations:
                logger.log_stat("opponent_model_loss_decode_observations", loss_obs.item(), t_env)
            if self.args.opponent_model_decode_actions:
                logger.log_stat("opponent_model_loss_decode_actions", loss_act.item(), t_env)
                logger.log_stat("opponent_model_decode_actions_accuracy", accuracy, t_env)
