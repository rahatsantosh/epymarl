from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def calculate_accuracy(predictions, targets):
    """
    Calculate the accuracy of predictions against the targets.
    
    Parameters:
    - predictions: torch.Tensor of shape (batch_size, num_classes), the predicted one-hot encoded vectors
    - targets: torch.Tensor of shape (batch_size, num_classes), the true one-hot encoded vectors

    Returns:
    - accuracy: float, the accuracy of the predictions
    """
    # Convert one-hot encoded vectors to class indices
    predicted_classes = torch.argmax(predictions, dim=1)
    true_classes = torch.argmax(targets, dim=1)
    
    # Calculate the number of correct predictions
    correct_predictions = (predicted_classes == true_classes).sum().item()
    
    # Calculate accuracy
    accuracy = correct_predictions / targets.size(0)
    
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
        
        # opp_reward_shape = list(batch['reward'].shape)
        # opp_reward_shape[-1] *= (args.n_agents - 1)
        # self.rewards = torch.empty(*opp_reward_shape, dtype=batch['reward'].dtype)
        # for i in range(args.n_agents):
        #     print(batch['reward'].shape)
        #     exit(0);
        #     a = torch.cat((batch['reward'][:,:,:i], batch['reward'][:,:,i+1:]), dim=2).view(self.rewards[:,:,i].shape)
        #     print(self.rewards[:,:,i].shape, a.shape)
        #     exit(0);
        #     self.rewards[:,:,i] = a
        
        self.ego_obs = torch.flatten(self.ego_obs, end_dim=-2)
        self.observations = torch.flatten(self.observations, end_dim=-2)
        self.actions = torch.flatten(self.actions, end_dim=-2)
        # self.rewards = torch.flatten(self.rewards, end_dim=2)

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
        input_shape = self._get_input_shape(scheme, args)
        reconstruction_dims = self._get_reconstruction_dims(scheme)
        
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

        self.decode_head1 = nn.Sequential(nn.Linear(64, int(reconstruction_dims[0])))
        self.decode_head2 = nn.Sequential(nn.Linear(64, int(reconstruction_dims[1])))
        # self.decode_head3 = nn.Sequential(nn.Linear(64, int(reconstruction_dims[2])))

        self.criterion = nn.HuberLoss()
        self.criterion_action = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        head1 = self.decode_head1(decoded)
        head2 = self.decode_head2(decoded)
        # head3 = self.decode_head3(decoded)
        return head1, head2#, head3
    
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
        reconstruction_dims = [
            scheme["obs"]["vshape"] * (self.args.n_agents-1), # Observations
            scheme["actions_onehot"]["vshape"][0] * (self.args.n_agents-1), # Actions
            # 1 * (self.args.n_agents-1), # Rewards
        ]
        return reconstruction_dims
    
    def learn(self, batch, logger, t_env, t, log_stats_t):
        dataset = OpponentDataset(self.args, batch, t)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        # Training loop
        for _ in range(self.args.epochs):
            for ego_obs, opp_obs, opp_acts in dataloader:
                self.optimizer.zero_grad()
                reconstructions1, reconstructions2 = self.forward(ego_obs)
                loss = 0.0
                if 'observation' in self.args.opponent_modelling:
                    loss += self.criterion(reconstructions1, opp_obs)
                if 'action' in self.args.opponent_modelling:
                    accuracy = calculate_accuracy(reconstructions2, opp_acts)
                    loss += self.criterion_action(reconstructions2, torch.argmax(opp_acts, dim=1))
                # if 'reward' in self.args.opponent_modelling:
                #     loss += self.criterion(reconstructions3, opp_rewards)
                loss.backward()
                self.optimizer.step()

        if t_env - log_stats_t >= self.args.learner_log_interval:
            logger.log_stat("opponent_model_loss", loss.item(), t_env)
            if 'action' in self.args.opponent_modelling:
                logger.log_stat("opponent_model_action_accuracy", accuracy, t_env)
