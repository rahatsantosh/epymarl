from torch import nn
import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

def calculate_multi_label_accuracy(predictions, targets, action_dim):
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    predictions = torch.softmax(predictions.view(-1, predictions.shape[-1]//action_dim, action_dim), dim=2)
    predicted_labels = torch.argmax(predictions, dim=2)
    
    correct_predictions = (predicted_labels == targets).float().sum(dim=1)
    accuracy_per_instance = correct_predictions / targets.size(1)
    
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
        
        if self.args.opponent_model_decode_rewards:
            opp_rew_shape = list(batch['reward'].shape)
            opp_rew_shape.append(1)
            opp_rew_shape[-1] *= (args.n_agents - 1)
            self.rewards = torch.empty(*opp_rew_shape, dtype=batch['reward'].dtype)
            for i in range(args.n_agents):
                self.rewards[:,:,i] = torch.cat((batch['reward'][:,:,:i], batch['reward'][:,:,i+1:]), dim=2).view(self.rewards[:,:,i].shape)
            self.rewards = torch.flatten(self.rewards, end_dim=-2)
        
        self.ego_obs = torch.flatten(self.ego_obs, end_dim=-2)
        self.observations = torch.flatten(self.observations, end_dim=-2)
        self.actions = torch.flatten(self.actions, end_dim=-2)

    def __len__(self):
        return self.ego_obs.shape[0]

    def __getitem__(self, idx):
        if self.args.opponent_model_decode_rewards:
            return self.ego_obs[idx], self.observations[idx], self.actions[idx], self.rewards[idx]
        else:
            return self.ego_obs[idx], self.observations[idx], self.actions[idx], torch.zeros_like(self.actions[idx])
    
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
    
    def append_data(self, batch, t):
        new_ego_obs = self._build_inputs(batch, t)

        new_opp_obs_shape = list(batch['obs'].shape)
        new_opp_obs_shape[-1] *= (self.args.n_agents - 1)
        new_observations = torch.empty(*new_opp_obs_shape, dtype=batch['obs'].dtype)
        for i in range(self.args.n_agents):
            new_observations[:,:,i] = torch.cat((batch['obs'][:,:,:i], batch['obs'][:,:,i+1:]), dim=2).view(new_observations[:,:,i].shape)

        new_opp_act_shape = list(batch['actions'].shape)
        new_opp_act_shape[-1] *= (self.args.n_agents - 1)
        new_actions = torch.empty(*new_opp_act_shape, dtype=batch['actions'].dtype)
        for i in range(self.args.n_agents):
            new_actions[:,:,i] = torch.cat((batch['actions'][:,:,:i], batch['actions'][:,:,i+1:]), dim=2).view(new_actions[:,:,i].shape)
        
        if self.args.opponent_model_decode_rewards:
            new_opp_rew_shape = list(batch['reward'].shape)
            new_opp_rew_shape.append(1)
            new_opp_rew_shape[-1] *= (self.args.n_agents - 1)
            new_rewards = torch.empty(*new_opp_rew_shape, dtype=batch['reward'].dtype)
            for i in range(self.args.n_agents):
                new_rewards[:,:,i] = torch.cat((batch['reward'][:,:,:i], batch['reward'][:,:,i+1:]), dim=2).view(new_rewards[:,:,i].shape)
            new_rewards = torch.flatten(new_rewards, end_dim=-2)
            self.rewards = torch.cat((self.rewards, torch.flatten(new_rewards, end_dim=-2)), dim=0)
        
        self.ego_obs = torch.cat((self.ego_obs, torch.flatten(new_ego_obs, end_dim=-2)), dim=0)
        self.observations = torch.cat((self.observations, torch.flatten(new_observations, end_dim=-2)), dim=0)
        self.actions = torch.cat((self.actions, torch.flatten(new_actions, end_dim=-2)), dim=0)



class OpponentModel(nn.Module):
    def __init__(self, scheme, args):
        super(OpponentModel, self).__init__()
        self.args = args
        self.action_dim = scheme["actions_onehot"]["vshape"][0]
        input_shape = self._get_input_shape(scheme, args)
        reconstruction_dims_obs, reconstruction_dims_act, reconstruction_dims_rew = self._get_reconstruction_dims(scheme)
        
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

        if self.args.opponent_model_decode_observations:
            self.decode_obs_head = nn.Sequential(nn.Linear(64, int(reconstruction_dims_obs)))
        if self.args.opponent_model_decode_actions:
            self.decode_act_head = nn.Sequential(nn.Linear(64, int(reconstruction_dims_act)))
        if self.args.opponent_model_decode_rewards:
            self.decode_rew_head = nn.Sequential(nn.Linear(64, int(reconstruction_dims_rew)))

        self.criterion = nn.MSELoss()
        self.criterion_action = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr_opponent_modelling)
        # self.fisher = {}
        # self.prev_params = {}
        # self.importance = args.opponent_ewc_importance  # EWC importance scaling factor
        self.dataset = None
    
    # def compute_fisher_information(self, dataloader, device):
    #     self.eval()
        
    #     for name, param in self.named_parameters():
    #         self.fisher[name] = torch.zeros_like(param)
    #         self.prev_params[name] = param.clone()
        
    #     for data in dataloader:
    #         ego_obs, obs_, acts_, rew_ = data
    #         self.optimizer.zero_grad()
    #         obs, acts, rew = self.forward(ego_obs)
    #         loss_obs, loss_act, loss_rew = 0.0, 0.0, 0.0
    #         loss = 0.0
    #         if self.args.opponent_model_decode_observations:
    #             loss_obs = self.criterion(obs, obs_)
    #         if self.args.opponent_model_decode_rewards:
    #             loss_rew = self.criterion(rew, rew_)
    #         if self.args.opponent_model_decode_actions:
    #             target_actions = torch.zeros(acts_.shape[0], acts_.shape[1] * self.action_dim, device=device)
    #             for i in range(acts_.shape[1]):
    #                 target_actions.scatter_(1, acts_[:, i].unsqueeze(1) + i * self.action_dim, 1)
    #             loss_act = self.criterion_action(acts, target_actions)
    #         loss = loss_obs + loss_act + loss_rew
    #         loss.backward()
            
    #         for name, param in self.named_parameters():
    #             self.fisher[name] += torch.square(param.grad.detach()) / len(dataloader)
    #     self.optimizer.zero_grad()
    #     self.train()
    
    # def ewc_penalty(self):
    #     loss = 0.0
    #     for name, param in self.named_parameters():
    #         fisher = self.fisher[name].detach()
    #         prev_param = self.prev_params[name].detach()
    #         loss += (fisher * torch.square(param.detach() - prev_param)).sum()
    #     return self.importance * loss
        
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        if self.args.opponent_model_decode_observations:
            obs_head = self.decode_obs_head(decoded)
        else:
            obs_head = torch.zeros_like(decoded)
        if self.args.opponent_model_decode_actions:
            act_head = self.decode_act_head(decoded)
        else:
            act_head = torch.zeros_like(decoded)
        if self.args.opponent_model_decode_rewards:
            rew_head = self.decode_rew_head(decoded)
        else:
            rew_head = torch.zeros_like(decoded) 
        return obs_head, act_head, rew_head
    
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
        reconstruction_dim_rew = 1 * (self.args.n_agents-1) # Rewards
        return reconstruction_dim_obs, reconstruction_dim_act, reconstruction_dim_rew
    
    def learn(self, batch, logger, t_env, t, log_stats_t):    
        self.train()
        if self.dataset is None:
            self.dataset = OpponentDataset(self.args, batch, t)
        else:
            self.dataset.append_data(batch, t)
        
        if t_env - log_stats_t >= self.args.learner_log_interval:
            dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size_opponent_modelling, shuffle=True)

            # Compute Fisher Information using the current dataloader before training
            # self.compute_fisher_information(dataloader, batch.device)
            loss_act_, loss_obs_, loss_rew_, accuracy_, ewc_loss_ = [], [], [], [], []

            # Training loop
            for _ in range(self.args.opponent_model_epochs):
                for ego_obs, opp_obs, opp_acts, opp_rew in dataloader:
                    self.optimizer.zero_grad()
                    reconstructions_obs, reconstructions_act, reconstructions_rew = self.forward(ego_obs)
                    loss_obs, loss_act, loss_rew = 0.0, 0.0, 0.0
                    loss = 0.0
                    if self.args.opponent_model_decode_observations:
                        loss_obs = self.criterion(reconstructions_obs, opp_obs)
                        loss_obs_.append(loss_obs.item())
                    if self.args.opponent_model_decode_rewards:
                        loss_rew = self.criterion(reconstructions_rew, opp_rew)
                        loss_rew_.append(loss_rew.item())
                    if self.args.opponent_model_decode_actions:
                        accuracy = calculate_multi_label_accuracy(reconstructions_act, opp_acts, self.action_dim)
                        reconstructions_act = reconstructions_act.view(-1, reconstructions_act.shape[-1]//self.action_dim, self.action_dim)
                        reconstructions_act = torch.swapaxes(reconstructions_act, 1, 2)
                        loss_act = self.criterion_action(reconstructions_act, opp_acts)
                        loss_act_.append(loss_act.item())
                        accuracy_.append(accuracy)
                    loss = loss_obs + loss_act + loss_rew

                    # Add EWC penalty to the loss
                    # ewc_loss = self.ewc_penalty()
                    # loss += ewc_loss

                    loss.backward()
                    self.optimizer.step()

            if self.args.opponent_model_decode_observations:
                logger.log_stat("opponent_model_loss_decode_observations", np.mean(loss_obs_), t_env)
                logger.log_stat("opponent_model_loss_decode_observations_std", np.std(loss_obs_), t_env)
            if self.args.opponent_model_decode_rewards:
                logger.log_stat("opponent_model_loss_decode_rewards", np.mean(loss_rew_), t_env)
                logger.log_stat("opponent_model_loss_decode_rewards_std", np.std(loss_rew_), t_env)
            if self.args.opponent_model_decode_actions:
                logger.log_stat("opponent_model_loss_decode_actions", np.mean(loss_act_), t_env)
                logger.log_stat("opponent_model_loss_decode_actions_std", np.std(loss_act_), t_env)
                logger.log_stat("opponent_model_decode_actions_accuracy", np.mean(accuracy_), t_env)
                logger.log_stat("opponent_model_decode_actions_accuracy_std", np.std(accuracy_), t_env)
            # logger.log_stat("opponent_model_decode_actions_accuracy", ewc_loss, t_env)
            self.dataset = None
