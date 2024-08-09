from torch import nn
import torch as th
import torch.nn.functional as F
import torch.distributions as distributions
from torch.utils.data import Dataset, DataLoader
import numpy as np

def calculate_multi_label_accuracy(predictions, targets, action_dim):
    if not isinstance(predictions, th.Tensor):
        predictions = th.tensor(predictions)
    if not isinstance(targets, th.Tensor):
        targets = th.tensor(targets)
    predictions = th.softmax(predictions.view(-1, predictions.shape[-1]//action_dim, action_dim), dim=2)
    predicted_labels = th.argmax(predictions, dim=2)
    
    correct_predictions = (predicted_labels == targets).float().sum(dim=1)
    accuracy_per_instance = correct_predictions / targets.size(1)
    
    accuracy = accuracy_per_instance.mean().item()
    
    return accuracy

def calculate_entropy(action_probs):
    # action_probs = th.softmax(action_probs.detach(), dim=2)
    dist = distributions.Categorical(probs=action_probs.detach())
    entropy = dist.entropy()
    
    return entropy

class OpponentDataset(Dataset):
    def __init__(self, args, batch, t):
        self.args = args
        self.ego_obs, self.ego_reward = self._build_inputs(batch, t)

        opp_obs_shape = list(batch['obs'].shape)
        opp_obs_shape[-1] *= (args.n_agents - 1)
        self.observations = th.empty(*opp_obs_shape, dtype=batch['obs'].dtype)
        for i in range(args.n_agents):
            self.observations[:,:,i] = th.cat((batch['obs'][:,:,:i], batch['obs'][:,:,i+1:]), dim=2).view(self.observations[:,:,i].shape)

        opp_act_shape = list(batch['actions'].shape)
        opp_act_shape[-1] *= (args.n_agents - 1)
        self.actions = th.empty(*opp_act_shape, dtype=batch['actions'].dtype)
        for i in range(args.n_agents):
            self.actions[:,:,i] = th.cat((batch['actions'][:,:,:i], batch['actions'][:,:,i+1:]), dim=2).view(self.actions[:,:,i].shape)
        
        opp_act_shape = list(batch['actions_onehot'].shape)
        opp_act_shape[-1] *= (args.n_agents - 1)
        self.action_logits = th.empty(*opp_act_shape, dtype=batch['actions_onehot'].dtype)
        for i in range(args.n_agents):
            self.action_logits[:,:,i] = th.cat((batch['actions_onehot'][:,:,:i], batch['actions_onehot'][:,:,i+1:]), dim=2).view(self.action_logits[:,:,i].shape)
        
        if self.args.opponent_model_decode_rewards:
            opp_rew_shape = list(batch['reward'].shape)
            opp_rew_shape.append(1)
            opp_rew_shape[-1] *= (args.n_agents - 1)
            self.rewards = th.empty(*opp_rew_shape, dtype=batch['reward'].dtype)
            for i in range(args.n_agents):
                self.rewards[:,:,i] = th.cat((batch['reward'][:,:,:i], batch['reward'][:,:,i+1:]), dim=2).view(self.rewards[:,:,i].shape)
            self.rewards = th.flatten(self.rewards, end_dim=-2)
        
        self.ego_obs = th.flatten(self.ego_obs, end_dim=-2)
        self.observations = th.flatten(self.observations, end_dim=-2)
        self.actions = th.flatten(self.actions, end_dim=-2)
        self.action_logits = th.flatten(self.action_logits, end_dim=-2)

    def __len__(self):
        return self.ego_obs.shape[0]

    def __getitem__(self, idx):
        if self.args.opponent_model_decode_rewards:
            return self.ego_obs[idx], self.observations[idx], self.actions[idx], self.rewards[idx], self.ego_reward[idx], self.action_logits[idx]
        else:
            return self.ego_obs[idx], self.observations[idx], self.actions[idx], th.zeros_like(self.actions[idx]), self.ego_reward[idx], self.action_logits[idx]
    
    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        rewards = [batch["reward"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.args.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.args.n_agents, -1) for x in inputs], dim=1)
        if self.args.opponent_model_decode_rewards:
            rewards = th.cat([x.reshape(bs*self.args.n_agents, -1) for x in rewards], dim=1)
        else:
            rewards = (th.cat([x.reshape(bs, -1) for x in rewards], dim=1)).repeat_interleave(self.args.n_agents, dim=0)
        return inputs, rewards
    
    def append_data(self, batch, t):
        new_ego_obs, new_ego_rewards = self._build_inputs(batch, t)

        new_opp_obs_shape = list(batch['obs'].shape)
        new_opp_obs_shape[-1] *= (self.args.n_agents - 1)
        new_observations = th.empty(*new_opp_obs_shape, dtype=batch['obs'].dtype)
        for i in range(self.args.n_agents):
            new_observations[:,:,i] = th.cat((batch['obs'][:,:,:i], batch['obs'][:,:,i+1:]), dim=2).view(new_observations[:,:,i].shape)

        new_opp_act_shape = list(batch['actions'].shape)
        new_opp_act_shape[-1] *= (self.args.n_agents - 1)
        new_actions = th.empty(*new_opp_act_shape, dtype=batch['actions'].dtype)
        for i in range(self.args.n_agents):
            new_actions[:,:,i] = th.cat((batch['actions'][:,:,:i], batch['actions'][:,:,i+1:]), dim=2).view(new_actions[:,:,i].shape)
        
        if self.args.opponent_model_decode_rewards:
            new_opp_rew_shape = list(batch['reward'].shape)
            new_opp_rew_shape.append(1)
            new_opp_rew_shape[-1] *= (self.args.n_agents - 1)
            new_rewards = th.empty(*new_opp_rew_shape, dtype=batch['reward'].dtype)
            for i in range(self.args.n_agents):
                new_rewards[:,:,i] = th.cat((batch['reward'][:,:,:i], batch['reward'][:,:,i+1:]), dim=2).view(new_rewards[:,:,i].shape)
            new_rewards = th.flatten(new_rewards, end_dim=-2)
            self.rewards = th.cat((self.rewards, th.flatten(new_rewards, end_dim=-2)), dim=0)
        
        self.ego_obs = th.cat((self.ego_obs, th.flatten(new_ego_obs, end_dim=-2)), dim=0)
        self.ego_reward = th.cat((self.ego_reward, th.flatten(new_ego_rewards, end_dim=-2)), dim=0)
        self.observations = th.cat((self.observations, th.flatten(new_observations, end_dim=-2)), dim=0)
        self.actions = th.cat((self.actions, th.flatten(new_actions, end_dim=-2)), dim=0)



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
            # nn.Dropout(p=0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(64, args.latent_dims),
        )
        # self.encode_mean = nn.Linear(64, args.latent_dims)
        # self.encode_std = nn.Linear(64, args.latent_dims)
        self.decode = nn.Sequential(
            nn.Linear(args.latent_dims, 64),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
        )

        if self.args.opponent_model_decode_observations:
            self.decode_obs_head = nn.Sequential(nn.Linear(64, int(reconstruction_dims_obs)))
        if self.args.opponent_model_decode_actions:
            self.decode_act_head = nn.Sequential(nn.Linear(64, int(reconstruction_dims_act)))
        if self.args.opponent_model_decode_rewards:
            self.decode_rew_head = nn.Sequential(nn.Linear(64, int(reconstruction_dims_rew)))

        self.criterion = nn.MSELoss()
        self.criterion_action = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss()
        self.optimizer = th.optim.Adam(self.parameters(), lr=args.lr_opponent_modelling)
        self.dataset = None
    
    # def reparameterize(self, mu, logvar):
    #     std = th.exp(0.5 * logvar)
    #     eps = th.randn_like(std)
    #     return mu + eps * std
        
    def forward(self, x):
        encoded = self.encode(x)
        # mu = self.encode_mean(encoded)
        # logvar = self.encode_std(encoded)
        # z = self.reparameterize(mu, logvar)
        decoded = self.decode(encoded)
        if self.args.opponent_model_decode_observations:
            obs_head = self.decode_obs_head(decoded)
        else:
            obs_head = th.zeros_like(decoded)
        if self.args.opponent_model_decode_actions:
            act_head = self.decode_act_head(decoded)
        else:
            act_head = th.zeros_like(decoded)
        if self.args.opponent_model_decode_rewards:
            rew_head = self.decode_rew_head(decoded)
        else:
            rew_head = th.zeros_like(decoded) 
        return obs_head, act_head, rew_head
    
    def encoder(self, x):
        self.eval()
        encoded = self.encode(x)
        # mu = self.encode_mean(encoded)
        # logvar = self.encode_std(encoded)
        # z = self.reparameterize(mu, logvar)
        return encoded
    
    def batch_encoder(self, batch, t):
        self.eval()
        x = self._build_inputs(batch, t)
        x_shape = list(x.shape)
        encoded = self.encode(x.view(-1, x_shape[-1])).view(x_shape[0], x_shape[1], x_shape[2], -1)
        # mu = self.encode_mean(encoded)
        # logvar = self.encode_std(encoded)
        # z = self.reparameterize(mu, logvar)
        return encoded
    
    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        episode_size = batch["obs"].shape[1]
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.args.n_agents, device=batch.device).unsqueeze(0).expand(bs, episode_size, -1, -1))

        inputs = th.cat(inputs, dim=-1)
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
        # if self.dataset is None:
        self.dataset = OpponentDataset(self.args, batch, t)
        # else:
        #     self.dataset.append_data(batch, t)
        
        dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size_opponent_modelling, shuffle=True)

        loss_act_, loss_obs_, loss_rew_, accuracy_ = [], [], [], []
        entropy = []

        # Training loop
        for _ in range(self.args.opponent_model_epochs):
            for ego_obs, opp_obs, opp_acts, opp_rew, ego_rew, action_logits in dataloader:
                self.optimizer.zero_grad()
                reconstructions_obs, reconstructions_act, reconstructions_rew = self.forward(ego_obs)
                loss_obs, loss_act, loss_rew = 0.0, 0.0, 0.0
                loss = 0.0
                if self.args.opponent_model_decode_observations:
                    loss_obs = self.criterion(reconstructions_obs, opp_obs).float()
                    loss_obs_.append(loss_obs.item())
                if self.args.opponent_model_decode_rewards:
                    loss_rew = self.criterion(reconstructions_rew, opp_rew).float()
                    loss_rew_.append(loss_rew.item())
                if self.args.opponent_model_decode_actions:
                    accuracy = calculate_multi_label_accuracy(reconstructions_act, opp_acts, self.action_dim)
                    reconstruction_shape = reconstructions_act.shape
                    reconstructions_act = reconstructions_act.view(-1, reconstructions_act.shape[-1]//self.action_dim, self.action_dim)
                    reconstructions_act = th.softmax(reconstructions_act, dim=2)
                    entropy.extend(calculate_entropy(reconstructions_act).view(-1))
                    reconstructions_act = reconstructions_act.view(reconstruction_shape)
                    # reconstructions_act = th.swapaxes(reconstructions_act, 1, 2)
                    # loss_act = self.criterion_action(reconstructions_act, opp_acts).float()
                    # target_acts = F.one_hot(opp_acts, num_classes=self.action_dim).view(-1, opp_acts.shape[-1]*self.action_dim).float()
                    # loss_act = self.criterion(reconstructions_act, action_logits).float()
                    loss_act = F.kl_div(reconstructions_act.log(), action_logits, reduction='batchmean')
                    loss_act_.append(loss_act.item())
                    accuracy_.append(accuracy)
                # if th.any(ego_rew != 0.0):
                #     print(ego_rew)
                # encoded_norm = th.norm(encoded, dim=-1)
                # reg_loss = th.mean(th.relu(1.0 - encoded_norm))
                loss = loss_obs + loss_act + loss_rew #+ 0.1*(reg_loss-th.mean(ego_rew * encoded).float())

                loss.backward()
                self.optimizer.step()

        if t_env - log_stats_t >= self.args.learner_log_interval:
            if self.args.opponent_model_decode_observations:
                logger.log_stat("opponent_model_loss_decode_observations", np.mean(loss_obs_), t_env)
                logger.log_stat("opponent_model_loss_decode_observations_std", np.std(loss_obs_), t_env)
            if self.args.opponent_model_decode_rewards:
                logger.log_stat("opponent_model_loss_decode_rewards", np.mean(loss_rew_), t_env)
                logger.log_stat("opponent_model_loss_decode_rewards_std", np.std(loss_rew_), t_env)
            if self.args.opponent_model_decode_actions:
                logger.log_stat("opponent_model_loss_decode_actions", np.mean(loss_act_), t_env)
                logger.log_stat("opponent_model_loss_decode_actions_std", np.std(loss_act_), t_env)
                logger.log_stat("opponent_model_entropy_decode_actions", np.mean(entropy), t_env)
                logger.log_stat("opponent_model_entropy_decode_actions_std", np.std(entropy), t_env)
                logger.log_stat("opponent_model_decode_actions_accuracy", np.mean(accuracy_), t_env)
                logger.log_stat("opponent_model_decode_actions_accuracy_std", np.std(accuracy_), t_env)
            # self.dataset = None


class OpponentModelNS(nn.Module):
    def __init__(self, scheme, args):
        super(OpponentModel, self).__init__()
        self.args = args
        self.action_dim = scheme["actions_onehot"]["vshape"][0]
        
        self.models = th.nn.ModuleList(
            [OpponentModel(scheme, args) for _ in range(args.n_agents)]
        )

        self.dataset = None
        
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        if self.args.opponent_model_decode_observations:
            obs_head = self.decode_obs_head(decoded)
        else:
            obs_head = th.zeros_like(decoded)
        if self.args.opponent_model_decode_actions:
            act_head = self.decode_act_head(decoded)
        else:
            act_head = th.zeros_like(decoded)
        if self.args.opponent_model_decode_rewards:
            rew_head = self.decode_rew_head(decoded)
        else:
            rew_head = th.zeros_like(decoded) 
        return obs_head, act_head, rew_head
    
    def encoder(self, x):
        return self.encode(x)
    
    def batch_encoder(self, batch, t):
        x = self._build_inputs(batch, t)
        x_shape = list(x.shape)
        return self.encode(x.view(-1, x_shape[-1])).view(x_shape[0], x_shape[1], x_shape[2], -1)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
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

            loss_act_, loss_obs_, loss_rew_, accuracy_ = [], [], [], []

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
                        reconstructions_act = th.swapaxes(reconstructions_act, 1, 2)
                        loss_act = self.criterion_action(reconstructions_act, opp_acts)
                        loss_act_.append(loss_act.item())
                        accuracy_.append(accuracy)
                    loss = loss_obs + loss_act + loss_rew

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
            self.dataset = None
