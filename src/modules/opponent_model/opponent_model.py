from torch import nn
import torch as th
import torch.nn.functional as F
import torch.distributions as distributions
from torch.utils.data import Dataset, DataLoader
import numpy as np

def calculate_multi_label_accuracy(predictions, targets, action_dim):
    """
    Calculates the accuracy of multi-label classification predictions.

    Args:
        predictions (torch.Tensor): The raw model outputs.
        targets (torch.Tensor): The ground truth labels.
        action_dim (int): The number of possible actions.

    Returns:
        float: The average accuracy across all instances.
    """
    if not isinstance(predictions, th.Tensor):
        predictions = th.tensor(predictions)
    if not isinstance(targets, th.Tensor):
        targets = th.tensor(targets)

    # Apply softmax to the predictions and reshape them
    predictions = th.softmax(predictions.view(-1, predictions.shape[-1]//action_dim, action_dim), dim=2)
    
    # Get the predicted labels by finding the indices of the maximum values
    predicted_labels = th.argmax(predictions, dim=2)
    
    # Calculate the number of correct predictions
    correct_predictions = (predicted_labels == targets).float().sum(dim=1)
    accuracy_per_instance = correct_predictions / targets.size(1)
    
    # Calculate the mean accuracy across all instances
    accuracy = accuracy_per_instance.mean().item()
    
    return accuracy

def calculate_entropy(action_probs):
    """
    Calculates the entropy of the action probabilities.

    Args:
        action_probs (torch.Tensor): The probability distribution over actions.

    Returns:
        torch.Tensor: The entropy of the distribution.
    """
    dist = distributions.Categorical(probs=action_probs.detach())
    entropy = dist.entropy()
    
    return entropy

class OpponentDataset(Dataset):
    """
    Dataset class for handling opponent observations, actions, and rewards.

    Args:
        args (Namespace): Configuration and arguments for dataset creation.
        batch (dict): Batch of data containing observations, actions, etc.
        t (int): Time step index.
    """
    def __init__(self, args, batch, t):
        self.args = args
        self.ego_obs, self.ego_reward = self._build_inputs(batch, t)

        # Prepare opponent observations by concatenating non-ego observations
        opp_obs_shape = list(batch['obs'].shape)
        opp_obs_shape[-1] *= (args.n_agents - 1)
        self.observations = th.empty(*opp_obs_shape, dtype=batch['obs'].dtype)
        for i in range(args.n_agents):
            self.observations[:,:,i] = th.cat((batch['obs'][:,:,:i], batch['obs'][:,:,i+1:]), dim=2).view(self.observations[:,:,i].shape)

        # Prepare opponent actions by concatenating non-ego actions
        opp_act_shape = list(batch['actions'].shape)
        opp_act_shape[-1] *= (args.n_agents - 1)
        self.actions = th.empty(*opp_act_shape, dtype=batch['actions'].dtype)
        for i in range(args.n_agents):
            self.actions[:,:,i] = th.cat((batch['actions'][:,:,:i], batch['actions'][:,:,i+1:]), dim=2).view(self.actions[:,:,i].shape)
        
        # Prepare opponent action logits by concatenating non-ego action logits
        opp_act_shape = list(batch['actions_onehot'].shape)
        opp_act_shape[-1] *= (args.n_agents - 1)
        self.action_logits = th.empty(*opp_act_shape, dtype=batch['actions_onehot'].dtype)
        for i in range(args.n_agents):
            self.action_logits[:,:,i] = th.cat((batch['actions_onehot'][:,:,:i], batch['actions_onehot'][:,:,i+1:]), dim=2).view(self.action_logits[:,:,i].shape)
        
        # Prepare opponent rewards if specified
        if self.args.opponent_model_decode_rewards:
            opp_rew_shape = list(batch['reward'].shape)
            opp_rew_shape.append(1)
            opp_rew_shape[-1] *= (args.n_agents - 1)
            self.rewards = th.empty(*opp_rew_shape, dtype=batch['reward'].dtype)
            for i in range(args.n_agents):
                self.rewards[:,:,i] = th.cat((batch['reward'][:,:,:i], batch['reward'][:,:,i+1:]), dim=2).view(self.rewards[:,:,i].shape)
            self.rewards = th.flatten(self.rewards, end_dim=-2)
        
        # Flatten inputs for efficient processing
        self.ego_obs = th.flatten(self.ego_obs, end_dim=-2)
        self.observations = th.flatten(self.observations, end_dim=-2)
        self.actions = th.flatten(self.actions, end_dim=-2)
        self.action_logits = th.flatten(self.action_logits, end_dim=-2)

    def __len__(self):
        """
        Returns the number of instances in the dataset.

        Returns:
            int: Number of instances.
        """
        return self.ego_obs.shape[0]

    def __getitem__(self, idx):
        """
        Returns the data instance at the given index.

        Args:
            idx (int): Index of the data instance.

        Returns:
            tuple: Data tuple containing ego observations, opponent observations, actions, rewards, and action logits.
        """
        if self.args.opponent_model_decode_rewards:
            return self.ego_obs[idx], self.observations[idx], self.actions[idx], self.rewards[idx], self.ego_reward[idx], self.action_logits[idx]
        else:
            return self.ego_obs[idx], self.observations[idx], self.actions[idx], th.zeros_like(self.actions[idx]), self.ego_reward[idx], self.action_logits[idx]
    
    def _build_inputs(self, batch, t):
        """
        Builds the inputs for the dataset from the batch at time step t.

        Args:
            batch (dict): Batch of data containing observations, actions, etc.
            t (int): Time step index.

        Returns:
            tuple: Processed inputs and rewards for the ego agent.
        """
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
        """
        Appends new data from the batch at time step t to the dataset.

        Args:
            batch (dict): Batch of data containing observations, actions, etc.
            t (int): Time step index.
        """
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
    """
    Neural network model for opponent modeling, utilizing autoencoder architecture.

    Args:
        scheme (dict): Scheme of the data containing input shapes and other configurations.
        args (Namespace): Configuration and arguments for model building and training.
    """
    def __init__(self, scheme, args):
        super(OpponentModel, self).__init__()
        self.args = args
        self.action_dim = scheme["actions_onehot"]["vshape"][0]
        input_shape = self._get_input_shape(scheme, args)
        reconstruction_dims_obs, reconstruction_dims_act, reconstruction_dims_rew = self._get_reconstruction_dims(scheme)
        
        # Encoder network
        self.encode = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, args.latent_dims),
        )

        # Decoder network with multiple heads for observations, actions, and rewards
        self.decode = nn.Sequential(
            nn.Linear(args.latent_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        if self.args.opponent_model_decode_observations:
            self.decode_obs_head = nn.Sequential(nn.Linear(64, int(reconstruction_dims_obs)))
        if self.args.opponent_model_decode_actions:
            self.decode_act_head = nn.Sequential(nn.Linear(64, int(reconstruction_dims_act)))
        if self.args.opponent_model_decode_rewards:
            self.decode_rew_head = nn.Sequential(nn.Linear(64, int(reconstruction_dims_rew)))

        self.criterion = nn.MSELoss()
        self.optimizer = th.optim.Adam(self.parameters(), lr=args.lr_opponent_modelling)
        self.dataset = None

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            tuple: Reconstructed observations, actions, and rewards.
        """
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
        """
        Encodes the input data into a latent representation.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Encoded latent representation.
        """
        self.eval()
        encoded = self.encode(x)
        return encoded
    
    def batch_encoder(self, batch, t):
        """
        Encodes a batch of data into latent representations.

        Args:
            batch (dict): Batch of data containing observations, actions, etc.
            t (int): Time step index.

        Returns:
            torch.Tensor: Encoded latent representations for the batch.
        """
        self.eval()
        x = self._build_inputs(batch, t)
        x_shape = list(x.shape)
        encoded = self.encode(x.view(-1, x_shape[-1])).view(x_shape[0], x_shape[1], x_shape[2], -1)
        return encoded
    
    def _build_inputs(self, batch, t):
        """
        Builds the inputs for the model from the batch at time step t.

        Args:
            batch (dict): Batch of data containing observations, actions, etc.
            t (int): Time step index.

        Returns:
            torch.Tensor: Processed input tensor.
        """
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
        """
        Calculates the input shape for the model based on the data scheme.

        Args:
            scheme (dict): Scheme of the data containing input shapes and other configurations.
            args (Namespace): Configuration and arguments for model building and training.

        Returns:
            int: The calculated input shape.
        """
        input_shape = scheme["obs"]["vshape"]
        if args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if args.obs_agent_id:
            input_shape += self.args.n_agents

        return input_shape
    
    def _get_reconstruction_dims(self, scheme):
        """
        Determines the reconstruction dimensions for observations, actions, and rewards.

        Args:
            scheme (dict): Scheme of the data containing input shapes and other configurations.

        Returns:
            tuple: Dimensions for reconstructing observations, actions, and rewards.
        """
        reconstruction_dim_obs = scheme["obs"]["vshape"] * (self.args.n_agents-1)
        reconstruction_dim_act = scheme["actions_onehot"]["vshape"][0] * (self.args.n_agents-1)
        reconstruction_dim_rew = 1 * (self.args.n_agents-1)
        return reconstruction_dim_obs, reconstruction_dim_act, reconstruction_dim_rew
    
    def learn(self, batch, logger, t_env, t, log_stats_t):
        """
        Training method for the opponent model.

        Args:
            batch (dict): Batch of data containing observations, actions, etc.
            logger (Logger): Logger to record training statistics.
            t_env (int): Current time step in the environment.
            t (int): Time step index.
            log_stats_t (int): Last time step statistics were logged.
        """
        self.train()
        self.dataset = OpponentDataset(self.args, batch, t)
        
        dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size_opponent_modelling, shuffle=True)

        loss_act_, loss_obs_, loss_rew_, accuracy_ = [], [], [], []
        entropy = [0]

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
                    try:
                        entropy.extend(calculate_entropy(reconstructions_act).view(-1))
                    except:
                        pass
                    reconstructions_act = reconstructions_act.view(reconstruction_shape)
                    loss_act = F.kl_div(reconstructions_act.log(), action_logits, reduction='batchmean')
                    loss_act_.append(loss_act.item())
                    accuracy_.append(accuracy)
                loss = loss_obs + loss_act + loss_rew

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

