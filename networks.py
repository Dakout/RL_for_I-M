import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import default


class Dueling_ADRQN(nn.Module):
    def __init__(self, name, **kwargs):
        super().__init__()

        """ Set class features """
        # network params
        self.n_actions      = kwargs.get('n_actions', default.N_ACTIONS)
        self.n_obs          = kwargs.get('n_obs', default.N_OBS)
        self.n_lstm_layers  = kwargs.get('n_lstm_layers', default.N_LSTM_LAYERS)
        self.lstm_input_dim = kwargs.get('lstm_input_dim', default.LSTM_INPUT_DIM)
        self.n_hidden       = kwargs.get('n_hidden', default.N_HIDDEN)
        self.lrelu_slope    = kwargs.get('lrelu_slope', default.LRELU_SLOPE)

        assert self.lstm_input_dim % 2 == 0

        # optimizer params
        self.learning_rate = kwargs.get('learning_rate', default.LEARNING_RATE)
        self.betas         = kwargs.get('betas', default.BETAS)
        self.opt_eps       = kwargs.get('eps', default.OPT_EPS)
        self.weight_decay  = kwargs.get('weight_decay', default.WEIGHT_DECAY)
        self.amsgrad       = kwargs.get('ams_grad', default.AMSGRAD)
        self.maximize      = kwargs.get('maximize', default.MAXIMIZE)
        self.lr_scheduler  = kwargs.get('lr_scheduler', default.LR_SCHEDULER)
        self.lr_step_size  = kwargs.get('lr_step_size', default.LR_STEP_SIZE)
        self.lr_gamma      = kwargs.get('lr_gamma', default.LR_GAMMA)

        # storage params
        self.name            = name
        self.checkpoint_dir  = kwargs.get('checkpoint_dir', default.CHECKPOINT_DIR)
        self.checkpoint_name = os.path.join(self.checkpoint_dir, name)

        # sets device - 'cuda:0' for gpu or 'cpu' for cpu
        self.device_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)


        """ Define network layers """
        # action fully-connected layers
        self.act_fc1 = nn.Linear(self.n_actions, 5*self.n_actions)
        self.act_fc2 = nn.Linear(5*self.n_actions, self.lstm_input_dim//2)

        # observation full_connected layers
        self.obs_fc1 = nn.Linear(self.n_obs, 5*self.n_actions)
        self.obs_fc2 = nn.Linear(5*self.n_actions, self.lstm_input_dim//2)

        # lstm layer
        self.lstm = nn.LSTM(input_size  = self.lstm_input_dim,
                            hidden_size = self.n_hidden,
                            num_layers  = self.n_lstm_layers)

        # another fc layer
        self.fc_layer = nn.Linear(self.n_hidden, 2*self.n_hidden)

        # value & advantage layers
        self.Value = nn.Linear(2*self.n_hidden, 1)
        self.Advantage = nn.Linear(2*self.n_hidden, self.n_actions)


        """ Define optimizer & Loss"""
        # Initialize optimizer and loss functions
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate,
                                    betas=self.betas, weight_decay=self.weight_decay,
                                    amsgrad=self.amsgrad, maximize=self.maximize)
        # include learning rate scheduler: lr is decreased periodically
        if self.lr_scheduler:
            self.lr_scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                          step_size=self.lr_step_size,
                                                          gamma=self.lr_gamma)

        self.loss = nn.MSELoss()
        self.to(self.device)


    def forward(self, previous_action_one_hot: torch.Tensor, observation: torch.Tensor,
                hidden_state: torch.Tensor, cell_state: torch.Tensor):

        act_output1   = F.leaky_relu(self.act_fc1(previous_action_one_hot), negative_slope=self.lrelu_slope)
        act_output2   = F.leaky_relu(self.act_fc2(act_output1), negative_slope=self.lrelu_slope)

        obs_output1   = F.leaky_relu(self.obs_fc1(observation), negative_slope=self.lrelu_slope)
        obs_output2   = F.leaky_relu(self.obs_fc2(obs_output1), negative_slope=self.lrelu_slope)

        concat_output = torch.cat((act_output2, obs_output2), -1).view(-1, self.lstm_input_dim)
        concat_output = concat_output.view(1, -1, self.lstm_input_dim) # This should be [1, batch, hidden input size]

        if (hidden_state == None) and (cell_state == None): # first step for lstm
             lstm_output, (next_hidden_state, next_cell_state) = self.lstm(concat_output)
        else:
            lstm_output, (next_hidden_state, next_cell_state) = self.lstm(concat_output, (hidden_state, cell_state))

        lstm_output   = F.leaky_relu(lstm_output.view(-1, self.n_hidden), negative_slope=self.lrelu_slope)

        fc_output     = F.leaky_relu(self.fc_layer(lstm_output), negative_slope=self.lrelu_slope)

        value         = self.Value(fc_output)
        advantage     = self.Advantage(fc_output)

        return value, advantage, next_hidden_state, next_cell_state


    def update_names(self, new_name):
        self.name = new_name
        self.checkpoint_name = os.path.join(self.checkpoint_dir, new_name)
        return


    def test(self):
        a = torch.tensor([0,0,0,1]).to(self.device)
        b = torch.tensor([0.1]).to(self.device)
        self.forward(a,b,None,None)


    def save_checkpoint(self):
        # Saves the checkpoint to the desired file.
        print(f"Saving to checkpoint {self.checkpoint_name}")
        torch.save(self.state_dict(), self.checkpoint_name)


    def load_checkpoint(self):
        # Loads the checkpoint from the saved file.
        print(f"Loading from checkpoint {self.checkpoint_name}")
        self.load_state_dict(torch.load(self.checkpoint_name, map_location=self.device))
