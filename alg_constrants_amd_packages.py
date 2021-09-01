import os
import time
from collections import namedtuple, deque

import gym
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import neptune.new as neptune
from neptune.new.types import File

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import loggers as pl_loggers


ENV = 'LunarLanderContinuous-v2'
# ENV='Pendulum-v0'
# ENV = "BipedalWalker-v3"
# ENV = "CarRacing-v0"
# ENV='MountainCarContinuous-v0'
NUMBER_OF_GAMES = 10
SAVE_RESULTS = True
PATH = 'actor_net.pt'

# NEPTUNE = True
NEPTUNE = False
# PLOT_LIVE = True
PLOT_LIVE = False
# ------------------------------------------- #
# ------------------FOR ALG:----------------- #
# ------------------------------------------- #

MAX_EPOCHS = 1000  # maximum epoch to execute
BATCH_SIZE = 128  # size of the batches
MAX_LENGTH_OF_A_GAME = 10000
LR = 3e-5  # learning rate
GAMMA = 0.99  # discount factor
ENTROPY_BETA = 0.001
REWARD_STEPS = 4
CLIP_GRAD = 0.1
VAL_CHECKPOINT_INTERVAL = 10
UPDATE_EVERY = 50
HIDDEN_SIZE = 256
REPLAY_BUFFER_SIZE = 1000
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'new_state'])

