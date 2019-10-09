import gym
from gym import wrappers

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PongEnv():
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    env = gym.make("Pong-v0").unwrapped
    self.env = env # wrappers.Monitor(env, 'tmp/pong', video_callable=False, force=True) 

  def get_screen(self):
    # Preprocessing from https://github.com/omkarv/pong-from-pixels/blob/master/pong-from-pixels.py
    # Returned screen requested by Pong is 210x160x3, reshape into 75x80
    screen = self.env.render(mode='rgb_array')
    screen = screen[35:185] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
    screen = screen[::3,::3,:] # downsample by factor of 3, keep last dimension
    screen[screen == 144] = 0 # erase background (background type 1)
    screen[screen == 109] = 0 # erase background (background type 2)
    screen[screen != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively

    screen = torch.from_numpy(screen.astype(np.float32))
    screen = screen.permute((2, 0, 1))
    return screen.unsqueeze(0).to(device) # add batch dimension by unsqueeze

  def show_example(self):
    self.env.reset()
    plt.figure()
    plt.imshow(self.get_screen().cpu().squeeze(0).permute(1,2,0).numpy(),
              interpolation='none')
    plt.title('Example extracted screen')
    plt.show()
    plt.pause(1)
    plt.close()
