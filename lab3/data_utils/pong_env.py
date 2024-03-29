import gym
from gym import wrappers

#import math
#import random
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
#from collections import namedtuple
#from itertools import count
#from PIL import Image

import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
#import torchvision.transforms as T

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PongEnv():

  # wrapper to force environment to stop once a point is scored
  class EnvWrapper():
    def __init__(self, env):
      self.env = env
    def __getattr__(self, attr):
      orig_attr = self.env.__getattribute__(attr)
      if attr == 'step':
        def func(*args, **kwargs):
          result = orig_attr(*args, **kwargs)
          if result[1] != 0:
            result = tuple([*result[:2], True, *result[3:]])
          return result
        return func
      else:
        return orig_attr

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    env = gym.make("PongDeterministic-v4").unwrapped
    self.env = self.EnvWrapper(env) # wrappers.Monitor(env, 'tmp/pong', video_callable=False, force=True) 

  def get_screen(self):
    # Preprocessing from https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
    # Returned screen requested by Pong is 210x160x3, reshape into 75x80
    screen = self.env.render(mode='rgb_array')
    screen = screen[35:185, :, :] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
    screen = screen[::2,::2, :] # downsample by factor of 3, keep last dimension
    screen[screen == 144] = 0 # erase background (background type 1)
    screen[screen == 109] = 0 # erase background (background type 2)
    screen[screen != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively

    screen = torch.from_numpy(screen.astype(np.float32))
    screen = screen.permute((2, 0, 1))
    return screen.unsqueeze(0).to(device) # add batch dimension by unsqueeze

  def show_example(self):
    self.env.reset()
    fig = plt.figure()
    screen = self.get_screen().cpu()
    print("%d bytes" % (screen.numpy().size * screen.numpy().itemsize))
    plt.imshow(screen.squeeze(0).permute(1,2,0).numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()
    fig.canvas.flush_events()
    plt.pause(0.001)
