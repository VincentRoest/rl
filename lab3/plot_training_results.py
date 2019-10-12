import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage.filters import convolve1d
import torch

def main(params):
  assert params.load_path != None, 'No input file given (use \'--load_path\' argument)'
  assert params.alpha >= 0.0 and params.alpha <= 1.0, \
      'Alpha value must be between 0 and 1'
  
  load_paths = params.load_path.split(',')
  
  figs = [plt.figure(i) for i in range(1, 3 + 1)]
  axes = [fig.add_subplot(1,1,1) for fig in figs]
  
  for load_path in load_paths:
    checkpoint = torch.load(load_path)
    episode_durations = np.asarray(checkpoint['episode_durations'], dtype=np.float)
    rewards = np.asarray(checkpoint['rewards'], dtype=np.float)
    loss = np.asarray(checkpoint['loss'])
    
    if params.smoothing is not None:
      smoothing_filter = np.ones(params.smoothing) / params.smoothing
      
      none_loss_filter = ~(loss == None)
      
      convolve_mode = 'reflect'
      episode_durations = convolve1d(episode_durations, smoothing_filter, mode=convolve_mode)
      rewards = convolve1d(rewards, smoothing_filter, mode=convolve_mode)
      loss[none_loss_filter] = convolve1d(np.asarray(loss[none_loss_filter], dtype=np.float),
          smoothing_filter, mode=convolve_mode)
    
    label = os.path.basename(load_path)
    x = np.arange(len(episode_durations))
    axes[0].plot(x, episode_durations, label=label, alpha=params.alpha)
    axes[1].plot(x, rewards, label=label, alpha=params.alpha)
    axes[2].plot(x, loss, label=label, alpha=params.alpha)
  
  for ax, ylabel in zip(axes, ['episode durations', 'rewards', 'loss']):
    ax.set_title('{} over episodes'.format(ylabel))
    ax.set_ylabel(ylabel)
    ax.set_xlabel('episode')
    ax.grid(True)
    ax.legend()
  
  for fig in figs:
    fig.tight_layout()
  
  plt.show()
  
  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='RL plot training results')
  
  parser.add_argument('--load_path', type=str, default=None,
      help='Path to load model checkpoint from')
  parser.add_argument('--smoothing', type=int, default=None,
      help='Amount of datapoints to smooth over (take mean)')
  parser.add_argument('--alpha', type=float, default=1.0,
      help='Alpha value for plotting curves')
  
  params, _ = parser.parse_known_args()
  
  main(params)


