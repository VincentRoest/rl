import argparse
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage.filters import convolve1d
import sys
import torch

def get_stats(data):
  manual = set(np.where(np.sum(np.isnan(data), axis=0) != 0)[0])
  other = set(range(data.shape[1])) - manual
  
  manual = list(manual)
  other = list(other)
  
  mean = np.mean(data[:, other], axis=0)
  std = np.std(data[:, other], axis=0)
  
  for ind in manual:
    dat = data[:, ind]
    dat = dat[~np.isnan(dat)]
    if len(dat) == 0:
      mean = np.insert(mean, ind, np.nan)
      std = np.insert(std, ind, np.nan)
    else:
      mean = np.insert(mean, ind, np.mean(dat))
      std = np.insert(std, ind, np.std(dat))
  
  return mean, std

def main(params):
  assert params.load_path != None, \
      'No input file given (use \'--load_path\' argument)'
  assert params.alpha >= 0.0 and params.alpha <= 1.0, \
      'Alpha value must be between 0 and 1'
  
  load_paths = [load_path_set.split('+')
      for load_path_set in params.load_path.split(',')]
  
  print('Input paths:')
  print(np.asarray(load_paths))
  print()
  
  to_continue = None
  while to_continue is None:
    answer = input('Continue? [y/n]').lower()
    if answer == 'y':
      to_continue = True
    elif answer == 'n':
      to_continue = False
    else:
      print('Unrecognised answer, use [y/n]')
  if to_continue == False:
    sys.exit(0)
  print()
  
  if params.labels == None:
    labels = []
    for load_path_set in load_paths:
      # determine commonality between file names
      label = os.path.basename(load_path_set[0])
      for load_file in [os.path.basename(path) for path in load_path_set[1:]]:
        match = type('TmpClass', (), {'size': 1})
        label_part = label
        strs = []
        while match.size != 0:
          match = SequenceMatcher(
              None, label_part, load_file).find_longest_match(
              0, len(label_part), 0, len(load_file))
          strs.append(label_part[match.a:match.a+match.size])
          label_part = label_part[:match.a] + '\\' \
              + label_part[match.a+match.size:]
          load_file = load_file[:match.b] + ' ' + load_file[match.b+match.size:]
        label = ''.join(strs)
      labels.append(label)
  else:
    labels = params.labels.split(',')
    print (labels)
    assert len(labels) == len(load_paths), \
        'Number of given labels does not match amount of lines in the plot'
  
  figs = [plt.figure(i) for i in range(1, 3 + 1)]
  axes = [fig.add_subplot(1,1,1) for fig in figs]
  
  print()
  
  for j, load_path_set in enumerate(load_paths):
    # collect data
    for i, load_path in enumerate(load_path_set):
      
      print('loading {}'.format(load_path))
      
      checkpoint = torch.load(load_path)
      episode_durations = np.asarray(checkpoint['episode_durations'],
          dtype=np.float)[None, ...]
      rewards = np.asarray(checkpoint['rewards'], dtype=np.float)[None, ...]
      loss = np.asarray(checkpoint['loss'], dtype=np.float)[None, ...]
      
      if params.n is not None:
        episode_durations = episode_durations[:, :params.n]
        rewards = rewards[:, :params.n]
        loss = loss[:, :params.n]
      
      if i == 0:
        episode_durations_arr = episode_durations
        rewards_arr = rewards
        loss_arr = loss
      else:
        episode_durations_arr = np.r_[episode_durations_arr, episode_durations]
        rewards_arr = np.r_[rewards_arr, rewards]
        loss_arr = np.r_[loss_arr, loss]
    
    print('calculating mean and standard deviation')
    
    #"""
    # smooth data
    if params.smoothing is not None:
      episode_durations_nan_filter = ~np.isnan(episode_durations_arr)
      rewards_nan_filter = ~np.isnan(rewards_arr)
      loss_nan_filter = ~np.isnan(loss_arr)
      
      smoothing_filter = np.ones(params.smoothing) / params.smoothing
      convolve_mode = 'reflect'
      
      for i in range(episode_durations_arr.shape[0]):
        episode_durations_arr[i, episode_durations_nan_filter[i]] = convolve1d(
            episode_durations_arr[i, episode_durations_nan_filter[i]],
            smoothing_filter, mode=convolve_mode)
        rewards_arr[i, rewards_nan_filter[i]] = convolve1d(
            rewards_arr[i, rewards_nan_filter[i]],
            smoothing_filter, mode=convolve_mode)
        loss_arr[i, loss_nan_filter[i]] = convolve1d(
            loss_arr[i, loss_nan_filter[i]],
            smoothing_filter, mode=convolve_mode)
    #"""
    
    # calculate mean and std
    episode_durations_mean, episode_durations_std = \
        get_stats(episode_durations_arr)
    rewards_mean, rewards_std = get_stats(rewards_arr)
    loss_mean, loss_std = get_stats(loss_arr)
    
    """
    # smooth data
    if params.smoothing is not None:
      episode_durations_nan_filter = ~np.isnan(episode_durations_mean)
      rewards_nan_filter = ~np.isnan(rewards_mean)
      loss_nan_filter = ~np.isnan(loss_mean)
      
      smoothing_filter = np.ones(params.smoothing) / params.smoothing
      convolve_mode = 'reflect'
      
      episode_durations_mean[episode_durations_nan_filter] = convolve1d(
          episode_durations_mean[episode_durations_nan_filter],
          smoothing_filter, mode=convolve_mode)
      rewards_mean[rewards_nan_filter] = convolve1d(
          rewards_mean[rewards_nan_filter],
          smoothing_filter, mode=convolve_mode)
      loss_mean[loss_nan_filter] = convolve1d(
          loss_mean[loss_nan_filter],
          smoothing_filter, mode=convolve_mode)
      
      episode_durations_std[episode_durations_nan_filter] = convolve1d(
          episode_durations_std[episode_durations_nan_filter],
          smoothing_filter, mode=convolve_mode)
      rewards_std[rewards_nan_filter] = convolve1d(
          rewards_std[rewards_nan_filter],
          smoothing_filter, mode=convolve_mode)
      loss_std[loss_nan_filter] = convolve1d(
          loss_std[loss_nan_filter],
          smoothing_filter, mode=convolve_mode)
    #"""
    
    print('plotting data\n')
    
    # plotting parameters
    n = len(episode_durations_mean)
    color = j % 10
    if params.std_interval == None:
      params.std_interval = max(1, int(n / 20))
    if params.std_offset == None:
      params.std_offset = max(1, int(params.std_interval / 10))
    off = j * params.std_offset
    
    # plot means
    x = np.arange(n)
    axes[0].plot(x, episode_durations_mean, label=labels[j], alpha=params.alpha,
        color='C{}'.format(color))
    axes[1].plot(x, rewards_mean, label=labels[j], alpha=params.alpha,
        color='C{}'.format(color))
    axes[2].plot(x, loss_mean, label=labels[j], alpha=params.alpha,
        color='C{}'.format(color))
    
    # plot standard deviations
    axes[0].errorbar(x[off::params.std_interval],
        episode_durations_mean[off::params.std_interval],
        yerr=episode_durations_std[off::params.std_interval],
        fmt='none', ecolor='k')
    axes[1].errorbar(x[off::params.std_interval],
        rewards_mean[off::params.std_interval],
        yerr=rewards_std[off::params.std_interval],
        fmt='none', ecolor='k')
    axes[2].errorbar(x[off::params.std_interval],
        loss_mean[off::params.std_interval],
        yerr=loss_std[off::params.std_interval],
        fmt='none', ecolor='k')
  
  for ax, ylabel in zip(axes, ['Episode Durations', 'Rewards', 'Loss']):
    ax.set_title('{} over Episodes'.format(ylabel))
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Episode')
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
  parser.add_argument('--n', type=int, default=None,
      help='Maximum amount of data-points to plot')
  parser.add_argument('--std_interval', type=int, default=None,
      help='Interval between plotted standard deviations.')
  parser.add_argument('--std_offset', type=int, default=None,
      help='Interval between standard deviations of different plotted lines.')
  parser.add_argument('--labels', type=str, default=None,
      help='Labels given to the different lines in the plots.')
  
  params, _ = parser.parse_known_args()
  
  main(params)


