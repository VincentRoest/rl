import os

from collections import namedtuple
from tqdm import tqdm

import matplotlib.pyplot as plt

#import traceback
#import warnings
#import sys

from itertools import count
import random
import math 

import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils import plot_durations, plot_screen

# TODO: this is now a duplicate code
#Transition = namedtuple('Transition',
#                        ('state', 'action', 'next_state', 'reward'))
from memory import ReplayMemory, Transition

steps_done = 0

# TODO: move ?where
def select_action(model, state, params, n_actions=2):
    global steps_done
    sample = random.random()
    eps_threshold = params.eps_end + (params.eps_start - params.eps_end) * \
        math.exp(-1. * steps_done / params.eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return model(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model(policy_net, target_net, memory, optimizer, params):
  if len(memory) < params.batch_size:
      return
  transitions = memory.sample(params.batch_size)
  # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
  # detailed explanation). This converts batch-array of Transitions
  # to Transition of batch-arrays.
  batch = Transition(*zip(*transitions))

  # Compute a mask of non-final states and concatenate the batch elements
  # (a final state would've been the one after which simulation ended)
  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
  non_final_next_states = torch.cat([s for s in batch.next_state
                                              if s is not None])
  state_batch = torch.cat(batch.state)
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)

  # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
  # columns of actions taken. These are the actions which would've been taken
  # for each batch state according to policy_net
  state_action_values = policy_net(state_batch).gather(1, action_batch)

  # Compute V(s_{t+1}) for all next states.
  # Expected values of actions for non_final_next_states are computed based
  # on the "older" target_net; selecting their best reward with max(1)[0].
  # This is merged based on the mask, such that we'll have either the expected
  # state value or 0 in case the state was final.
  next_state_values = torch.zeros(params.batch_size, device=device)

  if (params.target_update > -1):
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
  else:
    next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1)[0].detach()

  # Compute the expected Q values
  expected_state_action_values = (next_state_values * params.gamma) + reward_batch

  # Compute Huber loss
  loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

  # Optimize the model
  optimizer.zero_grad()
  loss.backward()

  if (params.clip_rewards):
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
  optimizer.step()

def train_model(env, optimizer, policy_net, target_net, params):

  os.makedirs(os.path.dirname(params.save_path), exist_ok=True)

  if params.load_path is not None:
    checkpoint = torch.load(params.load_path)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    memory = ReplayMemory(*checkpoint['memory_state'])
    episode_durations = checkpoint['episode_durations']
    rewards = checkpoint['rewards']
  else:
    memory = ReplayMemory(1000)
    episode_durations = []
    rewards = []

  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval()

  if params.show_screen == True:
    fig = plt.figure()

  for i_episode in tqdm(range(params.num_episodes)):
    episode_reward = 0
    # Initialize the environment and state
    env.env.reset()
    last_screen = env.get_screen()
    current_screen = env.get_screen()
    state = current_screen - last_screen
    for t in count():

      if params.show_screen == True:
        plot_screen(fig, current_screen)

      # Select and perform an action
      action = select_action(policy_net, state, params, n_actions=env.env.action_space.n)
      _, reward, done, _ = env.env.step(action.item())
      reward = torch.tensor([reward], device=device)

      episode_reward += reward.item()

      # Observe new state
      last_screen = current_screen
      current_screen = env.get_screen()
      if not done:
        next_state = current_screen - last_screen
      else:
        next_state = None

      # Store the transition in memory
      memory.push(state, action, next_state, reward)

      # Move to the next state
      state = next_state

      # Perform one step of the optimization (on the target network)
      optimize_model(policy_net, target_net, memory, optimizer, params)

      if done:
        rewards.append(episode_reward)
        # print(episode_reward)
        episode_durations.append(t + 1)
        # print (episode_durations)
        # plot_durations(episode_durations)
        break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % params.target_update == 0 and target_net and params.target_update >= 0:
      # print(episode_durations)
      print ('updating target')
      target_net.load_state_dict(policy_net.state_dict())

    if i_episode % params.save_every == 0 \
        or i_episode == params.num_episodes - 1:
      torch.save({
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'memory_state': memory.get_state(),
        'episode_durations': episode_durations,
        'rewards': rewards
        }, params.save_path)
  
  return episode_durations, rewards
