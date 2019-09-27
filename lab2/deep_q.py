import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

import random
import time
from collections import defaultdict
import numpy as np
import os
import sys
import gym
env = gym.envs.make("CartPole-v0")

import torchvision.transforms as T
from PIL import Image


class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        # add transation
        if (self.capacity < len(self.memory)):
            # cut the first memory offf
            self.memory = self.memory[1:]
        self.memory.append(transition)
            
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def select_action(model, state):
    # Samples an action according to the probability distribution induced by the model
    # Also returns the log_probability
    # log is simply the output of the model with log_softmax
    with torch.no_grad():
        log_p = model(torch.FloatTensor(state))
        action = torch.multinomial(torch.exp(log_p), 1)
        return action.item(), log_p[action]
        

def run_episode(env, model):
    state = env.reset()
    done = False
    episode = []
    while not done:
        action, log_p = select_action(model, state)
        next_state, reward, done, _ = env.step(action)
        episode.append((log_p, reward))
        state = next_state
  
    return episode

def compute_reinforce_loss(episode, discount_factor):
    # Compute the reinforce loss
    # Make sure that your function runs in LINEAR TIME
    # Don't forget to normalize your RETURNS (not rewards)
    # Note that the rewards/returns should be maximized 
    # while the loss should be minimized so you need a - somewhere
    print ('hi')
    def normalize(G):
        return (G - G.mean()) / G.std()
    
    G = np.zeros(len(episode))
    for i, (_, reward) in enumerate(reversed(episode)):
        G[i] = reward + discount_factor * G[i-1]


    return -sum([log_p * g for (log_p, _), g in zip(reversed(episode), normalize(G))])



def run_episodes_policy_gradient(model, env, num_episodes, discount_factor, learn_rate):
    
    optimizer = optim.Adam(model.parameters(), learn_rate)
    
    episode_durations = []
    for i in range(num_episodes):
        episode = run_episode(env, model)
        loss = compute_reinforce_loss(episode, discount_factor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode), '\033[92m' if len(episode) >= 195 else '\033[99m'))
        episode_durations.append(len(episode))
        
    return episode_durations

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

class CartPoleRawEnv(gym.Env):
    
    def __init__(self, *args, **kwargs):
        self._env = gym.make('CartPole-v0', *args, **kwargs)  #.unwrapped
        self.action_space = self._env.action_space
        screen_height, screen_width = 40, 80  # TODO
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(screen_height, screen_width, 3), dtype=np.uint8)
    
    def seed(self, seed=None):
        return self._env.seed(seed)
    
    def reset(self):
        s = self._env.reset()
        self.prev_screen = self.screen = self.get_screen()
        return self._get_observation()
    
    def step(self, action):
        s, r, done, info = self._env.step(action)
        self.prev_screen = self.screen
        self.screen = self.get_screen()
        return self._get_observation(), r, done, info
    
    def _get_observation(self):
        return self.screen - self.prev_screen
    
    def _get_cart_location(self, screen_width):
        _env = self._env.unwrapped
        world_width = _env.x_threshold * 2
        scale = screen_width / world_width
        return int(_env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        screen = self._env.unwrapped.render(mode='rgb_array').transpose(
            (2, 0, 1))  # transpose into torch order (CHW)
        # Strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, screen_height * 4 // 10:screen_height * 8 // 10]
        view_width = screen_height * 8 // 10
        cart_location = self._get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescare, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        #return screen.unsqueeze(0).to(device)
        return resize(screen).unsqueeze(0)
    
    def close(self):
        return self._env.close()

# Maybe you should make it a bit deeper?
class DeepPolicy(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        #self.fc = nn.Linear(40 * 80 * 3, 2)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=0)
        #self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=0)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=0)        
        self.fc = nn.Linear(20, 2)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        '''out = F.relu(self.conv1(x))
        print (out.size())
        out = F.relu(self.conv2(out))
        print (out.size())
        out = F.relu(self.conv(out))
        # Flatten
        '''
        out = self.conv1(x)
        print (out.size())
        
        return F.log_softmax(self.fc(out.view(out.size(0), -1)), -1)
    
def bonus_get_action(x):
    return policy(x).exp().multinomial(1)[:, 0]


if __name__ == '__main__':
    policy = DeepPolicy()
    filename = 'weights.pt'
    
    if os.path.isfile(filename):
        print(f"Loading weights from {filename}")
        weights = torch.load(filename, map_location='cpu')
        
        policy.load_state_dict(weights['policy'])
        
    else:
        # Train
        ### TODO some training here, maybe? Or run this on a different machine?
        torch.manual_seed(42)
        num_episodes = 20
        discount_factor = 0.9
        learn_rate = 0.005
        env = CartPoleRawEnv()
        env._env.close()

        run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate)
        
        env._env.close()

        print(f"Saving weights to {filename}")
        torch.save({
            # You can add more here if you need, e.g. critic
            'policy': policy.state_dict()  # Always save weights rather than objects
        },
        filename)
        env.close()
        