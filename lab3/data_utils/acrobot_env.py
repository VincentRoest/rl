import gym

import numpy as np
import matplotlib.pyplot as plt

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AcrobotEnv():

  class EnvWrapper():
    def __init__(self, env, T=200):
      self.env = env
      self.t = 0
      self.T = T
    def __getattr__(self, attr):
      orig_attr = self.env.__getattribute__(attr)
      if attr == 'step':
        self.t += 1
        if self.t >= self.T:
          def func(*args, **kwargs):
            result = orig_attr(*args, **kwargs)
            result = tuple([*result[:2], True, *result[3:]])
            return result
          return func
      if attr == 'reset':
        self.t = 0
      return orig_attr

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    env = gym.make('Acrobot-v1').unwrapped
    self.env = self.EnvWrapper(env)

  def reset(self):
    self.env.reset()
  def close(self):
    self.env.close()

  def get_screen(self):
    screen = self.env.render(mode='rgb_array')
    screen = screen[::10, ::10] # downsample
    screen[screen!=204] = 0 # remove background
    screen[screen!=0] = 255 # set all color to full
    screen = torch.from_numpy(screen.astype(np.float32)) / 255
    screen = screen.permute(2, 0, 1)
    return screen.unsqueeze(0).to(device)

  def show_example(self):
    self.env.reset()
    fig = plt.figure()
    screen = self.get_screen().cpu()
    print("%d bytes" % (screen.numpy().size * screen.numpy().itemsize))
    plt.imshow(screen.squeeze(0).permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()
    fig.canvas.flush_events()
    plt.pause(0.001)

if __name__ == '__main__':
  env = AcrobotEnv()
  env.show_example()
  env.close()


