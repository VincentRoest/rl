import torch
import matplotlib.pyplot as plt

def plot_progress(data, ylabel='Progress', fig_ind=2):
  fig = plt.figure(fig_ind)
  plt.clf()
  data_t = torch.tensor(data, dtype=torch.float)
  plt.title('Training...')
  plt.xlabel('Episode')
  plt.ylabel(ylabel)
  plt.plot(data_t.numpy())
  # Take 100 episode averages and plot them too
  if len(data_t) >= 100:
    means = data_t.unfold(0, 100, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(99), means))
    plt.plot(means.numpy())
  fig.canvas.flush_events()
  plt.pause(0.001)  # pause a bit so that plots are updated

def plot_screen(fig, screen):
  plt.clf()
  plt.imshow(screen.cpu().squeeze(0).permute(1,2,0).numpy(),
             interpolation='none')
  fig.canvas.flush_events()
  plt.pause(0.001)


