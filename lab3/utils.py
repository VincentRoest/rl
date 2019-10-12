import torch
import matplotlib.pyplot as plt

def plot_durations(durations):
  plt.figure(2)
  plt.clf()
  durations_t = torch.tensor(durations, dtype=torch.float)
  plt.title('Training...')
  plt.xlabel('Episode')
  plt.ylabel('Duration')
  plt.plot(durations_t.numpy())
  # Take 100 episode averages and plot them too
  if len(durations_t) >= 100:
    means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(99), means))
    plt.plot(means.numpy())
  plt.pause(0.001)  # pause a bit so that plots are updated

def plot_screen(fig, screen):
  plt.clf()
  plt.imshow(screen.cpu().squeeze(0).permute(1,2,0).numpy(),
             interpolation='none')
  fig.canvas.flush_events()
  plt.pause(0.001)


