import sys
import torch

def main(path):
  checkpoint = torch.load(path)
  remove = ['optimizer_state_dict', 'memory_state']
  checkpoint = dict(item for item in checkpoint.items()
      if item[0] not in remove)
  torch.save(checkpoint, path)
  return

if __name__ == '__main__':
  assert len(sys.argv) > 1, 'No input argument given'
  main(sys.argv[1])


