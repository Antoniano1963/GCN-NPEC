import os
import pickle
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import math

def read(filename):
    dataset = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip('\n')
            line = line.split(',')
            dataset.append(line)
    # print(dataset[32])
    return dataset
    

def show(dataset, num_of_epoch):
    cost = []
    epoch = []
    print(float(dataset[1][4]))
    for i in range(num_of_epoch):
        # index = i * 32 + 1
        index = (i + 1) * 32
        cost.append(float(dataset[index][4]))
        epoch.append(i + 1)
    cost = np.array(cost)
    epoch = np.array(epoch)
    plt.figure(1)
    plt.xlabel("epoch")
    plt.ylabel("cost")
    plt.title("cost-epoch")
    plt.plot(epoch, cost)
    plt.savefig('cost-epoch.jpg', dpi = 300)
    plt.show()


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }


class VRPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(VRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.
            }

            graph = np.random.rand(50000, 21, 2)
            dist = np.zeros((50000, 21, 21))
            for i in range(50000):
                for j in range(21):
                    for k in range(21):
                        dist[i][j][k] = get_dist(graph[i][j], graph[i][k])
            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'depot': torch.FloatTensor(2).uniform_(0, 1),
                    'graph':torch.FloatTensor(graph[i])
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def get_dist(n1, n2):
	x1,y1,x2,y2 = n1[0],n1[1],n2[0],n2[1]
	if isinstance(n1, torch.Tensor):
		return torch.sqrt((x2-x1).pow(2)+(y2-y1).pow(2))
	elif isinstance(n1, (list, np.ndarray)):
		return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
	else:
		raise TypeError
if __name__ == "__main__":
    delta = 0.1
    graph = np.random.rand(50000, 21, 2)
    dist = np.zeros((50000, 21, 21))
    for i in range(50000):
        for j in range(21):
            for k in range(21):
                dist[i][j][k] = get_dist(graph[i][j], graph[i][k]) #+ 0.1 * np.random.randn(1)
    demand = np.random.rand(50000, 20)
    depot_demand = np.zeros((50000,1))
    demand = np.concatenate((depot_demand, demand), axis = 1)
    np.savez('../tc/my-20-training.npz', graph = graph, demand = demand, dis = dist)