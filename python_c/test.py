import ctypes
from dis import dis
from cv2 import imencode
import numpy as np
import torch
import math
import time 

def get_dist(n1, n2):
	x1,y1,x2,y2 = n1[0],n1[1],n2[0],n2[1]
	if isinstance(n1, torch.Tensor):
		return torch.sqrt((x2-x1).pow(2)+(y2-y1).pow(2))
	elif isinstance(n1, (list, np.ndarray)):
		return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
	else:
		raise TypeError

if __name__ == "__main__":
    dll = ctypes.cdll.LoadLibrary('./Floyd.so')
    # FloydAlgorithm = dll.FloydAlgorithm
    # FloydAlgorithm.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags="C_CONTIGUOUS"),
    #               ctypes.c_int,
    #               ctypes.c_int,
    #               np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags="C_CONTIGUOUS")]
    batch = 10000
    customer_num = 20 + 1
    graph = np.random.rand(batch, 21, 2)
    dist = np.zeros((batch, 21, 21))
    print(graph[7][14])
    print(type(graph))
    graph_c = graph.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) 
    dist_c = dist.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    start = time.time() 
    dll.FloydAlgorithm(graph_c,batch,21,dist_c)
    print("c++ use {}".format(time.time() - start))
    dist1 = np.zeros((batch, 21, 21))
    start = time.time()
    for i in range(batch):
        for j in range(21):
            for k in range(21):
                dist1[i][j][k] = get_dist(graph[i][j], graph[i][k]) #+ 0.1 * np.random.randn(1)
    print("python use {}".format(time.time() - start))
    print((dist == dist1).all())
