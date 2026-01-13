"""
1 fix compute and fix model size, see how parameters affects loss
2 fix compute and vary model size, see how model size affects loss

first query select some data with training_flops less than 1e16
total is 2e18, need to use small models to estimate large models
let's use 1e16 find some trends first
"""

d_model = [64, 128, 256, 512]
num_layers = [2, 8, 16, 64]
num_heads = [2, 4, 8, 16]
batch_size = [128, 256]
learning_rate = 1e-3

"""
total is 2e18 compute for experiments
fix compute 3e16, total 2e18, can do 60 experiments

step 1: find optimal choices for num_heads, batch_size
fix compute (3e16), fix total_parameters
based on previous experience and paper,
C = 6 * N * D
D = 20 * N
N = 1.5e7 parameters

baseline
num_layers = 19
d_model = 256
num_heads = 8
batch_size = 128
learning rate 1e-3
total_parameters = 12 * 19 * 256 * 256 = 1.5e7

compare lr
learning rate
5e-4, 1e-4

compare batch_size
128, 256

compare num_heads
4, 8, 16

compare d_model and num_layers
(512, 5)

step 2: find d_model and num_layers
part a:
fix total parameters
num_layers = 24
d_model = 64
num_heads = 2
batch_size = 128
learning_rate = 1e-3

total_parameters = 1.2e6
fix compute 6e13

compare d_model and num_layers
(128, 6), (256, 2)

learning rate
5e-4, 1e-4

part b:
fix total parameters
num_layers = 20
d_model = 512
num_heads = 8
batch_size = 128
learning_rate = 5e-4

total_parameters = 6.3e7
fix compute 1e17

compare d_model and num_layers
(1024, 5)

learning rate
5e-4, 1e-4

batch size
256

step 3:
from step 1 and step 2
find how learning rate should be adjusted

step 4:
from step 3, find the best model

fix compute
1e17, 1e16, 6e15, 3e15

fix model size
3e7, 1e7, 6e6, 3e6, 1e6

find the optimal models
"""
total_parameters = 12 * num_layers * (d_model**2)
