# Pointer Network for TSP

This repository contains pytorch implementation for Pointer Network[1] and focusses on the usecase for TSP.

## Label Creation

Bellman Held Karp Heuristic [2]

## Model Details

### Encoder

1. Single LSTM Layer
2. 512 hidden units
3. Random Uniform weight initialization (-0.08 to 0.08)

### Decoder

1. Single LSTM Layer with hidden state from encoder
2. 512 hidden units
3. Random Uniform weight initialization (-0.08 to 0.08)

#### Optimizer

1. SGD
    1. learning rate 1.0
    1. L2 Gradient Clipping 2.0
    1. Batch Size 128

> Hardware

1. OS: macOS 14.0 23A344 arm64
1. Host: MacBookPro17,1
1. CPU: Apple M1
1. GPU: Apple M1
1. Memory: 16384MiB

> Environment

conda env create --name name_of_environment --file=environment.yml

#### Links

[1] Vinyals, O. et al.: Pointer Networks, <http://arxiv.org/abs/1506.03134>, (2017). <https://doi.org/10.48550/arXiv.1506.03134>.

[2] <https://github.com/Valdecy/pyCombinatorial>
