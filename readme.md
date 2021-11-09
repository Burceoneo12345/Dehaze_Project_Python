# Python Final Homework Project
This repo contains the codes and papers for haze removal.

| DCP | DehazeNet | GridDehazeNet |
| :----:| :----: | :----: |
| [Paper](https://ieeexplore.ieee.org/abstract/document/5567108) | [Paper](https://arxiv.org/abs/1601.07661) | [Paper](https://arxiv.org/abs/1908.03245) |
| Traditional Way | Deep Learning Way | Deep Learning Way |

## Introduction
- run ```./code/DCP/HazeRemoval.py``` to remove the haze from picture by DCP
- run ```./code/GridDehazeNet/Dehaze_tool.py``` to remove the haze from picture by GridDehazeNet. The Original Code is [here](https://github.com/proteus1991/GridDehazeNet)
- ```model.py``` defines the model of GridDehazeNet, and ```residual_dense_block.py``` builds the [RDB](https://arxiv.org/abs/1802.08797) block.
- The ```./doc/``` folder stores the paper of DCP、DehazeNet and GridDehazeNet.


