# Python Final Homework Project
This repo contains the codes and papers for haze removal.

| DCP | DehazeNet | GridDehazeNet |
| :----:| :----: | :----: |
| [Paper](https://ieeexplore.ieee.org/abstract/document/5567108) | [Paper](https://arxiv.org/abs/1601.07661) | [Paper](https://arxiv.org/abs/1908.03245) |
| Traditional Way | Deep Learning Way | Deep Learning Way |

## Introduction
* run ```./code/DCP/HazeRemoval.py``` to remove the haze from picture by DCP
  * change ```./code/DCP/images/test_list.txt``` and add the hazy images in the folder```./code/DCP/images/haze```
* run ```./code/GridDehazeNet/Dehaze_tool.py``` to remove the haze from picture by GridDehazeNet. The Original Code is [here](https://github.com/proteus1991/GridDehazeNet)
  * change ```./code/GridDehazeNet/test/test_list.txt``` and add the hazy images in the folder```./code/GridDehazeNet/test/haze```
* run ```./code/DehazeNet-caffe/DehazeNet.py``` to remove the haze from picture by DehazeNet.
* The ```./doc/``` folder stores the paper of DCP、DehazeNet and GridDehazeNet.


## Qualitative Comparisons
* Comparsion_1
![Comparsion_1](https://raw.githubusercontent.com/Burceoneo12345/Dehaze_Project_Python/master/%E6%AF%94%E8%BE%831.png)
* Comparsion_2
![Comparsion_2](https://raw.githubusercontent.com/Burceoneo12345/Dehaze_Project_Python/master/%E6%AF%94%E8%BE%832.png)
* Comparsion_3
![Comparsion_3](https://raw.githubusercontent.com/Burceoneo12345/Dehaze_Project_Python/master/%E6%AF%94%E8%BE%833.png)
* Comparsion_4
![Comparsion_4](https://raw.githubusercontent.com/Burceoneo12345/Dehaze_Project_Python/master/%E6%AF%94%E8%BE%834.png)