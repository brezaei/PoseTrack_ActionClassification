# Target Specific Action Classification using Pose Evolution Maps

This repository contains the code for the action classification associated to the following paper:

Rezaei, B., Christakis, Y., Ho, B., Thomas, K., Erb, K., Ostadabbas, S., Patel, S. (2019). Target-Specific Action Classification for Automated Assessment of Human Motor Behavior from Video. accepted for a special issue on “Sensors, Signal and Image Processing in Biomedicine and Assisted Living” of Sensors journal.

This work was done during Behnaz internship at Pfizer.

contact:

[Behnaz Rezaei](brezaei@ece.neu.edu),

[Shyamal Patel](Shyamal.Patel@pfizer.com)

## Contents   
*  [Requirements](#requirements)
*  [Running the code](#running-the-code)
*  [Citation](#citation)
*  [License](#license)
*  [Acknowledgements](#acknowledgements)

## Requirements
The classification network is coded in python 2.7 and used caffe2 framework.
For installing caffe2 you can use the following instruction, remember to set the USELMDB=ON while building the caffe2 from source.

https://caffe2.ai/docs/getting-started.html?platform=mac&configuration=prebuilt

Since we are using LMDB database in order to save the pose evolution representations. If you wish to use different data structure, you have to modify the input handling operation based on your desired data structure.

## Running the code
The code contains two parts:
Each of the snippets are self-contained with the clarifying explanations for each function and block of code
1. Generating pose evolution maps
```
${ROOT}/ActionClassification/tools/PoseEvoAugmentedLMDB.ipynb
```
2. Action classification network using the volumetric pose evolution representations
training
```
${ROOT}/ActionClassification/tools/PoseBased_ActionRec_Train.ipynb
```

deploying the pretrained model
```
${ROOT}/ActionClassification/tools/PoseBased_ActionRec_Test.ipynb
```

## Input
Input of the action classification is either pose heatmaps or keypoints of the target human whose action is to be classified in a video clip.


the length of the input video clips can be the same or varied.
#### Hint:
For the input of the action classification you can use the input of any pose estimation network on your raw video with slight changes in the global parameters of the code which are explained in the relative snippet. For example we used the pretrined network provided in the following repository in order to get the pose results.
https://github.com/facebookresearch/DetectAndTrack

## Citation
If you are using this code please consider citing following paper:
```
@article{rezaei2019target,
  title={Target-Specific Action Classification for Automated Assessment of Human Motor Behavior from Video},
  author={Rezaei, Behnaz and Christakis, Yiorgos and Ho, Bryan and Thomas, Kevin and Erb, Kelley and Ostadabbas, Sarah and Patel, Shyamal},
  journal={Sensors},
  volume={19},
  number={19},
  pages={4266},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
```

## License
This code is offered as-is without any warranty either expressed or implied.
## Acknowledgements
[1] R. Girdhar, G. Gkioxari, L. Torresani, M. Paluri, and D. Tran. Detect-and-track: Efficient pose estimation in videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 350–359, 2018.


[2] V. Choutas, P. Weinzaepfel, J. Revaud, and C. Schmid. Potion:Pose motion representation for action recognition. In CVPR
2018, 2018
