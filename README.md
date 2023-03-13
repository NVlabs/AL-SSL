# Not All Labels Are Equal:Rationalizing The Labeling Costs for Training Object Detection

This repository contains the official Pytorch implementation of training & evaluation code and the pretrained models for:

[Not All Labels Are Equal:Rationalizing The Labeling Costs for Training Object Detection] (https://openaccess.thecvf.com/content/CVPR2022/papers/Elezi_Not_All_Labels_Are_Equal_Rationalizing_the_Labeling_Costs_for_CVPR_2022_paper.pdf)

Ismail Elezi, Zhiding Yu, Anima Anandkumar, Laura Leal-Taixe, and Jose M. Alvarez.

CVPR 2023.


[Code](https://github.com/NVlabs/AL-SSL)


## Installation & Preparation
We experimented with the SSD in the PyTorch framework. To use our model, complete the installation & preparation on the [SSD pytorch homepage](https://github.com/amdegroot/ssd.pytorch)

#### prerequisites
- Python 3.6
- Pytorch 1.1.0

## Training
```Shell
python train.py
```

## Evaluation
```Shell
python eval.py
```



## License
Copyright © 2022-2023, NVIDIA Corporation and Affiliates. All rights reserved.

This work is made available under the Nvidia Source Code License-NC. Click [here](https://github.com/NVlabs/AL-SSL/blob/main/LICENSE) to view a copy of this license.

The pre-trained models are shared under CC-BY-NC-SA-4.0. If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).


## Citation

If you find this code useful, please consider citing the following paper:

````
@inproceedings{DBLP:conf/cvpr/EleziYALA22,
  author    = {Ismail Elezi and
               Zhiding Yu and
               Anima Anandkumar and
               Laura Leal{-}Taix{\'{e}} and
               Jose M. Alvarez},
  title     = {Not All Labels Are Equal: Rationalizing The Labeling Costs for Training
               Object Detection},
  booktitle = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
               {CVPR} 2022, New Orleans, LA, USA, June 18-24, 2022},
  pages     = {14472--14481},
  publisher = {{IEEE}},
  year      = {2022},
  url       = {https://doi.org/10.1109/CVPR52688.2022.01409},
  doi       = {10.1109/CVPR52688.2022.01409},
  timestamp = {Wed, 05 Oct 2022 16:31:19 +0200},
  biburl    = {https://dblp.org/rec/conf/cvpr/EleziYALA22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
````

We use the semi-consistency method CSD as developed by Jeong et al. Consider citing the following paper

````
@inproceedings{DBLP:conf/nips/JeongLKK19,
  author    = {Jisoo Jeong and
               Seungeui Lee and
               Jeesoo Kim and
               Nojun Kwak},
  editor    = {Hanna M. Wallach and
               Hugo Larochelle and
               Alina Beygelzimer and
               Florence d'Alch{\'{e}}{-}Buc and
               Emily B. Fox and
               Roman Garnett},
  title     = {Consistency-based Semi-supervised Learning for Object detection},
  booktitle = {Advances in Neural Information Processing Systems 32: Annual Conference
               on Neural Information Processing Systems 2019, NeurIPS 2019, December
               8-14, 2019, Vancouver, BC, Canada},
  pages     = {10758--10767},
  year      = {2019},
  url       = {https://proceedings.neurips.cc/paper/2019/hash/d0f4dae80c3d0277922f8371d5827292-Abstract.html},
  timestamp = {Mon, 16 May 2022 15:41:51 +0200},
  biburl    = {https://dblp.org/rec/conf/nips/JeongLKK19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
````
