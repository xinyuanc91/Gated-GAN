# Gated-GAN: Adversarial Gated Networks for Multi-Collection Style Transfer.

This repository contains the Torch code for our paper “Gated-GAN: Adversarial Gated Networks for Multi-Collection Style Transfer” on [IEEE Trans. On Image Processing 2019](https://ieeexplore.ieee.org/document/8463508) ([Arxiv](https://arxiv.org/pdf/1904.02296.pdf)). This code is based on the Torch implementation of CycleGAN provided by [Junyan Zhu](https://github.com/junyanz/CycleGAN). You may need to train several times as the quality of the results is sensitive to the initialization.

Our model architecture is defined as depicted below, please refer to the paper for more details: 

<img src='imgs/architecture.jpg' width="500px"/>

## Results

The results below are produced from a single network:  

<img src='imgs/multistyle.jpg' width="500px"/>

### Datasets
[Download Link](https://drive.google.com/drive/folders/10N972-REqb1R0rqkAB4jRFuNnFijTEgC?usp=sharing)

### Training


    bash train.sh


### Testing

    bash test.sh

## Citation
```bibtex
@ARTICLE{8463508,
  author={Chen, Xinyuan and Xu, Chang and Yang, Xiaokang and Song, Li and Tao, Dacheng},
  journal={IEEE Transactions on Image Processing}, 
  title={Gated-GAN: Adversarial Gated Networks for Multi-Collection Style Transfer}, 
  year={2019},
  volume={28},
  number={2},
  pages={546-560},
  keywords={Logic gates;Gallium nitride;Training;Generative adversarial networks;Decoding;Painting;Semantics;Multi-style transfer;adversarial generative networks},
  doi={10.1109/TIP.2018.2869695}}
```

