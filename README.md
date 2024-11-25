# LiVeDet: Lightweight Density-Guided Adaptive Transformer for Online On-Device Vessel Detection

### Zijie Zhang, Changhong Fu, Yongkang Cao, Mengyuan Li, Haobo Zuo

## Abstract
Vision-based online vessel detection boosts the automation of waterways monitoring, transportation management and navigation safety. However, a significant gap exists in on-device deployment between general high-performance PCs/servers and embedded AI processors. Existing state-of-the-art (SOTA) online vessel detectors lack sufficient accuracy and are prone to high latency on the edge AI camera, especially in scenarios with dense vessels and diverse distributions. To solve the above issues, a novel lightweight framework with density-guided adaptive Transformer (LiVeDet) is proposed for the edge AI camera to achieve online on-device vessel detection. Specifically, a new instance-aware representation extractor is designed to suppress cluttered background noise and capture instance-aware content information. Additionally, an innovative vessel distribution estimator is developed to direct superior feature representation learning by focusing on local regions with varying vessel density. Besides, a novel dynamic region embedding is presented to integrate hierarchical features represented by multi-scale vessels. A new benchmark comprising 100 high-definition, high-framerate video sequences from vessel-intensive scenarios is established to evaluate the efficacy of vessel detectors under challenging conditions prevalent in dynamic waterways. Extensive evaluations on this challenging benchmark demonstrate the robustness and efficiency of LiVeDet, achieving 32.9 FPS on the edge AI camera. Futhermore, real-world applications confirm the practicality of the proposed method.
![Workflow of our tracker](https://github.com/vision4robotics/LiVeDet/blob/main/images/main.png)

This figure shows the workflow of our LiVeDet.

## Demonstration running instructions
### 1. Requirements
This code has been tested on Jeston Orin NX 16GB with Jetpack 5.1.1.

1.Ubuntu 20.04

2.Python 3.8.10

3.CUDA 11.4

4.Pytorch 1.11.0 

5.torchvision 0.9.1

Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

### 2. Test

```bash 
python detect.py                                
```
The testing result will be saved in the `runs/detect/output` directory.

### 3. Contact
If you have any questions, please contact me.

Zijie Zhang

Email: [2410022@tongji.edu.cn](2410022@tongji.edu.cn)

For more evaluations, please refer to our paper.

## Acknowledgements 

We sincerely thank [RT-DETR](https://github.com/lyuwenyu/RT-DETR) and [ultralytics](https://github.com/ultralytics/ultralytics) for their efforts.
