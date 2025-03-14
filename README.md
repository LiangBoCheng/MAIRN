# MAIRN
This project provides the code and results for 'Multi-Scale Attention-Edge Interactive Refinement Network for Salient Object Detection', ESWA, 2025.
[Link](https://doi.org/10.1016/j.eswa.2025.127056)
# Network Architecture
![image text](https://github.com/LiangBoCheng/MAIRN/blob/main/model/MAIRN.png)
# Requirements
python 3.7 + pytorch 1.9.0 + imageio 2.22.2
# Saliency maps
MAIRN saliency maps: [Baidu](https://pan.baidu.com/s/1NhLZZWftV8RymKxAED7Nqg?pwd=MAIR) (code:MAIR)  
MAIRN-ResNet101 saliency maps: [Baidu](https://pan.baidu.com/s/1jPzY41MeDWIBCWtIWmxnyA?pwd=MAIR) (code:MAIR)  
MAIRN-Res2Net50 saliency maps: [Baidu](https://pan.baidu.com/s/1L2c0D7lNulmfLgLMbVwmOQ?pwd=MAIR) (code:MAIR)  
MAIRN-SwinB saliency maps: [Baidu](https://pan.baidu.com/s/1RIBtinHPaF61UD1E50fzxg?pwd=MAIR) (code:MAIR)
# Training
Run train_MAIRN.py.  
For MAIRN_Res2Net50, please modify paths of [Res2Net50_backbone](https://pan.baidu.com/s/1Ad1GccRH_QYX5JIMJ3Q_Gg?pwd=MAIR) (code: MAIR) in ./model/Res2Net.py.  
For MAIRN_SwinB, Run train_MAIRN_SwinB.py and please modify paths of [SwinB_backbone](https://pan.baidu.com/s/1sLpc5keVTzVl9WAyDdzo9A?pwd=MAIR) (code: MAIR) in ./model/MAIRN_SwinB.py.
# Pre-trained model and testing
Download the following pre-trained model and put it in ./models/, then run test_MAIRN.py  
[MAIRN_ResNet50](https://pan.baidu.com/s/1dLs2skmP-4CEJckZqgiVJQ?pwd=MAIR) (code:MAIR)  
[MAIRN_ResNet101](https://pan.baidu.com/s/1mq2Bo_UElTDXPGiSO9lxAw?pwd=MAIR) (code:MAIR)  
[MAIRN_SwinB](https://pan.baidu.com/s/1J9cyiujUfvYaR0WcMptcqw?pwd=MAIR) (code:MAIR). If you want to test MAIRN-SwinB, please change the testsize to 224.  
[MAIRN_Res2Net50](https://pan.baidu.com/s/1eNNhO_cB5bugJECxUm4WYA?pwd=MAIR) (code:MAIR)
# Evaluation Tool
You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.
