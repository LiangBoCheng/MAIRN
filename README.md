# MAIRN
# Requirements
python 3.7 + pytorch 1.9.0
# Saliency maps
MAIRN saliency maps: [Baidu](https://pan.baidu.com/s/1vaCSD5oxoqtN_ssO1X-Rhw?pwd=MAIR) (code:MAIR)  
MAIRN_Res2Net50 saliency maps: [Baidu](https://pan.baidu.com/s/1HZxzCxFjvzXkt2VatQ-qwA?pwd=MAIR) (code:MAIR)
# Training
Run train_MAIRN.py.  
For MAIRN_Res2Net50, please modify paths of [Res2Net50_backbone](https://pan.baidu.com/s/1Ad1GccRH_QYX5JIMJ3Q_Gg?pwd=MAIR) (code: MAIR) in ./model/Res2Net.py.
# Pre-trained model and testing
Download the following pre-trained model and put it in ./models/, then run test_MAIRN.py  
[MAIRN_DUTS-TR](https://pan.baidu.com/s/1tfb3PlmYOFK_0suKV6dRVw?pwd=MAIR) (code:MAIR)  
[MAIRN_Res2Net50](https://pan.baidu.com/s/1JUTY-RXhq8Xy6di7r5DcfA?pwd=MAIR) (code:MAIR)
# Evaluation Tool
You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.
