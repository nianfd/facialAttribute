# facialAttribute

## Repruduce the results presented in the paper

- Environments
  1. Pytorch 0.4.1
  2. Python 3.X
  
- CelebA dataset (All images are the original resolution of the aligned CelebA images)
  1. All codes folder: facialAttribute/CelebA 
  2. train.py: training the model
  3. predict.py: prediting the test set
  4. eva.py: computing the error rates
  5. training logs (backbone: LightCNN):
```lr: 0.01
Epoch: [0][0/2544]	Time 18.349 (18.349)	Data 17.233 (17.233)	Loss 74.2300 (74.2300)
Epoch: [0][100/2544]	Time 0.234 (1.058)	Data 0.000 (0.833)	Loss 18.7452 (32.6997)
Epoch: [0][200/2544]	Time 3.765 (0.979)	Data 3.562 (0.759)	Loss 18.3180 (25.4119)
Epoch: [0][300/2544]	Time 0.214 (0.928)	Data 0.000 (0.712)	Loss 18.2411 (22.9884)
Epoch: [0][400/2544]	Time 2.848 (0.897)	Data 2.653 (0.681)	Loss 17.7378 (21.7188)
Epoch: [0][500/2544]	Time 0.213 (0.871)	Data 0.001 (0.655)	Loss 17.6288 (20.9372)
Epoch: [0][600/2544]	Time 0.227 (0.840)	Data 0.001 (0.625)	Loss 17.2036 (20.4063)
Epoch: [0][700/2544]	Time 0.234 (0.819)	Data 0.000 (0.603)	Loss 17.8552 (19.9868)
Epoch: [0][800/2544]	Time 0.213 (0.794)	Data 0.000 (0.577)	Loss 17.3375 (19.6693)
Epoch: [0][900/2544]	Time 0.206 (0.773)	Data 0.000 (0.556)	Loss 17.1747 (19.3977)
Epoch: [0][1000/2544]	Time 0.203 (0.750)	Data 0.000 (0.533)	Loss 17.4203 (19.1730)
Epoch: [0][1100/2544]	Time 0.218 (0.730)	Data 0.000 (0.512)	Loss 15.6162 (18.9692)
Epoch: [0][1200/2544]	Time 0.207 (0.710)	Data 0.000 (0.493)	Loss 16.5509 (18.8017)
Epoch: [0][1300/2544]	Time 0.214 (0.693)	Data 0.000 (0.476)	Loss 16.2737 (18.6204)
Epoch: [0][1400/2544]	Time 0.239 (0.676)	Data 0.000 (0.459)	Loss 17.6197 (18.4570)
Epoch: [0][1500/2544]	Time 0.230 (0.660)	Data 0.001 (0.443)	Loss 16.6365 (18.3085)
Epoch: [0][1600/2544]	Time 0.242 (0.644)	Data 0.000 (0.426)	Loss 15.8182 (18.1534)
Epoch: [0][1700/2544]	Time 0.232 (0.628)	Data 0.000 (0.410)	Loss 15.0895 (18.0030)
Epoch: [0][1800/2544]	Time 0.214 (0.614)	Data 0.000 (0.396)	Loss 15.7540 (17.8546)
Epoch: [0][1900/2544]	Time 0.212 (0.600)	Data 0.000 (0.382)	Loss 15.1173 (17.7023)
Epoch: [0][2000/2544]	Time 0.220 (0.588)	Data 0.001 (0.370)	Loss 13.8108 (17.5519)
Epoch: [0][2100/2544]	Time 0.208 (0.576)	Data 0.000 (0.358)	Loss 14.6444 (17.3995)
Epoch: [0][2200/2544]	Time 0.232 (0.565)	Data 0.000 (0.347)	Loss 13.9772 (17.2645)
Epoch: [0][2300/2544]	Time 0.208 (0.555)	Data 0.000 (0.337)	Loss 13.4986 (17.1310)
Epoch: [0][2400/2544]	Time 0.238 (0.546)	Data 0.001 (0.328)	Loss 14.0403 (17.0069)
Epoch: [0][2500/2544]	Time 0.224 (0.538)	Data 0.000 (0.319)	Loss 13.8721 (16.8824)
```
- LFWA dataset (All images are preprocessed based on five landmarks, The processed dataset could be download from [@ BaiduNetdisk](https://pan.baidu.com/s/1-bxzom7IqhvXWWejS48P1Q) , password: xth9)
  1. train, val and test label files: facialAttribute/LFWA 
  2. codes are the same as CelebA (It should be noted that the number of the classes and the featuremap dimension should be changed).

- Equation (3) in our paper is implemented by ``` self.s = np.log((2.0**126)/2)/2```  in combineloss.py
