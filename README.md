# StrokeAI
This is an extraction version code of strokeAI dectection
## Directory
### StrokeAI
model

    |-- UNet3d_FL.py
    |-- losses.py
    |-- train.py
    |-- utiles.py
    |-- testModel.ipynb    

CT_Processed

    |-- utils.py
    |-- CT_Processed.py
    
## Tumor Task

Test our basic 3D Unet model on a public tumor task. The final goal is using this 3D attention Unet to apply on a stroke research. However, regarding it is an on-going project so we test the model on the public tumor set to ensure the basic performance.

### Training Result
#### Unet

![image](https://github.com/j217435/StrokeAI/blob/main/Figure/Unet.PNG)

#### Unet_Visualization

![image](https://github.com/j217435/StrokeAI/blob/main/Figure/307_unet_Axial.gif)

#### Attention-Unet
![image](https://github.com/j217435/StrokeAI/blob/main/Figure/AttUnet.PNG)

#### Attention-Unet_Visualization

![image](https://github.com/j217435/StrokeAI/blob/main/Figure/307_att_Axial.gif)
