# StrokeAI
This is an extraction version code of strokeAI dectection
## Directory
### StrokeAI
model

    |-- Att_Unet.py
    |-- UNet3d_FL.py
    |-- losses.py
    |-- train.py
    |-- utiles.py
    |-- testModel.ipynb    
    |-- unet_BCE_module.sh

CT_Processed

    |-- utils.py
    |-- CT_Processed.py
    
## Tumor Task

Test our basic 3D Unet model on a public tumor task

### Training Result
#### Unet

![image](https://github.com/j217435/StrokeAI/blob/main/Figure/Unet.PNG)

#### Unet_Visualization

![image](https://github.com/j217435/StrokeAI/blob/main/Figure/307_unet_Axial.gif)

#### Attention-Unet
![image](https://github.com/j217435/StrokeAI/blob/main/Figure/AttUnet.PNG)

#### Attention-Unet_Visualization

![image](https://github.com/j217435/StrokeAI/blob/main/Figure/307_att_Axial.gif)
