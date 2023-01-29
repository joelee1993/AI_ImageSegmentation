#!/bin/bash
 
#SBATCH -p gpuq
#SBATCH --gres=gpu:4
#SBATCH --job-name=para_BCE_module
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j217435@gmail.com

module load pytorch-extra-py37-cuda10.2-gcc8/1.8.1 pytorch-py37-cuda10.2-gcc8/1.8.1 pytorch-py37-cuda10.2-gcc/1.6.0
module load python37
module load nvhpc
module load cuda10.2/toolkit/10.2.89 
pip install simpleitk
python -c"import torch; device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu'); print(device)"

python UNet3d_para_BCEonehot1208.py --mode "train" \
                --model_save_name ""\
                --train_data_dir "" \
                --label_data_dir "" \
		    --model_save_dir ""\
                --lr 0.001 \
                --epochs 50\
                --bs 4\
