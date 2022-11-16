# Training and Certification commands for SPACTE

We list below the complete training and certification commands for SPACTE with the specified values of hyper-parameters for a convenient reproduction of our work. The 3 baseline methods, Gaussian, Consistency and SmoothMix, are all included.

All the commands are for `ResNet-110` on `CIFAR10`. The results of `ResNet-50` on `ImageNet` will be released soon.

## SPACTE-Gaussian

### sigma=0.25
```
cd codes/0Gaussian
# training
CUDA_VISIBLE_DEVICES=0 python train.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --noise_sd 0.25 --num_heads 5 --num_noise_vec 2 --lbdlast 1.2
# certification
CUDA_VISIBLE_DEVICES=0 python certify_mhead.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --model_path './save/CIFAR10/R110/noise-0.25/h-5-eps-0.8-m-2-lbdlast-1.2/epoch150.pth' --noise_sd 0.25 --skip 20 --num_heads 5
```
### sigma=0.50
```
cd codes/0Gaussian
# training
CUDA_VISIBLE_DEVICES=0 python train.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --noise_sd 0.5 --num_heads 5 --num_noise_vec 2 --lbdlast 0.8
# certification
CUDA_VISIBLE_DEVICES=0 python certify_mhead.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --model_path './save/CIFAR10/R110/noise-0.5/h-5-eps-0.8-m-2-lbdlast-0.8/epoch150.pth' --noise_sd 0.5 --skip 20 --num_heads 5
```
### sigma=1.00
```
cd codes/0Gaussian
# training
CUDA_VISIBLE_DEVICES=0 python train.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --noise_sd 1.0 --num_heads 5 --num_noise_vec 2 --lbdlast 0.8
# certification
CUDA_VISIBLE_DEVICES=0 python certify_mhead.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --model_path './save/CIFAR10/R110/noise-1.0/h-5-eps-0.8-m-2-lbdlast-0.8/epoch150.pth' --noise_sd 1.0 --skip 20 --num_heads 5
```
## SPACTE-Consistency

### sigma=0.25
```
cd codes/1Consistency
# training
CUDA_VISIBLE_DEVICES=0 python train.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --noise_sd 0.25 --num_heads 5 --num_noise_vec 2 --lbdlast 1.0 --lbd 20
# certification
CUDA_VISIBLE_DEVICES=0 python certify_mhead.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --model_path './save/CIFAR10/R110/noise-0.25/h-5-eps-0.8-m-2-lbdlast-1.0-lbd-20.0/epoch150.pth' --noise_sd 0.25 --skip 20 --num_heads 5
```
### sigma=0.50
```
cd codes/1Consistency
# training
CUDA_VISIBLE_DEVICES=0 python train.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --noise_sd 0.5 --num_heads 5 --num_noise_vec 2 --lbdlast 0.8 --lbd 10
# certification
CUDA_VISIBLE_DEVICES=0 python certify_mhead.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --model_path './save/CIFAR10/R110/noise-0.5/h-5-eps-0.8-m-2-lbdlast-0.8-lbd-10.0/epoch150.pth' --noise_sd 0.5 --skip 20 --num_heads 5
```
### sigma=1.00
```
cd codes/1Consistency
# training
CUDA_VISIBLE_DEVICES=0 python train.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --noise_sd 1.0 --num_heads 5 --num_noise_vec 2 --lbdlast 1.0 --lbd 10
# certification
CUDA_VISIBLE_DEVICES=0 python certify_mhead.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --model_path './save/CIFAR10/R110/noise-1.0/h-5-eps-0.8-m-2-lbdlast-1.0-lbd-10.0/epoch150.pth' --noise_sd 1.0 --skip 20 --num_heads 5
```

The argument `--lbd` is the hyper-parameter $\lambda$ in Consistency. Following the original paper of Consistency, we choose $\lambda=20$ for $\sigma=0.25$, and $\lambda=10$ for $\sigma=0.5$ and $\sigma=1.0$.

## SPACTE-SmoothMix

### sigma=0.25
```
cd codes/2SmoothMix
# training
CUDA_VISIBLE_DEVICES=0 python train.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --noise_sd 0.25 --num_heads 5 --num_noise_vec 2 --lbdlast 1.6 --num_steps 4 --attack_alpha 0.5 --mix_step 0 --eta 5.0
# certification
CUDA_VISIBLE_DEVICES=0 python certify_mhead.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --model_path './save/CIFAR10/R110/noise-0.25/h-5-eps-0.8-m-2-lbdlast-1.6/epoch150.pth' --noise_sd 0.25 --skip 20 --num_heads 5
```
### sigma=0.50
```
cd codes/2SmoothMix
# training
CUDA_VISIBLE_DEVICES=0 python train.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --noise_sd 0.5 --num_heads 5 --num_noise_vec 2 --lbdlast 1.0 --num_steps 4 --attack_alpha 1.0 --mix_step 1 --eta 5.0
# certification
CUDA_VISIBLE_DEVICES=0 python certify_mhead.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --model_path './save/CIFAR10/R110/noise-0.5/h-5-eps-0.8-m-2-lbdlast-1.0/epoch150.pth' --noise_sd 0.5 --skip 20 --num_heads 5
```
### sigma=1.00
```
cd codes/2SmoothMix
# training
CUDA_VISIBLE_DEVICES=0 python train.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --noise_sd 1.0 --num_heads 5 --num_noise_vec 2 --lbdlast 1.6 --num_steps 4 --attack_alpha 2.0 --mix_step 1 --eta 5.0
# certification
CUDA_VISIBLE_DEVICES=0 python certify_mhead.py --arch R110 --dataset CIFAR10 --data_dir $data_dir$ --model_path './save/CIFAR10/R110/noise-1.0/h-5-eps-0.8-m-2-lbdlast-1.6/epoch150.pth' --noise_sd 1.0 --skip 20 --num_heads 5
```
The arguments `--num_steps`, `--attack_alpha`, `--mix_step` and `--eta` are the hyper-parameters of SmoothMix. We follow the settings in the original paper of SmoothMix to set their values.