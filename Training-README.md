# 🧠 ImageNet Training Setup Guide

This document outlines the setup process for training ImageNet models on AWS using custom AMIs, EBS volumes, and Hugging Face datasets.

---

## 📦 1. ImageNet Data Download

### Instance Details
- **Instance Type:** `t3.medium`  
- **EBS Volume:** 350GB attached and mounted at `/Data`


1.1 **Setup EBS Volume**
```commandline
# Check the name for the 350GB disk. Can be nvme1n1, nvme2n1..
lsblk

sudo file -s /dev/nvme2n1
lsblk -f

# No need to do this while reattaching to train instance
sudo mkfs -t xfs /dev/nvme2n1

sudo mkdir /Data
sudo file -s /dev/nvme2n1
sudo mount /dev/nvme2n1 /Data
lsblk
```

Add volume to fstab so that it is mounted on startup     
```commandline
sudo cp /etc/fstab /etc/fstab.orig

# Get the UUID of this volume, and add it to fstab
sudo blkid
sudo vim /etc/fstab

(fstab entry:)
UUID=fda4420c-9eaa-4e57-bfe6-99f46f98d9f2  /Data  xfs  defaults,nofail  0  2
```

Testing mounting and unmounting 
```commandline
sudo umount /Data
lsblk
sudo mount -a
```


1.2 **Set up directories and environment**
   ```bash
   export HF_HOME=/Data/hf_cache
   ```
   - Hugging Face cache: /Data/hf_cache
   - Dataset cache: /Data/datasets/cache
1.3 **Download the dataset from Hugging Face**
    Download script:  
```python
from datasets import load_dataset
dataset = load_dataset(
    "ILSVRC/imagenet-1k",
    split="train",
    cache_dir="/Data/datasets_cache"  # 👈 forces download to EBS volume
)
```        

    - Estimated time: 
      - Data download: ~1 hour
      - Train/validation split generation: ~2 hours
1.4 Post-download
    - Detach the 350GB EBS volume (to reuse for training later).

## 💽 2. Creating a Training AMI

### Instance Details
- **Instance Type**: `g4dn.xlarge`
- **Purpose**: Create reusable AMI with GPU drivers, CUDA, and dependencies.
- **Root volume**: 30GB


### Step 2.1 — Install NVIDIA Drivers  
```commandline
uname -a
sudo apt-get update && sudo apt upgrade
sudo reboot  # if kernel updated

sudo apt-get install linux-headers-$(uname -r) 
sudo apt install gcc build-essential -y
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices
ubuntu-drivers list --gpgpu

# Install latest compatible driver
sudo apt install nvidia-driver-570-server
sudo reboot
nvidia-smi  # should show CUDA 12.8

```

### Step 2.2 — Install `uv` and Python Packages  

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version

# Create virtual environment
uv venv my-venv --python 3.12
source my-venv/bin/activate
```  

**Install dependencies**

```bash
uv pip install datasets
uv pip install -U "huggingface_hub[cli]"
uv pip install Pillow
uv pip install matplotlib
uv pip install albumentations
uv pip install tensorboard

# CUDA 12.8
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```   

### Step 2.3 — Clone GitHub Repository  

```commandline
ssh-keygen -t rsa -b 4096 -C "youremail@example.com"
chmod 400 ~/.ssh/id_rsa
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

git clone git@github.com:code4koustav/ERA_V4-S9.git
```

**Update your bash profile**
```commandline
export HF_HOME=/Data/hf_cache
source ~/.bashrc

```  

### Step 2.4 — Create AMI
- Exclude /Data EBS volume from the image.
- AMI Name: imagenet-train-ami
- Snapshot Size: ~16GB  


## ☁️ 3. Launching a Spot Instance for Training

### Instance Details
- **AMI**: imagenet-train-ami
- **Instance Type:** `g4dn.xlarge` to start with
    g5.xlarge next -> g5.2xlarge final
- **Network Settings -> Zone**: ap-south-1a (Same zone as the EBS volume containing the dataset, so that it can be attached)
- Select Spot instance option


### Step 3.1 - Attach and Mount the previoous Data EBS Volume

```commandline
    # Check the disk name for the 350GB disk
    lsblk
    sudo mkdir /Data
    sudo file -s /dev/nvme2n1
    sudo mount /dev/nvme2n1 /Data
    lsblk
```   

### Step 3.2 - Mount the EC2 instance's SSD volume and copy Imagenet data there  
   Check the disk name for the EC2 instance's volume -- should be nvme1n1 usually
   ```
   lsblk
   sudo mkfs -t ext4 /dev/nvme1n1
   sudo mkdir /Imagenet
   sudo mount /dev/nvme1n1 /Imagenet
   lsblk
   ```   

   Copy the huggingface datasets cache directory only   
   ```
   sudo apt-get install -y rsync 
   mkdir /Imagenet/datasets_cache
   #cp -r /Data/datasets_cache /Imagenet
   sudo rsync -ah --info=progress2 --inplace /Data/datasets_cache/ /Imagenet/datasets_cache/
   ```   

### Step 3 - Environment setup

- Add the Huggingface Dataset cache directory to .bashrc. No need to set HF_HOME  
```
   export HF_DATASETS_CACHE="/Imagenet/datasets_cache"
   #export HF_HOME=/Imagenet/hf_cache
   #unset HF_HOME
   source ~/.bashrc
```

- Virtual environment is at:
   source my-venv/bin/activate 

 - Some python packages were installed after AMI image was created, so they got missed..   
   uv pip install albumentations   
   uv pip install tensorboard
   uv pip install psutil pynvml

 - Pull latest master on the git repo
 - Update hyperparams in main.py
 - Create checkpoints dir under /Data ebs volume

### Step 4 - Start training

 - Start training inside screen/tmux session:
```commandline
   screen -S train
   source my-venv/bin/activate
   huggingface-cli login

   python main.py | tee /Data/run0.log
```

- Run tensorboard
```commandline
screen -S tf
source my-venv/bin/activate
tensorboard --logdir=/Data/tf_runs --host=0.0.0.0 --port=6006
```
Open port 6006 from inbound rules

