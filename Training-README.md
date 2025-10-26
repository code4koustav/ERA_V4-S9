# 🧠 ImageNet Training Setup Guide

This document outlines the setup process for training ImageNet models on AWS using custom AMIs, EBS volumes, and Hugging Face datasets.

---

## 📦 1. ImageNet Data Download

### Instance Details
- **Instance Type:** `t3.medium`  
- **EBS Volume:** 350GB attached and mounted at `/Data`


1.**Setup EBS Volume**
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


2.**Set up directories and environment**
   ```bash
   export HF_HOME=/Data/hf_cache
   ```
   - Hugging Face cache: /Data/hf_cache
   - Dataset cache: /Data/datasets/cache
3.**Download the dataset from Hugging Face**
    Download script:
    - Estimated time: 
      - Data download: ~1 hour
      - Train/validation split generation: ~2 hours
4.Post-download
    - Detach the 350GB EBS volume (to reuse for training later).

## 💽 2. Creating a Training AMI

### Instance Details
- **Instance Type**: `g4dn.xlarge`
- **Purpose**: Create reusable AMI with GPU drivers, CUDA, and dependencies.
- **Root volume**: 30GB


### Step 1 — Install NVIDIA Drivers  
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

### Step 2 — Install `uv` and Python Packages  

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

### Step 3 — Clone GitHub Repository  

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

### Step 4 — Create AMI
- Exclude /Data EBS volume from the image.
- AMI Name: imagenet-train-ami
- Snapshot Size: ~16GB  


## ☁️ 3. Launching a Spot Instance for Training

### Instance Details
- **AMI**: imagenet-train-ami
- **Instance Type:** `g4dn.xlarge` to start with
- **Zone**: ap-south-1a (Same zone as the EBS volume containing the dataset, so that it can be attached)
- Select Spot instance option


#### Step 1 - Attach and Mount the EBS Volume

```commandline
    # Check the disk name for the 350GB disk
    lsblk
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

#### Step 2 - Environment setup

- add HF_HOME to .bashrc
```commandline
   export HF_HOME=/Data/hf_cache
   source ~/.bashrc
```

- Virtual environment is at:
   source my-venv/bin/activate

 - Some python packages were installed after AMI image was created, so they got missed..   
   uv pip install albumentations   
   uv pip install tensorboard

 - Pull latest master on the git repo
 - Update hyperparams in main.py
 - Create checkpoints dir under /Data ebs volume
 - Start training inside screen/tmux session:
```commandline
   screen -S train
   source my-venv/bin/activate
   huggingface-cli login

```
