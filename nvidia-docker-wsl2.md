Here are the steps I used in setting up WSL to run nvidia-docker. It may or may not work on you.  Note this only works on Windows 11 (or Windows 10 build 22000 or newer)

1. Install WSL with Ubuntu Distro:  https://docs.microsoft.com/en-us/windows/wsl/install
2. Install WSL CUDA driver: https://developer.nvidia.com/cuda/wsl  (the filename is something like `510.06_gameready_win11_win10-dch_64bit_international`) - this driver works for both WSL as well as native windows. 
After install make sure you can do `nvidia-smi`
3. Inside WSL Ubuntu distro, check if you can do `nvidia-smi`. There is no need to install any driver inside Ubuntu distro.
4. Install NVidia docker support in WSL ubuntu distro using this guide: https://docs.nvidia.com/cuda/wsl-user-guide/index.html. Skip over the Windows 10 preview build thingy as Windows 11 is now available 
In particular, I follow these steps

```bash
$ wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
$ sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb
$ sudo dpkg -i cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb
$ sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-4-local/7fa2af80.pub
$ sudo apt-get update
$ sudo apt-get -y install cuda
```

Run test to make sure your CUDA is working:

```bash
$ cd /usr/local/cuda-11.4/samples/4_Finance/BlackScholes
$ make BlackScholes
$ ./BlackScholes
```

5. Install Docker container (**NOTE** I have tried to skip installing docker inside WSL distro, and instead using Windows Docker Desktop with its WSL integration, but failed to get the NVIDIA toolkit to work. So I have to remove the Windows Docker first and proceed with the following steps)

```bash
$ curl https://get.docker.com | sh 

$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2  

## test your nvidia docker 
$ docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark   
```

and also try following to make sure Tensorflow is working with GPU: 

```bash
docker run -it --gpus all -p 8888:8888 tensorflow/tensorflow:latest-gpu-py3-jupyter 
```

6. Finally, install your Windows Docker for Desktop, and amazingly, you can see the docker are somehow shared between Windows and WSL seamlessly.. You can see the same the docker images in both windows and WSL distro. 

