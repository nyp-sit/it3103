## Instructions for Running Docker container 



Pre-requisites: 

1. Enable WSL in your Windows 10/11.  You can refer to the following link for instructions on how to do it. 

   [Install WSL | Microsoft Docs](https://docs.microsoft.com/en-us/windows/wsl/install)

2. Install Docker for Desktop 

   [Get Started with Docker | Docker](https://www.docker.com/get-started)

3. (Optional) If you want to use GPU from within the docker container, you need to follow the following steps to set it up: 

   [Setup nvidia docker with WSL](https://github.com/nyp-sit/it3103/blob/main/nvidia-docker-wsl2.md)

   

### Run the docker container

From the command prompt, type the following to start the docker to use all GPUs on host, and mapped the local host data directory (e.g. `c:/Users/markk/balloon_project`) to the directory in container (e.g. `/home/ubuntu/balloon_project`), and also expose the port `8888` and port `6006` respectively for accessing the Jupyter server and Tensorboard running inside the container.

If you have NVIDIA docker support setup (point 3 above), then you can run:

```bash
docker run -it --gpus all -v  c:/Users/markk/balloon_project:/home/ubuntu/balloon_project -p 8888:8888 -p 6006:6006 --name dlcontainer --hostname it3103 ainyp/dlimage:1.0
```

Otherwise, run the following if you are using only CPU: 

```bash
docker run -it -v  c:/Users/markk/balloon_project:/home/ubuntu/balloon_project -p 8888:8888 -p 6006:6006 --name dlcontainer --hostname it3103 ainyp/dlimage:1.0
```


The first time you run the command, it will take a while as it needs to download the docker image (which is about 8 GB) from the docker hub. 

If you exit the docker, and try to do `docker run` again, you may encounter error saying dlcontainer already used, as you cannot run two containers with the same container name. 
You can either remove the exited docker container by the following:

```bash
docker rm dlcontainer 
```

Or if you want to continue with the previous exited container: 

```bash
docker start -i dlcontainer
```

#### Start the Jupyter server

From within the container, type the following

```
jupyter lab --ip=0.0.0.0 --no-browser 
```



You can use the terminal within the Jupyter server to do your training, evaluation and visualization. 



## Preserve the content within the docker container 

If you have installed anything within the container or downloaded any files that you want to preserve inside the container, you can commit the changes to a new docker image. Assuming you have a docker hub ID called  `myname`:

```bash
docker commit dlcontainer myname/myimage:1.0
```

This will create a docker image called `myname/myimage` with a tag 1.0.  The hub ID part is optional if you do not need to push your image to Docker hub. 


It is recommend to have all the project artifacts, e.g. your datasets, your models, downloaded pretrained models, config files, etc, to be resided within the mapped folder (e.g. `/home/ubuntu/balloon_project`), so that the contents are actually copied to local folder (e.g. `c:\Users\markk\balloon_project`).  

Once you are done with the testing, you can zip up the entire balloon_project on your host, and upload to the cloud VM to continue training using the GPU on the Cloud VM.  If you zip file is too big to upload through the Jupyter notebook, you can upload it to some cloud storage first and download it inside the cloud VM. 

