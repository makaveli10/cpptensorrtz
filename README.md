# cpptensorrtz

## Index
- [Getting Started](#1-getting-started)
- [Resources](#2-resources)

## 1. Getting Started
### Using docker (Preferred)
Note: This method is tested using Ubuntu Distros. This method doesn't work for Windows/MacOS as nvidia-docker isn't supported.

1. Install docker. (Skip this step if docker is already installed)
```
 $ sudo apt-get update
 $ sudo apt-get install docker-ce docker-ce-cli containerd.io
```

2. Install nvidia-container-toolkit to use gpus with docker. Reference - [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 
```
 $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
 $ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
 $ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
 $ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
 $ sudo systemctl restart docker
```

3. Pull TensorRT image from NVIDIA GPU cloud. 
```
 $ docker pull nvcr.io/nvidia/tensorrt:20.12-py3
```


## 2. Resources
1. [Docker quick-start](https://docker-curriculum.com/)
2. [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)
3. [TensorRT NVIDIA GPU Cloud](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)
