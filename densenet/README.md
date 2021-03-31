## How to Run

1. Generate "densenet121.wts" TensorRT weights file using [gen_trtwts.py](https://github.com/makaveli10/torchtrtz/blob/main/densenet/gen_trtwts.py).

2. Create a **build** folder. Place the weights in the this folder. 
```
 $ cd densenet
 $ mkdir build 
 $ cp /path/to/densenet121.wts /path/to/repo/densenet.
```
3. Use the below command to run the docker container that you have just downloaded. 
```
 $ docker images
 $ docker run --gpus all -d -it -v {/path/to/this/repo}:/home nvcr.io/nvidia/tensorrt:20.12-py3
```
> Note - Replace **nvcr.io/nvidia/tensorrt:20.12-py3** with the **repository_name:tag** that you see when you run __docker images__ command. 
4. Login to the container by running the following command:
```
 $ docker attach 2be
```
> Note - Replace **2be** with the **Container ID** that you will get by running the following command:
```
 $ docker ps
```

5. If you get the version number by running the above command that means the installation was successful. Next cd into the mounted directory.
```
 $ cd /home
```
6. Run the following commands to create a object file for densenet.cpp file. 
```
 $ cd build
 $ cmake ..
 $ make
```
7. If there are no errors, continue to create a serialized engine and run inference.
You'll need to modify densenet121.cpp to run on sample images. 
```
 $ ./densenet -r
```

## Verify output from [torch impl](https://github.com/makaveli10/torchtrtz/blob/main/densenet/README.md)
TensorRT output:
```
    [-0.587389, -0.329202, -1.83404, -1.89935, -0.928404]
```
