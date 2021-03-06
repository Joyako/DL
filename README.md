# Code in Pytorch
**This project consists of three parts, namely data processing, image processing and Deep Learing code.**

This is a [PyTorch](https://github.com/pytorch/pytorch)
implementation of Deep Learing code.

**Data Processing**  
This is mainly write data to LMDB or HDF5.

**Image Processing**  
The main function of this part that is implemented in C++ is adaptive binarization, line detection and 
attributes of gray level co-occurrence matrix. Also, I provide python interface.when you compile code , you 
need copy pybind11 to specify path.

1.A threshold T is calculated for every pixel in the image using the following formula:  
  Niblack local threshold:  
    T = m(x,y) - k * s(x,y)  
  Sauvola local threshold( Sauvola is a modification of Niblack technique):  
    T = m(x,y) * (1 + k * ((s(x,y) / R) - 1))  

**Loss**  
The three kind of loss are cetner loss, focal loss, quadratic kappa loss respectively.

## Installion
1. Download pybind11
    ```bash
	git clone https://github.com/pybind/pybind11.git
    ```
2. Pytorch
    ```bash
    conda install pytorch-nightly cuda92 -c pytorch
    ```

License: MIT license (MIT)
