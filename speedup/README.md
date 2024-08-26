# 1-bit FQT

# Build
~~~shell
cd <Repository Root>/csrc/binop
make

#Run
python test.py 

# Environment
## Software
* Python  3.5
* Pytorch 0.3.1
* CUDA    8.0
* gcc     5.4



# Reference
* [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/pdf/1602.02830.pdf)
* [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/pdf/1603.05279.pdf)
* https://github.com/jiecaoyu/XNOR-Net-PyTorch
* [cpu-gemm](http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/gemm/page02/index.html)
* [cpu-conv2d](https://github.com/pytorch/pytorch/blob/f23feca681c5066c70f0fe1516fc2e269d615e93/aten/src/THNN/generic/SpatialConvolutionMM.c)
* [gpu-gemm and gpu-conv2d](https://github.com/1adrianb/bnn.torch/blob/master/BinarySpatialConvolution.cu)
* [popcount](https://github.com/kimwalisch/libpopcnt)
* [Pytorch-XNOR-Net](https://github.com/cooooorn/Pytorch-XNOR-Net)