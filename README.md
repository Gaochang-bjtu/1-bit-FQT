1-bit FQT
====
This repository is the official implementation of [1-Bit FQT: Pushing the Limit of Fully Quantized
Training to 1-bit]

INSTALL
----
Tested with PyTorch 1.4.0 + CUDA 10.1.

Step 1: Install apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Step 2: Install this repo
```bash
# Make sure that your nvcc version is compatible with the cuda library version used by PyTorch
nvcc --version
cd pytorch_minimax
python setup.py install
cd ..
```

QAT
----

python imagenet.py --arch resnet18 

FQT
----
python cifar10_imagenet.py

# Reference
* [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/pdf/1603.05279.pdf)
* https://github.com/jiecaoyu/XNOR-Net-PyTorch
*[A Statistical Framework for Low-bitwidth Training of Deep Neural Networks(https://arxiv.org/abs/2010.14298)]
* https://github.com/cjf00000/StatQuant
