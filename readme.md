# OPTIMIZED CNN LENET-5 USING CUDA

## How to Run

### Google Colab

1. Open the file `run.ipynb` on **Google Colab** and follow the provided instructions.

### Local Environment on Google Colab

If you prefer to run the code locally on Google Colab, follow these steps:

```bash
%cd /content
!rm -rf cuda
!git clone -b main https://github.com/vietddude/cuda.git
```

Navigate to the `cuda` folder:

```bash
%cd /content/cuda
!make clean
!make setup
```

#### Run Build and Test for CPU:

```bash
!make clean_test_cpu
!make kernel_0
!make test_cpu
!./test_cpu
```

#### Run Build and Test for Kernel:

```bash
!make kernel_0
!make clean_test
!make test
!./test
```

You can perform the same steps for other kernels by replacing `kernel_0` with the appropriate kernel identifier. Follow the identical procedure for each kernel.

Video demo for this project: [Lenet-5 CNN CUDA](https://youtu.be/Q94Z-VZvkBI)
