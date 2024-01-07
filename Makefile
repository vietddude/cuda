# ##############################################################################
# test.o: test.cc
# 	nvcc -arch=sm_75 --compile test.cc -I./ -L/usr/local/cuda/lib64 -lcudart

# test: test.o
# 	nvcc -arch=sm_75 -o test -lm -lcuda -lrt test.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/optimizer/*.o src/layer/kernel/*.o -I./ -L/usr/local/cuda/lib64 -lcudart

# test_model: test
# 	./test
# ##############################################################################

# train: train.o 
# 	nvcc -arch=sm_75 -o train -lm -lcuda -lrt train.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/optimizer/*.o src/layer/kernel/*.o -I./ -L/usr/local/cuda/lib64 -lcudart

# train.o: train.cc
# 	nvcc -arch=sm_75 --compile train.cc -I./ -L/usr/local/cuda/lib64 -lcudart

# train_model: train
# 	./train

# ############################################################################
# dnn.o: dnn.cc
# 	nvcc -arch=sm_75 --compile dnn.cc -I./ -L/usr/local/cuda/lib64 -lcudart

# network.o: src/network.cc
# 	nvcc -arch=sm_75 --compile src/network.cc -o src/network.o -I./ -L/usr/local/cuda/lib64 -lcudart

# mnist.o: src/mnist.cc
# 	nvcc -arch=sm_75 --compile src/mnist.cc -o src/mnist.o  -I./ -L/usr/local/cuda/lib64 -lcudart

# layer: src/layer/conv.cc src/layer/ave_pooling.cc src/layer/fully_connected.cc src/layer/max_pooling.cc src/layer/relu.cc src/layer/sigmoid.cc src/layer/softmax.cc 
# 	nvcc -arch=sm_75 --compile src/layer/ave_pooling.cc -o src/layer/ave_pooling.o -I./ -L/usr/local/cuda/lib64 -lcudart
# 	nvcc -arch=sm_75 --compile src/layer/conv.cc -o src/layer/conv.o -I./ -L/usr/local/cuda/lib64 -lcudart
# 	nvcc -arch=sm_75 --compile src/layer/conv_kernel.cc -o src/layer/conv_kernel.o -I./ -L/usr/local/cuda/lib64 -lcudart
# 	nvcc -arch=sm_75 --compile src/layer/fully_connected.cc -o src/layer/fully_connected.o -I./ -L/usr/local/cuda/lib64 -lcudart
# 	nvcc -arch=sm_75 --compile src/layer/max_pooling.cc -o src/layer/max_pooling.o -I./ -L/usr/local/cuda/lib64 -lcudart
# 	nvcc -arch=sm_75 --compile src/layer/relu.cc -o src/layer/relu.o -I./ -L/usr/local/cuda/lib64 -lcudart
# 	nvcc -arch=sm_75 --compile src/layer/sigmoid.cc -o src/layer/sigmoid.o -I./ -L/usr/local/cuda/lib64 -lcudart
# 	nvcc -arch=sm_75 --compile src/layer/softmax.cc -o src/layer/softmax.o -I./ -L/usr/local/cuda/lib64 -lcudart

# kernel_0: 
# 	rm -f src/layer/kernel/*.o
# 	nvcc -arch=sm_75 --compile src/layer/kernel/kernel.cu -o src/layer/kernel/kernel.o -I./ -L/usr/local/cuda/lib64 -lcudart 
# 	nvcc -arch=sm_75 --compile src/layer/kernel/kernel_forward.cu -o src/layer/kernel/kernel_forward.o -I./ -L/usr/local/cuda/lib64 -lcudart

# kernel_1: 
# 	rm -f src/layer/kernel/*.o
# 	nvcc -arch=sm_75 --compile src/layer/kernel/kernel.cu -o src/layer/kernel/kernel.o -I./ -L/usr/local/cuda/lib64 -lcudart 
# 	nvcc -arch=sm_75 --compile src/layer/kernel/kernel_forward_1.cu -o src/layer/kernel/kernel_forward.o -I./ -L/usr/local/cuda/lib64 -lcudart

# kernel_2: 
# 	rm -f src/layer/kernel/*.o
# 	nvcc -arch=sm_75 --compile src/layer/kernel/kernel.cu -o src/layer/kernel/kernel.o -I./ -L/usr/local/cuda/lib64 -lcudart 
# 	nvcc -arch=sm_75 --compile src/layer/kernel/kernel_forward_2.cu -o src/layer/kernel/kernel_forward.o -I./ -L/usr/local/cuda/lib64 -lcudart

# loss: src/loss/cross_entropy_loss.cc src/loss/mse_loss.cc
# 	nvcc -arch=sm_75 --compile src/loss/cross_entropy_loss.cc -o src/loss/cross_entropy_loss.o -I./ -L/usr/local/cuda/lib64 -lcudart
# 	nvcc -arch=sm_75 --compile src/loss/mse_loss.cc -o src/loss/mse_loss.o -I./ -L/usr/local/cuda/lib64 -lcudart

# optimizer: src/optimizer/sgd.cc
# 	nvcc -arch=sm_75 --compile src/optimizer/sgd.cc -o src/optimizer/sgd.o -I./ -L/usr/local/cuda/lib64 -lcudart

# clean:
# 	rm -f train test
# 	rm -f *.o src/*.o src/layer/*.o src/loss/*.o src/optimizer/*.o src/layer/kernel/*.o

# setup:
# 	make network.o
# 	make mnist.o
# 	make layer
# 	make loss
# 	make optimizer


# Compiler options
NVCC = nvcc
NVCC_FLAGS = -arch=sm_75

# Directories
INCLUDE_DIRS = -I./ -I/usr/local/cuda/lib64
CUDA_LIB_DIR = -L/usr/local/cuda/lib64
CUDART_LIB = -lcudart

# Files
SRCS = test.cc train.cc dnn.cc
NETWORK_SRCS = src/network.cc
MNIST_SRCS = src/mnist.cc
LAYER_SRCS = src/layer/conv.cc src/layer/ave_pooling.cc src/layer/fully_connected.cc src/layer/max_pooling.cc src/layer/relu.cc src/layer/sigmoid.cc src/layer/softmax.cc
LOSS_SRCS = src/loss/cross_entropy_loss.cc src/loss/mse_loss.cc
OPTIMIZER_SRCS = src/optimizer/sgd.cc
KERNEL_SRCS = src/layer/kernel/kernel.cu src/layer/kernel/kernel_forward.cu src/layer/kernel/kernel_forward_1.cu src/layer/kernel/kernel_forward_2.cu

# Object files
OBJS = $(SRCS:.cc=.o)
NETWORK_OBJS = $(NETWORK_SRCS:.cc=.o)
MNIST_OBJS = $(MNIST_SRCS:.cc=.o)
LAYER_OBJS = $(LAYER_SRCS:.cc=.o)
LOSS_OBJS = $(LOSS_SRCS:.cc=.o)
OPTIMIZER_OBJS = $(OPTIMIZER_SRCS:.cc=.o)
KERNEL_OBJS = $(KERNEL_SRCS:.cu=.o)

# Targets
test: test.o $(OBJS) $(NETWORK_OBJS) $(MNIST_OBJS) $(LAYER_OBJS) $(LOSS_OBJS) $(OPTIMIZER_OBJS) $(KERNEL_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ -lm -lcuda -lrt $^ $(INCLUDE_DIRS) $(CUDA_LIB_DIR) $(CUDART_LIB)

train: train.o $(OBJS) $(NETWORK_OBJS) $(MNIST_OBJS) $(LAYER_OBJS) $(LOSS_OBJS) $(OPTIMIZER_OBJS) $(KERNEL_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ -lm -lcuda -lrt $^ $(INCLUDE_DIRS) $(CUDA_LIB_DIR) $(CUDART_LIB)

# Compile C++ source files
%.o: %.cc
	$(NVCC) $(NVCC_FLAGS) --compile $< $(INCLUDE_DIRS) $(CUDA_LIB_DIR) $(CUDART_LIB) -o $@

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) --compile $< $(INCLUDE_DIRS) $(CUDA_LIB_DIR) $(CUDART_LIB) -o $@

# Phony targets
.PHONY: clean setup

clean:
	rm -f train test
	rm -f *.o src/*.o src/layer/*.o src/loss/*.o src/optimizer/*.o src/layer/kernel/*.o

setup: network.o mnist.o layer loss optimizer

network.o: $(NETWORK_SRCS)
	$(NVCC) $(NVCC_FLAGS) --compile $^ $(INCLUDE_DIRS) $(CUDA_LIB_DIR) $(CUDART_LIB) -o $@

mnist.o: $(MNIST_SRCS)
	$(NVCC) $(NVCC_FLAGS) --compile $^ $(INCLUDE_DIRS) $(CUDA_LIB_DIR) $(CUDART_LIB) -o $@

layer: $(LAYER_OBJS)

loss: $(LOSS_OBJS)

optimizer: $(OPTIMIZER_OBJS)

kernel_0: $(KERNEL_OBJS)
	rm -f src/layer/kernel/*.o
	nvcc -arch=sm_75 --compile src/layer/kernel/kernel.cu -o src/layer/kernel/kernel.o $(INCLUDE_DIRS) $(CUDA_LIB_DIR) $(CUDART_LIB)
	nvcc -arch=sm_75 --compile src/layer/kernel/kernel_forward.cu -o src/layer/kernel/kernel_forward.o $(INCLUDE_DIRS) $(CUDA_LIB_DIR) $(CUDART_LIB)

kernel_1: $(KERNEL_OBJS)
	rm -f src/layer/kernel/*.o
	nvcc -arch=sm_75 --compile src/layer/kernel/kernel.cu -o src/layer/kernel/kernel.o $(INCLUDE_DIRS) $(CUDA_LIB_DIR) $(CUDART_LIB)
	nvcc -arch=sm_75 --compile src/layer/kernel/kernel_forward_1.cu -o src/layer/kernel/kernel_forward.o $(INCLUDE_DIRS) $(CUDA_LIB_DIR) $(CUDART_LIB)

kernel_2: $(KERNEL_OBJS)
	rm -f src/layer/kernel/*.o
	nvcc -arch=sm_75 --compile src/layer/kernel/kernel.cu