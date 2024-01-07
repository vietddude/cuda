##############################################################################
test: test.o
	rm -f test
	nvcc -arch=sm_75 -o test -lm -lcuda -lrt test.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/optimizer/*.o src/layer/kernel/*.o -I./ -L/usr/local/cuda/lib64 -lcudart

test.o: test.cc
	rm -f test.o
	nvcc -arch=sm_75 --compile test.cc -I./ -L/usr/local/cuda/lib64 -lcudart

##############################################################################

save_train_parameters: save_train_parameters.o 
	rm -f save_train_parameters
	nvcc -arch=sm_75 -o save_train_parameters -lm -lcuda -lrt save_train_parameters.o src/network.o src/mnist.o src/layer/*.o src/loss/*.o src/optimizer/*.o src/layer/kernel/*.o -I./ -L/usr/local/cuda/lib64 -lcudart

save_train_parameters.o: save_train_parameters.cc
	rm -f save_train_parameters.o
	nvcc -arch=sm_75 --compile save_train_parameters.cc -I./ -L/usr/local/cuda/lib64 -lcudart

############################################################################
network.o: src/network.cc
	nvcc -arch=sm_75 --compile src/network.cc -o src/network.o -I./ -L/usr/local/cuda/lib64 -lcudart

mnist.o: src/mnist.cc
	nvcc -arch=sm_75 --compile src/mnist.cc -o src/mnist.o  -I./ -L/usr/local/cuda/lib64 -lcudart

layer: src/layer/conv.cc src/layer/ave_pooling.cc src/layer/fully_connected.cc src/layer/max_pooling.cc src/layer/relu.cc src/layer/sigmoid.cc src/layer/softmax.cc 
	nvcc -arch=sm_75 --compile src/layer/ave_pooling.cc -o src/layer/ave_pooling.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/layer/conv.cc -o src/layer/conv.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/layer/conv_kernel.cc -o src/layer/conv_kernel.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/layer/fully_connected.cc -o src/layer/fully_connected.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/layer/max_pooling.cc -o src/layer/max_pooling.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/layer/relu.cc -o src/layer/relu.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/layer/sigmoid.cc -o src/layer/sigmoid.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/layer/softmax.cc -o src/layer/softmax.o -I./ -L/usr/local/cuda/lib64 -lcudart

kernel_0: 
	rm -f src/layer/kernel/*.o
	nvcc -arch=sm_75 --compile src/layer/kernel/kernel.cu -o src/layer/kernel/kernel.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc -arch=sm_75 --compile src/layer/kernel/kernel_forward_0.cu -o src/layer/kernel/kernel_forward.o -I./ -L/usr/local/cuda/lib64 -lcudart

kernel_1: 
	rm -f src/layer/kernel/*.o
	nvcc -arch=sm_75 --compile src/layer/kernel/kernel.cu -o src/layer/kernel/kernel.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc -arch=sm_75 --compile src/layer/kernel/kernel_forward_1.cu -o src/layer/kernel/kernel_forward.o -I./ -L/usr/local/cuda/lib64 -lcudart

kernel_2: 
	rm -f src/layer/kernel/*.o
	nvcc -arch=sm_75 --compile src/layer/kernel/kernel.cu -o src/layer/kernel/kernel.o -I./ -L/usr/local/cuda/lib64 -lcudart 
	nvcc -arch=sm_75 --compile src/layer/kernel/kernel_forward_2.cu -o src/layer/kernel/kernel_forward.o -I./ -L/usr/local/cuda/lib64 -lcudart

loss: src/loss/cross_entropy_loss.cc src/loss/mse_loss.cc
	nvcc -arch=sm_75 --compile src/loss/cross_entropy_loss.cc -o src/loss/cross_entropy_loss.o -I./ -L/usr/local/cuda/lib64 -lcudart
	nvcc -arch=sm_75 --compile src/loss/mse_loss.cc -o src/loss/mse_loss.o -I./ -L/usr/local/cuda/lib64 -lcudart

optimizer: src/optimizer/sgd.cc
	nvcc -arch=sm_75 --compile src/optimizer/sgd.cc -o src/optimizer/sgd.o -I./ -L/usr/local/cuda/lib64 -lcudart

clean:
	rm -f save_train_parameters test
	rm -f *.o src/*.o src/layer/*.o src/loss/*.o src/optimizer/*.o src/layer/kernel/*.o

setup:
	make network.o
	make mnist.o
	make layer
	make loss
	make optimizer