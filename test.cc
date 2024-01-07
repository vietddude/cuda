#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/conv_kernel.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"
#include "src/layer/kernel/kernel.h"
#include "src/dnn.h"

int main()
{
    Kernel kernel;
    kernel.printDeviceInfo();

    MNIST dataset("./data/fashion/");
    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();

    float accuracy = 0.0;
    Network dnn2 = dnn_Kernel();
    dnn2.load_parameters("./model/layer-parameters.bin");
    dnn2.forward(dataset.test_data);
    accuracy = compute_accuracy(dnn2.output(), dataset.test_labels);
    std::cout << "GPU accuracy: " << accuracy << std::endl;

    return 0;
}