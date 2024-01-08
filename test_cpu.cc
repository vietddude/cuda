#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv.h"
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
    std::cout << "mnist train number: " << n_train << std::endl;
    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;

    float accuracy = 0.0;
    Network dnn1 = dnn_CPU();
    dnn1.load_parameters("./model/layer-parameters.bin");
    dnn1.forward(dataset.test_data);
    accuracy = compute_accuracy(dnn1.output(), dataset.test_labels);
    std::cout << "----------------------------------------\n";
    std::cout << "CPU accuracy: " << accuracy << std::endl;
    return 0;
}