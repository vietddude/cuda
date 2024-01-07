#ifndef SRC_LAYER_KERNEL_FORWARD_H
#define SRC_LAYER_KERNEL_FORWARD_H
#include "./kernel.h"

class KernelInterface
{
public:
    void forward_kernel(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                        const int num_samples, const int output_channel, const int input_channel,
                        const int height_in, const int width_in, const int kernel_height);
};
#endif