#include "./kernel_interface.h"
#include <iostream>
#include <cmath>

#define TILE_WIDTH 32
#define CONV_KERNEL_SIZE 5

__global__ void forward_gpu_tiled(float *output, const float *input, const float *kernel,
                                  const int num_samples, const int output_channel, const int input_channel,
                                  const int height, const int width, const int kernel_size)
{
    int tileSize = TILE_WIDTH + kernel_size - 1;
    int height_out = height - kernel_size + 1;
    int width_out = width - kernel_size + 1;

    int height_grid = ceil(1.0 * height_out / TILE_WIDTH);
    int width_grid = ceil(1.0 * width_out / TILE_WIDTH);

    __shared__ float smem[(TILE_WIDTH + CONV_KERNEL_SIZE - 1) *
                              (TILE_WIDTH + CONV_KERNEL_SIZE - 1) +
                          (CONV_KERNEL_SIZE * CONV_KERNEL_SIZE)];

    float *x_s = (float *)&smem[0];
    float *k_s = (float *)&smem[tileSize * tileSize];

    int n = blockIdx.x;
    int m = blockIdx.y;

    // vertical base out data index for the block
    // blockIdx.z -> number of TILES needed for calculating entire output feature map
    int h_in = (blockIdx.z / width_grid); // TILE's index in output feature map
    // horizontal base out data index for the block
    int w_in = (blockIdx.z % width_grid); // TILE's index in output feature map
    // h_in and w_in used as shorthand for threadIdx.x and threadIdx.y
    int h0 = threadIdx.y;               // index in TILE
    int w0 = threadIdx.x;               // index in TILE
    int h_out = h_in * TILE_WIDTH + h0; // real index in output feature map
    int w_out = w_in * TILE_WIDTH + w0; // real index in output feature map
    // h and w is not center point, it's upper left corner point of Input image

    float acc = 0.0f;

    for (int channel = 0; channel < input_channel; channel++)
    {
        if (h0 < kernel_size && w0 < kernel_size)
        {
            k_s[h0 * kernel_size + w0] = kernel[m * (input_channel * kernel_size * kernel_size) +
                                                channel * (kernel_size * kernel_size) +
                                                h0 * kernel_size + w0];
        }
        __syncthreads();
        for (int i = h_out; i < h_in + tileSize; i += TILE_WIDTH)
        {
            for (int j = w_out; j < w_in + tileSize; j += TILE_WIDTH)
            {
                if (i - h_in < tileSize && j - w_in < tileSize)
                {
                    x_s[(i - h_in) * tileSize + j - w_in] = input[(n * (input_channel * height * width)) +
                                                                  channel * (height * width) +
                                                                  i * width + j];
                }
            }
        }
        __syncthreads();
        for (int i = 0; i < kernel_size; i++)
        {
            for (int j = 0; j < kernel_size; j++)
            {
                if (h_out < height_out && w_out < width_out)
                {
                    acc += x_s[(h0 + i) * tileSize + w0 + j] * k_s[i * kernel_size + j];
                }
            }
        }
        __syncthreads();
    }
    if (h_out < height_out && w_out < width_out)
    {
        output[n * (output_channel * height_out * width_out) +
               m * (height_out * width_out) +
               h_out * width_out + w_out] = acc;
    }
}

__host__ void KernelInterface::forward_kernel(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                                              const int num_samples, const int output_channel, const int input_channel,
                                              const int height_in, const int width_in, const int kernel_height)
{
    const int height_out = height_in - kernel_height + 1;
    const int width_out = width_in - kernel_height + 1;

    // Allocate device memory
    float *device_input, *device_output, *device_weight;
    cudaMalloc(&device_input, num_samples * input_channel * height_in * width_in * sizeof(float));              // input features map is input_channel
    cudaMalloc(&device_output, num_samples * output_channel * height_out * width_out * sizeof(float));          // output feature map is output_channel
    cudaMalloc(&device_weight, output_channel * input_channel * kernel_height * kernel_height * sizeof(float)); // input_channel * output_channel filter Maps of size kernel_height * kernel_height

    // Copy input and mask data to device
    cudaMemcpy(device_input, input_data, num_samples * input_channel * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight_data, output_channel * input_channel * kernel_height * kernel_height * sizeof(float), cudaMemcpyHostToDevice);

    //
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
    int height_grid = (height_out - 1) / TILE_WIDTH + 1;
    int width_grid = (width_out - 1) / TILE_WIDTH + 1;
    int z = height_grid * width_grid;
    dim3 gridSize(num_samples, output_channel, z);

    forward_gpu_tiled<<<gridSize, blockSize>>>(device_output, device_input, device_weight, num_samples, output_channel, input_channel, height_in, width_in, kernel_height);
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    // Copy the output back to host
    cudaMemcpy(output_data, device_output, num_samples * output_channel * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_weight);
}