#include "./kernel_interface.h"
#include <iostream>

#define TILE_WIDTH 32

__global__ void unroll_kernel(int channel_in, int height_in, int width_in, int kernel_size, int height_out, int width_out,
                              float *input, float *output)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int W_unroll = height_out * width_out;
    if (t < channel_in * W_unroll)
    {
        int c = t / W_unroll;        // if t < 28*28, c = 0  // output channel
        int w_unroll = t % W_unroll; // output height * output width
        int h_out = w_unroll / width_out;
        int w_out = w_unroll % width_out;
        int w_base = c * kernel_size * kernel_size;
        int z = c * (width_in * height_in);
        for (int p = 0; p < kernel_size; p++)
        {
            int x = (h_out + p) * width_in;
            for (int q = 0; q < kernel_size; q++)
            {
                int h_unroll = w_base + p * kernel_size + q;
                output[h_unroll * W_unroll + w_unroll] = input[z + x + w_out + q];
            }
        }
    }
}

__global__ void matrix_multiplication_kernel(float *A, float *B, float *C, int m, int n, int k)
{
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
    int r = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int c = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0;
    for (int i = 0; i < (n - 1) / TILE_WIDTH + 1; ++i)
    {
        if (r < m && i * TILE_WIDTH + threadIdx.x < n)
            s_A[threadIdx.y][threadIdx.x] = A[r + (i * TILE_WIDTH + threadIdx.x) * m];
        else
            s_A[threadIdx.y][threadIdx.x] = 0;
        if (i * TILE_WIDTH + threadIdx.y < n && c < k)
            s_B[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) + c * n];
        else
            s_B[threadIdx.y][threadIdx.x] = 0;
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; ++j)
            sum += s_A[threadIdx.y][j] * s_B[j][threadIdx.x];
        __syncthreads();
    }
    if (r < m && c < k)
        C[c * m + r] = sum;
}

__host__ void KernelInterface::forward_kernel(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                                              const int num_samples, const int output_channel, const int input_channel,
                                              const int height_in, const int width_in, const int kernel_height)
{
    const int height_out = height_in - kernel_height + 1;
    const int width_out = width_in - kernel_height + 1;

    // Allocate device memory
    float *device_input, *device_input_unroll, *device_output, *device_weight;
    cudaMalloc(&device_input, num_samples * input_channel * height_in * width_in * sizeof(float));              // input features map is input_channel
    cudaMalloc(&device_output, num_samples * output_channel * height_out * width_out * sizeof(float));          // output feature map is output_channel
    cudaMalloc(&device_weight, output_channel * input_channel * kernel_height * kernel_height * sizeof(float)); // input_channel * output_channel filter Maps of size kernel_height * kernel_height

    // Copy input and mask data to device
    cudaMemcpy(device_input, input_data, num_samples * input_channel * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, weight_data, output_channel * input_channel * kernel_height * kernel_height * sizeof(float), cudaMemcpyHostToDevice);

    // Set the kernel dimensions and call the kernel
    dim3 blockSize1(1024);
    dim3 gridSize1((input_channel * height_out * width_out - 1) / 1024 + 1);
    // Launch the kernel
    dim3 blockSize2(32, 32);
    int m = height_out * width_out;
    int n = input_channel * kernel_height * kernel_height;
    int k = output_channel;
    dim3 gridSize2((k - 1) / blockSize2.x + 1, (m - 1) / blockSize2.y + 1);
    cudaMalloc(&device_input_unroll, input_channel * kernel_height * kernel_height * height_out * width_out * sizeof(float));
    for (int i = 0; i < num_samples; i++)
    {
        // im2col
        int start_in = i * input_channel * height_in * width_in;
        int start_out = i * output_channel * height_out * width_out;

        unroll_kernel<<<gridSize1, blockSize1>>>(input_channel, height_in, width_in, kernel_height,
                                                 height_out, width_out, &device_input[start_in], device_input_unroll);

        matrix_multiplication_kernel<<<gridSize2, blockSize2>>>(device_input_unroll, device_weight, &device_output[start_out], m, n, k);
        // result.rowwise() += bias.transpose();
        // top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
    }

    // Copy the output back to host
    cudaMemcpy(output_data, device_output, num_samples * output_channel * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_input_unroll);
    cudaFree(device_output);
    cudaFree(device_weight);
}