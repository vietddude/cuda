#include "./kernel_interface.h"
#include <iostream>

#define TILE_WIDTH 32
#define CONST_MEM_SIZE 8192
__constant__ float dc_weight[CONST_MEM_SIZE];

__global__ void unroll_kernel(int channel_in, int height_in, int width_in, int kernel_size, int height_out, int width_out,
                              float *input, float *output)
{
    int batch_idx = blockIdx.z;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int W_unroll = height_out * width_out;

    if (t < channel_in * W_unroll)
    {
        int c = t / W_unroll; // if t < 28*28, c = 0  // output channel
        int start_in = batch_idx * channel_in * height_in * width_in + c * (width_in * height_in);
        int start_out = batch_idx * channel_in * kernel_size * kernel_size * W_unroll;

        int w_unroll = t % W_unroll; // output height * output width
        int h_out = w_unroll / width_out;
        int w_out = w_unroll % width_out;
        int w_base = c * kernel_size * kernel_size;

        for (int p = 0; p < kernel_size; p++)
        {
            int x = (h_out + p) * width_in;
            for (int q = 0; q < kernel_size; q++)
            {
                int h_unroll = w_base + p * kernel_size + q;
                output[start_out + h_unroll * W_unroll + w_unroll] = input[start_in + x + w_out + q];
            }
        }
    }
}

__global__ void matrix_multiplication_kernel(float *A, float *B, float *C, int m, int n, int k, const float *bias)
{
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
    int r = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int c = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int batch_idx = blockIdx.z;
    int start_A = batch_idx * m * n;
    int start_C = batch_idx * m * k;
    float sum = 0;
    for (int i = 0; i < (n - 1) / TILE_WIDTH + 1; ++i)
    {
        if (r < m && i * TILE_WIDTH + threadIdx.x < n)
            s_A[threadIdx.y][threadIdx.x] = A[start_A + r + (i * TILE_WIDTH + threadIdx.x) * m];
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
        C[start_C + c * m + r] = sum + bias[c];
}

__host__ void KernelInterface::forward_kernel(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                                              const int num_samples, const int output_channel, const int input_channel,
                                              const int height_in, const int width_in, const int kernel_height)
{
    std::cout << "GPU Convolution layer using unroll with matrix multiplication ." << std::endl;
    const int height_out = height_in - kernel_height + 1;
    const int width_out = width_in - kernel_height + 1;

    // Allocate device memory
    float *device_input_unroll, *device_weight, *device_bias;

    CHECK(cudaMalloc(&device_weight, output_channel * input_channel * kernel_height * kernel_height * sizeof(float))); // input_channel * output_channel filter Maps of size kernel_height * kernel_height
    CHECK(cudaMalloc((void **)&device_bias, output_channel * sizeof(float)));

    // Copy input and mask data to device
    // CHECK(cudaMemcpy(device_input, input_data, num_samples * input_channel * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_weight, weight_data, output_channel * input_channel * kernel_height * kernel_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_bias, bias_data, output_channel * sizeof(float), cudaMemcpyHostToDevice));
    // Set the kernel dimensions
    int batch_size = 128;
    dim3 blockSize1(1024);
    dim3 gridSize1((input_channel * height_out * width_out - 1) / 1024 + 1, 1, batch_size);
    dim3 blockSize2(32, 32);
    int m = height_out * width_out;
    int n = input_channel * kernel_height * kernel_height;
    int k = output_channel;
    dim3 gridSize2((k - 1) / blockSize2.x + 1, (m - 1) / blockSize2.y + 1, batch_size);
    // setting cuda streams
    int nStreams = 4;
    float **device_input = new float *[nStreams], **device_output = new float *[nStreams];
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++)
    {
        CHECK(cudaStreamCreate(&streams[i]));
        CHECK(cudaMalloc((void **)&device_input[i], batch_size * input_channel * height_in * width_in * sizeof(float)));
        CHECK(cudaMalloc((void **)&device_output[i], batch_size * output_channel * height_out * width_out * sizeof(float)));
    }
    CHECK(cudaMalloc(&device_input_unroll, batch_size * input_channel * kernel_height * kernel_height * height_out * width_out * sizeof(float)));

    // loop through each sample
    for (int stream = 0; stream < nStreams; stream++)
    {
        for (int i = stream * batch_size; i < num_samples; i += nStreams * batch_size)
        {
            int start_in = i * input_channel * height_in * width_in;
            int start_out = i * output_channel * height_out * width_out;
            // Launch the kernel
            // im2col (unroll)
            CHECK(cudaMemcpyAsync(device_input[stream], &input_data[start_in], min(batch_size, num_samples - i) * input_channel * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice, streams[stream]));

            unroll_kernel<<<gridSize1, blockSize1, 0, streams[stream]>>>(input_channel, height_in, width_in, kernel_height,
                                                                         height_out, width_out, device_input[stream], device_input_unroll);
            // gemm using shared mem
            matrix_multiplication_kernel<<<gridSize2, blockSize2, 0, streams[stream]>>>(device_input_unroll, device_weight, device_output[stream], m, n, k, device_bias);
            CHECK(cudaMemcpyAsync(&output_data[start_out], device_output[stream], min(batch_size, num_samples - i) * output_channel * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost, streams[stream]));
        }
    }
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    // Free device memory
    for (int i = 0; i < nStreams; i++)
    {
        CHECK(cudaStreamSynchronize(streams[i]));
        cudaStreamDestroy(streams[i]);
        CHECK(cudaFree(device_input[i]));
        CHECK(cudaFree(device_output[i]));
    }
    CHECK(cudaFree(device_input_unroll));
    CHECK(cudaFree(device_weight));
    CHECK(cudaFree(device_bias));
}