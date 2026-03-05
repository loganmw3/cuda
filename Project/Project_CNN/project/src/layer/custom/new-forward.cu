#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define CUDA_CHECK(call)                                                   \
{                                                                          \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                  << " -> " << cudaGetErrorString(err) << std::endl;       \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
}

#define TILE_WIDTH 16
#define LARGEST_K 7
#define LARGEST_CHANNEL 16
#define LARGEST_MAP_OUT 16
#define LARGEST_CONSTANT_MEM_SIZE LARGEST_K * LARGEST_K * LARGEST_CHANNEL * LARGEST_MAP_OUT

__constant__ float constant_mem [LARGEST_CONSTANT_MEM_SIZE];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    // Constants & Shared Memory
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int SHARED_WIDTH = TILE_WIDTH + K - 1;
    extern __shared__ float shared_mem[];

    // Macros
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define constant(i3, i2, i1, i0) constant_mem[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define shared(r, c) shared_mem[(r) * SHARED_WIDTH + (c)]

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;   // Pixel X
    int ty = threadIdx.y;   // Pixel Y

    int bx = blockIdx.x;   // flattened tile index
    int by = blockIdx.y;   // batch index
    int bz = blockIdx.z;   // map_out index

    int cur_batch = by;
    int cur_map = bz;

    int num_Width_Tiles = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH;

    int tile_x = bx % num_Width_Tiles;
    int tile_y = bx / num_Width_Tiles;

    int w_out = tile_x * TILE_WIDTH + tx;
    int h_out = tile_y * TILE_WIDTH + ty;


    float accumulator = 0.0f;

    const int h_in0 = tile_y * TILE_WIDTH;
    const int w_in0 = tile_x * TILE_WIDTH;

    // Loop over each Channel
    for(int c=0; c<Channel; ++c){
      // Load into shared memory
      for (int r = ty; r < SHARED_WIDTH; r += blockDim.y) {
        for (int col = tx; col < SHARED_WIDTH; col += blockDim.x) {
          int h_in = h_in0 + r;
          int w_in = w_in0 + col;
          if (h_in < Height && w_in < Width && h_in >= 0 && w_in >= 0)
            shared(r, col) = in_4d(cur_batch, c, h_in, w_in);
          else
            shared(r, col) = 0.0f;
        }
      }
      __syncthreads();

      // Apply the mask
      if(tx < TILE_WIDTH && ty < TILE_WIDTH){
        if(h_out < Height_out && w_out < Width_out){
          for(int j=0; j<K; ++j){
            for(int i=0; i<K; ++i){
              accumulator += shared(ty+j, tx+i) * constant(cur_map, c, j, i);
            }
          }
        }
      }
      __syncthreads();
    }
    // Write back
    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
        if (h_out < Height_out && w_out < Width_out) {
            out_4d(cur_batch, cur_map, h_out, w_out) = accumulator;
        }
    }
    

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef constant
    #undef shared
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    // Constants
    const int Width_out = Width - K + 1;
    const int Height_out = Height - K + 1;
    const int input_size =  sizeof(float) * Batch * Height * Width * Channel;
    const int output_size = sizeof(float) * Batch * Map_out * Height_out * Width_out;
    const int mask_size =   sizeof(float) * (K*K) * Map_out * Channel;

    // Malloc for input, output, mask
    CUDA_CHECK(cudaMalloc((void**)device_input_ptr, input_size));
    CUDA_CHECK(cudaMalloc((void**)device_output_ptr, output_size));
    CUDA_CHECK(cudaMalloc((void**)device_mask_ptr, mask_size));

    // Copy input and mask to GPU
    CUDA_CHECK(cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*device_mask_ptr, host_mask, mask_size  , cudaMemcpyHostToDevice));

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Constants
    const int Width_out = Width - K + 1;
    const int Height_out = Height - K + 1;
    const int num_Width_Tiles = (Width_out + TILE_WIDTH - 1)/TILE_WIDTH;
    const int num_Height_Tiles = (Height_out + TILE_WIDTH -1)/TILE_WIDTH;

    // Set the kernel dimensions and call the kernel
    // Each threas should compute one output pixel (h, w) for one (b, m)
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
    // grid.x = number of tiles across the output width (Can recover x_tile and y_tile in the kernel)
    // grid.y = Batch
    // grid.z = Map_out
    dim3 blocksPerGrid(num_Width_Tiles * num_Height_Tiles, Batch, Map_out);

    //Copy Mask to constant memory (Device-To-Device)
    const int mask_size = Map_out * Channel * K * K * sizeof(float);
    CUDA_CHECK(cudaMemcpyToSymbol(constant_mem, device_mask, mask_size, 0, cudaMemcpyDeviceToDevice));

    int SHARED_WIDTH = TILE_WIDTH + K - 1;
    size_t sharedBytes = SHARED_WIDTH * SHARED_WIDTH * sizeof(float);

    // Launch kernel
    conv_forward_kernel<<<blocksPerGrid, threadsPerBlock, sharedBytes>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    CUDA_CHECK(cudaGetLastError());

    // cudaDeviceSynchronize();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Constants
    const int Width_out = Width - K + 1;
    const int Height_out = Height - K + 1;
    const int output_size = sizeof(float) * Batch * Map_out * Height_out * Width_out;
    
    // Copy the output back to host
    CUDA_CHECK(cudaMemcpy(host_output, device_output, output_size ,cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_mask));
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}