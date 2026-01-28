// LAB 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < len) out[i] = in1[i] + in2[i];
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  
  wbLog(TRACE, "The input length is ", inputLength);
  
  //@@ Allocate GPU memory here
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;
  size_t size = inputLength * sizeof(float);
  cudaMalloc((void **)&deviceInput1, size);
  cudaMalloc((void **)&deviceInput2, size);
  cudaMalloc((void **)&deviceOutput, size);
  
  
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutput, hostOutput, size, cudaMemcpyHostToDevice);


  //@@ Initialize the grid and block dimensions here
  const float numThreadsPerBlock = 512.0;
  const unsigned int numBlocks = ceil(inputLength/numThreadsPerBlock);


  //@@ Launch the GPU Kernel here to perform CUDA computation
  vecAdd <<< numBlocks, numThreadsPerBlock >>> (deviceInput1, deviceInput2, deviceOutput, inputLength);


  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostInput1, deviceInput1, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostInput2, deviceInput2, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);


  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);


  wbSolution(args, hostOutput, inputLength);


  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
