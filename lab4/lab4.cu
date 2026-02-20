#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH  3
#define MASK_SIZE 27
#define RADIUS MASK_WIDTH/2
#define TILE_WIDTH 4
#define SHARED_WIDTH (TILE_WIDTH + 2*RADIUS)


//@@ Define constant memory for device kernel here
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  
  // __shared__ float N_ds [SHARED_WIDTH][SHARED_WIDTH][SHARED_WIDTH];
  __shared__ float N_ds[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // int bdx = blockDim.x;
  // int bdy = blockDim.y;
  // int bdz = blockDim.z;
  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;

  int xo = bx * TILE_WIDTH + tx;
  int yo = by * TILE_WIDTH + ty;
  int zo = bz * TILE_WIDTH + tz;
  
  int xi = xo - RADIUS;
  int yi = yo - RADIUS;
  int zi = zo - RADIUS;

  if ((yi >= 0) && (yi < y_size) && (xi >= 0) && (xi < x_size) && (zi >= 0) && (zi < z_size)) {
    N_ds[tz][ty][tx] = input[zi*(y_size*x_size) + yi*x_size + xi];
  } 
  else {
    N_ds[tz][ty][tx] = 0.0f;
  }
  __syncthreads();

  float accum = 0.0f;
  if (tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH) {
    for (int i=0; i<MASK_WIDTH; i++) {
      for (int j=0; j<MASK_WIDTH; j++) {
        for (int k=0; k<MASK_WIDTH; k++) {
          accum += Mc[i][j][k] * N_ds[i+tz][j+ty][k+tx];
        }
      }
    }
    if ((yo < y_size) && (xo < x_size) && (zo < z_size)) {
      output[zo*(y_size*x_size) + yo*(x_size) + xo] = accum;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.

  args = wbArg_read(argc, argv);

  // Import data
  hostInput  = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  // So this doesnt actually work. Use cudaMemcptToSymbol instead
  // for(int k=0; k<z_size; k++){
  //   for(int j=0; j<y_size; j++){
  //     for(int i=0; i<x_size; i++){
  //       Mc = hostKernel[k*z_size + j*y_size + i*x_size];
  //     }
  //   }
  // }


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first three elements were the dimensions
  float* deviceInput;
  float* deviceOutput;
  cudaMalloc((void**)&deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void**)&deviceOutput, (inputLength - 3) * sizeof(float));


  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpyToSymbol(Mc, hostKernel, kernelLength * sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput, (hostInput + 3), (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);


  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(x_size / (TILE_WIDTH * 1.0)), ceil(y_size / (TILE_WIDTH * 1.0)), ceil(z_size / (TILE_WIDTH * 1.0))); 
  dim3 DimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);
  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy((hostOutput + 3), deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);



  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostKernel);
  free(hostOutput);
  return 0;
}

