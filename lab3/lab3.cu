#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float A_s [TILE_WIDTH][TILE_WIDTH];
  __shared__ float B_s [TILE_WIDTH][TILE_WIDTH];

  int col = TILE_WIDTH * blockIdx.x + threadIdx.x;
  int row = TILE_WIDTH * blockIdx.y + threadIdx.y;

  int numTiles = (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH;

  float accum = 0.0;

  for(unsigned int tile=0; tile<numTiles; ++tile){
    // Check boundary for A
    if(row < numARows && tile*TILE_WIDTH+threadIdx.x < numAColumns) {
      A_s[threadIdx.y][threadIdx.x] = A[row*numAColumns + tile*TILE_WIDTH + threadIdx.x];
    }else{
      A_s[threadIdx.y][threadIdx.x] = 0;
    }
    // Check boundary for B
    if(tile*TILE_WIDTH+threadIdx.y < numBRows && col < numBColumns){
      B_s[threadIdx.y][threadIdx.x] = B[(tile*TILE_WIDTH + threadIdx.y)*numBColumns + col];
    }else{
      B_s[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

      for(unsigned int i=0; i<TILE_WIDTH; ++i){
        accum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
      }
    __syncthreads();
  }
  if(row < numCRows && col < numCColumns) {
    C[row*numCColumns + col] = accum;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;


  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  //@@ Allocate GPU memory here
  float *deviceA;
  float *deviceB;
  float *deviceC;

  cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void**)&deviceC, numCRows * numCColumns * sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, hostC, numCRows * numCColumns * sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(1.0 * numCColumns/16), ceil(1.0 * numCRows/16), 1);
  dim3 DimBlock(16, 16, 1);

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared  <<<DimGrid, DimBlock>>> (deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostA, deviceA, numARows * numAColumns * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostB, deviceB, numBRows * numBColumns * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
  

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@ Free the hostC matrix
  free(hostC);

  return 0;
}
