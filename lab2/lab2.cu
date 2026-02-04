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


// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  //@@ Implement matrix multiplication kernel here
  int x_index = blockDim.x * blockIdx.x + threadIdx.x;
  int y_index = blockDim.y * blockIdx.y + threadIdx.y;
  float accum = 0.0;
  if(x_index < numCRows && y_index < numCColumns){
    for(int i=0; i<numAColumns; i++){
      accum += A[x_index * numAColumns + i] * B[i * numBColumns + y_index];
    }
    C[x_index * numCColumns + y_index] = accum;
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
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

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


  //@@ cpy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, hostC, numCRows * numCColumns * sizeof(float), cudaMemcpyHostToDevice);


  //@@ Initialize the grid and block dimensions here
  dim3 numThreadsPerBlock(32, 32);
  dim3 numBlocks((numCRows + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (numCColumns + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y );

  //@@ Launch the GPU Kernel here
  matrixMultiply <<< numBlocks, numThreadsPerBlock >>> (deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  
  //@@ cpy the GPU memory back to the CPU here
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
  //@@Free the hostC matrix
  free(hostC);

  return 0;
}

