// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

  // Cast to unsigned
  __global__ void castToUnsignedChar(float *input, unsigned char *output, int len){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < len) output[i] = (unsigned char)(255 * input[i]);
  }
  
  // Convert image from RGB to Gray Scale
  __global__ void rgb2grayscale(unsigned char *input, unsigned char *output, int len){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len){
        unsigned char r = input[3 * i];
        unsigned char g = input[3 * i + 1];
        unsigned char b = input[3 * i + 2];
      output[i] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
  }
  
  // Compute the historgram
  __global__ void histo_kernel(unsigned char *buf, long size, unsigned int *histo){
    __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
    // warnning: this will not work correctly is there are fewer than 256 threads
    if (threadIdx.x < HISTOGRAM_LENGTH){
      histo_private[threadIdx.x] = 0;
    }
    __syncthreads();

    // Use private copies of the histo[] array to compute
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // stride is total number of threads
    int stride = blockDim.x * gridDim.x;
    while(i< size) {
      atomicAdd( &(histo_private[buf[i]]),1);
      i += stride;
    }

    // Copy from the histo[] arrays from each thread block to global memory
    __syncthreads();
    if (threadIdx.x < HISTOGRAM_LENGTH){
        atomicAdd(&(histo[threadIdx.x]), histo_private[threadIdx.x]);
    }
  }

  // Compute the scan (prefix sum) of the histogram (cpu)
  void scan(const unsigned int *histogram, float *cdf, int numPixels) {
    cdf[0] = (float)histogram[0] / numPixels;

    for (int i = 1; i < HISTOGRAM_LENGTH; i++) {
      cdf[i] = cdf[i - 1] + (float)histogram[i] / numPixels;
    }
  }

  // Apply equalization
  // Device Helper functions
  __device__ float clamp(float x, float start, float end){ return fminf(fmaxf(x, start), end); }

  __device__ unsigned char correct_color(unsigned char val, const float *cdf, float cdfmin) {
    float corrected = 255.0f * (cdf[val] - cdfmin) / (1.0f - cdfmin);
    corrected = clamp(corrected, 0.0f, 255.0f);
    return (unsigned char)corrected;
  }
  // actually do the equalization on the GPU using device helpers
  __global__ void hist_equalization(unsigned char *input, unsigned char *output, const float *cdf, float cdfmin, int len){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) output[i] = correct_color(input[i], cdf, cdfmin);
  }

  // Cast back to float
  __global__ void castBackToFloat(unsigned char* input, float* output, int len){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) output[i] = (float)(input[i]/255.0);
  }

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float* deviceInputImageData;
  float* deviceOutputImageData;
  unsigned char* deviceCastedChar;
  unsigned char* deviceGrayChar;
  unsigned int* deviceHisto;
  unsigned int hostHisto[HISTOGRAM_LENGTH];
  float hostScan[HISTOGRAM_LENGTH];
  float* deviceScan;
  unsigned char* deviceEqualizedChar;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);


  //@@ insert code here
  unsigned int numPixels = imageWidth * imageHeight;
  unsigned int numVals = imageWidth * imageHeight * imageChannels;
  cudaMalloc((void**)& deviceInputImageData, numVals * sizeof(float));
  cudaMalloc((void**)& deviceOutputImageData, numVals * sizeof(float));
  cudaMalloc((void**)& deviceCastedChar, numVals * sizeof(unsigned char));
  cudaMalloc((void**)& deviceGrayChar, numPixels * sizeof(unsigned char));
  cudaMalloc((void**)& deviceHisto, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset(deviceHisto, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void**)& deviceEqualizedChar, numVals * sizeof(unsigned char));

  cudaMalloc((void**)& deviceScan, HISTOGRAM_LENGTH * sizeof(float));

  cudaMemcpy(deviceInputImageData, hostInputImageData, numVals*sizeof(float), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(HISTOGRAM_LENGTH, 1, 1);
  dim3 blocksInGridVals((numVals + HISTOGRAM_LENGTH - 1)/HISTOGRAM_LENGTH , 1, 1);
  castToUnsignedChar<<<blocksInGridVals, threadsPerBlock>>>(deviceInputImageData, deviceCastedChar, numVals);
  cudaDeviceSynchronize();

  dim3 blocksInGridPixels((numPixels + HISTOGRAM_LENGTH - 1)/HISTOGRAM_LENGTH, 1, 1);
  rgb2grayscale<<<blocksInGridPixels, threadsPerBlock>>>(deviceCastedChar, deviceGrayChar, numPixels);
  cudaDeviceSynchronize();
  
  histo_kernel<<<blocksInGridPixels, threadsPerBlock>>>(deviceGrayChar, numPixels, deviceHisto);
  cudaDeviceSynchronize();

  // compute the scan CPU
  cudaMemcpy(hostHisto, deviceHisto, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  scan(hostHisto, hostScan, numPixels);
  float cdfmin = 0.0f;
  for (int i = 0; i < HISTOGRAM_LENGTH; i++) {
    if (hostScan[i] > 0.0f) {
      cdfmin = hostScan[i];
      break;
    }
  }
  cudaMemcpy(deviceScan, hostScan, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
  

  //
  hist_equalization<<<blocksInGridVals, threadsPerBlock>>>(deviceCastedChar, deviceEqualizedChar, deviceScan, cdfmin, numVals);
  cudaDeviceSynchronize();

  castBackToFloat<<<blocksInGridVals, threadsPerBlock>>>(deviceEqualizedChar, deviceOutputImageData, numVals);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutputImageData, numVals * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceCastedChar);
  cudaFree(deviceGrayChar);
  cudaFree(deviceHisto);
  cudaFree(deviceScan);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceEqualizedChar);


  return 0;
}

