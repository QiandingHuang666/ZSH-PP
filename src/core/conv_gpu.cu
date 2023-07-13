#include "core_gpu.hpp"


// CUDA并行实现图像卷积操作
__global__ void imageConvolution(const unsigned char* inputImage, unsigned char* outputImage,
                                 const float* filter, int width, int height, int channels, int FILTER_SIZE)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width)
    {
        for (int channel = 0; channel < channels; ++channel)
        {
            float sum = 0.0;
            int halfFilterSize = FILTER_SIZE / 2;

            for (int i = -halfFilterSize; i <= halfFilterSize; ++i)
            {
                for (int j = -halfFilterSize; j <= halfFilterSize; ++j)
                {
                    int imgRow = fminf(fmaxf(row + i, 0), height - 1);
                    int imgCol = fminf(fmaxf(col + j, 0), width - 1);
                    float pixelValue = static_cast<float>(inputImage[(imgRow * width + imgCol) * channels + channel]);
                    float filterValue = filter[(i + halfFilterSize) * FILTER_SIZE + (j + halfFilterSize)];
                    sum += pixelValue * filterValue;
                }
            }

            outputImage[(row * width + col) * channels + channel] = static_cast<unsigned char>(fminf(fmaxf(sum, 0), 255));
        }
    }
}

void imageConvolution(const cv::Mat& inputImage, const cv::Mat& kernel, cv::Mat& outputImage, int BLOCK_SIZE, int b)
{
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();
    int FILTER_SIZE = kernel.cols; 
    outputImage.create(height, width, inputImage.type());

    unsigned char* h_inputImage = inputImage.data;
    unsigned char* h_outputImage = outputImage.data;

    unsigned char* d_inputImage, * d_outputImage;
    cudaMalloc((void**)&d_inputImage, width * height * channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_outputImage, width * height * channels * sizeof(unsigned char));

    cudaMemcpy(d_inputImage, h_inputImage, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize;
    if (b == 0) {
        gridSize = dim3(1, 1);

    } else {
        gridSize = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    }
    

    const float* h_filter = kernel.ptr<float>(0);
    float* d_filter;
    cudaMalloc((void**)&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
    cudaMemcpy(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    imageConvolution<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, d_filter, width, height, channels, FILTER_SIZE);

    cudaMemcpy(h_outputImage, d_outputImage, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_filter);
}
