#include "gpuFilters.hpp"


// Gaussain Filter 
GPUGaussFilter::GPUGaussFilter(double sigma, int kernel_size, int blockSize) {
    kernel = cv::getGaussianKernel(kernel_size, sigma, CV_32F) * 
                cv::getGaussianKernel(kernel_size, sigma, CV_32F).t();
    block_size = blockSize;
}

void GPUGaussFilter::filtering(const cv::Mat& inputImage, cv::Mat& outputImage) {
    imageConvolution(inputImage, kernel, outputImage, block_size, 1);
}


// Laplacian Filter
GPULaplacianFilter::GPULaplacianFilter(int blockSize) {
    kernel = (cv::Mat_<float>(3, 3) 
                <<   0,  1,  0,
                     1, -4,  1,
                     0,  1,  0);
    block_size = blockSize;
}

void GPULaplacianFilter::filtering(const cv::Mat& inputImage, cv::Mat& outputImage) {
    imageConvolution(inputImage, kernel, outputImage, block_size, 1);
}


// Mean Filter
GPUMeanFilter::GPUMeanFilter(int kernel_size, int blockSize) {
    kernel =  cv::Mat::ones(kernel_size, kernel_size, CV_32F);
    kernel = kernel / kernel_size / kernel_size;
    block_size = blockSize;
}

void GPUMeanFilter::filtering(const cv::Mat& inputImage, cv::Mat& outputImage) {
    imageConvolution(inputImage, kernel, outputImage, block_size, 1);
}


// Sobel Filter
GPUSobelFilter::GPUSobelFilter(int blockSize) {
    sobel_x = (cv::Mat_<float>(3, 3) \
                <<  -1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1);
    sobel_y = (cv::Mat_<float>(3, 3) 
                <<  -1, -2, -1,
                     0,  0,  0,
                     1,  2,  1);
    block_size = blockSize;
}

void GPUSobelFilter::filtering(const cv::Mat& inputImage, cv::Mat& outputImage) {
    cv::Mat outXImage;
    cv::Mat outYImage;

    // 卷积操作
    imageConvolution(inputImage, sobel_x, outXImage, block_size, 1);
    imageConvolution(inputImage, sobel_y, outYImage, block_size, 1);

    // 合并归0~255
    outXImage.convertTo(outXImage, CV_32F);
    outYImage.convertTo(outYImage, CV_32F);
    cv::magnitude(outXImage, outYImage, outYImage);
    cv::normalize(outYImage, outYImage, 0, 255, cv::NORM_MINMAX);

    // 拷贝结果
    outYImage.copyTo(outputImage);
}


// Mutil Filter
GPUMutilFilter::GPUMutilFilter(int blockSize, int kernelSize, double siogma) 
    : guassFilter(siogma, kernelSize, blockSize),
      lapFilter(blockSize),
      meanFilter(kernelSize, blockSize),
      sobelFilter(blockSize) {}

void GPUMutilFilter::filtering(const cv::Mat& inputImage, 
                               cv::Mat& outputImage, int filterType) {
    // 0: gauss 1: lap 2: mean 3: sobel
    if (filterType == 0) {
        guassFilter.filtering(inputImage, outputImage);
    }
    else if (filterType == 1) {
        lapFilter.filtering(inputImage, outputImage);
    }
    else if (filterType == 2) {
        meanFilter.filtering(inputImage, outputImage);
    }
    else if (filterType == 3) {
        sobelFilter.filtering(inputImage, outputImage);
    }
}