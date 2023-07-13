#ifndef ZSH_GPU_FILERS
#define ZSH_GPU_FILERS


#include "core_gpu.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>


class GPUGaussFilter {
public:
    GPUGaussFilter(double siogma, int kernel_size, int blockSize);
    void filtering(const cv::Mat& inputImage, cv::Mat& outputImage);
private:
    cv::Mat kernel;
    int block_size;
};


class GPULaplacianFilter {
public:
    GPULaplacianFilter(int blockSize);
    void filtering(const cv::Mat& inputImage, cv::Mat& outputImage);
private:
    cv::Mat kernel;
    int block_size;
};


class GPUMeanFilter {
public:
    GPUMeanFilter(int kernel_size, int blockSize);
    void filtering(const cv::Mat& inputImage, cv::Mat& outputImage);

private:
    cv::Mat kernel;
    int block_size;
};


class GPUSobelFilter {
public:
    GPUSobelFilter(int blockSize);
    void filtering(const cv::Mat& inputImage, cv::Mat& outputImage);
private:
    cv::Mat sobel_x;
    cv::Mat sobel_y;
    int block_size;
};


class GPUMutilFilter {
public:
    GPUMutilFilter(int blockSize, int kernelSize, double siogma);
    void filtering(const cv::Mat& inputImage, cv::Mat& outputImage, int filterType);
private:
    GPUGaussFilter guassFilter;
    GPULaplacianFilter lapFilter;
    GPUMeanFilter meanFilter;
    GPUSobelFilter sobelFilter;
};

#endif