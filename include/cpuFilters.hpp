#ifndef ZSH_CPU_FILTER
#define ZSH_CPU_FILTER


#include "core_cpu.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>


class CPUGaussFilter {
public:
    CPUGaussFilter(double siogma, int kernel_size, int numThread);
    void filtering(cv::Mat& inputImage, cv::Mat& outputImage);
private:
    cv::Mat kernel;
    int numThread_;
};


class CPULaplacianFilter {
public:
    CPULaplacianFilter(int numThread);
    void filtering(cv::Mat& inputImage, cv::Mat& outputImage);
private:
    cv::Mat kernel;
    int numThread_;
};


class CPUMeanFilter {
public:
    CPUMeanFilter(int kernel_size, int numThread);
    void filtering(cv::Mat& inputImage, cv::Mat& outputImage);

private:
    cv::Mat kernel;
    int numThread_;
};


class CPUSobelFilter {
public:
    CPUSobelFilter(int numThread);
    void filtering(cv::Mat& inputImage, cv::Mat& outputImage);
private:
    cv::Mat sobel_x;
    cv::Mat sobel_y;
    int numThread_;
};


class CPUMutilFilter {
public:
    CPUMutilFilter(double siogma, int kernelSize, int numThread);
    void filtering(cv::Mat& inputImage, cv::Mat& outputImage, int filterType);
private:
    CPUGaussFilter guassFilter;
    CPULaplacianFilter lapFilter;
    CPUMeanFilter meanFilter;
    CPUSobelFilter sobelFilter;
    int numThread_;
};

#endif