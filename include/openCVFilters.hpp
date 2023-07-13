#ifndef ZSH_OPENCV_FILTERS
#define ZSH_OPENCV_FILTERS


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>


class OpenCVGaussFilter {
public:
    OpenCVGaussFilter(double sigma, int kernel_size);
    void filtering(const cv::Mat& inputImage, cv::Mat& outputImage);
private:
   cv::Size kernelSize;
   double sg;
};


class OpenCVLaplacianFilter {
public:
    void filtering(const cv::Mat& inputImage, cv::Mat& outputImage);
};


class OpenCVMeanFilter {
public:
    OpenCVMeanFilter(int kernel_size);
    void filtering(const cv::Mat& inputImage, cv::Mat& outputImage);
private:
   cv::Size kernelSize;
};


class OpenCVSobelFilter {
public:
    void filtering(const cv::Mat& inputImage, cv::Mat& outputImage);
};


class OpenCVMutilFilter {
public:
    OpenCVMutilFilter(int kernelSize, double siogma);
    void filtering(const cv::Mat& inputImage, cv::Mat& outputImage, int filterType);
private:
    OpenCVGaussFilter guassFilter;
    OpenCVLaplacianFilter lapFilter;
    OpenCVMeanFilter meanFilter;
    OpenCVSobelFilter sobelFilter;
};

#endif
