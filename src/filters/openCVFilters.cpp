#include "openCVFilters.hpp"


// Gaussain Filter 
OpenCVGaussFilter::OpenCVGaussFilter(double sigma, int kernel_size) {
    kernelSize = cv::Size(kernel_size, kernel_size);
    sg = sigma;
}

void OpenCVGaussFilter::filtering(const cv::Mat& inputImage, 
                                  cv::Mat& outputImage) {
    cv::GaussianBlur(inputImage, outputImage, kernelSize, sg);
}


// Laplacian Filter
void OpenCVLaplacianFilter::filtering(const cv::Mat& inputImage, 
                                      cv::Mat& outputImage) {

    cv::Laplacian(inputImage, outputImage, CV_32F, 3);
}


// Mean Filter
OpenCVMeanFilter::OpenCVMeanFilter(int kernel_size) {
    kernelSize = cv::Size(kernel_size, kernel_size);
}

void OpenCVMeanFilter::filtering(const cv::Mat& inputImage,
                                 cv::Mat& outputImage) {
    cv::blur(inputImage, outputImage, kernelSize);
}


// Sobel Filter
void  OpenCVSobelFilter::filtering(const cv::Mat& inputImage,
                                   cv::Mat& outputImage) {
    cv::Mat outXImage;
    cv::Mat outYImage;
    
    // 卷积操作
    cv::Sobel(inputImage, outXImage, CV_32F, 1, 0);
    cv::Sobel(inputImage, outYImage, CV_32F, 0, 1);

    // // 合并归0~255
    outXImage.convertTo(outXImage, CV_32F);
    outYImage.convertTo(outYImage, CV_32F);
    cv::magnitude(outXImage, outYImage, outYImage);
    cv::normalize(outYImage, outYImage, 0, 255, cv::NORM_MINMAX);

    // 拷贝结果
    outYImage.copyTo(outputImage);
}


// Mutil Filter
OpenCVMutilFilter::OpenCVMutilFilter(int kernelSize, double siogma) 
    : guassFilter(siogma, kernelSize),
      lapFilter(),
      meanFilter(kernelSize),
      sobelFilter() {}

void OpenCVMutilFilter::filtering(const cv::Mat& inputImage, 
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