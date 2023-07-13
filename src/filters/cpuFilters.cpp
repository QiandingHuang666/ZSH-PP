#include "cpuFilters.hpp"


// Gaussain Filter 
CPUGaussFilter::CPUGaussFilter(double sigma, int kernel_size, int numThread) {
    kernel = cv::getGaussianKernel(kernel_size, sigma, CV_32F) * 
                cv::getGaussianKernel(kernel_size, sigma, CV_32F).t();
    numThread_ = numThread;
}

void CPUGaussFilter::filtering(cv::Mat& inputImage, cv::Mat& outputImage) {
    pthread_filter(inputImage, kernel, outputImage, numThread_);
}


// Laplacian Filter
CPULaplacianFilter::CPULaplacianFilter(int numThread) {
    kernel = (cv::Mat_<float>(3, 3) 
                <<   0,  1,  0,
                     1, -4,  1,
                     0,  1,  0);
    numThread_ = numThread;
}

void CPULaplacianFilter::filtering(cv::Mat& inputImage, cv::Mat& outputImage) {
    pthread_filter(inputImage, kernel, outputImage, numThread_);
}


// Mean Filter
CPUMeanFilter::CPUMeanFilter(int kernel_size, int numThread) {
    kernel =  cv::Mat::ones(kernel_size, kernel_size, CV_32F);
    kernel = kernel / kernel_size / kernel_size;
    numThread_ = numThread;
}

void CPUMeanFilter::filtering(cv::Mat& inputImage, cv::Mat& outputImage) {
    pthread_filter(inputImage, kernel, outputImage, numThread_);
}


// Sobel Filter
CPUSobelFilter::CPUSobelFilter(int numThread) {
    sobel_x = (cv::Mat_<float>(3, 3) \
                <<  -1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1);
    sobel_y = (cv::Mat_<float>(3, 3) 
                <<  -1, -2, -1,
                     0,  0,  0,
                     1,  2,  1);
    numThread_ = numThread;
}

void CPUSobelFilter::filtering(cv::Mat& inputImage, cv::Mat& outputImage) {
    cv::Mat outXImage;
    cv::Mat outYImage;
    outXImage.create(inputImage.size(), CV_8UC3);
    outYImage.create(inputImage.size(), CV_8UC3);

    // 卷积操作
    pthread_filter(inputImage, sobel_x, outXImage, numThread_);
    pthread_filter(inputImage, sobel_y, outYImage, numThread_);
    // 合并归0~255
    outXImage.convertTo(outXImage, CV_32F);
    outYImage.convertTo(outYImage, CV_32F);

    cv::magnitude(outXImage, outYImage, outYImage);
    cv::normalize(outYImage, outYImage, 0, 255, cv::NORM_MINMAX);

    // 拷贝结果
    outYImage.copyTo(outputImage);
}


// Mutil Filter
CPUMutilFilter::CPUMutilFilter(double siogma, int kernelSize, int numThread) 
    : guassFilter(siogma, kernelSize, numThread),
      lapFilter(numThread),
      meanFilter(kernelSize, numThread),
      sobelFilter(numThread) {
    numThread_ = numThread;
}

void CPUMutilFilter::filtering(cv::Mat& inputImage, 
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

