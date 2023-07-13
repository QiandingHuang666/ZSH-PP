#ifndef ZSH_CORE
#define ZSH_CORE


#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>


void imageConvolution(const cv::Mat& inputImage, 
                      const cv::Mat& kernel, 
                      cv::Mat& outputImage, 
                      int BLOCK_SIZE, int b);

#endif