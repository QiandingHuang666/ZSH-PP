#ifndef ZSH_CORE_CPU
#define ZSH_CORE_CPU

#include <opencv2/opencv.hpp>
#include <iostream>
#include <pthread.h>
#include <vector>


typedef struct {
    int start_row;
    int end_row;
    cv::Mat channel;
    cv::Mat kernel_x;
    cv::Mat filteredChannel;
} ThreadData;

void* filter_thread(void* arg);
void pthread_filter(const cv::Mat& image, const cv::Mat& kernel_x, cv::Mat& result, int num_threads);


#endif