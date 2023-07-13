#include "core_cpu.hpp"


using namespace std;
using namespace cv;

void* filter_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    Mat channel = data->channel;
    Mat kernel_x = data->kernel_x;
    Mat filteredChannel = data->filteredChannel;

    // int padding = kernel_x.rows / 2;  // Padding size

    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < channel.cols; j++) {

            float sum_x = 0;
            for (int k = -kernel_x.rows / 2; k <= kernel_x.rows / 2; k++) {

                for (int l = -kernel_x.cols / 2; l <= kernel_x.cols / 2; l++) {
                    int row = i + k;  // Add padding offset
                    int col = j + l;  // Add padding offset

                    if (row < 0) {
                        row = 0;
                    }
                    if (row >= channel.rows) {
                        row = channel.rows -1;
                    }
                    if (col < 0) {
                        col = 0;
                    }
                    if (col >= channel.cols) {
                        col = channel.cols -1;
                    }

                    float pixel = channel.at<uchar>(row, col);
                    sum_x += pixel * kernel_x.at<float>(k + kernel_x.rows / 2, l + kernel_x.cols / 2);
                }
            }

            filteredChannel.at<uchar>(i, j) =
                    saturate_cast<uchar>(abs(sum_x));
        }
    }

    pthread_exit(NULL);
}

void pthread_filter(const Mat& image, const Mat& kernel_x, Mat& result, int num_threads) {
    vector<Mat> channels;
    split(image, channels);
    vector<Mat> filteredChannels(channels.size());

    // int padding = kernel_x.rows / 2;  // Padding size

    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    int chunk_size = image.rows / num_threads;
    int remaining_rows = image.rows % num_threads;
    int start_row = 0;

    for (int c = 0; c < channels.size(); c++) {
        start_row = 0;
        // cout<<image.rows<<"   "<<image.cols<<endl;
        
        Mat channel = channels[c];

        // Apply padding to the channel
        // Mat paddedChannel;
        // copyMakeBorder(channel, paddedChannel, padding, padding, padding, padding, BORDER_CONSTANT, Scalar(0));
        // cout<<paddedChannel.rows<<"   "<<paddedChannel.cols<<endl;
        Mat filteredChannel = Mat::zeros(channel.size(), CV_8UC1);

        for (int i = 0; i < num_threads; i++) {
            thread_data[i].start_row = start_row;
            thread_data[i].end_row = start_row + chunk_size;
            thread_data[i].channel = channel;  // Use the padded channel
            thread_data[i].kernel_x = kernel_x;
            thread_data[i].filteredChannel = filteredChannel;

            if (remaining_rows > 0) {
                thread_data[i].end_row++;
                remaining_rows--;
            }

            start_row = thread_data[i].end_row;

            pthread_create(&threads[i], NULL, filter_thread, &thread_data[i]);
        }

        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }

        filteredChannels[c] = filteredChannel;
        // cout<<filteredChannel.rows<<"   "<<filteredChannel.cols<<endl;
    }

    merge(filteredChannels, result);
}