#include "gpuFilters.hpp"
#include "utils.hpp"
#include <string>
#include <chrono>


int main(int argc, char** argv) {
    cv::VideoCapture cap;

    int blockSize = 1;
    int kernelSize = 3;
    double sigma = 0.0;
    bool openOrigin = false;
    bool saveOrigin = false;
    bool saveOutput = false;
    std::string originVideoPath("output/Video/originVideo.mp4");
    std::string outputVideoPath("output/Video/GaussVideo.mp4");

    // ckeck args
    if (argc == 8)
    {
        
        if (ends_with(std::string(argv[1]), ".mp4")) {
            cap.open(argv[1]);
            cap.open(argv[1],  cv::CAP_FFMPEG);
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('m', 'p', '4', 'v'));
        }
        else {
            cap.open(std::stoi(argv[1]));
        }
        
        if (!cap.isOpened()) {
            std::cout << "\033[1;31m" << "[ Deploy Gaussian Filter on GPU ]" << "\033[0m\n"
                      << " [Camera/Video] ERRO.\n";
        }

        blockSize = std::stoi(argv[2]);
        kernelSize = std::stoi(argv[3]);
        sigma = std::stod(argv[4]);
        if (std::string(argv[5]) == "on")
        {
            openOrigin = true;
        }
        if (std::string(argv[6]) == "on")
        {
            saveOrigin = true;
        }
        if (std::string(argv[7]) == "on")
        {
            saveOutput = true;
        }

    }
    else 
    {
        std::cout << "\033[1;31m" << "[ Deploy Gaussian Filter on GPU ]" << "\033[0m\n"
                  << "\033[1;33m" << "Must Specify Following Parameters:" 
                  << "\033[0m\n"
                  << "\t[Camera/Video] (Must)\n"
                  << "\t[Block Size] (Must)\n"
                  << "\t[Kernel Size] (Must)\n"
                  << "\t[Sigma] (Must)\n"
                  << "\t[Open Origin] : on/off (Only 'on' can work.)\n"
                  << "\t[Save Origin Video] : on/off (Only 'on' can work.)\n"
                  << "\t[Save Output Video] : on/off (Only 'on' can work.)\n"
                  << "\033[1;31m[ Deploy Gaussian Filter on GPU ]\033[0m\n";
        return 0;
    }

    if (openOrigin)
        cv::namedWindow("Origin", 1);
    cv::namedWindow("Deploy Gaussian Filtering", 1);
    cv::Mat frame, out_frame, deploy_frame;

    cap >> frame;
    if (frame.empty())
    {
        std::cout << "\033[1;31m" << "[ Deploy Gaussian Filter on GPU ]" << "\033[0m\n"
                  << " [Camera/Video] Frame is Empty!\n";
			return 0;
    }
    cv::Rect rect(119, 39, 402, 402);

    GPUGaussFilter filter(sigma, kernelSize, blockSize);
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    double seconds;
    TimeIndex times;
    char buffer[100];

    if (!saveOrigin)
        originVideoPath = "output/Video/originVideo.cache";
    if (!saveOutput)
        outputVideoPath = "output/Video/GaussVideo.cache";
    cv::VideoWriter originVideo(originVideoPath, 
                            cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                            25, 
                            frame.size());
    cv::VideoWriter outputVideo(outputVideoPath, 
                            cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                            25, 
                            frame.size());

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // show Origin
        if (openOrigin) {
            frame.copyTo(out_frame);
            cv::rectangle(out_frame, rect, cv::Scalar(0, 255, 0), 5, cv::LINE_8, 0);
		    cv::imshow("Origin", out_frame);
        }

        // write origin
        if (saveOrigin) {
            originVideo.write(frame);
        }

        // filtering
        filter.filtering(frame, deploy_frame);
        deploy_frame.convertTo(deploy_frame, CV_8U);

        // write output
        if (saveOutput) {
            outputVideo.write(deploy_frame);
        }

        // show output
        cv::rectangle(deploy_frame, rect, cv::Scalar(0, 255, 0), 5, cv::LINE_8, 0);
        end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        seconds = duration.count() / 1000000.0;
        sprintf(buffer, "FPS : %5.2f", 1/seconds);
        cv::putText(deploy_frame, buffer, cv::Point(15, 22), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, 
                    cv::Scalar(0, 255, 0), 2);
        cv::imshow("Deploy Gaussian Filtering", deploy_frame);

        if (seconds > times.max_time)
            times.max_time = seconds;
        if (seconds < times.min_time)
            times.min_time = seconds;
        start = end;
        times.time += seconds;
        times.num_frame++;

        if (cv::waitKey(1) == 'q') 
            break;
    }

    double cam_fps = cap.get(cv::CAP_PROP_FPS);
	cap.release();

    printf("\n\033[1;32m[ Deploy Gaussian Filter on GPU ]\033[0m Cam FPS: %10.4f\n",
                cam_fps);
    printf("\033[1;32m[ Deploy Gaussian Filter on GPU ]\033[0m Min FPS: %10.4f\n",
                1/times.max_time);
    printf("\033[1;32m[ Deploy Gaussian Filter on GPU ]\033[0m Max FPS: %10.4f\n",
                1/times.min_time);
    printf("\033[1;32m[ Deploy Gaussian Filter on GPU ]\033[0m Avg FPS: %10.4f\n\n",
                times.num_frame/times.time);

    if (saveOrigin)
        printf("\033[1;32m[ Deploy Gaussian Filter on GPU ]\033[0m"
               " Origin Video has been saved in " 
               "\033[1;36moutput/Video/originVideo.mp4\033[0m.\n");
    if (saveOutput)
        printf("\033[1;32m[ Deploy Gaussian Filter on GPU ]\033[0m"
               " Output Video has been saved in " 
               "\033[1;36moutput/Video/GaussVideo.mp4\033[0m.\n");

    return 0;
}