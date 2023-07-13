#include "utils.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <chrono>


int main(int argc, char** argv) {
    cv::VideoCapture cap;
    int fps = 25;
    std::string path;
    // ckeck args
    if (argc == 4)
    {
        cap.open(std::stoi(argv[1]));
        
        if (!cap.isOpened()) {
            std::cout << "\033[1;31m" << "[ Capture ]" << "\033[0m\n"
                      << " [Camera/Video] ERRO.\n";
            return 0;
        }

        fps = std::stoi(argv[2]);
        path = argv[3];
        if (!ends_with(path, ".mp4")) {
            std::cout << "\033[1;31m" << "[ Capture ]" << "\033[0m\n"
                      << "\033[1;33m" << "Must Specify Following Parameters:" 
                      << "\033[0m\n"
                      << "\t[Camera] (Must)\n"
                      << "\t[FPS] (Must)\n"
                      << "\t[Path to save Video] : (ends with '.mp4'.)\n"
                      << "\033[1;31m[ Capture ]\033[0m\n";
            return 0;
        }
    }
    else 
    {
        std::cout << "\033[1;31m" << "[ Capture ]" << "\033[0m\n"
                  << "\033[1;33m" << "Must Specify Following Parameters:" 
                  << "\033[0m\n"
                  << "\t[Camera] (Must)\n"
                  << "\t[FPS] (Must)\n"
                  << "\t[Path to save Video] : (ends with '.mp4'.)\n"
                  << "\033[1;31m[ Capture ]\033[0m\n";
        return 0;
    }

    cv::namedWindow("Capture", 1);
    cv::Mat frame;
    cap >> frame;
    if (frame.empty())
    {
        std::cout << "\033[1;31m" << "[ Capture ]" << "\033[0m\n"
                  << " [Camera/Video] Frame is Empty!\n";
			return 0;
    }

    cv::VideoWriter video(path, 
                            cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                            fps, 
                            frame.size());

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        cv::imshow("Capture", frame);
        video.write(frame);
        if (cv::waitKey(1) == 'q') 
            break;
    }

    cap.release();

    std::cout << "\033[1;32m[ Capture ]\033[0m"
              " Video has been saved in " 
              "\033[1;36m"
              << path
              << "\033[0m.\n";
}
