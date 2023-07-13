#include "openCVFilters.hpp"
#include "utils.hpp"
#include <string>
#include <chrono>


int main(int argc, char** argv) {
    if (argc < 2 || argc > 6)
    {
        std::cout << "\033[1;31m" << "[ Experiment on OpenCV ]" << "\033[0m\n"
                  << "\033[1;33m" << "Must Specify Following Parameters:" 
                  << "\033[0m\n"
                  << "\t[Filter] (Must) : gauss | lap | mean | sobel\n"
                  << "\t[Input (Image) Path] (Must)\n" 
                  << "\t[Output Path] (Must)\n"
                  << "\t[Kernel Size]\n"
                  << "\t[Sigma]\n" 
                  << "\033[1;31m[ Experiment on OpenCV ]\033[0m\n";
        return 0;
    }

    // [Sigme] [Kernal Size]
    double sigma = 0.0;
    int kernelSize = 3;
    int filterType;

    // check filter
    std::string filter(argv[1]);
    if (filter == "gauss") {
        sigma = std::stod(argv[5]);
        kernelSize = std::stoi(argv[4]);
        filterType = 0;
    }
    else if (filter == "mean") {
        kernelSize = std::stoi(argv[4]);
        filterType = 2;
    }
    else if (filter == "lap") {
        filterType = 1;
    }
    else if (filter == "sobel") {
        filterType = 3;
    }
    else {
        std::cout <<  "\033[1;31m" << "[ Experiment on OpenCV ]" << "\033[0m"
                  << "\033[1;33m" << "Filter Should be "
                     "gauss | lap | mean | sobel.\n" ;
        return 0;
    }
    
    std::cout <<  "\033[1;32m" << "[ Experiment on OpenCV ]" << "\033[0m"
                  << "\033[1;33m" << " Run " <<  filter << "..."
                  << "\033[0m\n" ;

    // get [Input (Image) Path] [Output Path] [Block Size]
    std::string inputPath(argv[2]);
    std::string outputPath(argv[3]);

    // get path to input and output
    // path to input
    std::vector<std::string> intputImgs;
    if (ends_with(inputPath, ".jpg") || ends_with(inputPath, ".png"))
    {
        intputImgs.push_back(inputPath);
    }
    else
    {   
        if (!ends_with(inputPath, "/"))
            inputPath = inputPath + "/";
        getAllImgs(inputPath, inputPath, intputImgs);
    }
    // path to output
    if (!ends_with(outputPath, "/"))
        outputPath = outputPath + "/";
    std::string result_path = outputPath;
    // Time index
    TimeIndex times;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    double seconds;

    // Filter
    OpenCVMutilFilter fter(kernelSize, sigma);

    cv::Mat in_img;
    cv::Mat out_img;
     for (auto it = intputImgs.begin(); it != intputImgs.end(); ++it) {
        in_img = cv::imread(*it, cv::IMREAD_COLOR);
        start = std::chrono::high_resolution_clock::now();
            fter.filtering(in_img, out_img, filterType);
        end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        seconds = duration.count() / 1000000.0;
        if (seconds > times.max_time)
            times.max_time = seconds;
        if (seconds < times.min_time)
            times.min_time = seconds;
        times.time += seconds;
        times.num_frame++;
        cv::imwrite(result_path + "[OpenCV]_[" + filter +
                        "]_kernelSize[" + 
                        std::to_string(kernelSize) + "]_" + 
                        "sigma[" + std::to_string(sigma) + "]_" + 
                        (*it).substr(
                            (*it).rfind("/")+1
                        ), 
                        out_img
                    );

        std::cout << "\033[1;32m[ Experiment on OpenCV ]\033[0m "
                  << "\033[1;34m" << *it << "\033[0m"
                  << " was processed "
                  << "\033[1;36m by " << filter
                  << " filter\033[0m.\n";      
    }

    printf("\n\033[1;32m[ Experiment on OpenCV ]\033[0m Kernel Size:\033[1;34m%10d\033[0m\n",
           kernelSize);
    printf("\033[1;32m[ Experiment on OpenCV ]\033[0m Max     FPS:\033[1;34m%10.4f\033[0m\n",
           1/times.min_time);
    printf("\033[1;32m[ Experiment on OpenCV ]\033[0m Min     FPS:\033[1;34m%10.4f\033[0m\n",
           1/times.max_time);
    printf("\033[1;32m[ Experiment on OpenCV ]\033[0m Avg     FPS:\033[1;34m%10.4f\033[0m\n\n",
           times.num_frame /times.time);

    return 0;
}