#ifndef ZSH_UTILS
#define ZSH_UTILS

#include <vector>
#include <dirent.h>
#include <iostream>

struct TimeIndex {
    long num_frame = 0;
    double time = 0.0;
    double max_time = 0.0;
    double min_time =5000000.0;
};


bool ends_with(const std::string& str, const std::string& suffix);

void getAllImgs(const std::string& root, const std::string& dir_path, 
                std::vector<std::string>& imgs);


#endif