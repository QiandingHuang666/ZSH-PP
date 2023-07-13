#include "utils.hpp"


bool ends_with(const std::string& str, const std::string& suffix) {
    if (str.length() >= suffix.length()) {
        return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
    } else {
        return false;
    }
}

void getAllImgs(const std::string& root, const std::string& dir_path, std::vector<std::string>& imgs) {
    DIR *dir = opendir(dir_path.c_str());
    if (dir != NULL) {
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) { 
            // 读取目录中的文件名和子目录名
            std::string file_name = entry->d_name;
            // 排除 "." 和 ".."
            if (file_name != "." && file_name != ".."  
                && (ends_with(file_name, ".jpg") || ends_with(file_name, ".png")))
            { 
                imgs.push_back(root + file_name);
            }
        }
        closedir(dir); // 关闭目录
    } else {
        std::cerr << "Failed to open directory " << dir_path << std::endl;
    }
}