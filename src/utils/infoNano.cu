#include <iostream>
#include <cuda_runtime_api.h>

int main() {
    int device_id = 0; // 设备 ID
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, device_id); // 获取设备属性

    std::cout << "\033[1;32mDevice name:\033[0m " 
              << prop.name << std::endl;
    std::cout << "\033[1;32mCompute capability:\033[0m " 
              << prop.major << "." << prop.minor << std::endl;
    std::cout << "\033[1;32mTotal global memory:\033[0m " 
              << prop.totalGlobalMem / (1024 * 1024) 
              << " MB" << std::endl;
    std::cout << "\033[1;32mTotal constant memory:\033[0m " 
              << prop.totalConstMem / 1024 
              << " KB" << std::endl;
    std::cout << "\033[1;32mMax shared memory per block:\033[0m " 
              << prop.sharedMemPerBlock / 1024 
              << " KB" << std::endl;
    std::cout << "\033[1;32mWarp size:\033[0m " 
              << prop.warpSize << std::endl;
    std::cout << "\033[1;32mMax threads per block:\033[0m " 
              << prop.maxThreadsPerBlock << std::endl;

    return 0;
}