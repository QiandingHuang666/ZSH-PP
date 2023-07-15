echo "\033[1;32m[Builder]\033[0m Please wait, Buiding start."

g++ src/exp/expCPU.cpp src/filters/cpuFilters.cpp src/utils/utils.cpp src/core/conv_cpu.cpp -Iinclude -O3 -w -pthread -o executable/exp/cpuFilter `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building cupFilter was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/exp/cpuFilter\033[0m to run it."

nvcc src/exp/expGPU.cpp src/filters/gpuFilters.cpp src/core/conv_gpu.cu src/utils/utils.cpp -Iinclude -O3 -w -o executable/exp/gpuFilter `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building gupFilter was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/exp/gpuFilter\033[0m to run it."

g++ src/exp/expOpenCV.cpp src/filters/openCVFilters.cpp src/utils/utils.cpp -Iinclude -O3 -w -o executable/exp/openCVFilter `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building openCVFilter was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/exp/openCVFilter\033[0m to run it."



g++ src/deploy/capture.cpp src/utils/utils.cpp -o executable/deploy/capture  -O3 -w -Iinclude `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building Capture was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/deploy/capture \033[0m to run it."

g++ -pthread src/deploy/cpuGauss.cpp src/filters/cpuFilters.cpp src/utils/utils.cpp src/core/conv_cpu.cpp -Iinclude -O3 -w -o executable/deploy/cpuGauss `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building cpuGauss was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/deploy/cpuGauss \033[0m to run it."

g++ -pthread src/deploy/cpuLap.cpp src/filters/cpuFilters.cpp src/utils/utils.cpp src/core/conv_cpu.cpp -Iinclude -O3 -w -o executable/deploy/cpuLap `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building cpuLap was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/deploy/cpuLap \033[0m to run it."

g++ -pthread src/deploy/cpuMean.cpp src/filters/cpuFilters.cpp src/utils/utils.cpp src/core/conv_cpu.cpp -Iinclude -O3 -w -o executable/deploy/cpuMean `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building cpuMean was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/deploy/cpuMean \033[0m to run it."

g++ -pthread src/deploy/cpuSobel.cpp src/filters/cpuFilters.cpp src/utils/utils.cpp src/core/conv_cpu.cpp -Iinclude -O3 -w -o executable/deploy/cpuSobel `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building cpuSobel was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/deploy/cpuSobel \033[0m to run it."

nvcc src/deploy/gpuGauss.cpp src/filters/gpuFilters.cpp src/utils/utils.cpp src/core/conv_gpu.cu -Iinclude -O3 -w -o executable/deploy/gpuGauss `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building gpuGauss was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/deploy/gpuGauss\033[0m to run it."

nvcc src/deploy/gpuLap.cpp src/filters/gpuFilters.cpp src/utils/utils.cpp src/core/conv_gpu.cu -Iinclude -O3 -w -o executable/deploy/gpuLap `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building gpuLap was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/deploy/gpuLap\033[0m to run it."

nvcc src/deploy/gpuMean.cpp src/filters/gpuFilters.cpp src/utils/utils.cpp src/core/conv_gpu.cu -Iinclude -O3 -w -o executable/deploy/gpuMean `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building gpuMean was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/deploy/gpuMean\033[0m to run it."

nvcc src/deploy/gpuSobel.cpp src/filters/gpuFilters.cpp src/utils/utils.cpp src/core/conv_gpu.cu -Iinclude -O3 -w -o executable/deploy/gpuSobel `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building gpuSobel was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/deploy/gpuSobel\033[0m to run it."


g++ src/deploy/openCVGauss.cpp src/filters/openCVFilters.cpp src/utils/utils.cpp -Iinclude -O3 -w -o executable/deploy/openCVGauss `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building openCVGauss was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/deploy/openCVGauss\033[0m to run it."

g++ src/deploy/openCVLap.cpp src/filters/openCVFilters.cpp src/utils/utils.cpp -Iinclude -O3 -w -o executable/deploy/openCVLap `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building openCVLap was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/deploy/openCVLap\033[0m to run it."

g++ src/deploy/openCVMean.cpp src/filters/openCVFilters.cpp src/utils/utils.cpp -Iinclude -O3 -w -o executable/deploy/openCVMean `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building openCVMean was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/deploy/openCVMean\033[0m to run it."

g++ src/deploy/openCVSobel.cpp src/filters/openCVFilters.cpp src/utils/utils.cpp -Iinclude -O3 -w -o executable/deploy/openCVSobel `pkg-config --cflags --libs opencv4`
echo "\033[1;32m[Builder]\033[0m Building openCVSobel was finished!"
echo "\033[1;32m[Builder]\033[0m Input \033[1;36m./executable/deploy/openCVSobel\033[0m to run it."
