CXX=g++

CUDA_INSTALL_PATH=/usr/local/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64 -lcudart

all:
	$(CXX) $(CFLAGS) -c main.cpp -o main.o
	$(CXX) $(CFLAGS) -c kernel.cpp -o kernel.o
	nvcc -c kernel.cu -o kernel_gpu.o
	$(CXX) $(LDFLAGS) main.o kernel_gpu.o kernel.o -o pso

clean:
	rm -f *.o pso

