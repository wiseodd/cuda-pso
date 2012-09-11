#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "kernel.h"


__device__ float tempParticle1[NUM_OF_DIMENSIONS];
__device__ float tempParticle2[NUM_OF_DIMENSIONS];

/*
 * 3-dimensional Levy function
 */
__device__ float fitness_function(float x[])
{
	float res = 0;
	float y1 = 1 + (x[0] - 1) / 4;
	float yn = 1 + (x[NUM_OF_DIMENSIONS - 1] - 1) / 4;

	res += pow(sin(phi * y1), 2);

	for (int i = 0; i < NUM_OF_DIMENSIONS - 1; i++)
	{
		float y = 1 + (x[i] - 1) / 4;
		float yp = 1 + (x[i + 1] - 1) / 4;

		res += pow(y - 1, 2) * (1 + 10 * pow(sin(phi * yp), 2)) + pow(yn - 1, 2);
	}

	return res;
}

/*
 * Kernel for update particle position and velocity
 */
__global__ void kernelUpdateParticle(float *positions, float *velocities, float *pBests, float *gBest, float r1, float r2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS)
        return;
	
	float rp = r1;
	float rg = r2;

	velocities[i] = OMEGA * velocities[i] + c1 * rp * (pBests[i] - positions[i])
			+ c2 * rg * (gBest[i % NUM_OF_DIMENSIONS] - positions[i]);

	// Update posisi particle
	positions[i] += velocities[i];
}

__global__ void kernelUpdatePBest(float *positions, float *pBests, float *gBest)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0)
        return;

	if (fitness_function(&positions[i]) < fitness_function(&pBests[i]))
	{
		for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
			pBests[i + k] = positions[i + k];
	}
}

extern "C" void cuda_pso(float *positions, float *velocities, float *pBests, float *gBest)
{
    // Use zero copy Memory
    cudaSetDeviceFlags(cudaDeviceMapHost);

    int size = NUM_OF_PARTICLES * NUM_OF_DIMENSIONS;
    
    float *devPos;
    float *devVel;
    float *hostPBest;
    float *hostGBest;
    float *devTemp;
    float *devPBest;
    float *devGBest;
		
	// Memory allocation
	cudaMalloc((void**)&devPos, sizeof(float) * size);
	cudaMalloc((void**)&devVel, sizeof(float) * size);
    cudaMalloc((void**)&devTemp, sizeof(float) * size);

    cudaHostAlloc((void**)&hostPBest, sizeof(float) * size, cudaHostAllocMapped);
	cudaHostAlloc((void**)&hostGBest, sizeof(float) * NUM_OF_DIMENSIONS, cudaHostAllocMapped);
	
	// Thread & Block number
	int threadsNum = 128;
	int blocksNum = (NUM_OF_PARTICLES * NUM_OF_DIMENSIONS / threadsNum) + 1;
	
    // Copy particle datas from host to device
	cudaMemcpy(devPos, positions, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devVel, velocities, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(hostPBest, pBests, sizeof(float) * size, cudaMemcpyHostToHost);
    cudaMemcpy(hostGBest, gBest, sizeof(float) * NUM_OF_DIMENSIONS, cudaMemcpyHostToHost);

    cudaHostGetDevicePointer(&devPBest, hostPBest, 0);
    cudaHostGetDevicePointer(&devGBest, hostGBest, 0);
    
    // PSO main function
    for (int iter = 0; iter < MAX_ITER; iter++)
	{	  
        // Update position and velocity
        kernelUpdateParticle<<<blocksNum, threadsNum>>>(devPos, devVel, devPBest, devGBest, getRandomClamped(), getRandomClamped());  
        // Update pBest
        kernelUpdatePBest<<<blocksNum, threadsNum>>>(devPos, devPBest, devGBest);
        
        // Update gBest
        for(int i = 0; i < size; i += NUM_OF_DIMENSIONS)
        {        
            float *temp = &hostPBest[i];

            if (host_fitness_function(temp) < host_fitness_function(hostGBest))
            {
                for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
                    hostGBest[k] = temp[k];
            }	
        }
    }
    
    // Retrieve particle datas from device to host
	cudaMemcpy(gBest, hostGBest, sizeof(float) * NUM_OF_DIMENSIONS, cudaMemcpyHostToHost); 
    
    // cleanup
    cudaFree(devPos);
	cudaFree(devVel);
	cudaFree(devPBest);
    cudaFree(devGBest);

    cudaFreeHost(hostPBest);
    cudaFreeHost(hostGBest);
}

