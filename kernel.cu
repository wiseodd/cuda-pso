#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "kernel.h"


__device__ float tempParticle1[NUM_OF_DIMENSIONS];
__device__ float tempParticle2[NUM_OF_DIMENSIONS];

// Fungsi yang dioptimasi
// Levy 3-dimensional
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

        res += pow(y - 1, 2) * (1 + 10 * pow(sin(phi * yp), 2)) 
                + pow(yn - 1, 2);
    }

    return res;
}

__global__ void kernelUpdateParticle(float *positions, float *velocities, 
                                     float *pBests, float *gBest, float r1, 
                                     float r2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS)
        return;

    //float rp = getRandomClamped();
    //float rg = getRandomClamped();
    
    float rp = r1;
    float rg = r2;

    velocities[i] = OMEGA * velocities[i] + c1 * rp * (pBests[i] - positions[i])
            + c2 * rg * (gBest[i % NUM_OF_DIMENSIONS] - positions[i]);

    // Update posisi particle
    positions[i] += velocities[i];
}

__global__ void kernelUpdatePBest(float *positions, float *pBests, float* gBest)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0)
        return;

    for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
    {
        tempParticle1[j] = positions[i + j];
        tempParticle2[j] = pBests[i + j];
    }

    if (fitness_function(tempParticle1) < fitness_function(tempParticle2))
    {
        for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
            pBests[i + k] = positions[i + k];
    }
}


extern "C" void cuda_pso(float *positions, float *velocities, float *pBests, 
                         float *gBest)
{
    int size = NUM_OF_PARTICLES * NUM_OF_DIMENSIONS;
    
    float *devPos;
    float *devVel;
    float *devPBest;
    float *devGBest;
    
    float temp[NUM_OF_DIMENSIONS];
        
    // Memory allocation
    cudaMalloc((void**)&devPos, sizeof(float) * size);
    cudaMalloc((void**)&devVel, sizeof(float) * size);
    cudaMalloc((void**)&devPBest, sizeof(float) * size);
    cudaMalloc((void**)&devGBest, sizeof(float) * NUM_OF_DIMENSIONS);
    
    // Thread & Block number
    int threadsNum = 32;
    int blocksNum = NUM_OF_PARTICLES / threadsNum;
    
    // Copy particle datas from host to device
    cudaMemcpy(devPos, positions, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devVel, velocities, sizeof(float) * size, 
               cudaMemcpyHostToDevice);
    cudaMemcpy(devPBest, pBests, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devGBest, gBest, sizeof(float) * NUM_OF_DIMENSIONS, 
               cudaMemcpyHostToDevice);
    
    // PSO main function
    for (int iter = 0; iter < MAX_ITER; iter++)
    {     
        // Update position and velocity
        kernelUpdateParticle<<<blocksNum, threadsNum>>>(devPos, devVel, 
                                                        devPBest, devGBest, 
                                                        getRandomClamped(), 
                                                        getRandomClamped());  
        // Update pBest
        kernelUpdatePBest<<<blocksNum, threadsNum>>>(devPos, devPBest, 
                                                     devGBest);
        
        // Update gBest
        cudaMemcpy(pBests, devPBest, 
                   sizeof(float) * NUM_OF_PARTICLES * NUM_OF_DIMENSIONS, 
                   cudaMemcpyDeviceToHost);
        
        for(int i = 0; i < size; i += NUM_OF_DIMENSIONS)
        {
            for(int k = 0; k < NUM_OF_DIMENSIONS; k++)
                temp[k] = pBests[i + k];
        
            if (host_fitness_function(temp) < host_fitness_function(gBest))
            {
                for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
                    gBest[k] = temp[k];
            }   
        }
        
        cudaMemcpy(devGBest, gBest, sizeof(float) * NUM_OF_DIMENSIONS, 
                   cudaMemcpyHostToDevice);
    }
    
    // Retrieve particle datas from device to host
    cudaMemcpy(positions, devPos, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities, devVel, sizeof(float) * size, 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(pBests, devPBest, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gBest, devGBest, sizeof(float) * NUM_OF_DIMENSIONS, 
               cudaMemcpyDeviceToHost); 
    
    
    // cleanup
    cudaFree(devPos);
    cudaFree(devVel);
    cudaFree(devPBest);
    cudaFree(devGBest);
}

