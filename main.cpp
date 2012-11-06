#include "kernel.h"

int main(int argc, char** argv)
{
    // Particle
    float positions[NUM_OF_PARTICLES * NUM_OF_DIMENSIONS];
    float velocities[NUM_OF_PARTICLES * NUM_OF_DIMENSIONS];
    float pBests[NUM_OF_PARTICLES * NUM_OF_DIMENSIONS];

    // gBest
    float gBest[NUM_OF_DIMENSIONS];
    
    srand((unsigned) time(NULL));

	// Initialize particles
	for (int i = 0; i < NUM_OF_PARTICLES * NUM_OF_DIMENSIONS; i++)
	{
		positions[i] = getRandom(START_RANGE_MIN, START_RANGE_MAX);
		pBests[i] = positions[i];
		velocities[i] = 0;
	}

	for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
		gBest[k] = pBests[k];        

    clock_t begin = clock();
    
	// PSO main function
    cuda_pso(positions, velocities, pBests, gBest);
    
    clock_t end = clock();
    
    printf("==================== GPU =======================\n");
            
    printf("Time elapsed : %10.3lf ms\n", 
           (double)(end - begin) / CLOCKS_PER_SEC);
	
    
	// gBest berisi nilai minimum
	for (int i = 0; i < NUM_OF_DIMENSIONS; i++)
		printf("x%d = %f\n", i, gBest[i]);

	printf("Minimum = %f\n", host_fitness_function(gBest));
	
    // ======================== END OF GPU ====================== //
    

    // Initialize particles
	for (int i = 0; i < NUM_OF_PARTICLES * NUM_OF_DIMENSIONS; i++)
	{
		positions[i] = getRandom(START_RANGE_MIN, START_RANGE_MAX);
		pBests[i] = positions[i];
		velocities[i] = 0;
	}

	for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
		gBest[k] = pBests[k];        

    begin = clock();
    
	// PSO main function
    pso(positions, velocities, pBests, gBest);
    
    end = clock();
    
    printf("==================== CPU =======================\n");
            
    printf("Time elapsed : %10.3lf ms\n", 
           (double)(end - begin) / CLOCKS_PER_SEC);
	
    
	// gBest berisi nilai minimum
	for (int i = 0; i < NUM_OF_DIMENSIONS; i++)
		printf("x%d = %f\n", i, gBest[i]);

	printf("Minimum = %f\n", host_fitness_function(gBest));
	
    // ======================== END OF GPU ====================== //

	return 0;
}

