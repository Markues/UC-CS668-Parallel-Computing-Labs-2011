#include <stdio.h>
#include <stdlib.h>
#include <cutil_inline.h>

/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/
#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else
bool InitCUDA(void)
{
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
	cudaSetDevice(i);

	printf("CUDA initialized.\n");
	return true;
}

#endif
__global__ static void findsmallest(int * primes,int f)
{ 
 int idx = blockIdx.x * blockDim.x + threadIdx.x; 
 if (idx > 1) {
	 for(int i=idx+idx;i < f;i+=idx) 
		 primes[i] = 1;
 }
}

int main(int argc, char* argv[])
{

	int  *primes;
	int sieve[100000]; 
	double n = sizeof(sieve)/sizeof(int);
	
	if(!InitCUDA()) {
		return 0;
	} 
	cudaMalloc((void**) &primes, sizeof(int) * n); 

	findsmallest<<<1, 512, 16000>>>(primes, n);
	cudaThreadSynchronize();  
	cudaMemcpy(&sieve, primes, sizeof(int) * n, cudaMemcpyDeviceToHost);
	
	cudaFree(primes);

	for(int k=2;k < n;++k) 
		if (sieve[k] == 0)
			printf("%d is prime\n",k);
	
	return 0;
}
