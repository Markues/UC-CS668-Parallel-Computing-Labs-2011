#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

__global__ static void findsmallest(int * primes,int f)
{ 
 int idx = blockIdx.x * blockDim.x + threadIdx.x; 
 if (idx > 1) {
  for(int i=idx+idx;i < f;i+=idx) 
   primes[i] = 1;
 }
}
/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/
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
  if(cudaGetDeviceProperties(∝, i) == cudaSuccess) {
   if(prop.major >= 1) {
    break;
   }
  }
 }
 if(i == count) {
  fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
  return false;
 }
 cudaSetDevice(i);
 return true;
}
int main(int argc, char** argv)
{
  
 int  *primes;
 int host_sieve[1000]; 
 double n = sizeof(host_sieve)/sizeof(int);

 if(!InitCUDA()) {
  return 0;
 } 
 cudaMalloc((void**) &primes, sizeof(int) * n); 

 findsmallest<<<1, sqrt(n), 0>>>(primes, n);
 cudaThreadSynchronize();  
 cudaMemcpy(&host_sieve, primes, sizeof(int) * n, cudaMemcpyDeviceToHost);
 
 cudaFree(primes);
 
 for(int k=2;k < n;++k) 
  if (host_sieve[k] == 0)
   printf("%d is prime\n",k);
 

 return 0;
}
