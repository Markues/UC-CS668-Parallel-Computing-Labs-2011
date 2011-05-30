#include <stdio.h>
#include <stdlib.h>
#include <cutil_inline.h>
#include <math.h>
//http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html

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

	printf("\nCUDA initialized.\n");
	return true;
}
#endif
////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! WA is A's width and WB is B's width
////////////////////////////////////////////////////////////////////////////////
__global__ static void radixSort(int* C, int* A,int N, int bit)
{    
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	extern __shared__ int sortList[];
	int numFalses = 0;
	for(int i = idx; i < N; i++){ //for each item
		if(((int)(A[i]/pow(2.0,bit)) % 2) == 1){
			sortList[i] = 0;
		}
		else
		{
			numFalses++;
			sortList[i] = 1;					
		}
	}
	C[0] = sortList[0];
	for(int i=idx;i<N;i++){
		C[i] = C[i-1] + sortList[i]; 
	}
	for(int i = idx; i < N; i++){		
		if(sortList[i] == 0)
		{
			C[i] = i - C[i] + numFalses;							
		}
		
	}
}
int main(int argc, char* argv[])
{
	int  *c, *a, *d_a, *d_c; //host and device arrays
	const int n = 10; // num elements in array
	size_t size = n * sizeof(int); //size of array
	int block_size = 512;
	
	a = (int *)malloc(size);//allocate host array
	c = (int *)malloc(size);//allocate host array
	cudaMalloc((void**) &d_a, size); //allocate device array
	cudaMalloc((void**) &d_c, size); //allocate device array

	//init host array
	for(int i = 0; i < n;i++) a[i] = (int)rand() % 10;	
	printf(" OLD LIST: \n");
	for(int k=0;k < n;++k) printf("%d ",a[k]);

	//copy it to the device 
	
	
	for(int bit = 0; bit >= 0;bit--){ //for each bit
		cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_c,c,size,cudaMemcpyHostToDevice);
		radixSort<<<ceil((double)n/block_size),block_size,size>>>(d_c,d_a, n, bit);
		cudaMemcpy(a, d_a, sizeof(int) * n, cudaMemcpyDeviceToHost);
		cudaMemcpy(c, d_c, sizeof(int) * n, cudaMemcpyDeviceToHost);
		//parallel prefix sum
	}
	//for(int i=1;i<n;i++){
	//	a[i] += a[i-1];
	//}
	printf("\n");
	for(int k=0;k < n;++k) printf("%d ",c[k]);
	cudaFree(c);
	free(a); 
	return 0;
}
