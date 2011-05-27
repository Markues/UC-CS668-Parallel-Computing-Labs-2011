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
__global__ static void radixSort(int* C, int N, int bit)
{    
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	__shared__ int sortList[4000];
	int numFalses = 0;
	for(int i = idx; i < N; i++){ //for each item
		if(((int)(C[i]/pow(2.0,bit)) % 2) == 1){
			numFalses ++;
			sortList[i] = 0;
		}
		else
		{
			sortList[i] = 1;					
		}
	}
	/*for(int i = 1; i < N; i++){
		sortList[i] += sortList[i-1];
	}
	int rollingSum = 0;
	for(int i = idx; i < N; i++){		
		if(sortList[i] == 0)
		{
			C[i] = i - rollingSum + numFalses;							
		}
		else
		{			
			C[i] = rollingSum;
			rollingSum += 1;
		}
		
	}*/
}
int main(int argc, char* argv[])
{
	int  *c,*a, *temp; //host and device arrays
	const int n = 4000; // num elements in array
	size_t size = n * sizeof(int); //size of array

	a = (int *)malloc(size);//allocate host array
	temp = (int *)malloc(size);//allocate host array 2
	cudaMalloc((void**) &c, size); //allocate device array

	//init host array
	for(int i = 0; i < n;i++) a[i] = (int)rand() % 1024;	
	for(int i = 0; i < n;i++) temp[i] = a[i];	
	printf(" OLD LIST: \n");
	for(int k=0;k < n;++k) printf("%d ",a[k]);

	//copy it to the device 
	cudaMemcpy(c,a,size,cudaMemcpyHostToDevice);

	int block_size = 512;
	int n_blocks = n/block_size + (n%block_size == 0 ? 0:1);
	
	for(int bit = 0; bit >= 0;bit--){ //for each bit
		radixSort<<<1, 1>>>(c, n, bit);
		cudaMemcpy(a, c, sizeof(int) * n, cudaMemcpyDeviceToHost);
		//parallel prefix sum
	}
	
	//for(int k=0;k < n;++k)printf("%d ",a[k]);

	free(a); cudaFree(c);
	return 0;
}