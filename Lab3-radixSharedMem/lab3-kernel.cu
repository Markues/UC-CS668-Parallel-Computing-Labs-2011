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
__global__ static void radixSort(int *A, int *C, int *prefix, int N, int bit)
{
    int thidx = 16*threadIdx.x;
	int numFalses = prefix[N-1]+1;
	for(int i = thidx;i < thidx+16;i++){
		if(i < 4001){
			if((1-((int)(A[i]/pow(2.0,bit))%2)) == 1){
				C[prefix[i]] = A[i];
			}
			else{
				C[i-prefix[i]+numFalses] = A[i];				
			}
		}
	}
}
__device__ void uniformAdd(int *g_data, int *uniforms)
{
	int thidx = 16*threadIdx.x;
	if(thidx != 0){
		for(int thid = threadIdx.x-1;thid >= 0;thid--){
			for(int i = thidx;i < thidx+16;i++){
				g_data[i] += uniforms[thid];
			}
		}
	}    
}
__global__ void scan(int *g_odata, int *g_idata, int *C, int *sums, int n, int bit)
{
	extern __shared__ int temp[]; 
	int thidx = 16*threadIdx.x;
	for(int thid = thidx;thid < thidx + 16; thid++){
		temp[thid] = (thid > 0) ? (1-((int)(g_idata[thid]/pow(2.0,bit)) % 2)) : 0;	
		__syncthreads();
		g_odata[thid] = temp[thid];	
	}
	__syncthreads();
	for(int thid = thidx;thid < thidx + 16; thid++){
		if(thid % 16 > 0){
			if(temp[thid-1] == 1){
				g_odata[thid] = g_odata[thid-1]+1;				
			}
			else{
				g_odata[thid] = g_odata[thid-1];
			}
		}
		else{
			if(thidx != 0){
				g_odata[thid] = temp[thid-1];
			}
			else{
				g_odata[thid] = 0;
			}
		}
		__syncthreads();
	}	
	__syncthreads();
	
	sums[threadIdx.x] = g_odata[thidx+15];

	__syncthreads();
	uniformAdd(g_odata, sums);

	__syncthreads();
}
int main(int argc, char* argv[])
{

	int  *c, *a, *d_a, *d_c, *d_prefix, *h_prefix, *h_sums, *d_sums; //host and device arrays
	const int n = 4000; // num elements in array
	size_t size = n * sizeof(int); //size of array
	
	a = (int *)malloc(size);//allocate host array
	c = (int *)malloc(size);//allocate host array
	h_sums = (int *)malloc(256*sizeof(int));//allocate host array
	h_prefix = (int *)malloc(size);//allocate host array
	cudaMalloc((void**) &d_a, size); //allocate device array
	cudaMalloc((void**) &d_sums, 256*sizeof(int)); //allocate device array
	cudaMalloc((void**) &d_prefix, size); //allocate device array
	cudaMalloc((void**) &d_c, size); //allocate device array	

	//init host array
	for(int i = 0; i < n;i++) a[i] = (int)rand() % 4294967297;	
	for(int i = 0; i < 256;i++) h_sums[i] = 0;
	printf(" OLD LIST: \n");
	for(int k=0;k < n/2;k++) printf("%d ",a[k]);
	int sum = 0;
	for(int k=0;k < n;k++) sum+=a[k];
	double gpuTime;
	unsigned int timer = 0;
	cutCreateTimer( &timer );
	cutStartTimer( timer );  // Start timer
	for(int bit = 0; bit <= 33;bit++){ //for each bit
		cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_c,c,size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_sums,h_sums,256*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_prefix,h_prefix,size,cudaMemcpyHostToDevice);
		scan<<<1,256,size>>>(d_prefix,d_a,d_c,d_sums,n,bit);
		cudaMemcpy(h_prefix, d_prefix, sizeof(int) * n, cudaMemcpyDeviceToHost);
		cudaMemcpy(d_prefix,h_prefix,size,cudaMemcpyHostToDevice);
		radixSort<<<1,256>>>(d_a,d_c,d_prefix,n,bit);
		cudaMemcpy(a, d_c, sizeof(int) * n, cudaMemcpyDeviceToHost);		
	}
	//printf("\nPREFIX: \n");
	//for(int k=0;k < n;k++) printf("%d ",h_prefix[k]);
	cutStopTimer( timer ); 
	printf("\nORDERED LIST: \n");
	
	for(int k=0;k < n;k++) printf("%d ",a[k]);
	printf("\nCUDA execution time = %f ms\n",cutGetTimerValue( timer ));
	cudaFree(d_a);
	cudaFree(d_prefix);
	cudaFree(d_c);
	cudaFree(d_sums);
	free(a); 
	return 0;
}
