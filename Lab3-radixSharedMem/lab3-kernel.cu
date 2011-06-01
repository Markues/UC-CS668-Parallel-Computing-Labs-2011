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
__device__ static void radixSort(int *A, int *C, int N, int bit)
{
	extern  __shared__  int temp[];
    int idx = threadIdx.x;
	int numFalses = temp[N-1]+1-((int)(A[idx]/pow(2.0,bit)) % 2);
	for(int i=idx;i<N;i++){
		if(1-((int)(A[i]/pow(2.0,bit)) % 2) == 1){
			C[temp[i]] = A[i]; 
		}
		else{
			C[i - temp[i] + numFalses] = A[i];
		}
	}
}
__global__ void bigscan(int *g_odata, int *g_idata,int *C, int *f, int n, int bit)
{
    // Dynamically allocated shared memory for scan kernels
    extern  __shared__  int temp[];
    int thid = threadIdx.x;
	int sectionAdd = 0;
	
	int offset = 1;

	// Cache the computational window in shared memory
	temp[2*thid]   = 1-((int)(g_idata[2*thid]/pow(2.0,bit)) % 2);
	temp[2*thid+1] = 1-((int)(g_idata[2*thid+1]/pow(2.0,bit)) % 2);

	// build the sum in place up the tree
	for (int d = n>>1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (thid < d)      
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;

			temp[bi] += temp[ai];
		}

		offset *= 2;
	}

	// scan back down the tree

	// clear the last element
	if (thid == 0)
	{
		temp[n - 1] = 0;
	}   

	// traverse down the tree building the scan in place
	for (int d = 1; d < n; d *= 2)
	{
		offset >>= 1;
		__syncthreads();

		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;

			float t   = temp[ai];
			temp[ai]  = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	// write results to global memory
	g_odata[2*thid]   = temp[2*thid];
	g_odata[2*thid+1] = temp[2*thid+1];

	__syncthreads();
	radixSort(g_idata,C,n, bit);
}

int main(int argc, char* argv[])
{
	int  *c, *a, *d_a, *d_c, *d_prefix, *h_prefix, *h_f, *d_f; //host and device arrays
	const int n = 1024; // num elements in array
	size_t size = n * sizeof(int); //size of array
	
	a = (int *)malloc(size);//allocate host array
	c = (int *)malloc(size);//allocate host array
	h_f = (int *)malloc(size);//allocate host array
	h_prefix = (int *)malloc(size);//allocate host array
	cudaMalloc((void**) &d_a, size); //allocate device array
	cudaMalloc((void**) &d_prefix, size); //allocate device array
	cudaMalloc((void**) &d_c, size); //allocate device array
	cudaMalloc((void**) &d_f, size); //allocate device array

	//init host array
	for(int i = 0; i < n;i++) a[i] = (int)rand() % 4294967297;	
	printf(" OLD LIST: \n");
	for(int k=0;k < n;++k) printf("%d ",a[k]);

	//copy it to the device 
	
	
	for(int bit = 0; bit <= 32;bit++){ //for each bit
		cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_c,c,size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_prefix,h_prefix,size,cudaMemcpyHostToDevice);
		cudaMemcpy(d_f,h_f,size,cudaMemcpyHostToDevice);
		
		bigscan<<<1,512,size>>>(d_prefix,d_a,d_c,d_f,n,bit);
		//radixSort<<<1,512,size>>>(d_c,d_a, d_prefix, d_f, n);		
		cudaMemcpy(h_prefix, d_prefix, sizeof(int) * n, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_f, d_f, sizeof(int) * n, cudaMemcpyDeviceToHost);
		cudaMemcpy(a, d_c, sizeof(int) * n, cudaMemcpyDeviceToHost);		
	}

	printf("\nPREFIX: \n");
	for(int k=0;k < n;++k) printf("%d ",h_prefix[k]);
	printf("\nORDERED LIST: \n");
	
	for(int k=0;k < n;++k) printf("%d ",a[k]);
	cudaFree(c);
	free(a); 
	return 0;
}
