/* Sieve of Eratosthenes
 * Device Code
 */

#ifndef _PRIMES_KERNEL_H_
#define _PRIMES_KERNEL_H_

#include <stdio.h>

#define SDATA( index)     cutilBankChecker(sdata, index)

#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

#define __register__ volatile

/////////////////////////////////////////////////////////////
///////////////////
//! Sieve of Eratosthenes on GPU
/////////////////////////////////////////////////////////////
///////////////////
__global__ void
primesSMKernel( unsigned char *g_prepattern, unsigned char *g_bitbuffer, unsigned int bitbufsize, unsigned int NUM )
{
        extern unsigned char __shared__ sdata[];
        if (NUM > 0)
        {
                __register__ unsigned int *isdata = (unsigned int*)&sdata[0];

                __register__ unsigned int *g_ibitbuffer = (unsigned int*)g_bitbuffer;
                __register__ unsigned int *g_iprepattern   = (unsigned int*)g_prepattern;
                __register__ unsigned int num   = bitbufsize / 4;  // round down
                __register__ unsigned int remain = bitbufsize % 4;  // remainder
                __register__ const unsigned int idx = threadIdx.x;

                // initialize shared memory with precomputed bit pattern for primes 2, 3, 5, 7, 11, 13
                for (__register__ int i=0; i < num; i+= blockDim.x)
                        if (i+idx < num) isdata[i+idx] = g_iprepattern[i+idx];
                if (idx < remain) sdata[4*num+idx] = g_prepattern[4*num+idx];

                __syncthreads();

                unsigned int __shared__ firstprime;
                __register__ unsigned int sqrt_N = ceil(sqrtf((float)NUM));
        
                if (threadIdx.x == 0)
                {
                        firstprime = 17;  // start marking multiples of primes beginning with prime 11
                        sdata[0] = 0x53;  // 2 is prime, 3 is prime, 5 is prime, 7 is prime, the rest in this byte isn't
                        sdata[1] = 0xd7;  // 11 is prime, 13 is prime
                }
                __syncthreads();

                while (firstprime <= sqrt_N)
                {
                        // mark out all multiples of "firstprime" starting with firstprime squared.
                        for (unsigned int i = (firstprime+idx) * firstprime; i < NUM; i += firstprime*blockDim.x)
                                sdata[i>>3] |= (1<<(i&7));

                        __syncthreads();

                        // search for next prime (unmarked number) in the bit array using a single thread
                        if (threadIdx.x == 0)
                                for (firstprime = firstprime + 1; firstprime < NUM; firstprime++)
                                        if ((sdata[firstprime>>3] & (1<<(firstprime&7))) == 0) break;

                        __syncthreads();
                }

                // coalesced and bank-conflict free 32 bit integer copy from shared to global
                for (__register__ int i=0; i < num; i+= blockDim.x)
                        if (i+idx < num) g_ibitbuffer[i+idx] = isdata[i+idx];

                // copy remaining bytes
                if (idx < remain) g_bitbuffer[4*num+idx] = sdata[4*num+idx];
        }
}

__device__ __constant__ unsigned char d_prebitbuffer[65536];

__global__ void
primesSMKernel2( unsigned char *g_prepattern, unsigned char *g_bitbuffer, unsigned int bitbufsize, unsigned int NUM )
{
        extern unsigned char __shared__ sdata[];

        if (NUM > 0)
        {
                __register__ unsigned int *isdata = (unsigned int*)&sdata[0];

                __register__ unsigned int *g_ibitbuffer = (unsigned int*)g_bitbuffer;
                __register__ unsigned int *g_iprepattern   = (unsigned int*)g_prepattern;
                __register__ unsigned int num   = bitbufsize / 4;  // round down
                __register__ unsigned int remain = bitbufsize % 4;  // remainder
                __register__ const unsigned int idx = threadIdx.x;

                // initialize shared memory with precomputed bit pattern for primes 2, 3, 5, 7, 11, 13
                for (__register__ int i=0; i < num; i+= blockDim.x)
                        if (i+idx < num) isdata[i+idx] = g_iprepattern[i+idx];
                if (idx < remain) sdata[4*num+idx] = g_prepattern[4*num+idx];

                // K is the block-specific offset
                unsigned long long K = NUM * blockIdx.x + NUM;

                __syncthreads();

                unsigned int __shared__ firstprime;
                __register__ unsigned int sqrt_KN = ceil(sqrtf((float)(K+NUM)));
        
                if (threadIdx.x == 0)
                {
                        firstprime = 17;  // start marking multiples of primes beginning with prime 17
                }
                __syncthreads();

                while (firstprime <= sqrt_KN)
                {
                        // compute an offset such that we're instantly entering the range of
                        // K...K+N in the loop below
                        // Because 64 bit division is costly, use only the first thread
                        unsigned int __shared__ offset;
                        if (threadIdx.x == 0)
                        {
                                offset = 0;
                                if (K >= firstprime*firstprime) offset = (K-firstprime*firstprime) / firstprime;
                        }
                        __syncthreads();

                        // mark out all multiples of "firstprime" that fall into this thread block, starting
                        // with firstprime squared.
                        for (unsigned long long i = (offset+firstprime+idx) * firstprime;
                                 i < K+NUM;
                                 i += firstprime*blockDim.x)
                                if (i >= K) sdata[(i-K)>>3] |= (1<<((i-K)&7));

                        __syncthreads();


                        // search for next prime (unmarked number) in the reference bit array using a single thread
                        if (threadIdx.x == 0)
                                for (firstprime = firstprime + 1; firstprime < NUM; firstprime++)
                                        if ((d_prebitbuffer[firstprime>>3] & (1<<(firstprime&7))) == 0) break;

                        __syncthreads();
                }

                // byte-by-byte uncoalesced and bank-conflict-prone copy from shared to global memory
                // TODO: create a generic coalesced copy routine that works with arbitrary
                //         output byte offsets on compute 1.0 and 1.1 hardware
                __register__ unsigned int byteoff = bitbufsize * blockIdx.x + bitbufsize;
                for (__register__ int i=0; i < bitbufsize; i+= blockDim.x)
                        if (i+idx < bitbufsize) g_bitbuffer[byteoff+i+idx] = sdata[i+idx];
        }
}
#endif // #ifndef _PRIMES_KERNEL_H_