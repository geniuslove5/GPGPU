#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct isAlphabet{
	__host__ __device__ int operator()(const char text){
		if(text>=97 && text<=122)return 1;
		else return 0;
	}
};

__global__ void isAlphabetKernel(const char* text, int* pos, int* d_pos, int text_size){
	int position=blockDim.x*blockIdx.x+threadIdx.x;
	if(position<text_size && text[position]>=97 && text[position]<=122)d_pos[position]=pos[position]=1;
	else if(position<text_size)d_pos[position]=pos[position]=0;
	
}

__global__ void reductionPos(int *pos, int* d_pos, int text_size, int offset){
	int position=blockDim.x*blockIdx.x+threadIdx.x;
	if(position<text_size){
		if(pos[position] && position>0 && pos[position]==pos[position-1])d_pos[position]+=pos[position-offset];
	}
}

__global__ void update(int *pos, int* d_pos, int text_size){
	int position=blockDim.x*blockIdx.x+threadIdx.x;
	if(position<text_size){
		pos[position]=d_pos[position];
	}
}

void CountPosition1(const char *text, int *pos, int text_size){
	thrust::device_ptr<const char> ptr_text(text);
	thrust::device_ptr<int> ptr_pos(pos);
	thrust::transform(ptr_text, ptr_text+text_size, ptr_pos, isAlphabet());
	thrust::inclusive_scan_by_key(thrust::device, ptr_pos, ptr_pos+text_size, ptr_pos, ptr_pos);
}

void CountPosition2(const char *text, int *pos, int text_size){
	int threadsPerBlock=256;
	int blocksPerGrid=CeilDiv(text_size, threadsPerBlock);
	int *d_pos;
	cudaMalloc(&d_pos, text_size*sizeof(int));
	isAlphabetKernel<<<blocksPerGrid, threadsPerBlock>>>(text, pos, d_pos, text_size);
	for(int i=0;i<10;i++){
		reductionPos<<<blocksPerGrid, threadsPerBlock>>>(pos, d_pos, text_size, 1<<i);
		update<<<blocksPerGrid, threadsPerBlock>>>(pos, d_pos, text_size);
		
	}

	cudaFree(d_pos);
}
