#include "emd.h"
#include <cstdio>
#include <cstdlib>
#include <cuda.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }


__global__ void TurnGray(float *output, const int wb, const int hb){
	const int yb = blockIdx.y * blockDim.y + threadIdx.y;
	const int xb = blockIdx.x * blockDim.x + threadIdx.x;
	const int curb = wb*yb+xb;

	if(yb<hb and xb<wb){
		const int gray = (output[curb*3+0] + output[curb*3+1] + output[curb*3+2]) / 3;
		output[curb*3+0]=gray;
		output[curb*3+1]=gray;
		output[curb*3+2]=gray;
	}
}

__global__ void calculateMMNumber(float* output, const int wb, const int hb, int* mmnumber){
	const int yb = blockIdx.y * blockDim.y + threadIdx.y;
	const int xb = blockIdx.x * blockDim.x + threadIdx.x;
	const int curb = wb*yb+xb;

	//four boundary
        const int upB = (yb-1)>=0 ? (yb-1) : 0;
        const int lowB = (yb+1)<=(hb-1) ? (yb+1) : (hb-1);
        const int leftB = (xb-1)>=0 ? (xb-1) : 0;
        const int rightB = (xb+1)<=(wb-1) ? (xb+1) : (wb-1);

        int max=-10000, min=10000;
        if( xb<wb and yb<hb ){
                for(int i=leftB;i<=rightB;i++){
                        for(int j=upB;j<=lowB;j++){
                                int curt = j*wb+i;
                                if(output[curt*3+0]>max)max=output[curt*3+0];
                                if(output[curt*3+0]<min)min=output[curt*3+0];
                        }
                }
                if(max==output[curb*3+0] and output[curb*3+0]>min)atomicAdd(mmnumber, 1);
                if(min==output[curb*3+0] and max>output[curb*3+0])atomicAdd(mmnumber, 1);
        }
}

__global__ void calculateWindowSize(float* windowSize, int* mmnumber, int wb, int hb){
	*windowSize = 2*sqrtf( (float)(wb*hb)/(*mmnumber) );
}

__global__ void generateEnv(float* env, float* output, int wb, int hb, float* windowSize){
	const int yb = blockIdx.y * blockDim.y + threadIdx.y;
	const int xb = blockIdx.x * blockDim.x + threadIdx.x;
	const int curb = yb*wb+xb;
	
	//four boundary
	const int half_windowSize = floor(*windowSize/2);
	const int upB = (yb-half_windowSize)>=0 ? (yb-half_windowSize) : 0;
	const int lowB = (yb+half_windowSize)<=(hb-1) ? (yb+half_windowSize) : (hb-1);
	const int leftB = (xb-half_windowSize)>=0 ? (xb-half_windowSize) : 0;
	const int rightB = (xb+half_windowSize)<=(wb-1) ? (xb+half_windowSize) : (wb-1);

	int max=-10000, min=10000;
	if( xb<wb and yb<hb ){
		for(int i=leftB;i<=rightB;i++){
			for(int j=upB;j<=lowB;j++){
				int curt = j*wb+i;
				if(output[curt*3+0]>max)max=output[curt*3+0];
				if(output[curt*3+0]<min)min=output[curt*3+0];
			}
		}
		env[curb*3+0]=(max+min)/2;
		env[curb*3+1]=(max+min)/2;
		env[curb*3+2]=(max+min)/2;
	}
}

__global__ void smoothEnv(float* env, int wb, int hb, float* windowSize){
	const int yb = blockIdx.y * blockDim.y + threadIdx.y;
	const int xb = blockIdx.x * blockDim.x + threadIdx.x;
	const int curb = yb*wb+xb;
	
	//four boundary
	const int half_windowSize = floor(*windowSize/2);
	const int upB = (yb-half_windowSize)>=0 ? (yb-half_windowSize) : 0;
	const int lowB = (yb+half_windowSize)<=(hb-1) ? (yb+half_windowSize) : (hb-1);
	const int leftB = (xb-half_windowSize)>=0 ? (xb-half_windowSize) : 0    ;
	const int rightB = (xb+half_windowSize)<=(wb-1) ? (xb+half_windowSize) : (wb-1);

	float all=0;
	if( xb<wb and yb<hb ){
		for(int i=leftB;i<=rightB;i++){
			for(int j=upB;j<=lowB;j++){
				int curt = j*wb+i;
				all += env[curt*3+0];		
			}
		}
		const float mean = all/((rightB-leftB+1)*(lowB-upB+1));
		env[curb*3+0]=mean;
		env[curb*3+1]=mean;
		env[curb*3+2]=mean;
	}
}

__global__ void calculateDev(float* env, float* output, int wb, int hb, float* dev){
	float max=0, min=0;
	for(int i=0;i<wb;i++){
		for(int j=0;j<hb;j++){
			max += output[(j*wb+i)*3+0]*output[(j*wb+i)*3+0];
			min += env[(j*wb+i)*3+0]*env[(j*wb+i)*3+0];
			output[(j*wb+i)*3+0] -= env[(j*wb+i)*3+0];
			output[(j*wb+i)*3+1] -= env[(j*wb+i)*3+1];
			output[(j*wb+i)*3+2] -= env[(j*wb+i)*3+2];
		}
	}
	*dev=min/max;
}

__global__ void findMaxMinToRescale(float* src, int wb, int hb, float* max, float* min){
	*max=-10000, *min=10000;
	for(int i=0;i<wb;i++){
		for(int j=0;j<hb;j++){
			if(*max<src[(j*wb+i)+0])*max=src[(j*wb+i)+0];
			if(*min>src[(j*wb+i)+0])*min=src[(j*wb+i)+0];	
		}
	}
}
__global__ void rescale(float* input, float* output, int wb, int hb, float* max, float* min){
	const int yb = blockIdx.y * blockDim.y + threadIdx.y;
	const int xb = blockIdx.x * blockDim.x + threadIdx.x;
	const int curb = yb*wb+xb;
	if( xb<wb and yb<hb ){
		output[curb*3+0] = (input[curb*3+0]-*min)/(*max-*min)*255;
		output[curb*3+1] = (input[curb*3+1]-*min)/(*max-*min)*255;
		output[curb*3+2] = (input[curb*3+2]-*min)/(*max-*min)*255;
	}
}

__global__ void clone(float* dst, float* src, int wb, int hb){
	const int yb = blockIdx.y * blockDim.y + threadIdx.y;
	const int xb = blockIdx.x * blockDim.x + threadIdx.x;
	const int curb = yb*wb+xb;
	if( xb<wb and yb<hb ){
		dst[curb*3+0] = src[curb*3+0];
		dst[curb*3+1] = src[curb*3+1];
		dst[curb*3+2] = src[curb*3+2];
	}
}

__global__ void calculateOutput(float* output, float* background, int wb, int hb){
	const int yb = blockIdx.y * blockDim.y + threadIdx.y;
	const int xb = blockIdx.x * blockDim.x + threadIdx.x;
	const int curb = yb*wb+xb;
	if( xb<wb and yb<hb ){
		output[curb*3+0] = background[curb*3+0] - output[curb*3+0];
		output[curb*3+1] = background[curb*3+1] - output[curb*3+1];
		output[curb*3+2] = background[curb*3+2] - output[curb*3+2];
		background[curb*3+0] = output[curb*3+0];
		background[curb*3+1] = output[curb*3+1];
		background[curb*3+2] = output[curb*3+2];
	}
}
void EMD(
	float *background,
	float *output,
	const int wb, const int hb, int* iter
)
{
	//set up
	float *windowSize, *env, *dev, *testimf, *rMax, *rMin, *input;
	int *MMNumber;
	cudaMalloc(&windowSize, sizeof(float));
	cudaMalloc(&MMNumber, sizeof(int));

	cudaMalloc(&env, wb*hb*sizeof(float)*3);
	cudaMalloc(&testimf, wb*hb*sizeof(float)*3);
	cudaMalloc(&input, wb*hb*sizeof(float)*3);

	cudaMalloc(&dev, sizeof(float));
	cudaMalloc(&rMax, sizeof(float));
	cudaMalloc(&rMin, sizeof(float));
	
	float *h_dev = (float*)malloc(sizeof(float));
	int *h_MMNumber = (int*)malloc(sizeof(int));
	float *h_windowSize = (float*)malloc(sizeof(float));

	
	//copy the image back
	cudaMemcpy(output, background, 3*wb*hb*sizeof(float), cudaMemcpyDeviceToDevice);
	
	//EMD
	dim3 gdim1(CeilDiv(wb, 32), CeilDiv(hb, 16)), bdim1(32,16);
	if(*iter==0)TurnGray<<<gdim1, bdim1>>>(output, wb, hb);
	clone<<<gdim1, bdim1>>>(testimf, output, wb, hb);
	clone<<<gdim1, bdim1>>>(input, output, wb, hb);
//	if(!=0){
//		calculateOutput<<<gdim1, bdim1>>>(input, testimf, wb, hb);
//		findMaxMinToRescale<<<1, 1>>>(output, wb, hb, rMax, rMin);
//		rescale<<<gdim1, bdim1>>>(output, input, wb, hb, rMax, rMin);
//	}
	*h_dev=10;
	float threshold = 0.01;
	while(*h_dev > threshold){
		*h_MMNumber=0;
		cudaMemcpy(MMNumber, h_MMNumber, sizeof(int), cudaMemcpyHostToDevice);
		calculateMMNumber<<<gdim1, bdim1>>>(input, wb, hb, MMNumber);
		calculateWindowSize<<<1, 1>>>(windowSize, MMNumber, wb, hb);
		generateEnv<<<gdim1, bdim1>>>(env, input, wb, hb, windowSize);
		smoothEnv<<<gdim1, bdim1>>>(env, wb, hb, windowSize);
		calculateDev<<<1, 1>>>(env, input, wb, hb, dev);

		cudaMemcpy(h_dev, dev, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_windowSize, windowSize, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_MMNumber, MMNumber, sizeof(int), cudaMemcpyDeviceToHost);
		printf("windowSize = %f, number of extrema = %d\n", *h_windowSize, *h_MMNumber);
		break;
	}
	(*iter)++;
	findMaxMinToRescale<<<1, 1>>>(input, wb, hb, rMax, rMin);
	rescale<<<gdim1, bdim1>>>(input, output, wb, hb, rMax, rMin);
	cudaMemcpy(h_MMNumber, MMNumber, sizeof(int), cudaMemcpyDeviceToHost);
	calculateOutput<<<gdim1, bdim1>>>(input, background, wb, hb);
	if(*h_MMNumber<=20)*iter=1000;

//	calculateOutput<<<gdim1, bdim1>>>(input, testimf, wb, hb);
//	clone<<<gdim1, bdim1>>>(output, input, wb, hb);

//	cudaMemcpy(output, env, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	//clean up
	cudaFree(windowSize);
	cudaFree(MMNumber);
	cudaFree(env);   
	cudaFree(testimf);
	cudaFree(input);
	cudaFree(dev);
	cudaFree(rMax);
	cudaFree(rMin);
	
}
