#include "lab1.h"
#include <math.h>

#define S 1000
#define fps 24

static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;
__device__ const unsigned w = 640;
__device__ const unsigned h = 480;
int *tempx;
int *tempy;


struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};


__device__ void draw(int x1, int y1, uint8_t *yuv) {
	if(x1>=0 && x1<w && y1>=0 && y1<h)
		yuv[x1+y1*w] = 255;
}

__global__ void printStripe(int *x, int *y, int time, uint8_t *yuv, int len) {
	int idx = blockIdx.x+blockDim.x + threadIdx.x;
	if(idx > S) return;
	int x1 = x[idx]*4;
	int y1 = y[idx]*4;
	int dx = x1 - w/2;
	int dy = y1 - h/2;
	if(abs(dx) > abs(dy)) {
		if(x1 < w) {
			for(int i=x1; i>(x1 -len); i--)
				draw(i, y1+dy*(i-x1)/dx, yuv);
		} else {
			for(int i=x1; i<(x1 +len); i++) 
				draw(i, y1+dy*(i-x1)/dx, yuv);
		}
	} else {
		if(y1 < h) {
			for(int i=y1; i>(y1 -len); i--)
				draw(x1+dx*(i-y1)/dy, i, yuv);
		} else {
			for(int i=y1; i<(y1 +len); i++)
				draw(x1+dx*(i-y1)/dy, i, yuv);
		}
	}
}



void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	int *x;
	int *y;

	cudaMalloc(&x, S*sizeof(int));
	cudaMemcpy(x, tempx, S*sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&y, S*sizeof(int));
	cudaMemcpy(y, tempy, S*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(yuv, 0, W*H);
	
	printStripe<<<S/4+1, 16>>>(x, y, 0, yuv, impl->t);

	cudaMemset(yuv+W*H, 128, W*H/2);

	cudaDeviceSynchronize();
	cudaMemset(yuv+W*H, 128, W*H/2);
	++(impl->t);
}
