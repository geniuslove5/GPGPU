CC = nvcc
INCLUDE = -I../utils
CFLAGS= --std=c++11 -arch sm_20 -O2

all: main.cu emd.cu pgm.cpp
	$(CC) ${INCLUDE} ${CFLAGS} main.cu emd.cu pgm.cpp 

clean:
	rm -f a.out

run:
	./a.out img1.ppm
