CC = nvcc
INCLUDE = -I../utils
CFLAGS= --std=c++11 -arch sm_30 -O2

all: main.cu lab3.cu pgm.cpp
	$(CC) ${INCLUDE} ${CFLAGS} main.cu lab3.cu pgm.cpp 

clean:
	rm -f a.out

run:
	./a.out img_background.ppm img_target.ppm img_mask.pgm 130 600 output.ppm
