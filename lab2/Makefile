CC = nvcc
INCLUDE = -I../utils
CFLAGS= --std=c++11 -arch sm_30 -O2

all: main.cu counting.cu
	$(CC) ${INCLUDE} ${CFLAGS} main.cu counting.cu

clean:
	rm -f a.out

run:
	./a.out
