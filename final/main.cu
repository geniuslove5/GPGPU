#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <algorithm>
#include "SyncedMemory.h"
#include "pgm.h"
#include "emd.h"
using namespace std;

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

int main(int argc, char **argv)
{
	if (argc != 2) {
		printf("Usage: %s <background> \n", argv[0]);
		abort();
	}
	bool sucb;
	int wb, hb, cb;
	auto imgb = ReadNetpbm(wb, hb, cb, sucb, argv[1]);
	if (not (sucb)) {
		puts("Something wrong with reading the input image files.");
		abort();
	}

	const int SIZEB = wb*hb*3;
	MemoryBuffer<float> background(SIZEB), output(SIZEB);
	auto background_s = background.CreateSync(SIZEB);
	auto output_s = output.CreateSync(SIZEB);

	float *background_cpu = background_s.get_cpu_wo();
	copy(imgb.get(), imgb.get()+SIZEB, background_cpu);

	
	int iter = 0;
	while( iter!=1000 ){
		EMD(
			background_s.get_gpu_rw(),
			output_s.get_gpu_wo(),
			wb, hb, &iter
		);
	
		unique_ptr<uint8_t[]> o(new uint8_t[SIZEB]);
		const float *o_cpu = output_s.get_cpu_ro();
		transform(o_cpu, o_cpu+SIZEB, o.get(), [](float f) -> uint8_t { return max(min(int(f+0.5f), 255), 0); });
		string s = to_string(iter);		
		if(iter!=1000)WritePPM(o.get(), wb, hb, ("imf"+s+".ppm").c_str());
		else WritePPM(o.get(), wb, hb, "residue.ppm");
//		printf("%d", iter);
	}
	return 0;
}
