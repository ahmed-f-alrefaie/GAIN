#include "common.h"
#include "cuda_objects.cuh"
#include "cuda_host.cuh"
#include "dipole_kernals.cuh"
#include <cstdio>
#include <cmath>
#include "fields.h"
#include "trove_functions.h"







int main(int argc,char** argv)
{

	printf("%12.6f\n",three_j(2,3,1,1,-1,0));
	FintensityJob test_intensity;

	dipole_initialise(&test_intensity);
	//dipole_do_intensities(test_intensity);
	dipole_do_intensities_async(test_intensity,0);
	exit(0);
/*

	return 0;

}
