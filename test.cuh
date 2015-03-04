#ifndef _CUDA_TEST_H
#define _CUDA_TEST_H

#include "trove_objects.h"
#include "cuda_objects.cuh"
#include "cuda_host.cuh"
#include "cublas_v2.h"
#include "Util.h"
#include <cstdio>
#include <cstdlib>
#include <omp.h>

__host__ void benchmark_half_ls(FintensityJob & intensity,int no_initial_states);

#endif
