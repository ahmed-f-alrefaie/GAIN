#ifndef _CUDA_HOST_H
#define _CUDA_HOST_H
#include "cuda_objects.cuh"
#include "fields.h"
#include "trove_functions.h"
#include "dipole_kernals.cuh"
#include <cstring>
//Copies relevant information needed to do intensity calculations onto the gpu
//Arguments p1: The bset_contr to copy p2: A device memory pointer to copy to
void CheckCudaError(const char* tag);
size_t copy_bset_contr_to_gpu(TO_bset_contrT* bset_contr,cuda_bset_contrT* bset_gptr,int sym_nrepres,int*sym_degen);
void copy_threej_to_gpu(double* threej,double** threej_gptr, int jmax);
void copy_array_to_gpu(void* arr,void** arr_gpu,size_t arr_size,const char* arr_name);
size_t create_and_copy_bset_contr_to_gpu(TO_bset_contrT* bset_contr,cuda_bset_contrT** bset_gptr,int* ijterms,int sym_nrepres,int*sym_degen);
void dipole_initialise(FintensityJob* intensity);
void dipole_do_intensities(FintensityJob & intensity);
void get_cuda_info(FintensityJob & intensity);
void dipole_initialise_cpu(FintensityJob* intensity);
void dipole_do_intensities_async_omp(FintensityJob & intensity,int device_id,int num_devices);
int count_free_devices();
void do_1st_half_ls(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,int dimenMax,int idegI,int igammaI,double* dipole_me,double* vecI,double* vec,double* threej,double* half_ls,cudaStream_t stream);
void do_1st_half_ls_blocks(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,int dimenMax,int idegI,int igammaI,double* gpu_dipole,FDipole_ptrs &  dipole_me,double* vecI,double* vec,double* threej,double* half_ls,cudaStream_t stream);
void do_1st_half_ls_branch(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,int dimenMax,int idegI,int igammaI,double* gpu_dipole,FDipole_ptrs & dipole_me,double* vecI,double* vec,double* threej,double* half_ls,cudaStream_t stream);
void do_1st_half_ls_shared_blocks(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,int jF,int j0dimen,int dimenMax,int idegI,int igammaI,double* gpu_dipole,FDipole_ptrs &  dipole_me,double* vecI,double* vec,double* threej,double* half_ls,cudaStream_t stream);
void do_1st_half_ls_shared(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,int jF,int j0dimen,int dimenMax,int idegI,int igammaI,double* dipole_me,double* vecI,double* vec,double* threej,double* half_ls,cudaStream_t stream);
int get_free_device(int last);
void dipole_initialise_gpu(FintensityJob * intensity, FGPU_ptrs & g_ptrs,int device_id);
#endif
