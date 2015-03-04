#include "cuda_objects.cuh"
#pragma once

__host__ void copy_intensity_info(intensity_info* int_inf);
__global__ void device_correlate_vectors(cuda_bset_contrT* bset_contr,int idegI,int igammaI,const double* vecI,double* vec);
__global__ void device_compute_1st_half_ls(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,double* dipole_me,int igammaI,double* vector,double* threej,double* half_ls);
__global__ void device_compute_1st_half_ls_flipped_dipole(const cuda_bset_contrT* bset_contrI,const cuda_bset_contrT* bset_contrF,const double* __restrict__ dipole_me,const double* vector,const double*  __restrict__ threej,double* half_ls);
__global__ void device_compute_1st_half_ls_flipped_dipole_branch(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,double* dipole_me,double* vector,double* threej,double* half_ls);
__global__ void device_clear_vector(double* vec,int N);
__global__ void device_compute_1st_half_ls_2D(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,double* dipole_me,int igammaI,double* vector,double* threej,double* half_ls);
__global__ void device_complete_half_ls_dimenF(cuda_bset_contrT* bset_contrF,double* half_ls);
__global__ void device_compute_1st_half_ls_dimenF(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,double* dipole_me,int irootF,int igammaI,int idegI,double* vector,double* threej,double* half_ls);
__global__ void device_compute_1st_half_ls_flipped_dipole_blocks(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,int startF,int endF,int ncontrF,double* dipole_me,double* vector,double* threej,double* half_ls);
__global__ void device_compute_1st_half_ls_flipped_dipole_shared_blocks(const cuda_bset_contrT* bset_contrI,const cuda_bset_contrT* bset_contrF,int start_idx,int startF,int endF,int ncontrF,const double* __restrict__ dipole_me,const double* vector,const double*  __restrict__ threej,double* half_ls);
__global__ void device_compute_1st_half_ls_flipped_dipole_shared(const cuda_bset_contrT* bset_contrI,const cuda_bset_contrT* bset_contrF,int start_idx,const double* __restrict__ dipole_me,const double* vector,const double*  __restrict__ threej,double* half_ls);
__global__ void device_compute_1st_half_ls_flipped_dipole_shared_tau(const cuda_bset_contrT* bset_contrI,const cuda_bset_contrT* bset_contrF,int start_idx,const double* __restrict__ dipole_me,const double* vector,const double*  __restrict__ threej,double* half_ls);
