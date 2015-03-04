#include "trove_objects.h"
#include "cuda_objects.cuh"
#include <cstring>
#pragma once

//SYMMTRYIRES

#define C2VM_ID 1
#define C3VM_ID 2
#define D3HM_ID 3
#define C2HM_ID 4
#define D2HM_ID 5
struct Fmole_type{
	//number of modes
	int nmodes;
	int nclasses;

	//Other info
	int sym_nrepres;
	int* sym_degen;
	int sym_maxdegen;
	char** c_sym;
	int sym_type;
};	//Intensity job information

struct FDipole_block{
	double* dipole_me;
	int startF;
	int endF;
	size_t size;
	int ncontrF;
};
struct FDipole_ptrs{
	int parts;
	FDipole_block* dip_block;
};


struct FGPU_ptrs{
	double* dipole_me;
	double* threej;
	cuda_bset_contrT** bset_contr;
	size_t avail_mem;
	
};


struct FintensityJob{		
	double ZPE;
	double erange[2];
	double erange_lower[2];
	double erange_upper[2];
	double freq_window[2];
	int* jvals;
	double q_stat;
	double temperature;
	bool* isym_do;
	int* isym_pairs;
	double* gns;
	Fmole_type molec;
	char filename[1024];
	char output_file[1024];
	TO_bset_contrT* bset_contr;
	TO_PTeigen* eigen;
	int Neigenlevels, Neigenroots;
	double* dipole_me;
	size_t dip_size;
	double* threej;
	FGPU_ptrs g_ptrs;
	int* igamma_pair;
	unsigned int dimenmax;
	double thresh_linestrength;
	unsigned int nsizemax;
	int** quanta_lower;
	int** quanta_upper;
	bool host_dipole;
	//Will b replaced with something better
	unsigned long cpu_memory;
	unsigned long gpu_memory;
	FDipole_ptrs dipole_blocks;
	bool split_dipole;
	bool reduced;
	unsigned long dipole_max;
	//Multiple J
	int nJ;

};

void read_fields(FintensityJob* intensity);
