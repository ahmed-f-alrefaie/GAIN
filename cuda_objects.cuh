#include "trove_objects.h"

#pragma once
//A smaller version of bset_contr wih only relevant info needed to perform computations
typedef struct
{
	int 	jval;
	int 	Maxsymcoeffs;
	int 	max_deg_size;
	int 	Maxcontracts;
	int 	Nclasses;
	int* 	icontr2icase;
	int*	iroot_correlat_j0;
	int* 	k;
	int*	ktau;
	double** irr_repres; //Will be changed to a 1d object
	int* N;
	int* Ntotal;
	int* ijterms;
	
} cuda_bset_contrT;

typedef struct
{
	int sym_nrepres;
	int* sym_degen;
	int jmax;
	int dip_stride_1;
	int dip_stride_2;
	int dimenmax;
	double sq2;
}intensity_info;

//Device globals

//Holds important constants in the intensit calculation



