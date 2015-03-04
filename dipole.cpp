
#include "trove_functions.h"
#include "dipole.h"
#include "cuda_objects.cuh"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdio>
#include <cmath>

void dipole_initialise(FintensityJob* intensity){
	printf("Begin Input\n");
	read_fields(intensity);
	printf("End Input\n");

	//Wake up the gpu//
	printf("Wake up gpu\n");
	cudaFree(0);
	printf("....Done!\n");
	
	int jmax = max(intensity->jvals[0],intensity->jvals[1]);

	//Now create the bset_contrs
	bset_contr_factory(&(intensity->bset_contr[0]),0,intensity->molec.sym_degen,intensity->molec.sym_nrepres);
	bset_contr_factory(&(intensity->bset_contr[1]),intensity->jvals[0],intensity->molec.sym_degen,intensity->molec.sym_nrepres);
	bset_contr_factory(&(intensity->bset_contr[2]),intensity->jvals[1],intensity->molec.sym_degen,intensity->molec.sym_nrepres);

	//Correlate them 
	correlate_index(intensity->bset_contr[0],intensity->bset_contr[0]);
	correlate_index(intensity->bset_contr[0],intensity->bset_contr[1]);
	correlate_index(intensity->bset_contr[0],intensity->bset_contr[2]);
	
	printf("Reading dipole\n");
	//Read the dipole
	read_dipole(&(intensity->dipole_me),intensity->dipsize);
	printf("Computing threej\n");
	//Compute threej
	precompute_threej(&(intensity->threej),jmax);
	//ijterms
	printf("Computing ijerms\n");
	compute_ijterms(&(intensity->bset_contr[1]),&(intensity->bset_contr[1].ijterms),intensity->molec.sym_nrepres);
	compute_ijterms(&(intensity->bset_contr[2]),&(intensity->bset_contr[2].ijterms),intensity->molec.sym_nrepres);
	
	read_eigenvalues((*intensity));

	//Begin GPU related initalisation////////////////////////////////////////////////////////
	intensity_info int_gpu;
	//Copy over constants to GPU
	int_gpu.sym_nrepres = intensity->molec.sym_nrepres;
	int_gpu.jmax = jmax;
	copy_array_to_gpu((void*)intensity->molec.sym_degen,(void**)&int_gpu.sym_degen,sizeof(int)*intensity->molec.sym_nrepres,"sym_degen");
	int_gpu.dip_stride_1 = intensity->bset_contr[0];
	int_gpu.dip_stride_2 = intensity->bset_contr[0].Maxcontracts*intensity->bset_contr[0].Maxcontracts;
	copy_intensity_info(&int_gpu);

};
