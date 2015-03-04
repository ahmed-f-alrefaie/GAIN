

#include "trove_objects.h"

#include "fields.h"
#include <cstring>

#pragma once

void read_dipole(TO_bset_contrT & bsetj0,double** dipole_me,size_t & dip_size);
void read_dipole_flipped(TO_bset_contrT & bsetj0,double** dipole_me,size_t & dip_size);
void read_dipole_flipped_blocks(TO_bset_contrT & bsetj0,FDipole_ptrs & dipole_block, int parts);
double fakt(double a);
double three_j(int j1,int j2,int j3,int k1,int k2,int k3);
void bset_contr_factory(TO_bset_contrT* bset_contr,uint jval,int* sym_degen,int sym_nrepres);
void correlate_index(TO_bset_contrT & bset_contrj0, TO_bset_contrT & bset_contr);
void destroy_bset_contr(TO_bset_contrT* bset_contr,int sym_nrepres);
void precompute_threej(double** threej,int jmax);
void compute_ijterms(TO_bset_contrT & bset_contr, int** ijterm,int sym_nrepres);
void host_correlate_vectors(TO_bset_contrT* bset_contr,int idegI,int igammaI,int* ijterms,int* sym_degen,const double* vecI,double* vec);
void read_eigenvalues(FintensityJob & job);
void find_igamma_pair(FintensityJob & intensity);
bool energy_filter_lower(FintensityJob & job,int J,double energy, int* quanta);
bool energy_filter_upper(FintensityJob & job,int J,double energy, int* quanta);
bool intensity_filter(FintensityJob & job,int jI,int jF,double energyI,double energyF,int igammaI,int igammaF,int* quantaI,int* quantaF);
const char* branch(int jI,int jF);
bool indF_filter(FintensityJob & job,int jI,int jF,double energyI,int igammaI,int* quantaI);
bool degeneracy_filter(FintensityJob & job, int gammaI,int gammaF,int idegI,int idegF);

extern "C" double c_three_j(int * j1,int * j2,int * j3,int * k1,int * k2,int * k3);
