#pragma once
#include "common.h"
/*

 type bset_contrT
    integer(ik)                 :: jval            ! rotational quantum number that correspond to the contr. basis set
    integer(ik)                 :: Maxsymcoeffs
    integer(ik)                 :: max_deg_size
    integer(ik)                 :: Maxcontracts
    integer(ik)                 :: Nclasses
    integer(ik),        pointer :: icontr2icase(:, :)
    integer(ik),        pointer :: icase2icontr(:, :)
    integer(ik),        pointer :: ktau(:)
    integer(ik),        pointer :: k(:)
    integer(ik),        pointer :: contractive_space(:, :)
    type(PTintcoeffsT), pointer :: index_deg(:)
    type(PTrotquantaT), pointer :: rot_index(:, :)
    integer(ik),        pointer :: icontr_correlat_j0(:, :)
    integer(ik),        pointer :: iroot_correlat_j0(:)
    integer(ik),        pointer :: nsize(:)
    type(PTrepresT),    pointer :: irr(:)
 end type bset_contrT
 
 */

struct TO_PTrotquantaT
{
     int	 j;  
     int	 k;
     int	tau;
};
struct TO_PTintcoeffsT 
{      
      
      char*	type;
      int*	icoeffs;  //iCoeffs indexes - arbitrary information
      int	size1,size2;
      
};

struct TO_PTrepresT
{
	double* repres;
	int* N;
};
 

//A replica of trove's bset_contr type [tran.f90]
struct TO_bset_contrT
{
    int 	jval;
    int 	Maxsymcoeffs;
    int 	max_deg_size;
    int 	Maxcontracts;
    int 	Nclasses;
    int* 	icontr2icase;
    int*	icase2icontr;
    int* 	ktau;
    int* 	k;
    int*	contractive_space;
    TO_PTintcoeffsT*  index_deg;
    TO_PTrotquantaT* rot_index;
    int*	icontr_correlat_j0;
    int*	 iroot_correlat_j0;
    int* 	nsize;
    TO_PTrepresT* irr;
    int* Ntotal;
    int mat_size;
    int* ijterms;
    int ncases;
    int nlambdas;
};

struct TO_PTeigen{
    int ndeg;
    int igamma;
    int* irec;
    int* iroot;
    int ilevel;
    int* quanta;
    int* normal;
    char** cgamma;
    double energy;
    int jind;
    int jval;
    int krot;
    int taurot;
};

