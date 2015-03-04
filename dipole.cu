#include "dipole.cuh"




__global__ void compute_1st_half_ls(int jI,int jF,int indI, int indF,int igammaI,int idegI,double* vector,int jmax,double* threej,double* half_ls)
{
	//
	
	int idx = threadIdx.x*blockDim.x + blockIdx;
	int dimenI = bset_contr[indI].Maxcontracts;
	int dimenF = bset_contr[indF].Maxcontracts;
	
	half_ls[idx] = 0;
	
	double sq2 = 1.0/sqrt(2.0);
	
	double ls = 0.0;
	int icontrI,sigmaI,tauI;
	int icontrF = bset_contr[indF].iroot_correlat_j0[idx];
	int ktau = bset_contr[indF].ktau[idx];
	int tauF = ktau%2;
	int sigmaF = (kF%3)*tauF;
	
	for(int i =0; i < dimenI; i++)
	{
		/*
		  kI = bset_contr(indI)%k(irootI)
                  !
                  if (abs(kF - kI)>1) cycle loop_I
                  !
                  icontrI = bset_contr(indI)%iroot_correlat_j0(irootI)
                  !
                  !irlevelI = bset_contr(indI)%ktau(irootI)
                  !irdegI   = bset_contr(indI)%k(irootI)
                  !
                  ktau = bset_contr(indI)%ktau(irootI)
                  tauI  = mod(ktau,2_ik)
                  !
                  sigmaI = mod(kI, 3)*tauI
                  !
                  f3j  =  threej(jI, kI, jF - jI, kF - kI)   
		*/
	    kI = bset_contr[indI].k[i]
	    
	    if(abs(kF-kI) > 1) continue;
	    /*
	                      icontrI = bset_contr(indI)%iroot_correlat_j0(irootI)
                  !
                  !irlevelI = bset_contr(indI)%ktau(irootI)
                  !irdegI   = bset_contr(indI)%k(irootI)
                  !
                  ktau = bset_contr(indI)%ktau(irootI)
                  tauI  = mod(ktau,2_ik)
                  !
                  sigmaI = mod(kI, 3)*tauI
                  !
                  f3j  =  threej(jI, kI, jF - jI, kF - kI)                 
                  ! 
                  ! 3j-symbol selection rule
                  !
                  if (abs(f3j)<intensity%threshold%coeff) cycle loop_I
	    */
	    
	    icontrI = bset_contr[indI].iroot_correlat_j0[i];
	    ktau = bset_contr[indI].ktau[i];
	    tauI = ktau%2;
	    sigmaI = (kI%3)*tauI;
	    
	    f3j = threej[kF-kI,jF-jI,kI,jI];
	    
	    ls = 0.0
	    
	    if(kF==kI)
	    	ls = double(tauF-tauI)*dipole_me[3,icontrF,icontrI];
	    else 
	    {
	    	if(tauF != tauI)
	    		ls = double((kF-kI)*(tauF-tauI))*dipole_me[1,icontrF,icontrI];
	    	else if(tauF==tauI)
	    		ls =  -dipole_me[2,icontrF,icontrI];
	    	if(kI*kF != 0) ls*=sq2;
	    }	
	    
	    half_ls[idx] += pow(-1,sigmaI+kI)*ls*f3j*vector[i]
	}
	
	half_ls[idx]*=pow(-1,sigmaF);
	

}
