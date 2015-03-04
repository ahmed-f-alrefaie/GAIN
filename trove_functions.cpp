#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>
#include "common.h"
#include "trove_functions.h"
#include "trove_objects.h"
#include "Util.h"
#include "fields.h"
#include <iostream>
using namespace std;

//int sym_nrepres = 4; 
//int sym_degen[4] = {1,1,1,1}; 

//C++ version of fortran's NINT
#define NINT(a) ((a) >= 0.0 ? (int)((a)+0.5) : (int)((a)-0.5))

#define NDEBUG

const char* extFmat_file = "j0_extfield.chk";//"contr_extfield.chk";
const char* j0eigen_filebase = "j0eigen";
const char* j0eigen_quanta_filebase = "j0eigen_quanta%i.chk";
const char* j0eigen_vector_gamma_filebase = "j0eigen_vectors%i_%i.chk";
const char* j0eigen_descr_gamma_filebase = "j0eigen_descr%i_%i.chk";

void read_dipole(TO_bset_contrT & bsetj0,double** dipole_me,size_t & dip_size)
{
	char buff[20];
	int imu,imu_t;
	int ncontr_t;
   	size_t matsize,rootsize,rootsize_t,rootsize2;
   	
	FILE* extF = fopen(extFmat_file,"r");
	
	if(extF == NULL)
	{
		printf("[read_dipole] Error: checkpoint file of name %s not found",extFmat_file);
		fprintf(stderr,"[read_dipole] Error: checkpoint file not found");
		exit(0);
	}
	
	ReadFortranRecord(extF, buff);
	
	if(memcmp(buff,"Start external field",20) != 0)
	{
		printf("[read_dipole] Error: checkpoint file of name %s has bogus header %s",extFmat_file,buff);
		fprintf(stderr,"[read_dipole] bogus header");
		exit(0);
	}
	
	ReadFortranRecord(extF, &ncontr_t);
	
	//Check basis size here//
	//printf("ncontr_t = %i ",ncontr_t);
	if(ncontr_t != bsetj0.Maxcontracts)
	{
			printf("[read_dipole] Actual and stored basis sizes at J=0 do not agree  %i /= %i",bsetj0.Maxcontracts,ncontr_t);
			fprintf(stderr,"[read_dipole] Actual and stored basis sizes at J=0 do not agree");
			exit(0);
	}
	
	
	
	/////////////////////////
	
	rootsize  = ncontr_t*(ncontr_t+1)/2;
   	rootsize2 = ncontr_t*ncontr_t;
	matsize = rootsize2*3;
	(*dipole_me) = new double[matsize];
	
	for(int i = 0 ; i< 3; i++)
	{
		ReadFortranRecord(extF, &imu_t);
		if(imu_t != (i+1))
		{
			printf("[read_dipole] has bogus imu - restore_vib_matrix_elements: %i /= %i",imu_t,(i+1));
			fprintf(stderr,"[read_dipole] bogus imu");
			exit(0);
		}
		
		ReadFortranRecord(extF, (*dipole_me) + i*rootsize2);
				
			 
	}
		
	ReadFortranRecord(extF, buff);		 
		if(memcmp(buff,"End external field",18) != 0)
	{
		printf("[read_dipole] Error: checkpoint file of name %s has bogus footer %s",extFmat_file,buff);
		fprintf(stderr,"[read_dipole] bogus footer");
		exit(0);
	}
	
	fclose(extF);

	#ifdef DEBUG
	//for(int i = 0; i < ncontr_t; i++)
//		for(int j = 0; j < ncontr_t; j++)//
			//for(int k = 0; k < 3; k++)
		//		printf("dipole[%i,%i,%i] = %16.8e\n",i,j,k,(*dipole_me)[i + j*ncontr_t + k*ncontr_t*ncontr_t]);
	//exit(0);
	#endif
	dip_size = sizeof(double)*matsize;
	
	
}


void read_dipole_flipped(TO_bset_contrT & bsetj0,double** dipole_me,size_t & dip_size)
{
	char buff[20];
	int imu,imu_t;
	int ncontr_t;
   	size_t matsize,rootsize,rootsize_t,rootsize2;
   	
	FILE* extF = fopen(extFmat_file,"r");
	
	if(extF == NULL)
	{
		printf("[read_dipole] Error: checkpoint file of name %s not found",extFmat_file);
		fprintf(stderr,"[read_dipole] Error: checkpoint file not found");
		exit(0);
	}
	
	ReadFortranRecord(extF, buff);
	
	if(memcmp(buff,"Start external field",20) != 0)
	{
		printf("[read_dipole] Error: checkpoint file of name %s has bogus header %s",extFmat_file,buff);
		fprintf(stderr,"[read_dipole] bogus header");
		exit(0);
	}
	
	ReadFortranRecord(extF, &ncontr_t);
	
	//Check basis size here//
	//printf("ncontr_t = %i ",ncontr_t);
	if(ncontr_t != bsetj0.Maxcontracts)
	{
			printf("[read_dipole] Actual and stored basis sizes at J=0 do not agree  %i /= %i",bsetj0.Maxcontracts,ncontr_t);
			fprintf(stderr,"[read_dipole] Actual and stored basis sizes at J=0 do not agree");
			exit(0);
	}
	
	
	
	/////////////////////////
	
	rootsize  = ncontr_t*(ncontr_t+1)/2;
   	rootsize2 = ncontr_t*ncontr_t;
	matsize = rootsize2*3;
	double* temp_dipole = new double[matsize];
	(*dipole_me) = new double[matsize];
	
	for(int i = 0 ; i< 3; i++)
	{
		ReadFortranRecord(extF, &imu_t);
		if(imu_t != (i+1))
		{
			printf("[read_dipole] has bogus imu - restore_vib_matrix_elements: %i /= %i",imu_t,(i+1));
			fprintf(stderr,"[read_dipole] bogus imu");
			exit(0);
		}
		
		ReadFortranRecord(extF, (temp_dipole) + i*rootsize2);
				
			 
	}
		
	ReadFortranRecord(extF, buff);		 
		if(memcmp(buff,"End external field",18) != 0)
	{
		printf("[read_dipole] Error: checkpoint file of name %s has bogus footer %s",extFmat_file,buff);
		fprintf(stderr,"[read_dipole] bogus footer");
		exit(0);
	}
	
	fclose(extF);

	#ifdef DEBUG
	//for(int i = 0; i < ncontr_t; i++)
//		for(int j = 0; j < ncontr_t; j++)//
			//for(int k = 0; k < 3; k++)
		//		printf("dipole[%i,%i,%i] = %16.8e\n",i,j,k,(*dipole_me)[i + j*ncontr_t + k*ncontr_t*ncontr_t]);
	//exit(0);
	#endif
	int zeros = 0;
	dip_size = sizeof(double)*matsize;
	printf("Flipping dipole........");
	for(int i = 0; i < ncontr_t; i++)
		for(int f = 0; f < ncontr_t; f++)
			for(int k = 0; k < 3; k++){
				(*dipole_me)[f + i*ncontr_t + k*ncontr_t*ncontr_t] = temp_dipole[i + f*ncontr_t + k*ncontr_t*ncontr_t];
				if((*dipole_me)[f + i*ncontr_t + k*ncontr_t*ncontr_t] == 0.0)
					zeros++;
			}



	
	delete[] temp_dipole;
	printf("done!\n");
	printf("Counted %i zeros we could have saved %12.6f GB\n",zeros,double(zeros)*8.0/(1024.0*1024.0*1024.0));
	
}

void read_dipole_flipped_blocks(TO_bset_contrT & bsetj0,FDipole_ptrs & dipole_block, int parts)
{
	char buff[20];
	int imu,imu_t;
	int ncontr_t;
   	size_t matsize,rootsize,rootsize_t,rootsize2;
	FILE* extF = fopen(extFmat_file,"r");
	
	if(extF == NULL)
	{
		printf("[read_dipole] Error: checkpoint file of name %s not found",extFmat_file);
		fprintf(stderr,"[read_dipole] Error: checkpoint file not found");
		exit(0);
	}
	
	ReadFortranRecord(extF, buff);
	
	if(memcmp(buff,"Start external field",20) != 0)
	{
		printf("[read_dipole] Error: checkpoint file of name %s has bogus header %s",extFmat_file,buff);
		fprintf(stderr,"[read_dipole] bogus header");
		exit(0);
	}
	
	ReadFortranRecord(extF, &ncontr_t);
	
	//Check basis size here//
	//printf("ncontr_t = %i ",ncontr_t);
	if(ncontr_t != bsetj0.Maxcontracts)
	{
			printf("[read_dipole] Actual and stored basis sizes at J=0 do not agree  %i /= %i",bsetj0.Maxcontracts,ncontr_t);
			fprintf(stderr,"[read_dipole] Actual and stored basis sizes at J=0 do not agree");
			exit(0);
	}
	
	
	
	/////////////////////////
	
	rootsize  = ncontr_t*(ncontr_t+1)/parts;
   	rootsize2 = ncontr_t*ncontr_t;
	int contr_div = ceil(float(ncontr_t)/float(parts));
	matsize = rootsize2*3;
	double* temp_dipole = new double[matsize];
	//(*dipole_me) = new double[matsize];
	
	for(int i = 0 ; i< 3; i++)
	{
		ReadFortranRecord(extF, &imu_t);
		if(imu_t != (i+1))
		{
			printf("[read_dipole] has bogus imu - restore_vib_matrix_elements: %i /= %i",imu_t,(i+1));
			fprintf(stderr,"[read_dipole] bogus imu");
			exit(0);
		}
		
		ReadFortranRecord(extF, (temp_dipole) + i*rootsize2);
				
			 
	}
		
	ReadFortranRecord(extF, buff);		 
		if(memcmp(buff,"End external field",18) != 0)
	{
		printf("[read_dipole] Error: checkpoint file of name %s has bogus footer %s",extFmat_file,buff);
		fprintf(stderr,"[read_dipole] bogus footer");
		exit(0);
	}
	
	fclose(extF);

	#ifdef DEBUG
	//for(int i = 0; i < ncontr_t; i++)
//		for(int j = 0; j < ncontr_t; j++)//
			//for(int k = 0; k < 3; k++)
		//		printf("dipole[%i,%i,%i] = %16.8e\n",i,j,k,(*dipole_me)[i + j*ncontr_t + k*ncontr_t*ncontr_t]);
	//exit(0);
	#endif
	int n_contr_block = ceil(float(ncontr_t)/float(parts));
	int cur_block_size = 0;
	int startF=0,endF=0,ncontrF=0;
	//Allocate X parts
	dipole_block.dip_block=new FDipole_block[parts];
	dipole_block.parts=parts;	
	printf("Flipping dipole...and blocking int %i parts\n",parts);
	for(int blocks = 0; blocks < parts; blocks++){
		startF=blocks*n_contr_block;
		
		endF = (blocks+1)*n_contr_block;
		endF = min(endF,ncontr_t);
		ncontrF = endF-startF;
		//Allocate the dipole
		dipole_block.dip_block[blocks].startF = startF;
		dipole_block.dip_block[blocks].endF = endF;
		dipole_block.dip_block[blocks].dipole_me = new double[ncontrF*ncontr_t*3];
		dipole_block.dip_block[blocks].size = sizeof(double)*ncontrF*ncontr_t*3;
		printf("Size of part %i is %zu\n",blocks,dipole_block.dip_block[blocks].size);
		dipole_block.dip_block[blocks].ncontrF = ncontrF;
		printf("startF =%i, blocks = %i, n_contr_block = %i\n",startF,blocks,ncontrF);
		for(int i = 0; i < ncontr_t; i++)
			for(int f = startF; f < endF; f++)
				for(int k = 0; k < 3; k++)
					dipole_block.dip_block[blocks].dipole_me[f-startF + i*ncontrF + k*ncontr_t*ncontrF] = temp_dipole[i + f*ncontr_t + k*ncontr_t*ncontr_t]; 

	}


	delete[] temp_dipole;
	printf("done!\n");
	
	
}


//Ported trove's fakt function [moltype.f90] 
double fakt(double a)
{
      double ax,f;
      int i,ic	;
      
      ax=a;
      f=1.0;
      if(abs(ax)<pow(10.0,-24)) return f;
      f=0.1;
      if(ax < 0.0){
         printf(" fkt.err  negative argument for functi on fakt. argument = %10.4e)",ax);
         fprintf(stderr,"fkt.err  negative argument");
         exit(0);
      }
      
      ic=NINT(ax);
      ax=ax/10.0;
      f=ax;
      for(i=1; i< ic; i++)
        f*=(ax-double(i)*0.1);
        
      return f;
}       
      


//Ported trove's three_j subroutine [moltype.f90]
double three_j(int j1,int j2,int j3,int k1,int k2,int k3)
{

	/*
	int newmin,newmax,_new,iphase;
	double a,b,c,al,be,ga,delta,clebsh,minus;
        double term,term1,term2,term3,summ,dnew,term4,term5,term6;
        
      	a = j1;
      	b = j2;
      	c = j3;
      	al= k1;
      	be= k2;
      	ga= k3;

      	double three_j=0.0;
      	
      	if(c > a+b) return three_j;
        if(c < abs(a-b)) return three_j;
        if(a < 0.0 || b < 0.0 || c < 0.0) return three_j;
        if(a < abs(al) || b < abs(be) || c < abs(ga)) return three_j;
        if(-1.0*ga != al+be) return three_j;
      	
      	delta=sqrt(fakt(a+b-c)*fakt(a+c-b)*fakt(b+c-a)/fakt(a+b+c+1.0));     
      	term1=fakt(a+al)*fakt(a-al);
      	term2=fakt(b-be)*fakt(b+be);
      	term3=fakt(c+ga)*fakt(c-ga);
      	term=sqrt((2.0*c+1.0)*term1*term2*term3);
      	
      	newmin=NINT(max(max((a+be-c),(b-c-al)),0.0));
        newmax=NINT(min(min((a-al),(b+be)),(a+b-c))) ;       
        
        summ=0;
        
        for(_new=newmin; _new <=newmax; _new++)
        {
        	dnew=double(_new);
        	term4=fakt(a-al-dnew)*fakt(c-b+al+dnew);
        	term5=fakt(b+be-dnew)*fakt(c-a-be+dnew);
        	term6=fakt(dnew)*fakt(a+b-c-dnew);
        	summ+=pow(-1.0,_new)/(term4*term5*term6);
        }
        
         clebsh=delta*term*summ/sqrt(10.0);
         
         iphase=NINT(a-b-ga);
      	 minus = -1.0;
      	if (iphase%2==0) minus = 1.0;
      	three_j=minus*clebsh/sqrt(2.0*c+1.0);
      	
      	return three_j;
	*/
	return c_three_j(&j1,&j2,&j3,&k1,&k2,&k3);
        
}

void bset_contr_factory(TO_bset_contrT* bset_contr,uint jval,int* sym_degen,int sym_nrepres){
	

	int mat_size;
	int* Ntotal;
	char filename[1024];
	string line;
	char* line_ptr;
	int ncases,nlambdas,ncontr,nclasses,icontr,iroot;
	if(bset_contr == NULL)
	{
		printf("[bset_contr_factory]: bset is null");
		fprintf(stderr,"[bset_contr_factory]: bset is null");
		exit(0);
	}
	//Null pointer to be safe :3c
	bset_contr->icontr2icase=NULL;
    	bset_contr->icase2icontr=NULL;
   	bset_contr->ktau=NULL;
    	bset_contr->k=NULL;
    	bset_contr->contractive_space=NULL;
    	bset_contr->index_deg=NULL;
    	bset_contr->rot_index=NULL;
    	bset_contr->icontr_correlat_j0=NULL;
    	bset_contr->iroot_correlat_j0=NULL;
    	bset_contr->nsize=NULL;
    	bset_contr->irr=NULL;
    	bset_contr->Ntotal=NULL;
	
	
	
	//Get the filename
	sprintf(filename,j0eigen_quanta_filebase,jval);
	
	//Open file
	ifstream eig_qu(filename);
	
	//Begin reading
	getline(eig_qu,line);
	
	if(trim(line).compare("Start Primitive basis set")!=0)
	{
		printf("[bset_contr_factory]: bad header %s",filename);
		fprintf(stderr,"[bset_contr_factory]: bad header");
		exit(0);
	}	

	getline(eig_qu,line);
	/*
	       read(iounit, '(4i8)') ncases, nlambdas, ncontr, nclasses

       bset_contr(jind)%Maxsymcoeffs = ncases
       bset_contr(jind)%max_deg_size = nlambdas
       bset_contr(jind)%Maxcontracts = ncontr
       bset_contr(jind)%nclasses     = nclasses
	*/
	bset_contr->jval = jval;
	bset_contr->Maxsymcoeffs  = ncases = strtol(line.c_str(),&line_ptr,0);
	bset_contr->max_deg_size = nlambdas = strtol(line_ptr,&line_ptr,0);
	bset_contr->Maxcontracts = ncontr = strtol(line_ptr,&line_ptr,0);
	bset_contr->Nclasses = nclasses = strtol(line_ptr,&line_ptr,0);
	
	
	
	//Allocate memory
	/*
	       allocate(bset_contr(jind)%index_deg(ncases),bset_contr(jind)%contractive_space(0:nclasses, ncases),stat = info)
       call ArrayStart('bset_contr',info,size(bset_contr(jind)%contractive_space),kind(bset_contr(jind)%contractive_space))
       !
       allocate(bset_contr(jind)%nsize(sym%Nrepresen),stat = info)
       call ArrayStart('bset_contr',info,size(bset_contr(jind)%nsize),kind(bset_contr(jind)%nsize))
       !
       allocate(bset_contr(jind)%icontr2icase(ncontr, 2),bset_contr(jind)%icase2icontr(ncases, nlambdas),stat = info)
       call ArrayStart('bset_contr',info,size(bset_contr(jind)%icontr2icase),kind(bset_contr(jind)%icontr2icase))
       call ArrayStart('bset_contr',info,size(bset_contr(jind)%icase2icontr),kind(bset_contr(jind)%icase2icontr))
       */
       bset_contr->index_deg = new TO_PTintcoeffsT[ncases];
       bset_contr->contractive_space = new int[(nclasses+1)*ncases];
       bset_contr->nsize = new int[sym_nrepres];
       bset_contr->icontr2icase = new int[ncontr*2];
       bset_contr->icase2icontr = new int[ncases*nlambdas];
	//cout<<bset_contr->nsize[0]<<endl;
       	#ifdef DEBUG
		printf("%i %i %i %i\n",bset_contr->Maxsymcoeffs,bset_contr->max_deg_size,bset_contr->Maxcontracts,bset_contr->Nclasses);
	#endif
       icontr = 0;
       iroot = 0;
       
       for(int icase = 0; icase < ncases; icase++)
       {
       		getline(eig_qu,line);
       		//read(iounit,*) nlambdas, bset_contr(jind)%contractive_space(0:nclasses, icase)
        	//bset_contr(jind)%index_deg(icase)%size1 = nlambdas
        	nlambdas = strtol(line.c_str(),&line_ptr,0);
        	for(int i = 0; i <=nclasses; i++)
        		 bset_contr->contractive_space[i + icase*(nclasses+1)] = strtol(line_ptr,&line_ptr,0)-1;
        	
        	bset_contr->index_deg[icase].size1 = nlambdas;
		#ifdef DEBUG
			printf("nlambdas = %i contr_space_1 = %i contr_space_2 = %i \n", nlambdas, bset_contr->contractive_space[0 + icase*(nclasses+1)], bset_contr->contractive_space[1 + icase*(nclasses+1)]);
		#endif
        	
        	//    allocate(bset_contr(jind)%index_deg(icase)%icoeffs(0:nclasses, nlambdas))
        	bset_contr->index_deg[icase].icoeffs = new int[(nclasses+1)*nlambdas]; 
        	
        	for(int ilambda = 0; ilambda < nlambdas; ilambda++)
        	{
        		getline(eig_qu,line);
        		//read(iounit, '(i8, i6, <nclasses>i6)') iroot, bset_contr(jind)%index_deg(icase)%icoeffs(0:nclasses, ilambda)
        		iroot = strtol(line.c_str(),&line_ptr,0)-1;
        		for(int i = 0; i <=nclasses; i++)
        			bset_contr->index_deg[icase].icoeffs[i+ ilambda*(nclasses+1)] = strtol(line_ptr,&line_ptr,0)-1;
        	
        		#ifdef DEBUG
				printf("iroot = %i icoeffs_1 = %i icoeffs_2 = %i \n", iroot, bset_contr->index_deg[icase].icoeffs[0+ ilambda*(nclasses+1)], bset_contr->index_deg[icase].icoeffs[1+ ilambda*(nclasses+1)]);
			#endif	
        		

             		bset_contr->icontr2icase[icontr + 0*ncontr]      = icase;
             		bset_contr->icontr2icase[icontr + 1*ncontr]      = ilambda;
        		#ifdef DEBUG
				printf("icontr = %i icase = %i ilambda = %i \n", icontr,icase,ilambda);
			#endif	
             		bset_contr->icase2icontr[icase + ilambda*ncases] = icontr;
        	
        		
        		
        		if(iroot != icontr)
        		{
        			fprintf(stderr,"[bset_contr_factory] wrong indexing icontr = %i iroot = %i\n",icontr,iroot);
        			exit(0);
        		}
        		icontr = icontr + 1;
        	}
        	  
       }
	//if(jval==1){printf("icontr2icase[:,2]\n");     
	//for(int i = 0; i < ncontr; i++)
	//	printf("%i\n",bset_contr->icontr2icase[i + 1*ncontr]);

	//exit(0);
		
	//}

	if (icontr != ncontr)
	{
		fprintf(stderr,"[bset_contr_factory] wrong indexing\n");
		exit(0);
	}
	
	//read(iounit, '(2i8)') ncases, nlambdas
	getline(eig_qu,line);
	bset_contr->ncases = ncases = strtol(line.c_str(),&line_ptr,0);
	bset_contr->nlambdas =nlambdas = strtol(line_ptr,&line_ptr,0);
	
	//allocate(bset_contr(jind)%rot_index(ncases, nlambdas), stat = info) 
	bset_contr->rot_index=new TO_PTrotquantaT[ncases*nlambdas];
	printf("j= %i rot_index size =%i,ncases=%i, nlambdas = %i\n",jval,ncases*nlambdas,ncases,nlambdas); 
	int icase,ilambda;
	while(getline(eig_qu,line))
	{
		if( trim(line).compare("End Primitive basis set")==0)
			break;
		/*read(buf, *) icase, ilambda, bset_contr(jind)%rot_index(icase, ilambda)%j,                                       &
                                       bset_contr(jind)%rot_index(icase, ilambda)%k,                                       &
                                       bset_contr(jind)%rot_index(icase, ilambda)%tau
		
		*/
		icase = strtol(line.c_str(),&line_ptr,0)-1;
		ilambda = strtol(line_ptr,&line_ptr,0)-1;
		bset_contr->rot_index[icase + ilambda*ncases].j = strtol(line_ptr,&line_ptr,0);
		bset_contr->rot_index[icase + ilambda*ncases].k = strtol(line_ptr,&line_ptr,0);
		bset_contr->rot_index[icase + ilambda*ncases].tau = strtol(line_ptr,&line_ptr,0);
			
		#ifdef DEBUG
			printf("icase = %i ilambda = %i j = %i k = %i tau = %i\n",icase,ilambda,bset_contr->rot_index[icase + ilambda*ncases].j,bset_contr->rot_index[icase + ilambda*ncases].k,bset_contr->rot_index[icase + ilambda*ncases].tau);
		#endif
	
	}
	
	
		//Begin reading
	getline(eig_qu,line);
	//cout<<line<<endl;
	if(trim(line).compare("Start irreducible transformation")!=0)
	{
		printf("[bset_contr_factory]: wrong sym-footer");
		fprintf(stderr,"[bset_contr_factory]: wrong sym-footer");
		exit(0);
	}
	
	bset_contr->irr = new TO_PTrepresT[sym_nrepres];
	bset_contr->Ntotal = new int[sym_nrepres];
	getline(eig_qu,line);

	mat_size = strtol(line.c_str(),&line_ptr,0);

	printf("matsize = %i\n",mat_size);

	//cout<<line<<endl;
	bset_contr->mat_size = mat_size;

	//read(iounit,*) Ntotal(1:sym%Nrepresen)
	getline(eig_qu,line);

	bset_contr->Ntotal[0] = strtol(line.c_str(),&line_ptr,0); 

	//printf("Ntotal[0]=%i\n",bset_contr->Ntotal[0]);

	//cout<<line<<endl;
	for(int i = 1; i < sym_nrepres; i++){

		bset_contr->Ntotal[i] = strtol(line_ptr,&line_ptr,0);
		//printf("Ntotal[%i]=%i\n",i,bset_contr->Ntotal[i]);
	}
	//(bset_contr(jind)%irr(igamma)%N(bset_contr(jind)%Maxsymcoeffs),bset_contr(jind)%irr(igamma)%repres(Ntotal(igamma),sym%degen(igamma),mat_size)
	//printf("nrepres = %i",sym_nrepres);
	for(int igamma = 0; igamma < sym_nrepres; igamma++)
	{
		bset_contr->irr[igamma].N = new int[bset_contr->Maxsymcoeffs];
		bset_contr->irr[igamma].repres = new double[bset_contr->Ntotal[igamma]*sym_degen[igamma]*mat_size];
		
		//           do icoeff = 1,bset_contr(jind)%Maxsymcoeffs
                //		!
                // 		read(iounit,*) bset_contr(jind)%irr(igamma)%N(icoeff)
                //		!
                //	     enddo
                
                for(int icoeff = 0; icoeff <  bset_contr->Maxsymcoeffs; icoeff++)
                {
                		getline(eig_qu,line);
				//cout<<line<<endl;
				bset_contr->irr[igamma].N[icoeff] = strtol(line.c_str(),&line_ptr,0);
		}
                
                //do icoeff = 1,Ntotal(igamma)
              //		!
              //		do ideg = 1,sym%degen(igamma)
              //   	!
              //   	read(iounit,*) bset_contr(jind)%irr(igamma)%repres(icoeff,ideg,1:mat_size)
              //   	!
              //		enddo
              //		!
              //	enddo
	//	printf("irred-part\n");
              for(int icoeff = 0; icoeff <  bset_contr->Ntotal[igamma]; icoeff++)
              {
              		for(int ideg =0; ideg <  sym_degen[igamma]; ideg++)
              		{
              			getline(eig_qu,line);	
				//cout<<line<<endl;
              			bset_contr->irr[igamma].repres[icoeff + ideg*bset_contr->Ntotal[igamma]] = strtod(line.c_str(),&line_ptr);
				#ifdef DEBUG
					printf("irr[%i].repres[%i,%i,%i]=%11.4e\n",igamma,icoeff,ideg,0,bset_contr->irr[igamma].repres[icoeff + ideg*bset_contr->Ntotal[igamma] + 0*bset_contr->Ntotal[igamma]*sym_degen[igamma]]);
				#endif
              			for(int i = 1; i < mat_size; i++){
					bset_contr->irr[igamma].repres[icoeff + ideg*bset_contr->Ntotal[igamma] + i*bset_contr->Ntotal[igamma]*sym_degen[igamma]] = strtod(line_ptr,&line_ptr);
					#ifdef DEBUG
						printf("irr[%i].repres[%i,%i,%i]=%11.4e\n",igamma,icoeff,ideg,i,bset_contr->irr[igamma].repres[icoeff + ideg*bset_contr->Ntotal[igamma] + i*bset_contr->Ntotal[igamma]*sym_degen[igamma]]);
					#endif
				}
			}
	      }

		
			
	}
	getline(eig_qu,line);
	
	if(trim(line).compare("End irreducible transformation")!=0)
	{
		printf("[bset_contr_factory]: wrong irrep-footer");
		fprintf(stderr,"[bset_contr_factory]: wrong irrep-footer");
		exit(0);
	}
	
	printf("Done!\n");

	
}
//Ported correlation [trans.f90]
void correlate_index(TO_bset_contrT & bset_contrj0, TO_bset_contrT & bset_contr)
{
	
	printf("Establish the correlation between the indexes of J=0 and J=%i contr. basis funct.\n",bset_contr.jval);	
	
	if(bset_contrj0.jval != 0)
	{
		printf("[correlate_index] index_correlation: bset_contrj0 is not for J=0\n");
		exit(0);
	}
	
	int nclasses = bset_contr.Nclasses;
	
	if(bset_contrj0.Nclasses != nclasses)
	{
		fprintf(stderr,"[index_correlation]: Nclasses are different for diff. J");
		exit(0);
	}
	
	int icase,jcase,ilambda,jlambda,icontr,jcontr,info, iroot,jroot;
	int ilevel,ideg,k,tau,dimen,irow,icol;

	int*	cnu_i, *cnu_j;

	bool found;
	
	//allocate(cnu_i(1:nclasses),cnu_j(1:nclasses),stat = info)
	cnu_i = new int[nclasses];
	cnu_j = new int[nclasses];
	printf("Maxsymj0 = %i %i %i \n",bset_contrj0.Maxcontracts,bset_contrj0.Maxsymcoeffs,	bset_contrj0.max_deg_size);
	printf("Maxsym = %i %i %i \n",bset_contr.Maxcontracts,bset_contr.Maxsymcoeffs,bset_contr.max_deg_size);

	//allocate(bset_contr(jind)%icontr_correlat_j0(bset_contr(jind)%Maxsymcoeffs,bset_contr(jind)%max_deg_size), stat = info)
	bset_contr.icontr_correlat_j0 = new int[bset_contr.Maxsymcoeffs*bset_contr.max_deg_size];
	//printf("Do I root correlate\n");
	//printf("Start_correlation\n");
	//bool contract_space;
	//bool index_deg;
	//do icase = 1, bset_contr(jind)%Maxsymcoeffs
	for(icase = 0; icase < bset_contr.Maxsymcoeffs; icase++)
	{	
 		//printf("icase=%i\n",icase);
		//cnu_i(1:nclasses) = bset_contr(jind)%contractive_space(1:nclasses, icase)
		for(int c_i = 0; c_i < nclasses; c_i++)
			cnu_i[c_i] = bset_contr.contractive_space[(c_i+1) + icase*(nclasses+1)];
		//memcpy(cnu_i,bset_contr.contractive_space + icase*(nclasses+1) + 1,sizeof(int)*nclasses);
		//do ilambda = 1, bset_contr(jind)%index_deg(icase)%size1
		for(ilambda = 0; ilambda < bset_contr.index_deg[icase].size1; ilambda++)
		{
			found = false;

			//do jcase = 1, bset_contr(1)%Maxsymcoeffs
			for(jcase = 0; jcase < bset_contrj0.Maxsymcoeffs; jcase++)
			{
				//cnu_j(1:nclasses) = bset_contr(1)%contractive_space(1:nclasses, jcase)
				for(int c_j = 0; c_j < nclasses; c_j++)
					cnu_j[c_j] = bset_contrj0.contractive_space[(c_j+1) + jcase*(nclasses+1)];
				//memcpy(cnu_j,bset_contrj0.contractive_space + (1 + jcase*(nclasses+1)),sizeof(int)*nclasses);
				//do jlambda = 1, bset_contr(1)%index_deg(jcase)%size1
				for(jlambda = 0; jlambda < bset_contrj0.index_deg[jcase].size1; jlambda++)
				{
					/*			                 if (all(cnu_i(:) == cnu_j(:))  .and.   &
                   				  all(bset_contr(   1)%index_deg(jcase)%icoeffs(1:nclasses,jlambda) == &
                  				       bset_contr(jind)%index_deg(icase)%icoeffs(1:nclasses,ilambda))) then 
                  				       !
                  				       found = .true.
                 			        exit l_jcase
                			         !
                 					endif
               					  */  
					//for(int i = 0; i < nclasses; i++)
					//	printf(" %i ",cnu_i[i]);
					//printf("\t|\t");
					//for(int i = 0; i < nclasses; i++)
					//	printf(" %i ",cnu_j[i]);
					//printf("\n");					
					if(memcmp(cnu_i,cnu_j,sizeof(int)*nclasses)==0 && memcmp(bset_contrj0.index_deg[jcase].icoeffs + jlambda*(nclasses+1) + 1,
												 bset_contr.index_deg[icase].icoeffs + ilambda*(nclasses+1) + 1,
												 sizeof(int)*nclasses)==0)
					{
						found = true;
						break;
					}
				}//jlambda
				if(found) break;
			}//jcase
			if(!found)
			{
				printf("icase =%i ilambda = %i\n",icase,ilambda);
				printf("[index_correlation] not found for J = %i -> problems with checkpoints?\n", bset_contr.jval);
				fprintf(stderr,"[index_correlation] No correlation for J = %i\n", bset_contr.jval);
				exit(0);
			}
			//printf("Done!");
			//jcontr = bset_contr(1)%icase2icontr(jcase,jlambda)
			jcontr = bset_contrj0.icase2icontr[jcase + jlambda*bset_contrj0.Maxsymcoeffs];
			//printf("jcontr = %i\n",jcontr);
			bset_contr.icontr_correlat_j0[icase + ilambda*bset_contr.Maxsymcoeffs] = jcontr;
			//             jcontr = bset_contr(1)%icase2icontr(jcase,jlambda)
            // !
            // bset_contr(jind)%icontr_correlat_j0(icase, ilambda) = jcontr
		}//ilambda
	}//icase
	//printf("Finished correlation\n");
//	       allocate(bset_contr(jind)%iroot_correlat_j0(bset_contr(jind)%Maxcontracts), stat = info)
//       call ArrayStart('bset_contr',info,size(bset_contr(jind)%iroot_correlat_j0),kind(bset_contr(jind)%iroot_correlat_j0))
//       allocate(bset_contr(jind)%ktau(bset_contr(jind)%Maxcontracts), stat = info)
//       call ArrayStart('bset_contr',info,size(bset_contr(jind)%ktau),kind(bset_contr(jind)%ktau))
//       allocate(bset_contr(jind)%k(bset_contr(jind)%Maxcontracts), stat = info)
//       call ArrayStart('bset_contr',info,size(bset_contr(jind)%k),kind(bset_contr(jind)%k))
	//printf("Allocate iroot j0\n");
	bset_contr.iroot_correlat_j0 = new int[bset_contr.Maxcontracts];
	//printf("Allocate ktau\n");	
	bset_contr.ktau = new int[bset_contr.Maxcontracts];
	//printf("Allocate k\n");
	bset_contr.k = new int[bset_contr.Maxcontracts];
	
/*	       do iroot = 1, bset_contr(jind)%Maxcontracts
          !
          icase   = bset_contr(jind)%icontr2icase(iroot, 1)
          ilambda = bset_contr(jind)%icontr2icase(iroot, 2)
          !
          jcontr = bset_contr(jind)%icontr_correlat_j0(icase, ilambda)
          !
          bset_contr(jind)%iroot_correlat_j0(iroot) = jcontr
          !
          ilevel  = bset_contr(jind)%contractive_space(0,icase)
          ideg    = bset_contr(jind)%index_deg(icase)%icoeffs(0,ilambda)
          !
          k      = bset_contr(jind)%rot_index(ilevel,ideg)%k
          tau    = bset_contr(jind)%rot_index(ilevel,ideg)%tau
          !
          bset_contr(jind)%ktau(iroot) = 2*k+tau
          bset_contr(jind)%k(iroot)    = k
*/
	int ncases = bset_contr.ncases;
	int ncontr = bset_contr.Maxcontracts;
	//printf("Doing k j zero stuf ncases = %i nlambda=%i\n",ncases,bset_contr.max_deg_size);
	for(int iroot = 0; iroot < bset_contr.Maxcontracts; iroot++)
	{
		//if(bset_contr.jval==23&&iroot > 14346) printf("Reached beginning\n");
          	icase   = bset_contr.icontr2icase[iroot];
		//if(bset_contr.jval==23&&iroot > 14346) printf("icase =%i\n",icase);
         	ilambda = bset_contr.icontr2icase[iroot + ncontr];
		//if(bset_contr.jval==23&&iroot > 14346) printf("ilambda =%i\n",ilambda);			
		jcontr = bset_contr.icontr_correlat_j0[icase + ilambda*bset_contr.Maxsymcoeffs];
		//if(bset_contr.jval==23&&iroot > 14346) printf("jcontr =%i\n",jcontr);
		bset_contr.iroot_correlat_j0[iroot] = jcontr;
		//if(bset_contr.jval==23&&iroot > 14346) printf("done\n");

		ilevel  = bset_contr.contractive_space[0 + icase*(nclasses+1)];
		//if(bset_contr.jval==23&&iroot > 14346) printf("ilevel = %i\n",ilevel);
		ideg    = bset_contr.index_deg[icase].icoeffs[0 + (nclasses+1)*ilambda];
		//printf("ideg = %i\n",ideg);
		//if(bset_contr.jval==23&&iroot > 14346) printf("rot parts\n");
		k      = bset_contr.rot_index[ilevel+ideg*ncases].k;
		//if(bset_contr.jval==23&&iroot > 14346) printf("k = %i\n",k);
          	tau    = bset_contr.rot_index[ilevel+ideg*ncases].tau;
		//if(bset_contr.jval==23&&iroot > 14346) printf("tau=%i\n",tau);
		bset_contr.ktau[iroot] = 2*k+tau;
          	bset_contr.k[iroot]    = k;
		if(bset_contr.iroot_correlat_j0[iroot] >= bset_contrj0.Maxcontracts){
			printf("Error in root correlats %i %i \n",bset_contr.iroot_correlat_j0[iroot],bset_contrj0.Maxcontracts);
			exit(0);
		}
          	#ifdef DEBUG
          	 printf("iroot = %i ilevel = %i icase = %i ideg=%i ilambda = %i jcontr = %i k = %i, tau = %i\n",iroot,ilevel,icase,ideg,ilambda,jcontr,k,tau); fflush(0);
          	#endif 
		//printf("iroot = %i ilevel = %i icase = %i ideg=%i ilambda = %i jcontr = %i k = %i, tau = %i\n",iroot,ilevel,icase,ideg,ilambda,jcontr,k,tau);
		//printf("iroot = %i j0root = %i\n",iroot,jcontr);
          	
         }
	printf("Cleaning up\n");
	delete[] cnu_j;
	delete[] cnu_i;
	printf("done\n");
	
}	




void destroy_bset_contr(TO_bset_contrT* bset_contr,int sym_nrepres)
{
    #ifdef DEBUG
	printf("Destroying bset J=%i......",bset_contr->jval);
    #endif 
    destroy_arr_valid((void**)&bset_contr->icontr2icase);
    destroy_arr_valid((void**)&bset_contr->icase2icontr);
    destroy_arr_valid((void**)&bset_contr->ktau);
    destroy_arr_valid((void**)&bset_contr->k);
    destroy_arr_valid((void**)&bset_contr->contractive_space);
    if(bset_contr->index_deg !=NULL)
    {
	    for(int  i =0; i < bset_contr->Maxsymcoeffs; i++)
	    	delete[] bset_contr->index_deg[i].icoeffs;
    }
    destroy_arr_valid((void**)&bset_contr->index_deg);
    destroy_arr_valid((void**)&bset_contr->rot_index);
    destroy_arr_valid((void**)&bset_contr->icontr_correlat_j0);
    destroy_arr_valid((void**)&bset_contr->iroot_correlat_j0);
    destroy_arr_valid((void**)&bset_contr->nsize);
    if(bset_contr->irr!= NULL)
    {
	    for(int  i =0; i < sym_nrepres; i++)
	    {
	    	delete[] bset_contr->irr[i].repres;
	    	delete[] bset_contr->irr[i].N;
	    }

    	delete[] bset_contr->irr;    
    }
    destroy_arr_valid((void**)&bset_contr->Ntotal);
    #ifdef DEBUG
	printf("Done!\n");
    #endif 
};

void precompute_threej(double** threej,int jmax)
{
	//allocate(threej(0:jmax,0:jmax,-1:1,-1:1), stat = info)
	(*threej) = new double[(jmax+1)*(jmax+1)*3*3];
	//printf("threejsize = %i\n",(jmax+1)*(jmax+1)*3*3);
	//    do jI = 0,jmax
      //do jF = max(jI-1,0),min(jI+1,jmax)
      //  do kI = 0,jI
        //  do kF = max(kI-1,0),min(kI+1,jF)
        printf("Pre-computing threej...");
        for(int jI=0; jI <= jmax; jI++)
		for(int jF=max(jI-1,0); jF <= min(jI+1,jmax); jF++)
			for(int kI=0; kI <= jI; kI++)
				for(int kF=max(kI-1,0); kF <= min(kI+1,jF); kF++)
				{
					//printf("%i %i %i %i\n",jI,kI,jF-jI,kF-kI);
					//printf("Array position = %i\n",jI + kI*(jmax+1) +(jF-jI + 1)*(jmax+1)*(jmax+1) +  (kF-kI + 1)*(jmax+1)*(jmax+1)*3);
					//threej(jI, kI, jF - jI, kF - kI) = three_j(jI, 1, jF, kI, kF - kI, -kF)
					(*threej)[jI + kI*(jmax+1) +(jF-jI + 1)*(jmax+1)*(jmax+1) +  (kF-kI + 1)*(jmax+1)*(jmax+1)*3] = three_j(jI, 1, jF, kI, kF - kI, -kF);
					//printf("three-j[%i,%i,%i,%i] = %12.6f\n",jI,jF,kI,kF,three_j(jI, 1, jF, kI, kF - kI, -kF));
				}
				
        	
	//exit(0);
	//printf("Done!\n");
	
};

void compute_ijterms(TO_bset_contrT & bset_contr, int** ijterm,int sym_nrepres)
{
	/*
	      allocate (ijterm(jind)%kmat(bset_contr(jind)%Maxsymcoeffs,sym%Nrepresen),stat=info)
      call ArrayStart('ijterm',info,size(ijterm(jind)%kmat),kind(ijterm(jind)%kmat))
      !
      do igammaI = 1,sym%Nrepresen
         !
         Nterms = 0 
         !
         do iterm = 1,bset_contr(jind)%Maxsymcoeffs
           !
           ijterm(jind)%kmat(iterm,igammaI) = Nterms
           !
           Nterms = Nterms + bset_contr(jind)%irr(igammaI)%N(iterm) 
           !
         enddo
         !
      enddo
    enddo
    */
    printf("Computing ijterms for J= %i\n",bset_contr.jval);
    (*ijterm) = new int[bset_contr.Maxsymcoeffs*sym_nrepres];
    
    for(int igammaI= 0; igammaI < sym_nrepres; igammaI++)
    {
    	int Nterms = 0;
    	for(int iterm = 0; iterm < bset_contr.Maxsymcoeffs; iterm++)
    	{
    		#ifdef DEBUG
			printf("iterm,igamma = %i,%i Nterms = %i!\n",iterm,igammaI,Nterms);
    		#endif 
    		(*ijterm)[iterm + igammaI*bset_contr.Maxsymcoeffs] = Nterms;
    		Nterms += bset_contr.irr[igammaI].N[iterm];
    	}
    }
    
    printf(".....Done!\n");
    
};

void host_correlate_vectors(TO_bset_contrT* bset_contr,int idegI,int igammaI,int* ijterms,int* sym_degen,const double* vecI,double* vec)
{
/*
              do irootI = 1,dimenI
                 !
                 irow = bset_contr(indI)%icontr2icase(irootI,1)
                 ib   = bset_contr(indI)%icontr2icase(irootI,2)
                 !
                 iterm = ijterm(indI)%kmat(irow,igammaI)
                 !
                 dtemp0 = 0
                 !
                 nelem = bset_contr(indI)%irr(igammaI)%N(irow)
                 !
                 do ielem = 1,nelem
                    !
                    isrootI = iterm+ielem 
                    !
                    dtemp0 = dtemp0 + vecI(isrootI)*bset_contr(indI)%irr(igammaI)%repres(isrootI,idegI,ib)
                    !
                 enddo
                 !
                 vec(irootI) = dtemp0
                 !
              enddo
*/	int dimenI = bset_contr->Maxcontracts;
	//#pragma omp parallel for shared(bset_contr,sym_degen,ijterms,vec,vecI) firstprivate(idegI,igammaI,dimenI)
	for(int irootI = 0; irootI < dimenI; irootI++)
	{
	
	
			int irow,ib,iterm,nelem,isrootI,Ntot,sdeg;
			double dtemp0 = 0.0;
			irow = bset_contr->icontr2icase[irootI];
			ib = bset_contr->icontr2icase[irootI + bset_contr->Maxcontracts];
	
			iterm = ijterms[irow + igammaI*bset_contr->Maxsymcoeffs];
	
			nelem = bset_contr->irr[igammaI].N[irow];
	
			Ntot = bset_contr->Ntotal[igammaI];
			sdeg = sym_degen[igammaI]-1;
	
			for(int i = 0; i < nelem; i++)
			{
				isrootI = iterm+i;
				dtemp0 +=  vecI[isrootI]*bset_contr->irr[igammaI].repres[isrootI + idegI*Ntot + ib*sdeg*Ntot];
			}
	
			vec[irootI] = dtemp0;
			//printf("vec[%i]=%16.8e\n",irootI,dtemp0);
	}
	//exit(0);


}    

inline bool filter(FintensityJob & job, double energy,int igamma)
{
	return job.isym_do[igamma] & ((energy-job.ZPE)>= job.erange[0]) & ((energy-job.ZPE)<= job.erange[1]);
}

int sort_pteigen_func (const void * a, const void * b)
{
	TO_PTeigen* eiga,*eigb;
	eiga = (TO_PTeigen*)a;
	eigb = (TO_PTeigen*)b;
//	printf("nu a = %11.6f nu b = %11.6f\n",nu_a,nu_b);
  if (eiga->energy <  eigb->energy ) return -1;
  if ( eiga->energy == eigb->energy ) return 0;
  if ( eiga->energy >  eigb->energy ) return 1;
}

void find_igamma_pair(FintensityJob & intensity)
{
	int ngamma=0;
	printf ("Do igammas here");
	for(int igammaI = 0; igammaI < intensity.molec.sym_nrepres; igammaI++)
	{
		ngamma=0;
		intensity.igamma_pair[igammaI] = igammaI;
		printf("igammaI = %i sym_nrepres = %i\n",igammaI,intensity.molec.sym_nrepres);
		for(int igammaF = 0; igammaF < intensity.molec.sym_nrepres; igammaF++)
		{
			if(igammaI!=igammaF && intensity.isym_pairs[igammaI]==intensity.isym_pairs[igammaF])
			{
				intensity.igamma_pair[igammaI] = igammaF;
				ngamma++;
				if(ngamma>1){
					printf("find_igamma_pair: Assumption that selection rules come in pairs is not fulfilled!\n");
					exit(0);
					
				}
			}
				
		}
		//
		if(intensity.gns[igammaI] != intensity.gns[intensity.igamma_pair[igammaI]]){
			printf("find_igamma_pair: selection rules do not agree with Gns\n");
			exit(0);		
		}
		
	}

}

//Thus us a port of the read_eigenvalues function in Trove; This will store every single state into a single array  [read_eigenvalues - tran.f90]  
void read_eigenvalues(FintensityJob & job){
/*
    !
    implicit none
    !
    integer(ik), intent(in) :: njval, jval(njval)

    integer(ik)             :: jind, nmodes, nroots, ndeg, nlevels,  iroot, irec, igamma, ilevel, jlevel, &
                               ideg, ilarge_coef,k0,tau0,nclasses,nsize,   &
                               iounit, info, quanta(0:FLNmodes), iline, nroots_t, nu(0:FLNmodes),normal(0:FLNmodes),Npolyad_t
    integer(ik),allocatable :: ilevel_new(:,:,:),ktau_rot(:,:),isym(:)
    !
    real(rk)                :: energy,energy_t
    !
    character(cl)           :: filename, ioname, buf
    character(4)            :: jchar,gchar
    character(500)          :: buf500
    !
    logical                 :: passed
    logical                 :: normalmode_input = .false.
    integer(ik)             :: jind_t,maxdeg,gamma
    !
    type(PTeigenT)          :: eigen_t   ! temporal object used for sorting 'eigen'
    ! 
    if (job%verbose>=2) write(out,"(/'Read and sort eigenvalues in increasing order...')")
    !
    call TimerStart('Read eigenvalues')

    if (.not. allocated(bset_contr)) stop 'read_eigenval error: associated(bset_contr) = .false.'
    !
    nmodes = FLNmodes
    !nclasses = PTNclasses
    !
    nclasses = bset_contr(1)%nclasses


*/
	char filename[1024];
	string line;
	//The firs process is to count how many states pass the initial tests
	int nlevels = 0;
	int nroots = 0;
	int nroots_t=0;
	int jVal = 0;	
	int dim_basis=0;
	char* line_ptr;
	int igamma,ideg,maxdeg,irec, ilevel,ilarge_coef;
	double energy;
	int maxj = 0;
	int iroot = 0;
	maxdeg = 0;
	int* ilevel_new;
	//int ilevel_new(njval,sym%Nrepresen,nroots_t)
	int nJ = job.nJ;
	for(int i = 0; i < nJ; i++)
		maxj = max(maxj,job.jvals[i]);


	for(int i= 0; i < nJ; i++)
		nroots_t = max(job.bset_contr[i].Maxcontracts,nroots_t);

	ilevel_new = new int[nJ*job.molec.sym_nrepres*nroots_t];	

	//printf("Step");
	int (*ktau_rot)[2] = new int[1 + (2*maxj)][2];
	for(int i = 0; i < nJ; i++)
	{
		for(int gamma = 0; gamma < job.molec.sym_nrepres; gamma++)
		{
			if(!job.isym_do[gamma]) continue;
			jVal = job.jvals[i]; 
			//get the filename;
			sprintf(filename,j0eigen_descr_gamma_filebase,jVal,(gamma+1));
			ifstream descr_file(filename);
			if(!descr_file)
			{
				printf("Error! couldn';t open %s!!!!\n",filename);
				exit(0);
			}

			printf("I'm sorry but I can't read the fingerprints of %s yet so I'm skipping them :( please be careful ;_;\n",filename);

			//
			while(trim(line).compare("Start Quantum numbers and energies")!=0)
			{
				getline(descr_file,line);
				if(descr_file.eof()){
					printf("Error! malformed descr file %s!!!!\n",filename);
					exit(0);
				}
					
			}
			getline(descr_file,line);
			//Get the nroots
			getline(descr_file,line);
			getline(descr_file,line);
			nroots_t = strtol(line.c_str(),&line_ptr,0);
			dim_basis = strtol(line_ptr,&line_ptr,0);
			printf("J=%i nroots = %i, dim_basis = %i\n",jVal,nroots_t,dim_basis);

			//Check the max conracts
			//if(job.bset_contr[i+1].Maxcontracts != nroots_t)
			//{
			//	printf("Max contracts do not agree!!! %i != %i\n",job.bset_contr[i+1].Maxcontracts,nroots_t);
			//	exit(0);
			//}
			//Otherwise lets start running through and counting
			while(getline(descr_file,line)){
				//
				if(trim(line).compare("End Quantum numbers and energies")==0 || descr_file.eof())
					break;
				
				//Otherwise we read

				/*read(buf500, *) irec, igamma, ilevel, ideg, energy, nu(0:nmodes)*/
				strtol(line.c_str(),&line_ptr,0);//irec
				igamma = strtol(line_ptr,&line_ptr,0); //igamma
				if((igamma) != gamma+1){
					printf("Gammas dont match what the fuck bro\n");
					exit(0);
				}
				ilevel = strtol(line_ptr,&line_ptr,0)-1;
				ideg=strtol(line_ptr,&line_ptr,0); //ideg
				//Get the energy
				energy = strtod(line_ptr,&line_ptr);
				//if (job%ZPE<0.and.igamma==1.and.Jval(jind)==0) job%zpe = energy
				if(job.ZPE < 0 && igamma==1 && jVal==0) job.ZPE = energy;
				if(filter(job,energy,gamma)){
					if(ideg==1) {
						
						ilevel_new[i + gamma*nJ + ilevel*job.molec.sym_nrepres*2] = nlevels;
						nlevels++;
					}
					nroots += job.molec.sym_degen[gamma];
					maxdeg = max(maxdeg,(job.molec.sym_degen[gamma]));

				} 

			}

		}
	}
	
	if (nroots ==0){
		printf("No roots, filters too tight!! \n");
		exit(0);
	}

	job.Neigenlevels = nlevels;
	job.Neigenroots = nroots;
	//Otherwise allocate
	job.eigen = new TO_PTeigen[job.Neigenlevels];
	if(job.eigen == NULL){
		printf("Allocation error: job.eigen\n");
		exit(0);
	}
	printf("Number of eigenvalues = %i\n",nlevels);
/*
	    do ilevel = 1,Neigenlevels
      !
      allocate(eigen(ilevel)%irec(maxdeg),eigen(ilevel)%iroot(maxdeg),eigen(ilevel)%quanta(0:nmodes),&
               eigen(ilevel)%normal(0:nmodes),eigen(ilevel)%cgamma(0:nclasses), stat = info)
      if (info /= 0) stop 'read_eigenval allocation error: eigen%irec, eigen%quanta - out of memory'
      eigen(ilevel)%ndeg   = 0
      eigen(ilevel)%iroot = 0
      eigen(ilevel)%quanta = 0
      !
    enddo
	*/
	printf("Maxdeg = %i\n",maxdeg);
	//exit(0);
	//allocate memory
	for(int i = 0; i < job.Neigenlevels; i++)
	{

		job.eigen[i].irec = new int[maxdeg];
		job.eigen[i].iroot = new int[maxdeg];
		job.eigen[i].quanta = new int[job.molec.nmodes+1];
		job.eigen[i].normal = new int[job.molec.nmodes+1];	
		job.eigen[i].cgamma = new char*[job.molec.nclasses+1];
		job.eigen[i].igamma     = 0;
		job.eigen[i].ndeg = 0;
		//job.eigen[i].energy=99999999.9;
	}

	/*
	    do jind = 1, njval
       !
       ! reconstruct k_rot and tau_rot from a 1d distribution
       !
       ktau_rot(0,1) = 0
       ktau_rot(0,2) = mod(Jval(jind),2)
       !
       iroot = 0
       !
       do k0 = 1,Jval(jind)
         !
         do tau0 = 0,1
           !
           iroot = iroot + 1
           ktau_rot(iroot,1) = k0
           ktau_rot(iroot,2) = tau0
           !
         enddo 
         !
       enddo
*/	
	nlevels = 0;
	nroots = 0;
	for(int jind = 0; jind < nJ; jind++)
	{
		ktau_rot[0][0] = 0;
		ktau_rot[0][1] = job.jvals[jind] % 2;       
	//ktau_rot(0,1) = 0
        //ktau_rot(0,2) = mod(Jval(jind),2)
		iroot = 0;
		/*
			do k0 = 1,Jval(jind)
		*/
		for(int k0= 1; k0 <= job.jvals[jind]; k0++)
		{
			for(int tau0 = 0; tau0 <=1; tau0++){
			   iroot = iroot + 1;
			   ktau_rot[iroot][0] = k0;
			   ktau_rot[iroot][1] = tau0;
			   
			}
		}


		//Get the J_value
		jVal = job.jvals[jind];
		for(int gamma = 0; gamma < job.molec.sym_nrepres; gamma++)
		{
			if(!job.isym_do[gamma]) continue;
			//get the filename;
			sprintf(filename,j0eigen_descr_gamma_filebase,jVal,(gamma+1));
			ifstream descr_file(filename);
			if(!descr_file)
			{
				printf("Error! couldn';t open %s!!!!\n",filename);
				exit(0);
			}

			printf("I'm sorry but I can't read the fingerprints of %s yet so I'm skipping them :( please be careful ;_;\n",filename);

			//
			while(trim(line).compare("Start Quantum numbers and energies")!=0)
			{
				getline(descr_file,line);
				
				if(descr_file.eof()){
					printf("Error! malformed descr file %s!!!!\n",filename);
					exit(0);
				}
					
			}
			getline(descr_file,line);
			//Get the nroots
			getline(descr_file,line);
			getline(descr_file,line);
			nroots_t = strtol(line.c_str(),&line_ptr,0);
			dim_basis = strtol(line_ptr,&line_ptr,0);
			printf("J=%i nroots = %i, dim_basis = %i\n",jVal,nroots_t,dim_basis);
			//cout<<line<<endl;
			//cout<<job.bset_contr[jind+1].Maxcontracts<<endl;
			job.bset_contr[jind+1].nsize[gamma] = max(dim_basis,1);
			//cout<<job.bset_contr[jind+1].nsize[gamma]<<endl;
			//Check the max conracts
			//if(job.bset_contr[jind+1].Maxcontracts != nroots_t)
			//{
			//	printf("Max contracts do not agree!!! %i != %i\n",job.bset_contr[jind+1].Maxcontracts,nroots_t);
			//	exit(0);
			//}
			//Otherwise lets start running through and counting
			while(getline(descr_file,line)){
				//
				
				if(trim(line).compare("End Quantum numbers and energies")==0 || descr_file.eof())
					break;		
				irec = strtol(line.c_str(),&line_ptr,0);//irec
				igamma = strtol(line_ptr,&line_ptr,0); //igamma
				if((igamma) != gamma+1){
					printf("Gammas dont match what the fuck bro\n");
					exit(0);
				}
				ilevel = strtol(line_ptr,&line_ptr,0)-1; //ilevel
				ideg=strtol(line_ptr,&line_ptr,0); //ideg
				//Get the energy
				energy = strtod(line_ptr,&line_ptr);
				//if (job%ZPE<0.and.igamma==1.and.Jval(jind)==0) job%zpe = energy
				if(filter(job,energy,gamma)){
					//okay it passed right?
					nlevels = ilevel_new[jind + gamma*nJ + ilevel*job.molec.sym_nrepres*2];
					job.eigen[nlevels].irec[ideg-1]  = irec;
					if(ideg==1) {
						//Assign
						
						job.eigen[nlevels].jind       = jind;
						job.eigen[nlevels].jval       = jVal;
						job.eigen[nlevels].ilevel     = ilevel;
						job.eigen[nlevels].energy     = energy;
						job.eigen[nlevels].igamma     = gamma;
						//printf("ilevel = %i nlevels=%i nroots=%i gamma=%i indI=%i\n",ilevel,nlevels, nroots,job.eigen[nlevels].igamma,job.eigen[nlevels].jind);
						for(int q = 0; q < job.molec.nmodes+1; q++)
							job.eigen[nlevels].quanta[q]  = strtol(line_ptr,&line_ptr,0);
						strtol(line_ptr,&line_ptr,0); //large_coef part
						//isym
						for(int q = 0; q < job.molec.nclasses+1; q++){
							job.eigen[nlevels].cgamma[q]  = job.molec.c_sym[strtol(line_ptr,&line_ptr,0)-1];
							//printf("cgamma = %s\n",job.eigen[nlevels].cgamma[q]);
						}
						for(int q = 0; q < job.molec.nmodes+1; q++)
							job.eigen[nlevels].normal[q]  = strtol(line_ptr,&line_ptr,0);


						job.eigen[nlevels].krot       = ktau_rot[job.eigen[nlevels].quanta[0]][0];
						job.eigen[nlevels].taurot     = ktau_rot[job.eigen[nlevels].quanta[0]][1];
					
					}


					nroots += job.molec.sym_degen[gamma];
					job.eigen[nlevels].ndeg  = job.molec.sym_degen[gamma];
				}


			}


			

			


		}	
		
		
	}

	printf("nlevels=%i\n",nlevels);
	//Sort the energies
	//Sort the energies
	qsort(job.eigen,nlevels,sizeof(TO_PTeigen),sort_pteigen_func);
	//Print them out if debug

	//
	//#ifdef DEBUG
	//printf("Output states\n");
	//for(int i = 0; i < job.Neigenlevels; i++)
	//	printf("%12.6f %i %i %i %i %i %i\n",job.eigen[i].energy,job.eigen[i].jind,job.eigen[i].jval,job.eigen[i].ilevel,job.eigen[i].igamma,job.eigen[i].krot,job.eigen[i].taurot);
	//#endif	
	
	delete[] ktau_rot;
	delete[] ilevel_new;

	
	
}   


bool energy_filter_lower(FintensityJob & job,int J,double energy, int* quanta)
{
	for(int i = 0; i < job.molec.nmodes; i++){
		if(quanta[i+1] < job.quanta_lower[0][i])
			return false;
		if(quanta[i+1] > job.quanta_lower[1][i])
			return false;
	}
	return J >= job.jvals[0] &&  J <= job.jvals[1] && (energy-job.ZPE) >= job.erange_lower[0] &&  (energy-job.ZPE) <= job.erange_lower[1];
}
bool energy_filter_upper(FintensityJob & job,int J,double energy, int* quanta)
{
	for(int i = 0; i < job.molec.nmodes; i++){
		if(quanta[i+1] < job.quanta_upper[0][i])
			return false;
		if(quanta[i+1] > job.quanta_upper[1][i])
			return false;
	}
	return J >= job.jvals[0] &&  J <= job.jvals[job.nJ-1] && (energy-job.ZPE) >= job.erange_upper[0] &&  (energy-job.ZPE) <= job.erange_upper[1];
}
bool intensity_filter(FintensityJob & job,int jI,int jF,double energyI,double energyF,int igammaI,int igammaF,int* quantaI,int* quantaF){
	double nu_if = energyF - energyI ;
	bool passed = (job.gns[igammaI] > 0.0) && (nu_if >=job.freq_window[0]) && (nu_if <=job.freq_window[1]) && energy_filter_lower(job,jI,energyI,quantaI) && energy_filter_upper(job,jF,energyF,quantaF);
	
	if(passed){
		passed = passed && ((jF != job.jvals[0]) || (jI != job.jvals[0])) && (job.isym_pairs[igammaI] == job.isym_pairs[igammaF]) && (job.igamma_pair[igammaI]==igammaF) && (abs(jF-jI) <=1) && (jI+jF>=1);
	}

	return passed;



}

const char* branch(int jF,int jI){
	int delta_j = jF-jI;
	switch(delta_j){
		case 0: return "Q";
		case 1: return "R";
		case -1: return "P";
	}
		
};

bool indF_filter(FintensityJob & job,int jI,int jF,double energyI,int igammaI,int* quantaI){

	double energyF;
	int* quantaF;
	int igammaF;
	for(int ilevelF=0; ilevelF < job.Neigenlevels; ilevelF++){
		if(job.eigen[ilevelF].jval != jF) continue;
		energyF = job.eigen[ilevelF].energy;
		igammaF = job.eigen[ilevelF].igamma;
		quantaF = job.eigen[ilevelF].quanta;
		if(intensity_filter(job,jI,jF,energyI,energyF,igammaI,igammaF,quantaI,quantaF)) return true;
		
	}
	
	return false;


}

bool degeneracy_filter(FintensityJob & job, int gammaI,int gammaF,int idegI,int idegF){
	if(job.molec.sym_degen[gammaI]==1 || job.molec.sym_degen[gammaF]==1) return true;
	if(!job.reduced) return true;

	switch(job.molec.sym_type){
		case C3VM_ID:
		case D3HM_ID:
			if(idegI==1 && idegF==0) return true; else return false;
		break;
		default:
			return true;
	
	}
 

}
	//
