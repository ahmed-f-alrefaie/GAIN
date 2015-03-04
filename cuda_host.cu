#include "trove_objects.h"
#include "cuda_objects.cuh"
#include "cuda_host.cuh"
#include "cublas_v2.h"
#include "Util.h"
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#ifdef KEPLER
#define CUDA_STREAMS 32
#else
#define CUDA_STREAMS 16
#endif

#ifdef KEPLER
const size_t max_memory =5l*1024l*1024l*1024l; 
#else
const size_t max_memory = 4l*1024l*1024l*1024l; 
#endif
int itransit=0;


double pi = 4.0 * atan2(1.0,1.0);
double A_coef_s_1 = 64.0*pow(10.0,-36.0) * pow(pi,4.0)  / (3.0 * 6.62606896*pow(10.0,-27.0));
double planck = 6.62606896*pow(10.0,-27.0);
double avogno = 6.0221415*pow(10.0,23.0);
double vellgt = 2.99792458*pow(10.0,10.0);
double intens_cm_mol  = 8.0*pow(10.0,-36.0) * pow(pi,3.0)*avogno/(3.0*planck*vellgt);
double boltz = 1.380658*pow(10.0,-16.0);
    //beta = planck * vellgt / (boltz * intensity%temperature)

void CheckCudaError(const char* tag){
  // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("[%s] CUDA error: %s\n", tag,cudaGetErrorString(error));
    exit(-1);
  }
}


#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %u\n",  devProp.totalConstMem);
    printf("Texture alignment:             %u\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

void get_cuda_info(FintensityJob & intensity){
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);
    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }    // Iterate through devices

}

int count_free_devices(){
    int devCount;
    int free_devices=0;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);
    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
	cudaSetDevice(i);
	if(cudaFree(0)==cudaSuccess){	
		free_devices++;
		cudaThreadExit();
	}

    }    // Iterate through devices
   return free_devices;
}

int get_free_device(int last){
	int device_id=-1;
	last++;
	int devCount;
	cudaGetDeviceCount(&devCount);
	for(int i=last; i< devCount; i++){
		cudaSetDevice(i);
		if(cudaFree(0)==cudaSuccess){
			cudaThreadExit();
			return i;
		}	
	}

	return -1;


}

__host__ void copy_dipole_host(double* dipole_me,double** dipole_me_host,size_t & dip_size)
{
	printf("Alloc");
	cudaMallocHost(dipole_me_host,dip_size); //Malloc to pinned memory
	printf("memcpy");	
memcpy(dipole_me_host,dipole_me,dip_size);
	
	
}




__host__ void copy_array_to_gpu(void* arr,void** arr_gpu,size_t arr_size,const char* arr_name)
{

		//Malloc dipole
		if(cudaSuccess != cudaMalloc(arr_gpu,arr_size))
		{
			fprintf(stderr,"[copy_array_to_gpu]: couldn't malloc for %s \n",arr_name);
			CheckCudaError(arr_name);
			
			exit(0);			
		}
	if(cudaSuccess != cudaMemcpy((*arr_gpu),arr,arr_size,cudaMemcpyHostToDevice))
	{
		fprintf(stderr,"[copy_array_to_gpu]: error copying %s \n",arr_name);
		exit(0);
	}
};


//Copies relevant information needed to do intensity calculations onto the gpu
//Arguments p1: The bset_contr to copy p2: A device memory pointer to copy to
//Returns how much memory was used in bytes
__host__ size_t copy_bset_contr_to_gpu(TO_bset_contrT* bset_contr,cuda_bset_contrT* bset_gptr,int* ijterms,int sym_nrepres,int*sym_degen)
{
	size_t memory_used = 0;
	printf("Copying bset_contr for J=%i to gpu........",bset_contr->jval);
	//construct a gpu_bset_contr
	cuda_bset_contrT to_gpu_bset;
	printf("copy easy part\n");
	//Copy easy stuff
	to_gpu_bset.jval = bset_contr->jval;
	to_gpu_bset.Maxsymcoeffs = bset_contr->Maxsymcoeffs;
	to_gpu_bset.max_deg_size = bset_contr->max_deg_size;
	to_gpu_bset.Maxcontracts = bset_contr->Maxcontracts;
	to_gpu_bset.Nclasses = bset_contr->Nclasses;
	
	printf("copy icontr\n");
	//GPU pointer to icontr2icase///////////////////////////////////////
	int* icontr_gptr;
	//Malloc in the gpu
	if(cudaSuccess != cudaMalloc(&icontr_gptr,sizeof(int)*bset_contr->Maxcontracts*2))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't allocate memory for icontr2icase for J=%i\n",to_gpu_bset.jval);
		exit(0);
	}
	memory_used += sizeof(int)*bset_contr->Maxcontracts*2;

	//give the pointer to the cuda object
	to_gpu_bset.icontr2icase = icontr_gptr;
	
	//Copy over
	if(cudaSuccess != cudaMemcpy(icontr_gptr,bset_contr->icontr2icase,sizeof(int)*bset_contr->Maxcontracts*2,cudaMemcpyHostToDevice))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't copy icontr2icase to gpu for J=%i\n",to_gpu_bset.jval);
	}

	////////////////////////////////////////////////////////////////////////
	printf("copy iroot\n");
	////////////////////////////////Same for iroot_correlat_j0///////////////////////////////////////////////////
	int* iroot_corr_gptr;
	//Malloc in the gpu
	if(cudaSuccess != cudaMalloc (&iroot_corr_gptr , sizeof(int)*bset_contr->Maxcontracts ) )
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't allocate memory for iroot_correlat_j0 for J=%i\n",to_gpu_bset.jval);
		exit(0);
	}
	//give the pointer to the cuda object
	to_gpu_bset.iroot_correlat_j0 = iroot_corr_gptr;

	memory_used += sizeof(int)*bset_contr->Maxcontracts;	//Add memory used

	//Copy over
	if(cudaSuccess != cudaMemcpy(iroot_corr_gptr,bset_contr->iroot_correlat_j0,sizeof(int)*bset_contr->Maxcontracts,cudaMemcpyHostToDevice))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't copy iroot_correlat_j0 to gpu for J=%i\n",to_gpu_bset.jval);
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////		K		////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	printf("copy K\n");
	int* k_gptr;
	//Malloc in the gpu
	if(cudaSuccess != cudaMalloc(&k_gptr,sizeof(int)*bset_contr->Maxcontracts))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't allocate memory for K for J=%i\n",to_gpu_bset.jval);
		exit(0);
	}
	//give the pointer to the cuda object
	to_gpu_bset.k = k_gptr;
	memory_used += sizeof(int)*bset_contr->Maxcontracts;
	
	//Copy over
	if(cudaSuccess != cudaMemcpy(k_gptr,bset_contr->k,sizeof(int)*bset_contr->Maxcontracts,cudaMemcpyHostToDevice))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't copy k to gpu for J=%i\n",to_gpu_bset.jval);
	}	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////		KTau		////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	printf("copy Ktau\n");	
	int* kt_gptr;
	//Malloc in the gpu
	if(cudaSuccess != cudaMalloc(&kt_gptr,sizeof(int)*bset_contr->Maxcontracts))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't allocate memory for Ktau for J=%i\n",to_gpu_bset.jval);
		exit(0);
	}
	//give the pointer to the cuda object
	to_gpu_bset.ktau = kt_gptr;
	memory_used += sizeof(int)*bset_contr->Maxcontracts;
	
	//Copy over
	if(cudaSuccess != cudaMemcpy(kt_gptr,bset_contr->ktau,sizeof(int)*bset_contr->Maxcontracts,cudaMemcpyHostToDevice))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't copy ktau to gpu for J=%i\n",to_gpu_bset.jval);
		exit(0);
	}	
	///////////////////////////////////////////////N///////////////////////////////////////////////////////////////////
	printf("copy N\n");
	int* N_gptr;
	if(cudaSuccess != cudaMalloc(&N_gptr,sizeof(int)*sym_nrepres*bset_contr->Maxsymcoeffs))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't allocate memory for N for J=%i\n",to_gpu_bset.jval);
	}
	memory_used += sizeof(int)*sym_nrepres*bset_contr->Maxsymcoeffs;

	to_gpu_bset.N = N_gptr;

	printf("Malloc\n");

	int* Ncopy = (int*)malloc(sizeof(int)*sym_nrepres*bset_contr->Maxsymcoeffs);

	printf("Make copy\n");

	for(int i = 0; i < sym_nrepres; i++){
		for(int j = 0; j < bset_contr->Maxsymcoeffs; j++)
		{
			Ncopy[ i + (j*sym_nrepres)] = bset_contr->irr[i].N[j];
			//printf("N[%i,%i] = %i %i\n",i,j,Ncopy[ i + (j*sym_nrepres)],bset_contr->irr[i].N[j]);
		}
	}
	printf("Copy\n");		
	cudaMemcpy(N_gptr,Ncopy,sizeof(int)*sym_nrepres*bset_contr->Maxsymcoeffs,cudaMemcpyHostToDevice);

	to_gpu_bset.N = N_gptr;
	
	free(Ncopy);
	////////////////////////////////////////////////////////////////////////////////////////
	printf("copy Ntotal\n");
	//////////////////////////////N total////////////////////////////////////////////////////////
	int* Ntot_gptr;
	copy_array_to_gpu((void*)bset_contr->Ntotal,(void**)&Ntot_gptr,sizeof(int)*sym_nrepres,"Ntotal");
	to_gpu_bset.Ntotal = Ntot_gptr;
	///////////////////////////////////////////
	printf("copy irr_repres\n");
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////		irre_repres		////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	double** irr_gptr;
	if(cudaSuccess != cudaMalloc(&irr_gptr,sizeof(double*)*sym_nrepres))
	{
		fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't allocate memory for irreducible representation for J=%i\n",to_gpu_bset.jval);
		exit(0);
	}
	memory_used += sizeof(double*)*sym_nrepres;	
	
	to_gpu_bset.irr_repres = irr_gptr;
	
	//Hold pointers to doubles
	double** d_ptr = (double**)malloc(sizeof(double*)*sym_nrepres);
	
	for(int i =0; i < sym_nrepres; i++)
	{
		
		if(cudaSuccess != cudaMalloc(&d_ptr[i],sizeof(double)*bset_contr->Ntotal[i]*sym_degen[i]*bset_contr->mat_size))
		{
			fprintf(stderr,"[copy_bset_contr_to_gpu]: Couldn't allocate memory for irreducible representation for J=%i\n",to_gpu_bset.jval);
			exit(0);
		}
		memory_used += sizeof(double)*bset_contr->Ntotal[i]*sym_degen[i]*bset_contr->mat_size;
		//copy repres to irr_repres
		cudaMemcpy(d_ptr[i],bset_contr->irr[i].repres,sizeof(double)*bset_contr->Ntotal[i]*sym_degen[i]*bset_contr->mat_size,cudaMemcpyHostToDevice);
	}
	
	//copy pointerlist to irr_gptr;
	cudaMemcpy(irr_gptr,d_ptr,sizeof(double*)*sym_nrepres,cudaMemcpyHostToDevice);	
	free(d_ptr); //clear memory and pointer
	d_ptr = 0;
	
	printf("copy ijterms size = %i\n",bset_contr->Maxsymcoeffs*sym_nrepres);
	//Copy ijterms
	copy_array_to_gpu((void*)ijterms,(void**)&(to_gpu_bset.ijterms),sizeof(int)*bset_contr->Maxsymcoeffs*sym_nrepres,"ijterms");
	memory_used += sizeof(int)*bset_contr->Maxsymcoeffs*sym_nrepres;
	printf("copy final bset\n");
	/////////////////////////////////copy object over////////////////////////////////
	cudaMemcpy(bset_gptr,&to_gpu_bset,sizeof(cuda_bset_contrT),cudaMemcpyHostToDevice);
	
	printf(".....done!\n");

	return memory_used;

};

__host__ size_t create_and_copy_bset_contr_to_gpu(TO_bset_contrT* bset_contr,cuda_bset_contrT** bset_gptr,int* ijterms,int sym_nrepres,int*sym_degen)
{
	if(cudaSuccess != cudaMalloc(bset_gptr,sizeof(cuda_bset_contrT) ) )
	{
		fprintf(stderr,"[create_and_copy_bset_contr_to_gpu]: Couldn't allocate memory for bset\n");
		exit(0);
	}
	return copy_bset_contr_to_gpu( bset_contr,*bset_gptr,ijterms,sym_nrepres,sym_degen);
}

//Copy threej
__host__ void copy_threej_to_gpu(double* threej,double** threej_gptr, int jmax)
{
	copy_array_to_gpu((void*)threej,(void**) threej_gptr, (jmax+1)*(jmax+1)*3*3*sizeof(double),"three_j");
	
};


///////////Dipole stuff now

__host__ void dipole_initialise(FintensityJob* intensity){
	printf("Begin Input\n");
	read_fields(intensity);
	printf("End Input\n");

	//Wake up the gpu//
	printf("Wake up gpu\n");
	cudaFree(0);
	printf("....Done!\n");
	
	int jmax = max(intensity->jvals[0],intensity->jvals[1]);


	bset_contr_factory(&(intensity->bset_contr[0]),0,intensity->molec.sym_degen,intensity->molec.sym_nrepres);
	bset_contr_factory(&(intensity->bset_contr[1]),intensity->jvals[0],intensity->molec.sym_degen,intensity->molec.sym_nrepres);
	bset_contr_factory(&(intensity->bset_contr[2]),intensity->jvals[1],intensity->molec.sym_degen,intensity->molec.sym_nrepres);

	//Correlate them 
	correlate_index(intensity->bset_contr[0],intensity->bset_contr[0]);
	correlate_index(intensity->bset_contr[0],intensity->bset_contr[1]);
	correlate_index(intensity->bset_contr[0],intensity->bset_contr[2]);
	
	printf("Reading dipole\n");
	//Read the dipole
	read_dipole(intensity->bset_contr[0],&(intensity->dipole_me),intensity->dip_size);
	printf("Computing threej\n");
	//Compute threej
	precompute_threej(&(intensity->threej),jmax);
	//ijterms
	printf("Computing ijerms\n");
	compute_ijterms((intensity->bset_contr[1]),&(intensity->bset_contr[1].ijterms),intensity->molec.sym_nrepres);
	compute_ijterms((intensity->bset_contr[2]),&(intensity->bset_contr[2].ijterms),intensity->molec.sym_nrepres);

	//Read eigenvalues
	read_eigenvalues((*intensity));
	unsigned int dimenmax = 0;
	unsigned int nsizemax = 0;
	intensity->dimenmax = 0;
	intensity->nsizemax = 0;
	//Find nsize
	for(int i =0; i < intensity->molec.sym_nrepres; i++){
		if(intensity->isym_do[i]){
			nsizemax= max(intensity->bset_contr[1].nsize[i],nsizemax);
			nsizemax = max(intensity->bset_contr[2].nsize[i],nsizemax);
		}
	}

	printf("Biggest vector dimensiton is %u \n",nsizemax);
	for(int i = 0; i < 3; i++){
		printf("dimenmax = %u J=%i Maxcontracts =%i\n",dimenmax,intensity->bset_contr[i].jval,intensity->bset_contr[i].Maxcontracts);
		dimenmax = max(intensity->bset_contr[i].Maxcontracts,dimenmax);
	}
	//intensity->dimenmax = max(intensity->bset_contr[2].Maxcontracts,intensity->dimenmax);
	printf("Biggest max contraction is is %u \n",dimenmax);
	intensity->dimenmax = dimenmax;
	intensity->nsizemax = nsizemax;
	printf("Find igamma pairs\n");
	find_igamma_pair((*intensity));
	printf("done!\n");
	//Begin GPU related initalisation////////////////////////////////////////////////////////
	intensity_info int_gpu;
	//Copy over constants to GPU
	int_gpu.sym_nrepres = intensity->molec.sym_nrepres;
	int_gpu.jmax = jmax+1;
	int_gpu.dip_stride_1 = intensity->bset_contr[0].Maxcontracts;
	int_gpu.dip_stride_2 = intensity->bset_contr[0].Maxcontracts*intensity->bset_contr[0].Maxcontracts;
	int_gpu.dimenmax = intensity->dimenmax;
	int_gpu.sq2 = 1.0/sqrt(2.0);

	copy_array_to_gpu((void*)intensity->molec.sym_degen,(void**)&int_gpu.sym_degen,sizeof(int)*intensity->molec.sym_nrepres,"sym_degen");

	CheckCudaError("Pre-initial");
	printf("Copy intensity information\n");	
	copy_intensity_info(&int_gpu);
	printf("done\n");
	CheckCudaError("Post-initial");
	printf("Copying bset_contrs to GPU...\n");
	intensity->g_ptrs.bset_contr = new cuda_bset_contrT*[2];
	create_and_copy_bset_contr_to_gpu(&intensity->bset_contr[1],&(intensity->g_ptrs.bset_contr[0]),intensity->bset_contr[1].ijterms,intensity->molec.sym_nrepres,intensity->molec.sym_degen);
	create_and_copy_bset_contr_to_gpu(&intensity->bset_contr[2],&(intensity->g_ptrs.bset_contr[1]),intensity->bset_contr[2].ijterms,intensity->molec.sym_nrepres,intensity->molec.sym_degen);

	printf("Done\n");
	
	printf("Copying threej...\n");
	copy_threej_to_gpu(intensity->threej,&(intensity->g_ptrs.threej), jmax);
	printf("done\n");

	printf("Copying dipole\n");
	copy_array_to_gpu((void*)intensity->dipole_me,(void**)&(intensity->g_ptrs.dipole_me),intensity->dip_size,"dipole_me");
	printf("Done..");
	//exit(0);
	//Will improve
	intensity->gpu_memory = 1l*1024l*1024l*1024l;
	intensity->cpu_memory = 1l*1024l*1024l*1024l;
	
};

__host__ void dipole_do_intensities(FintensityJob & intensity){

	//Prinf get available cpu memory
	unsigned long available_cpu_memory = intensity.cpu_memory;
	unsigned long available_gpu_memory = intensity.gpu_memory;

	//Compute how many inital state vectors and final state vectors
	unsigned long no_final_states_cpu = ((available_cpu_memory)/8l - long(2*intensity.dimenmax))/(3l*intensity.dimenmax);//(Initial + vec_cor + half_ls)*dimen_max
	unsigned long no_final_states_gpu = ((available_gpu_memory)/8l - long(2*intensity.dimenmax))/(3l*intensity.dimenmax);//(Initial + vec_cor + half_ls)*dimen_max
	printf("No of final states in gpu_memory: %d  cpu memory: %d\n",no_final_states_gpu,no_final_states_cpu);

	//The intial state vector
	double* initial_vec = new double[intensity.dimenmax];

	double* gpu_initial_vec=NULL;

	copy_array_to_gpu((void*)initial_vec,(void**)&(gpu_initial_vec),sizeof(double)*intensity.dimenmax,"gpu_initial_vec");
	printf("%p\n",gpu_initial_vec);

	double* final_vec = new double[intensity.dimenmax];
	double* gpu_final_vec=NULL;
	
	copy_array_to_gpu((void*)final_vec,(void**)&(gpu_final_vec),sizeof(double)*intensity.dimenmax,"gpu_final_vec");

	double* corr_vec = new double[intensity.dimenmax];
	double* gpu_corr_vec=NULL;

	copy_array_to_gpu((void*)corr_vec,(void**)&(gpu_corr_vec),sizeof(double)*intensity.dimenmax,"gpu_corr_vec");

	double* half_ls = new double[intensity.dimenmax];
	double** gpu_half_ls=new double*[2];

	copy_array_to_gpu((void*)half_ls,(void**)&(gpu_half_ls[0]),sizeof(double)*intensity.dimenmax,"gpu_half_ls1");
	copy_array_to_gpu((void*)half_ls,(void**)&(gpu_half_ls[1]),sizeof(double)*intensity.dimenmax,"gpu_half_ls2");
	double line_str =0.0;


	char filename[1024];
	//Get the filename
	printf("Open vector unit\n");
	FILE** eigenvec_unit = new FILE*[2*intensity.molec.sym_nrepres];
	for(int i =0; i< 2; i++){
		for(int j = 0; j < intensity.molec.sym_nrepres; j++)
		{

			sprintf(filename,j0eigen_vector_gamma_filebase,intensity.jvals[i],j+1);
			printf("Reading %s\n",filename);
			eigenvec_unit[i + j*2] = fopen(filename,"r");
			if(eigenvec_unit[i + j*2] == NULL)
			{
				printf("error opening %s \n",filename);
				exit(0);
			}
		}
	}
	
	//Opened all units, now lets start compuing
	
	//Initialise cublas
	cublasHandle_t handle;
	cublasStatus_t stat;
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("CUBLAS initialization failed\n");
		return;
	}
	
	CheckCudaError("Initialisation");

			    // Number of threads in each thread block
    	int blockSize =256;
 
    	// Number of thread blocks in grid
    	int gridSize = (int)ceil((float)intensity.dimenmax/blockSize);

			printf("Nu_if\tJf Kf quantaF\t <-- \tJI KI tauI quantaI\t Ein_A\tLine_str\n");
	//Run
	for(int ilevelI = 0; ilevelI < intensity.Neigenlevels; ilevelI++){
	
			    //  ! start measuring time per line
	   //   !
	      int indI = intensity.eigen[ilevelI].jind;
	  //    !
	  //    !dimension of the bases for the initial states
	  //    !
	     int dimenI = intensity.bset_contr[indI+1].Maxcontracts;
	   //   !
	    //  !energy, quanta, and gedeneracy order of the initial state
	    //  !
	      int jI = intensity.eigen[ilevelI].jval;
	      double energyI = intensity.eigen[ilevelI].energy;
	      int igammaI  = intensity.eigen[ilevelI].igamma;
	      int * quantaI = intensity.eigen[ilevelI].quanta;
	      int * normalI = intensity.eigen[ilevelI].normal;
	      int ndegI   = intensity.eigen[ilevelI].ndeg;
	      int nsizeI = intensity.bset_contr[indI+1].nsize[igammaI];

	      FILE* unitI = eigenvec_unit[ indI + (igammaI)*2]; 
	    //   printf("Ilevel = %i\n",ilevelI);

	      if(!energy_filter_lower(intensity,jI,energyI,quantaI)) continue;
	      fseek(unitI,(intensity.eigen[ilevelI].irec[0]-1)*nsizeI*sizeof(double),SEEK_SET);


		//Read vector from file
	    //  printf("Read vector\n");
	     int tread =  fread(initial_vec,sizeof(double),nsizeI,unitI);

		//for(int i=0; i< nsizeI; i++){
		//	printf("vec[%i]=%16.8e\n",i,initial_vec[i]);}
		//printf("read = %i\n",tread);
		//Transfer it to the GPU
	//	printf("Transfer vector\n");
	        stat = cublasSetVector(intensity.dimenmax, sizeof(double),initial_vec, 1, gpu_initial_vec, 1);
		CheckCudaError("Set Vector I");

		cudaDeviceSynchronize();

	  //    printf("Correlating vectors\n");
		//for(int ideg = 0; ideg < ndegI; ideg++){
		//host_correlate_vectors(&intensity.bset_contr[indI+1],0,igammaI,intensity.bset_contr[indI+1].ijterms,intensity.molec.sym_degen,initial_vec,corr_vec);
	      	
		device_correlate_vectors<<<gridSize,blockSize>>>(intensity.g_ptrs.bset_contr[indI],0,igammaI, gpu_initial_vec,gpu_corr_vec);
			CheckCudaError("device correlate I");
			cudaDeviceSynchronize();

//
		//printf("Done\n");
		printf("J= %i energy = %11.4f\n",jI,energyI);

		

		printf("----------------------------------\n");
	     for(int indF=0; indF <2; indF++){
	     	device_compute_1st_half_ls<<<gridSize,blockSize>>>(intensity.g_ptrs.bset_contr[indI],intensity.g_ptrs.bset_contr[indF],intensity.g_ptrs.dipole_me,igammaI,gpu_corr_vec,intensity.g_ptrs.threej,gpu_half_ls[indF]);
			//CheckCudaError("compute half ls I");
			//cudaDeviceSynchronize();
			//cublasGetVector(dimenI, sizeof(double),gpu_half_ls[indF], 1, half_ls, 1);
			//for(int i=0; i< dimenI; i++){
			//  printf("half_ls[%i]=%16.8e\n",i,half_ls[i]);}
			//printf("----------------------------------\n");

		}
	
		


			
	
		//Final states
		for(int ilevelF = 0; ilevelF < intensity.Neigenlevels; ilevelF++){

					    //  ! start measuring time per line
		   //   !
		      int indF = intensity.eigen[ilevelF].jind;
		  //    !
			//printf("indF=%i",indF);
		  //    !dimension of the bases for the initial states
		  //    !
		     int dimenF = intensity.bset_contr[indF+1].Maxcontracts;
		   //   !
		    //  !energy, quanta, and gedeneracy order of the initial state
		    //  !
		      int jF = intensity.eigen[ilevelF].jval;
		      double energyF = intensity.eigen[ilevelF].energy;
		      int igammaF  = intensity.eigen[ilevelF].igamma;
		      int * quantaF = intensity.eigen[ilevelF].quanta;
		      int * normalF = intensity.eigen[ilevelF].normal;
		      int ndegF   = intensity.eigen[ilevelF].ndeg;
		      int nsizeF = intensity.bset_contr[indF+1].nsize[igammaF];

			FILE* unitF = eigenvec_unit[ indF + (igammaF)*2]; 

		      if(!energy_filter_upper(intensity,jF,energyF,quantaF)) continue;

			for(int i = 0; i < intensity.dimenmax; i++){
				final_vec[i]=0.0;
			}

		      fseek(unitF,(intensity.eigen[ilevelF].irec[0]-1)*nsizeF*sizeof(double),SEEK_SET);
			//Read vector from file
		      fread(final_vec,sizeof(double),nsizeF,unitF);
		
			//for(int i=0; i< dimenF; i++){
			//	printf("ivec[%i]=%16.8e\n",i,final_vec[i]);}

			if(!intensity_filter(intensity,jI,jF,energyI,energyF,igammaI,igammaF,quantaI,quantaF)) continue;
			//device_clear_vector<<<gridSize,blockSize>>>(gpu_final_vec);
			//Transfer it to the GPU
		      stat = cublasSetVector(intensity.dimenmax, sizeof(double),final_vec, 1, gpu_final_vec, 1);

			if (stat != CUBLAS_STATUS_SUCCESS) {
				printf ("CUBLAS SetVector F failed\n");
				printf ("Error code: %s\n",_cudaGetErrorEnum(stat));
				return;
			}

			double nu_if = energyF - energyI; 
			//for(int ideg = 0; ideg < ndegF; ideg++){
		        device_correlate_vectors<<<gridSize,blockSize>>>(intensity.g_ptrs.bset_contr[indF],0,igammaF, gpu_final_vec,gpu_corr_vec);
			CheckCudaError("correlate final vector");
		        cudaDeviceSynchronize();
			
			//cublasGetVector(dimenF, sizeof(double),gpu_corr_vec, 1, corr_vec, 1);
			//for(int i=0; i< dimenF; i++){
			//	printf("ivec[%i]=%16.8e\n",i,corr_vec[i]);}
			
			//}

//
			cudaDeviceSynchronize();
			//Compute ls
		//	for(int i = 0; i < dimenF; i++)
		//			printf("%11.4e\n",corr_vec[i]);
		//	//exit(0);
			line_str = 0;
			//cublasDdot (handle,intensity.dimenmax,gpu_half_ls[indF], 1,gpu_corr_vec, 1,&line_str);
			cublasDdot (handle, intensity.dimenmax, gpu_corr_vec, 1, gpu_half_ls[indF], 1, &line_str);
			//cublasDdot (handle, intensity.dimenmax, gpu_half_ls[indF], 1, gpu_half_ls[indF], 1, &line_str);
			double orig_ls = line_str;
			//Print intensitys
			line_str *= line_str;
			//printf("line_str %11.4e\n",line_str);
			double A_einst = A_coef_s_1*double((2*jI)+1)*line_str*pow(abs(nu_if),3);
			 line_str = line_str * intensity.gns[igammaI] * double( (2*jI + 1)*(2 * jF + 1) );

			//if(line_str < 0.00000000001) continue;
			/*

               write(out, "( (i4, 1x, a4, 3x),'<-', (i4, 1x, a4, 3x),a1,&
                            &(2x, f11.4,1x),'<-',(1x, f11.4,1x),f11.4,2x,&
                            &'(',1x,a3,x,i3,1x,')',1x,'(',1x,<nclasses>(x,a3),1x,<nmodes>(1x, i3),1x,')',1x,'<- ',   &
                            &'(',1x,a3,x,i3,1x,')',1x,'(',1x,<nclasses>(x,a3),1x,<nmodes>(1x, i3),1x,')',1x,   &
                            & 3(1x, es16.8),2x,(1x,i6,1x),'<-',(1x,i6,1x),i8,1x,i8,&
                            1x,'(',1x,<nmodes>(1x, i3),1x,')',1x,'<- ',1x,'(',1x,<nmodes>(1x, i3),1x,')',1x,& 
                            <nformat>(1x, es16.8))")  &
                            !
                            jF,sym%label(igammaF),jI,sym%label(igammaI),branch, &
                            energyF-intensity%ZPE,energyI-intensity%ZPE,nu_if,                 &
                            eigen(ilevelF)%cgamma(0),eigen(ilevelF)%krot,&
                            eigen(ilevelF)%cgamma(1:nclasses),eigen(ilevelF)%quanta(1:nmodes), &
                            eigen(ilevelI)%cgamma(0),eigen(ilevelI)%krot,&
                            eigen(ilevelI)%cgamma(1:nclasses),eigen(ilevelI)%quanta(1:nmodes), &
                            linestr,A_einst,absorption_int,&
                            eigen(ilevelF)%ilevel,eigen(ilevelI)%ilevel,&
                            itransit,istored(ilevelF),normalF(1:nmodes),normalI(1:nmodes),&
                            linestr_deg(1:ndegI,1:ndegF)
             endif

			*/


			printf("%11.4f\t(%i %i ) ( ",nu_if,jF,intensity.eigen[ilevelF].krot);
			for(int i = 0; i < intensity.molec.nmodes+1; i++)
				printf("%i ",quantaF[i]);
			printf(")\t <-- \t(%i %i ) ",jI,intensity.eigen[ilevelI].krot);
			for(int i = 0; i < intensity.molec.nmodes+1; i++)
				printf("%i ",quantaI[i]);	
			printf("\t %16.8e    %16.8e %16.8e\n",A_einst,line_str,orig_ls);

			//exit(0);		



			

			
		}
		
		
		


	}


	

		
	
}

__host__ void do_1st_half_ls(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,int dimenMax,int idegI,int igammaI,double* dipole_me,double* vecI,double* vec,double* threej,double* half_ls,cudaStream_t stream = 0){

		  int blockSize = 512;
		  int gridSize = gridSize = (int)ceil((float)dimenMax/blockSize);
		
		  device_correlate_vectors<<<gridSize,blockSize,0,stream>>>(bset_contrI,idegI,igammaI, vecI,vec);
		  #ifdef KEPLER
		  blockSize =576;
		  #else
		  blockSize = 640;
		  #endif

		  int numSMs;
		  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
		  #ifdef KEPLER
		  //numSMs*=2;
		  #endif
		  int block_num = (int)ceil((float)dimenMax/blockSize); 
	     	  device_compute_1st_half_ls_flipped_dipole<<<block_num,blockSize,0,stream>>>(bset_contrI,bset_contrF,
								   dipole_me,vec,threej,
								   half_ls);				
}
__host__ void do_1st_half_ls_shared(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,int jF,int j0dimen,int dimenMax,int idegI,int igammaI,double* dipole_me,double* vecI,double* vec,double* threej,double* half_ls,cudaStream_t stream = 0){//(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,int* gpu_dipole_idx,
		/*  double* gpu_dipole_factor,
		  double* gpu_sigmaF,
		  int dimenMax, int j0dimen,int idegI,int igammaI,double* dipole_me,double* vecI,double* vec,double* threej,double* half_ls,cudaStream_t stream = 0){

		  int blockSize = 512;
		  int gridSize = gridSize = (int)ceil((float)dimenMax/(float)blockSize);
		  int j_size = (dimenMax/j0dimen);		
		  device_correlate_vectors<<<gridSize,blockSize,0,stream>>>(bset_contrI,idegI,igammaI, vecI,vec);


		//  printf("dimenMax = %i j0dimen = %i Jsize = %i\n",dimenMax,j0dimen,j_size);

		
		  blockSize=448;
		  gridSize = (int)ceil((float)j_size/float(blockSize));

		  device_precompute_k_dipole<<<gridSize,blockSize,0,stream>>>(bset_contrI,bset_contrF,j0dimen,gpu_dipole_idx,gpu_dipole_factor,gpu_sigmaF,threej);
		 // CheckCudaError("Precompute_K");
		 

		  #ifdef KEPLER
		  blockSize = 1024;
		  #else
		  blockSize = 1024;
		  #endif
		  int numSMs;
		  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
		  gridSize = (int)ceil(dimenMax/float(blockSize));
		  
		//  size_t shared_count = (3*sizeof(double))*max(blockSize,200);
		  
		device_compute_1st_half_ls_flipped_dipole_high_occ<<<gridSize,blockSize,0,stream>>>(bset_contrI,bset_contrF,j0dimen,gpu_dipole_idx,gpu_dipole_factor,gpu_sigmaF,dipole_me,vec,half_ls);
		
*/
		  int blockSize = 512;
		  int gridSize = gridSize = (int)ceil((float)dimenMax/blockSize);
		
		  device_correlate_vectors<<<gridSize,blockSize,0,stream>>>(bset_contrI,idegI,igammaI, vecI,vec);
		  #ifdef KEPLER
		  blockSize = 128;
		  #else
		  blockSize = 128;
		  #endif
		  int numSMs;
		  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
		//  int block_num = 1+(7642/blockSize);
		  gridSize = (int)ceil((float)j0dimen/blockSize);
		  
	         //Create streams
		/* cudaStream_t half_ls_str[CUDA_STREAMS];
		  for(int i = 0; i < CUDA_STREAMS; i++)
			cudaStreamCreate(&half_ls_str[i]);

		  int stream_count = 0;
*/
		  for(int i = 0; i < (2*jF)+1; i++){
			//printf("i = %i End = %i\n",i,(2*jF)+1);
	     	  	device_compute_1st_half_ls_flipped_dipole_shared<<<gridSize,blockSize,0,stream>>>(bset_contrI,bset_contrF,i,
								   dipole_me,vec,threej,
								   half_ls);
			//stream_count++;
			//if(stream_count >= CUDA_STREAMS)
			//	stream_count = 0;
			//cudaDeviceSynchronize();

		  }
		/*
		cudaDeviceSynchronize();	
		
		double* shared_vector_check = new double[dimenMax];
		double* vector_check = new double[dimenMax];
		cublasGetVector(dimenMax,sizeof(double),half_ls,1,shared_vector_check,1);
		cudaDeviceSynchronize();
		//stat = cublasSetVector(nsizeI, sizeof(double),initial_vector, 1, gpu_initial_vector, 1);


	     	  device_compute_1st_half_ls_flipped_dipole<<<gridSize,blockSize,0,stream>>>(bset_contrI,bset_contrF,
								   dipole_me,vec,threej,
								   half_ls);
		cudaDeviceSynchronize();
		cublasGetVector(dimenMax,sizeof(double),half_ls,1,vector_check,1);

		printf ("Shared\tOriginal\tPercentDif\tirootF\tj,k block\n");		
		for(int i = 0; i< dimenMax; i++){
			printf("%16.8E %16.8E %12.6f \\ %10i %10i %10i \n",shared_vector_check[i],vector_check[i],(vector_check[i]-shared_vector_check[i])*100.0/vector_check[i],i,(i/j0dimen), j0dimen);
		
		}
		delete[] vector_check;
				
		exit(0);
		*/
		
	//CheckCudaError("1st_high_OCC");
		//cudaDeviceSynchronize();		
		//CheckCudaError("Clear vectors");			
}

__host__ void do_1st_half_ls_blocks(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,int dimenMax,int idegI,int igammaI,double* gpu_dipole,FDipole_ptrs &  dipole_me,double* vecI,double* vec,double* threej,double* half_ls,cudaStream_t stream = 0){

		  int blockSize = 512;
		  int gridSize = (int)ceil((float)dimenMax/blockSize);
		
		  device_correlate_vectors<<<gridSize,blockSize,0,stream>>>(bset_contrI,idegI,igammaI, vecI,vec);
		CheckCudaError("correlate");
		  blockSize = 64;
		  gridSize = (int)ceil((float)dimenMax/blockSize);
		  int parts = dipole_me.parts;
		 
		 // printf("parts = %i\n",parts);
		  for(int i = 0; i < parts; i++){
			//  printf("i=%i\n startF = %i endF = %i ncontr = %i",i, dipole_me.dip_block[i].startF, dipole_me.dip_block[i].endF ,dipole_me.dip_block[i].ncontrF );
		     	  device_compute_1st_half_ls_flipped_dipole_blocks<<<gridSize,blockSize,0,stream>>>(bset_contrI,bset_contrF,
									   dipole_me.dip_block[i].startF,dipole_me.dip_block[i].endF,dipole_me.dip_block[i].ncontrF,gpu_dipole,vec,threej,
									   half_ls);		//Compute half ls
											//Transfer next block
			//cudaDeviceSynchronize();
			//CheckCudaError("half_ls");
			if((i+1) >= parts){
			// printf("memcopy %i/%i\n",i+1,parts);
			// printf("size=%zu\n gpu_dipole = %p dipole_me = %p \n",dipole_me.dip_block[0].size,gpu_dipole,dipole_me.dip_block[0].dipole_me);
			 cudaMemcpyAsync(gpu_dipole,dipole_me.dip_block[0].dipole_me,dipole_me.dip_block[0].size,cudaMemcpyHostToDevice,stream) ;
			}else{
			//  printf("memcopy %i\n",i);
			 // printf("size=%zu\n gpu_dipole = %p dipole_me = %p \n",dipole_me.dip_block[i+1].size,gpu_dipole,dipole_me.dip_block[i+1].dipole_me);
			  cudaMemcpyAsync(gpu_dipole,dipole_me.dip_block[i+1].dipole_me,dipole_me.dip_block[i+1].size,cudaMemcpyHostToDevice,stream) ;
			}
			//CheckCudaError("Memcpy");
			//cudaDeviceSynchronize();
		}
		//exit(0);
			//exit(0);


					
}
__host__ void do_1st_half_ls_shared_blocks(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,int jF,int j0dimen,int dimenMax,int idegI,int igammaI,double* gpu_dipole,FDipole_ptrs &  dipole_me,double* vecI,double* vec,double* threej,double* half_ls,cudaStream_t stream = 0){
		  int blockSize = 512;
		  int gridSize = (int)ceil((float)dimenMax/blockSize);
		
		  device_correlate_vectors<<<gridSize,blockSize,0,stream>>>(bset_contrI,idegI,igammaI, vecI,vec);
		CheckCudaError("correlate");
		  blockSize = 128;
		  gridSize = (int)ceil((float)j0dimen/blockSize);
		  int parts = dipole_me.parts;
		 
		 // printf("parts = %i\n",parts);
		  for(int i = 0; i < parts; i++){
			//  printf("i=%i\n startF = %i endF = %i ncontr = %i",i, dipole_me.dip_block[i].startF, dipole_me.dip_block[i].endF ,dipole_me.dip_block[i].ncontrF );
			 for(int j = 0; j < (2*jF)+1; j++){
		     		  device_compute_1st_half_ls_flipped_dipole_shared_blocks<<<gridSize,blockSize,0,stream>>>(bset_contrI,bset_contrF,j,
									   dipole_me.dip_block[i].startF,dipole_me.dip_block[i].endF,dipole_me.dip_block[i].ncontrF,gpu_dipole,vec,threej,
									   half_ls);	
			 }				//Compute half ls
											//Transfer next block
			
			//CheckCudaError("half_ls");
			if((i+1) >= parts){
			// printf("memcopy %i/%i\n",i+1,parts);
			// printf("size=%zu\n gpu_dipole = %p dipole_me = %p \n",dipole_me.dip_block[0].size,gpu_dipole,dipole_me.dip_block[0].dipole_me);
			 cudaMemcpyAsync(gpu_dipole,dipole_me.dip_block[0].dipole_me,dipole_me.dip_block[0].size,cudaMemcpyHostToDevice,stream) ;
			}else{
			// printf("memcopy %i\n",i);
			// printf("size=%zu\n gpu_dipole = %p dipole_me = %p \n",dipole_me.dip_block[i+1].size,gpu_dipole,dipole_me.dip_block[i+1].dipole_me);
			  cudaMemcpyAsync(gpu_dipole,dipole_me.dip_block[i+1].dipole_me,dipole_me.dip_block[i+1].size,cudaMemcpyHostToDevice,stream) ;
			}
			//CheckCudaError("Memcpy");
			
		}
				
		
}




__host__ void do_1st_half_ls_branch(cuda_bset_contrT* bset_contrI,cuda_bset_contrT* bset_contrF,int dimenMax,int idegI,int igammaI,double* dipole_me,double* vecI,double* vec,double* threej,double* half_ls,cudaStream_t stream = 0){

		  int blockSize = 512;
		  int gridSize = gridSize = (int)ceil((float)dimenMax/blockSize);
		
		  device_correlate_vectors<<<gridSize,blockSize,0,stream>>>(bset_contrI,idegI,igammaI, vecI,vec);
		  blockSize = 256;
		  int numSMs;
		  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
		  
	     	  device_compute_1st_half_ls_flipped_dipole_branch<<<numSMs/2,blockSize,0,stream>>>(bset_contrI,bset_contrF,
								   dipole_me,vec,threej,
								   half_ls);				
}

/////////////////////////////--------------------Multi-threaded verstions--------------------///////////////////////////////////////////////

__host__ void dipole_initialise_cpu(FintensityJob* intensity){
	printf("Begin Input\n");
	read_fields(intensity);
	printf("End Input\n");
	
	int jmax=0;
	for(int i = 0; i < intensity->nJ; i++)
		jmax = max(intensity->jvals[i],jmax);

	//Now create the bset_contrs
	printf("Sym_nrepres = %i jmax = %i\n",intensity->molec.sym_nrepres, jmax);
	//Now create the bset_contrs
	bset_contr_factory(&(intensity->bset_contr[0]),0,intensity->molec.sym_degen,intensity->molec.sym_nrepres);
	intensity->molec.nclasses = intensity->bset_contr[0].Nclasses;

	for(int i = 0; i < intensity->nJ; i++){
		//printf("We are using J = %i\n",intensity->jvals[i]);
		bset_contr_factory(&(intensity->bset_contr[i+1]),intensity->jvals[i],intensity->molec.sym_degen,intensity->molec.sym_nrepres);
	}

	//Correlate them 

	for(int i =0; i < intensity->nJ+1; i++)
		correlate_index(intensity->bset_contr[0],intensity->bset_contr[i]);
//	correlate_index(intensity->bset_contr[0],intensity->bset_contr[2]);

	
	printf("Check dipole size\n");
	size_t dipole_size = GetFilenameSize("j0_extfield.chk");
	if(dipole_size > intensity->dipole_max){
		printf("Splitting dipole\n");
		int parts = ceil(double(dipole_size)/double(intensity->dipole_max));
		read_dipole_flipped_blocks(intensity->bset_contr[0],intensity->dipole_blocks,parts);
		intensity->split_dipole = true;
	}else{
	//Read the dipole
		read_dipole_flipped(intensity->bset_contr[0],&(intensity->dipole_me),intensity->dip_size);
		intensity->split_dipole = false;
	}
	printf("Computing threej\n");
	//Compute threej
	precompute_threej(&(intensity->threej),jmax);

	//ijterms
	printf("Computing ijerms\n");

	for(int i =0; i < intensity->nJ; i++){
		compute_ijterms((intensity->bset_contr[i+1]),&(intensity->bset_contr[i+1].ijterms),intensity->molec.sym_nrepres);
		//compute_ijterms((intensity->bset_contr[2]),&(intensity->bset_contr[2].ijterms),intensity->molec.sym_nrepres);
	}
	

	printf("Read eigenvalues\n");
	//Read eigenvalues
	read_eigenvalues((*intensity));

	intensity->dimenmax = 0;
	intensity->nsizemax = 0;
	//Find nsize
	for(int i =0; i < intensity->molec.sym_nrepres; i++){
		if(intensity->isym_do[i]){
			for(int j =0; j < intensity->nJ; j++)
				intensity->nsizemax= max(intensity->bset_contr[j+1].nsize[i],intensity->nsizemax);
			//intensity->nsizemax = max(intensity->bset_contr[2].nsize[i],intensity->nsizemax);
		}
	}

	printf("Biggest vector dimensiton is %i \n",intensity->nsizemax);
	for(int i =0; i < intensity->nJ; i++)
			intensity->dimenmax = max(intensity->bset_contr[i+1].Maxcontracts,intensity->dimenmax);
	//intensity->dimenmax = max(intensity->bset_contr[2].Maxcontracts,intensity->dimenmax);
	printf("Biggest max contraction is is %i \n",intensity->dimenmax);
	printf("Find igamma pairs\n");
	find_igamma_pair((*intensity));
	printf("done!\n");

	
};

__host__ void dipole_initialise_gpu(FintensityJob * intensity, FGPU_ptrs & g_ptrs,int device_id){
	int jmax = max(intensity->jvals[0],intensity->jvals[1]);

	//Get available memory
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
	g_ptrs.avail_mem = size_t(double(devProp.totalGlobalMem)*0.95);
	printf("Available gpu memory = %2.4f GB",float(g_ptrs.avail_mem)/(1024.0f*1024.0f*1024.0f));
	printf("Total global memory:           %zu\n",  devProp.totalGlobalMem);
	//Begin GPU related initalisation////////////////////////////////////////////////////////
	intensity_info int_gpu;
	//Copy over constants to GPU
	int_gpu.sym_nrepres = intensity->molec.sym_nrepres;
	int_gpu.jmax = jmax+1;
	int_gpu.dip_stride_1 = intensity->bset_contr[0].Maxcontracts;
	int_gpu.dip_stride_2 = intensity->bset_contr[0].Maxcontracts*intensity->bset_contr[0].Maxcontracts;
	int_gpu.dimenmax = intensity->dimenmax;
	int_gpu.sq2 = 1.0/sqrt(2.0);
	printf("Sym max_degen = %i\n",intensity->molec.sym_maxdegen);
	copy_array_to_gpu((void*)intensity->molec.sym_degen,(void**)&int_gpu.sym_degen,sizeof(int)*intensity->molec.sym_nrepres,"sym_degen");
	g_ptrs.avail_mem -= sizeof(int)*intensity->molec.sym_nrepres;

	CheckCudaError("Pre-initial");
	printf("Copy intensity information...");	
	copy_intensity_info(&int_gpu);
	printf("done...");
	CheckCudaError("Post-initial");
	printf("Copying bset_contrs to GPU...");
	g_ptrs.bset_contr = new cuda_bset_contrT*[intensity->nJ];
	for(int i =0 ; i < intensity->nJ; i++)
		g_ptrs.avail_mem -= create_and_copy_bset_contr_to_gpu(&intensity->bset_contr[i+1],&(g_ptrs.bset_contr[i]),intensity->bset_contr[i+1].ijterms,intensity->molec.sym_nrepres,intensity->molec.sym_degen);
	//g_ptrs.avail_mem -= create_and_copy_bset_contr_to_gpu(&intensity->bset_contr[2],&(g_ptrs.bset_contr[1]),intensity->bset_contr[2].ijterms,intensity->molec.sym_nrepres,intensity->molec.sym_degen);

	printf("Done..");
	
	printf("Copying threej...");
	copy_threej_to_gpu(intensity->threej,&(g_ptrs.threej), jmax);
	g_ptrs.avail_mem -=(jmax+1)*(jmax+1)*3*3*sizeof(double);
	printf("done..");
	

	/*
	if(intensity->dip_size > g_ptrs.avail_mem)
	{
		printf("Dipole too large to fit into gpu memory, leaving on host gpu_avail = %zu dipole_size = %zu\n",g_ptrs.avail_mem,intensity->dip_size);
		if(omp_get_thread_num()==0) intensity->host_dipole=true;
	}else{

	printf("Copying dipole...");
	copy_array_to_gpu((void*)intensity->dipole_me,(void**)&(g_ptrs.dipole_me),intensity->dip_size,"dipole_me");
	g_ptrs.avail_mem -=intensity->dip_size;
	intensity->host_dipole=false;
	}

	#pragma omp barrier
	if(intensity->host_dipole && omp_get_thread_num()==0){
		printf("Copying dipole\n");
		
		//double* replacement_dipole;
		printf("Allocing memory....");
		if(cudaSuccess != cudaHostRegister(intensity->dipole_me,intensity->dip_size,cudaHostAllocPortable |  cudaHostAllocMapped | cudaHostAllocWriteCombined)) printf("Could not malloc!!!\n");
		CheckCudaError("Dipole!");
		printf("copying....");
		//memcpy(replacement_dipole,intensity->dipole_me,intensity->dip_size);
		//copy_dipole_host(intensity->dipole_me,&replacement_dipole,intensity->dip_size);
		printf("Done");
		//Clear dipole from memory
		//delete[] intensity->dipole_me;
		//Put new dipole
		//intensity->dipole_me = replacement_dipole;
	}
	*/
	intensity->host_dipole=false;
	if(intensity->split_dipole){
		size_t malloc_size = intensity->dipole_blocks.dip_block[0].size;
		//for(int i = 0; i < intensity->dipole_blocks.parts; i++){
		//	malloc_size = max((unsigned long long)malloc_size,(unsigned long long)intensity->dipole_blocks.dip_block[i].size);
		//}
		printf("maqlloc size is %zu\n",malloc_size);
		printf("-------------------------------------------UTILIZING DIPOLE SPLITTING----------------------------------------------------------------");
		copy_array_to_gpu((void*)intensity->dipole_blocks.dip_block[0].dipole_me,(void**)&(g_ptrs.dipole_me),malloc_size,"dipole_me_block");	
		g_ptrs.avail_mem -=malloc_size ;

	}else{
		copy_array_to_gpu((void*)intensity->dipole_me,(void**)&(g_ptrs.dipole_me),intensity->dip_size,"dipole_me");
		g_ptrs.avail_mem -=intensity->dip_size;
		intensity->host_dipole=false;
		//remove dipole from memory since we dont need it anymore
		delete[] intensity->dipole_me;
	}	
	
	
	printf("Left over memory is %zu bytes\n",g_ptrs.avail_mem);

	printf("Done\n");
	
}

__host__ void dipole_do_intensities_async_omp(FintensityJob & intensity,int device_id,int num_devices){

	//cudaThreadExit(); // clears all the runtime state for the current thread
	///cudaSetDevice(device_id); //Set the device name
	//Wake up the gpu//
	//printf("Wake up gpu\n");
	cudaFree(0);
	//printf("....Done!\n");
	int current_stream = 0;
	char buffer[1024];
	int nJ = intensity.nJ;
	int process_id;
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	//fflush(0);
	
	char post_fix[] = "_%i_.out";

	char output_format[1024];
	char output_filename[1024];
	strcpy(output_format,intensity.output_file);
	printf("%s",output_format);
	strcat(output_format,post_fix);
	sprintf(output_filename,output_format,process_id);

	printf("\nOutputting to file %s\n",output_filename);

	FILE* output_intensity = fopen(output_filename,"w");

	//timing information
	unsigned long int half_ls_count = 0;
	unsigned long int ddot_count = 0;
	double half_ls_total_time = 0.0;
	double ddot_total_time = 0.0;
	double io_total_time = 0.0;
	double cur_io_time = 0.0;
	
	double current_time = 0.0;
	int io_calls=0;

	//Setup the gpu pointers
	FGPU_ptrs g_ptrs;
	dipole_initialise_gpu(&intensity,g_ptrs,device_id); // Initialise the gpu pointers
	//Prinf get available cpu memory
	//unsigned long available_cpu_memory = intensity.cpu_memory;
	size_t available_gpu_memory = g_ptrs.avail_mem;
	//Compute how many inital state vectors and final state vectors

	//unsigned long no_final_states_cpu = ((available_cpu_memory)/8l - long(2*intensity.dimenmax))/(3l*intensity.dimenmax);//(Initial + vec_cor + half_ls)*dimen_max
	size_t no_final_states_gpu = available_gpu_memory/sizeof(double);

	no_final_states_gpu -=	intensity.nsizemax + intensity.dimenmax*nJ*intensity.molec.sym_maxdegen + CUDA_STREAMS*intensity.dimenmax;

	no_final_states_gpu /= ( intensity.nsizemax + intensity.molec.sym_maxdegen*intensity.molec.sym_maxdegen);
	printf("We can fit %zu states in the GPU memory\n",no_final_states_gpu);
	//no_final_states_gpu /=2;

	//no_final_states_gpu = 10;
	printf("%zu\n",no_final_states_gpu);
	no_final_states_gpu = min((unsigned int )intensity.Neigenlevels,(unsigned int )no_final_states_gpu);

	printf("%d\n",no_final_states_gpu);



	//Create Stream variables/////////l
	cudaStream_t st_ddot_vectors[CUDA_STREAMS];
	cudaEvent_t st_vec_done[CUDA_STREAMS];
	cudaStream_t f_memcpy;
	cudaStream_t f_linestr;
	//cudaEvent_t half_ls_done = new cudaStream_t


	//Half linestrength related variable
	cudaStream_t* st_half_ls = new cudaStream_t[nJ*intensity.molec.sym_maxdegen]; 	//Concurrently run half_ls computations on this many of the half_ls's
	double* gpu_half_ls;


	//Create initial vector holding point
	double* initial_vector = new double[intensity.nsizemax];
	double* gpu_initial_vector;
	
	//Final vectors
	//Streams for each final vector computation

	double* final_vectors;
	cudaMallocHost(&final_vectors,sizeof(double)*intensity.nsizemax*no_final_states_gpu, cudaHostAllocWriteCombined);
	
	//= new double[intensity.dimenmax*no_final_states_gpu]; //Pin this memory in final build
	//int* vec_ilevelF = new int[no_final_states_gpu];

	double* gpu_corr_vectors;
	double* gpu_final_vectors;
	double* gpu_line_str =0;
	//double* check_vector = new double[intensity.dimenmax];
	int** vec_ilevel_buff = new int*[2];
	vec_ilevel_buff[0] = new int[no_final_states_gpu];
	vec_ilevel_buff[1] = new int[no_final_states_gpu];

	double* line_str; //= new double[no_final_states_gpu*intensity.molec.sym_maxdegen*intensity.molec.sym_maxdegen];
        cudaMallocHost(&line_str,sizeof(double)*no_final_states_gpu*intensity.molec.sym_maxdegen*intensity.molec.sym_maxdegen);
	//double* gpu_line_str;
	//Track which vectors we are using
	unsigned long int vector_idx=0;
	int vector_count=0;	
	int ilevel_total=0;
	int ilevelF=0,start_ilevelF=0;

	printf("Finished host side allocation\n");
	
	printf("Copying intial vectors\n");
	//Copy them to the gpu
	copy_array_to_gpu((void*)initial_vector,(void**)&(gpu_initial_vector),sizeof(double)*intensity.nsizemax,"gpu_initial_vector");
	available_gpu_memory -= sizeof(double)*intensity.nsizemax;

	printf("Copying final vectors\n");
	copy_array_to_gpu((void*)final_vectors,(void**)&(gpu_final_vectors),sizeof(double)*intensity.nsizemax*no_final_states_gpu,"gpu_final_vectors");
	available_gpu_memory -= sizeof(double)*intensity.nsizemax*no_final_states_gpu;


	printf("Create correlation vectors\n");
	cudaMalloc((void**)&(gpu_corr_vectors),sizeof(double)*intensity.dimenmax*CUDA_STREAMS);
	CheckCudaError("Init correlation");
	available_gpu_memory -= sizeof(double)*intensity.dimenmax*CUDA_STREAMS;


	printf("Create Half ls vector\n");
	cudaMalloc((void**)&(gpu_half_ls),sizeof(double)*intensity.dimenmax*nJ*intensity.molec.sym_maxdegen);
	available_gpu_memory -= sizeof(double)*intensity.dimenmax*nJ*intensity.molec.sym_maxdegen;
	CheckCudaError("Init half ls");
	printf("Create line_str\n");
	copy_array_to_gpu((void*)line_str,(void**)&(gpu_line_str),sizeof(double)*no_final_states_gpu*intensity.molec.sym_maxdegen*intensity.molec.sym_maxdegen,"gpu_half_ls");
	//cudaMalloc((void**)&(gpu_line_str),sizeof(double)*no_final_states_gpu*intensity.molec.sym_maxdegen*intensity.molec.sym_maxdegen);
	available_gpu_memory -= sizeof(double)*no_final_states_gpu*intensity.molec.sym_maxdegen*intensity.molec.sym_maxdegen;
	CheckCudaError("Init line_str");

	//copy_array_to_gpu((void*)line_str,(void**)&(gpu_line_str),sizeof(double)*intensity.dimenmax*nJ*intensity.molec.sym_maxdegen,"gpu_line_str");

	//A HACK to host the dipole in CPU memory, will slow stuff down considerably
	/*if(intensity.host_dipole){
		printf("Device pointer fun!!!");
		if(cudaSuccess != cudaHostGetDevicePointer((void **)&g_ptrs.dipole_me, (void *)intensity.dipole_me, 0)){
			printf("Device pointer is not fun :(!!");
		}
		printf("\n\n GPU-> Host pointer: %p\n",g_ptrs.dipole_me);
	}*/
	printf("Finished gpu copying\n");
	//
	//Open the eigenvector units
	char filename[1024];

	//Get the filename1552 bytes stack frame, 24 bytes spill stores, 24 bytes spill loads
	printf("Open vector units\n");

	int* eigenvec_unit = new int[nJ*intensity.molec.sym_nrepres];

	for(int i =0; i< nJ; i++){
		for(int j = 0; j < intensity.molec.sym_nrepres; j++)
		{
			if(intensity.isym_do[j] == false) continue;
			sprintf(filename,j0eigen_vector_gamma_filebase,intensity.jvals[i],j+1);
			printf("Reading %s\n",filename);
			eigenvec_unit[i + j*nJ] = open(filename,O_RDONLY);
			if(eigenvec_unit[i + j*nJ] == -1)
			{
				printf("error opening %s \n",filename);
				exit(0);
			}
		}
	}
	
	//Initialise cublas
	cublasHandle_t handle;
	cublasStatus_t stat;
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf ("CUBLAS initialization failed\n");
		return;
	}	
	
	
	//Create the streams
	//Intial state
	for(int i = 0; i < intensity.molec.sym_maxdegen; i++)
		for(int j=0; j < nJ; j++)
			cudaStreamCreate(&st_half_ls[j + i*nJ]);

	//Final states
	cudaStreamCreate(&f_memcpy);
	for(int i = 0; i < CUDA_STREAMS; i++){
		cudaStreamCreate(&st_ddot_vectors[i]);
		cudaEventCreate(&st_vec_done[i],cudaEventDisableTiming);
	}

	int last_ilevelF= 0;
	//////Begin the computation//////////////////////////////////////////////////////////////////////////////
	CheckCudaError("Initialisation");
	printf("NUM: %i",num_devices);
	//If zero then itll progress normally otherwise with 4 devices it will go like this
	//Thread 0 = 0 4 8 12
	//Thread 1 = 1 5 9 13
	//Thread 2 = 2 6 10 14
	//Thread 3 = 3 7 11 15
	//Run


	double total_memory = 0.0;
	double no_transitions=0.0;
	int no_initial_states=0;
	//#pragma omp barrier
	MPI_Barrier(MPI_COMM_WORLD);
	if(process_id==0){
		no_initial_states = 0;
		printf("\n\nPredicting how many transitions to compute.....");
		for(int ilevelI=0; ilevelI < intensity.Neigenlevels; ilevelI++){
			int jI = intensity.eigen[ilevelI].jval;
			double energyI = intensity.eigen[ilevelI].energy;
			int igammaI  = intensity.eigen[ilevelI].igamma;
			int * quantaI = intensity.eigen[ilevelI].quanta;
		//	int * normalI = intensity.eigen[ilevelI].normal;

			if(!energy_filter_lower(intensity,jI,energyI,quantaI)) continue;

			for(int ilevelF=0; ilevelF < intensity.Neigenlevels; ilevelF++){
		      		int jF = intensity.eigen[ilevelF].jval;
		      		double energyF = intensity.eigen[ilevelF].energy;
		     	 	int igammaF  = intensity.eigen[ilevelF].igamma;
		     	 	int * quantaF = intensity.eigen[ilevelF].quanta;
		    	  	//int * normalF = intensity.eigen[ilevelF].normal;
				if(!energy_filter_upper(intensity,jF,energyF,quantaF)) continue;

				if(!intensity_filter(intensity,jI,jF,energyI,energyF,igammaI,igammaF,quantaI,quantaF)) continue;

				no_transitions++;

			}
			no_initial_states++;


		}
		printf(".....We have counted %i transitions with %i initial states\n\n",int(no_transitions),no_initial_states);
		printf("Linestrength S(f<-i) [Debye**2], Transition moments [Debye],Einstein coefficient A(if) [1/s],and Intensities [cm/mol]\n\n\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
	//#pragma omp barrier
	//constants
	double beta = planck * vellgt / (boltz * intensity.temperature);
	double boltz_fc=0.0;
	double absorption_int = 0.0;
	cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE);


	for(int ilevelI = process_id; ilevelI < intensity.Neigenlevels; ilevelI+=num_devices){
		//printf("new I level!\n");
		//Get the basic infor we need
	    //  printf("ilevelI = %i\n",ilevelI);
	      int indI = intensity.eigen[ilevelI].jind;
		
	      int dimenI = intensity.bset_contr[indI+1].Maxcontracts;

	      int jI = intensity.eigen[ilevelI].jval;
	      double energyI = intensity.eigen[ilevelI].energy;
	      int igammaI  = intensity.eigen[ilevelI].igamma;
	      int * quantaI = intensity.eigen[ilevelI].quanta;
	      int * normalI = intensity.eigen[ilevelI].normal;
	      unsigned long int ndegI   = intensity.eigen[ilevelI].ndeg;
	      unsigned long int irecI = intensity.eigen[ilevelI].irec[0]-1;
	    // printf("ilevelI=%i jI=%i energyI=%11.4f igammaI=%i ndegI=%i\n",ilevelI,jI,energyI,igammaI,ndegI);
	      unsigned long int nsizeI = intensity.bset_contr[indI+1].nsize[igammaI];

	      int unitI = eigenvec_unit[ indI + (igammaI)*nJ]; 
		//Check filters
		
	      if(!energy_filter_lower(intensity,jI,energyI,quantaI)) continue;
		//If success then read
	      lseek(unitI,irecI*nsizeI*sizeof(double),SEEK_SET);
	      read(unitI,initial_vector,sizeof(double)*nsizeI);

	      stat = cublasSetVector(nsizeI, sizeof(double),initial_vector, 1, gpu_initial_vector, 1);
               int idegF_t = 0;
	       int igammaF_t = intensity.igamma_pair[igammaI];
	      CheckCudaError("Set Vector I");


		current_time = GetTimeMs64();
   		//Do first half ls
	for(int indF =0; indF < nJ; indF++){
			int jF= intensity.jvals[indF];
		if(!indF_filter(intensity,jI,jF,energyI,igammaI,quantaI))continue;
	      for(int ideg=0; ideg < ndegI; ideg++){
			if(!degeneracy_filter(intensity,igammaI,igammaF_t,ideg,idegF_t)) continue;
			if(intensity.split_dipole == false){
			//	do_1st_half_ls(g_ptrs.bset_contr[indI],g_ptrs.bset_contr[indF],
				//		intensity.dimenmax,ideg,igammaI,g_ptrs.dipole_me,gpu_initial_vector,gpu_corr_vectors + intensity.dimenmax*ideg,
				//						g_ptrs.threej,
				//							gpu_half_ls + indF*intensity.dimenmax + ideg*intensity.dimenmax*nJ
					//							,st_half_ls[indF]);
				do_1st_half_ls_shared(g_ptrs.bset_contr[indI],g_ptrs.bset_contr[indF],jF,intensity.bset_contr[0].Maxcontracts,intensity.dimenmax,ideg,igammaI,g_ptrs.dipole_me,gpu_initial_vector,gpu_corr_vectors + intensity.dimenmax*ideg,
										g_ptrs.threej,
											gpu_half_ls + indF*intensity.dimenmax + ideg*intensity.dimenmax*nJ
												,st_half_ls[indF]);
//
				}else{
					//do_1st_half_ls_blocks(g_ptrs.bset_contr[indI],g_ptrs.bset_contr[indF],
					//	intensity.dimenmax,ideg,igammaI,g_ptrs.dipole_me,intensity.dipole_blocks,gpu_initial_vector,gpu_corr_vectors + intensity.dimenmax*ideg,
					//					g_ptrs.threej,
					//						gpu_half_ls + indF*intensity.dimenmax + ideg*intensity.dimenmax*nJ
					//							,st_half_ls[0]);
					do_1st_half_ls_shared_blocks(g_ptrs.bset_contr[indI],g_ptrs.bset_contr[indF],jF,intensity.bset_contr[0].Maxcontracts,
						intensity.dimenmax,ideg,igammaI,g_ptrs.dipole_me,intensity.dipole_blocks,gpu_initial_vector,gpu_corr_vectors + intensity.dimenmax*ideg,
										g_ptrs.threej,
											gpu_half_ls + indF*intensity.dimenmax + ideg*intensity.dimenmax*nJ
												,st_half_ls[0]);
				}
				half_ls_count++;
			
			}

			 //wait for the next batch
			
	      }


		vector_idx=0;	
		ilevelF=0;
		int current_buff = 0;
		cur_io_time = GetTimeMs64();
		//While the half_ls is being computed, lets load up some final state vectors
		while(vector_idx < no_final_states_gpu && ilevelF < intensity.Neigenlevels)
		{
					   //   !
		      	int indF = intensity.eigen[ilevelF].jind;
		  //    !
			//printf("indF=%i",indF);
		  //    !dimension of the bases for the initial states
		  //    !
		   //   !
		      //!energy, quanta, and gedeneracy order of the initial state
		     // !
		      	int jF = intensity.eigen[ilevelF].jval;
		      	double energyF = intensity.eigen[ilevelF].energy;
		      	int igammaF  = intensity.eigen[ilevelF].igamma;
		      	int * quantaF = intensity.eigen[ilevelF].quanta;
		      	int * normalF = intensity.eigen[ilevelF].normal;
		     	unsigned long int nsizeF = intensity.bset_contr[indF+1].nsize[igammaF];
			unsigned long int irec = intensity.eigen[ilevelF].irec[0]-1;
			int unitF = eigenvec_unit[ indF + (igammaF)*nJ]; 			

			ilevelF++;
			if(!energy_filter_upper(intensity,jF,energyF,quantaF)) {continue;}
			if(!intensity_filter(intensity,jI,jF,energyI,energyF,igammaI,igammaF,quantaI,quantaF)) continue;
 			// store the level
			vec_ilevel_buff[0][vector_idx] = ilevelF-1;
			//printf("ilevelF=%i\n",vec_ilevel_buff[0][vector_idx]);
			//Otherwise load the vector to a free slot
			lseek(unitF,irec*nsizeF*sizeof(double),SEEK_SET);
			read(unitF,final_vectors + vector_idx*((unsigned long int)(intensity.nsizemax)),sizeof(double)*nsizeF);
			total_memory += sizeof(double)*nsizeF;
			//Increment
			vector_idx++;
		}
		cur_io_time = GetTimeMs64()-cur_io_time;
		io_calls++;
		io_total_time += cur_io_time;
		vector_count = vector_idx;
		
	
		//printf("memcopy");
		//Memcopy it in one go
		//cudaDeviceSynchronize();
		cudaMemcpyAsync(gpu_final_vectors,final_vectors,sizeof(double)*size_t(intensity.nsizemax)*size_t(vector_count),cudaMemcpyHostToDevice,f_memcpy) 	;


		cudaDeviceSynchronize(); //Wait till we're set up


		current_time = GetTimeMs64() - current_time;
		half_ls_total_time += current_time - cur_io_time;

		CheckCudaError("Batch final vectors");	
		//printf("vector_count = %i\n",vector_count);

		while(vector_count != 0)
		{
			current_time = GetTimeMs64();
			int stream_count = 0;
			for(int i = 0; i < vector_count; i++){
				ilevelF = vec_ilevel_buff[int(current_buff)][i];
				//printf("ilevelF=%i\n",ilevelF);
				int indF = intensity.eigen[ilevelF].jind;
			      	int jF = intensity.eigen[ilevelF].jval;
			      	double energyF = intensity.eigen[ilevelF].energy;
			      	int igammaF  = intensity.eigen[ilevelF].igamma;
			      	int * quantaF = intensity.eigen[ilevelF].quanta;
			      	int * normalF = intensity.eigen[ilevelF].normal;
			     	unsigned long int nsizeF = intensity.bset_contr[indF+1].nsize[igammaF];
				//int irec = intensity.eigen[ilevelF].irec[0]-1;
				int dimenF = intensity.bset_contr[indF+1].Maxcontracts;
				int ndegF   = intensity.eigen[ilevelF].ndeg;
				int blockSize =512;
				int gridSize = (int)ceil((float)intensity.dimenmax/blockSize);
				//for(int i = 0; i < ndeg
				//Correlate the vectors
				for(int idegF = 0; idegF < ndegF; idegF++){
					//gridSize = (int)ceil((float)dimenF/blockSize);
					for(int idegI=0; idegI < ndegI; idegI++)
						line_str[i + idegI*no_final_states_gpu + idegF*no_final_states_gpu*intensity.molec.sym_maxdegen] = 0.0;

					if(intensity.reduced && idegF!=0) continue;
					for(int idegI=0; idegI < ndegI; idegI++){
						//line_str[i + idegI*no_final_states_gpu + idegF*no_final_states_gpu*intensity.molec.sym_maxdegen]=0.0;
						//line_str[i + idegI*no_final_states_gpu + idegF*no_final_states_gpu*intensity.molec.sym_maxdegen] = 0.0;
						if(!degeneracy_filter(intensity, igammaI,igammaF,idegI,idegF)) continue;
						device_correlate_vectors<<<gridSize,blockSize,0,st_ddot_vectors[stream_count]>>>(g_ptrs.bset_contr[indF],idegF,igammaF, (gpu_final_vectors + i*intensity.nsizemax),gpu_corr_vectors + intensity.dimenmax*stream_count);
						CheckCudaError("Correlate final vector");
						cublasSetStream(handle,st_ddot_vectors[stream_count]);
						//printf("%i\n",i + idegI*no_final_states_gpu + idegF*no_final_states_gpu*intensity.molec.sym_maxdegen);
						cublasDdot (handle, dimenF,gpu_corr_vectors + intensity.dimenmax*stream_count, 1, gpu_half_ls + indF*intensity.dimenmax + idegI*intensity.dimenmax*nJ, 1, 

														gpu_line_str + i + idegI*no_final_states_gpu + idegF*no_final_states_gpu*intensity.molec.sym_maxdegen);
						ddot_count++;
						CheckCudaError("ddot final vector");


					}
				}
				stream_count++;
				if(stream_count >=CUDA_STREAMS) stream_count=0;

			
			}


			//Record the events for synchronization
			for(int i = 0; i < CUDA_STREAMS; i++){
				cudaEventRecord(st_vec_done[i],st_ddot_vectors[i]);
				cudaStreamWaitEvent ( f_memcpy,st_vec_done[i],0); //Make this stream wait for the event
				cudaStreamWaitEvent ( f_linestr,st_vec_done[i],0);
			}
		

			current_buff = 1-current_buff;
			vector_idx = 0;
			ilevelF++;
			cur_io_time = GetTimeMs64();
			//While the line_Strength is being computed, lets load up some final state vectors
			while(vector_idx < no_final_states_gpu && ilevelF < intensity.Neigenlevels)
			{
						   //   !
			      	int indF = intensity.eigen[ilevelF].jind;
			  //    !
				//printf("indF=%i",indF);
			  //    !dimension of the bases for the initial states
			  //    !
			   //   !
			      //!energy, quanta, and gedeneracy order of the initial state
			     // !
			      	int jF = intensity.eigen[ilevelF].jval;
			      	double energyF = intensity.eigen[ilevelF].energy;
			      	int igammaF  = intensity.eigen[ilevelF].igamma;
			      	int * quantaF = intensity.eigen[ilevelF].quanta;
			      	int * normalF = intensity.eigen[ilevelF].normal;
			     	unsigned long int nsizeF = intensity.bset_contr[indF+1].nsize[igammaF];
				unsigned long int irec = intensity.eigen[ilevelF].irec[0]-1;
				
				int unitF = eigenvec_unit[ indF + (igammaF)*nJ]; 			

				ilevelF++;
				if(!energy_filter_upper(intensity,jF,energyF,quantaF)) {continue;}
				if(!intensity_filter(intensity,jI,jF,energyI,energyF,igammaI,igammaF,quantaI,quantaF)) continue;
				 // store the level				
				vec_ilevel_buff[current_buff][vector_idx] = ilevelF-1;
				//load the vector to a free slot
				lseek(unitF,irec*nsizeF*sizeof(double),SEEK_SET);
				read(unitF,final_vectors + vector_idx*((unsigned long int)(intensity.nsizemax)),sizeof(double)*nsizeF);
				//cudaMemcpyAsync(gpu_final_vectors,final_vectors,sizeof(double)*intensity.dimenmax*vector_count,cudaMemcpyHostToDevice,st_ddot_vectors[vector_idx]) ;
				//Increment
				total_memory += sizeof(double)*nsizeF;
				vector_idx++;
			}
			cur_io_time = GetTimeMs64()-cur_io_time;
			io_total_time += cur_io_time;
			io_calls++;
			last_ilevelF=ilevelF;

			
			cudaMemcpyAsync(gpu_final_vectors,final_vectors,sizeof(double)*size_t(intensity.nsizemax)*size_t(vector_count),cudaMemcpyHostToDevice,f_memcpy) ;
			cudaMemcpyAsync(line_str,gpu_line_str,sizeof(double)*size_t(no_final_states_gpu*intensity.molec.sym_maxdegen*intensity.molec.sym_maxdegen),cudaMemcpyDeviceToHost,f_linestr) ;
			CheckCudaError("Copy final vector");
			//We'e done now lets output
			
			cudaStreamSynchronize(f_linestr);	//wait for all events to be completed
			for(int ivec = 0; ivec < vector_count; ivec++)
			{
				ilevelF = vec_ilevel_buff[1-current_buff][ivec];
				//printf("ilevelF=%i\n",ilevelF);
				int indF = intensity.eigen[ilevelF].jind;
			      	int jF = intensity.eigen[ilevelF].jval;
			      	double energyF = intensity.eigen[ilevelF].energy;
			      	int igammaF  = intensity.eigen[ilevelF].igamma;
			      	int * quantaF = intensity.eigen[ilevelF].quanta;
			      	int * normalF = intensity.eigen[ilevelF].normal;
				int ndegF   = intensity.eigen[ilevelF].ndeg;
				//cudaStreamSynchronize(st_ddot_vectors[ivec]);
				double ls=0.0;
				double linestr=0.0;
			        for(int idegF=0; idegF < ndegF; idegF++){
				      for(int idegI=0; idegI < ndegI; idegI++){
						if(!degeneracy_filter(intensity, igammaI,igammaF,idegI,idegF)) continue;
						linestr=line_str[ivec + idegI*no_final_states_gpu + idegF*no_final_states_gpu*intensity.molec.sym_maxdegen];
						ls +=(linestr*linestr);		//line_str[ivec + idegI*no_final_states_gpu + idegF*no_final_states_gpu*intensity.molec.sym_maxdegen]*line_str[ivec + idegI*no_final_states_gpu + idegF*no_final_states_gpu*intensity.molec.sym_maxdegen];
						
					}
				}
				//printf("ls = %16.9e\n");
				//printf("ndegI=%i\n",ndegI);
				ls /= double(ndegI);
				if (intensity.reduced && ndegF!=1 && ndegI != 1) ls  *= double(ndegI);
				double final_ls = ls;
				double nu_if = energyF - energyI; 
             			boltz_fc = abs(nu_if) * exp(-(energyI-intensity.ZPE) * beta) * (1.0 - exp(-abs(nu_if) * beta))/ intensity.q_stat;
				//Print intensitys
				//printf("line_str %11.4e\n",line_str);
				double A_einst = A_coef_s_1*double((2*jI)+1)*final_ls*pow(abs(nu_if),3);

				final_ls = final_ls * intensity.gns[igammaI] * double( (2*jI + 1)*(2 * jF + 1) );
				absorption_int = final_ls * intens_cm_mol * boltz_fc;


				
				if(final_ls < intensity.thresh_linestrength || absorption_int <  intensity.thresh_linestrength ) continue;
	/*
				printf("%11.4f\t(%i %i ) ( ",nu_if,jF,intensity.eigen[ilevelF].krot);

				for(int i = 0; i < intensity.molec.nmodes+1; i++)
					printf("%i ",quantaF[i]);

				printf(")\t <-- \t(%i %i ) ",jI,intensity.eigen[ilevelI].krot);

				for(int i = 0; i < intensity.molec.nmodes+1; i++)
					printf("%i ",quantaI[i]);	

				printf("\t %16.8e    %16.8e %16.8e\n",A_einst,final_ls,orig_ls);	
*/

			/*	               write(out, "( (i4, 1x, a4, 3x),'<-', (i4, 1x, a4, 3x),a1,&
                            &(2x, f11.4,1x),'<-',(1x, f11.4,1x),f11.4,2x,&
                            &'(',1x,a3,x,i3,1x,')',1x,'(',1x,<nclasses>(x,a3),1x,<nmodes>(1x, i3),1x,')',1x,'<- ',   &
                            &'(',1x,a3,x,i3,1x,')',1x,'(',1x,<nclasses>(x,a3),1x,<nmodes>(1x, i3),1x,')',1x,   &
                            & 3(1x, es16.8),2x,(1x,i6,1x),'<-',(1x,i6,1x),i8,1x,i8,&
                            1x,'(',1x,<nmodes>(1x, i3),1x,')',1x,'<- ',1x,'(',1x,<nmodes>(1x, i3),1x,')',1x,& 
                            <nformat>(1x, es16.8))")  &
                            !
                            jF,sym%label(igammaF),jI,sym%label(igammaI),branch, &
                            energyF-intensity%ZPE,energyI-intensity%ZPE,nu_if,                 &
                            eigen(ilevelF)%cgamma(0),eigen(ilevelF)%krot,&
                            eigen(ilevelF)%cgamma(1:nclasses),eigen(ilevelF)%quanta(1:nmodes), &
                            eigen(ilevelI)%cgamma(0),eigen(ilevelI)%krot,&
                            eigen(ilevelI)%cgamma(1:nclasses),eigen(ilevelI)%quanta(1:nmodes), &
                            linestr,A_einst,absorption_int,&
                            eigen(ilevelF)%ilevel,eigen(ilevelI)%ilevel,&
                            itransit,istored(ilevelF),normalF(1:nmodes),normalI(1:nmodes),&
                            linestr_deg(1:ndegI,1:ndegF)
			*/	

			  
				//itransit++;
			   sprintf(buffer,"%4i %4s   <-%4i %4s   %1s  %11.4f <- %11.4f %11.4f  ( %3s %3i ) ( ",jF,intensity.molec.c_sym[igammaF],jI,intensity.molec.c_sym[igammaI],branch(jF,jI),energyF-intensity.ZPE,energyI-intensity.ZPE,abs(nu_if),intensity.eigen[ilevelF].cgamma[0],intensity.eigen[ilevelF].krot);
			   for(int i = 1; i <= intensity.molec.nclasses; i++)
					sprintf(buffer + strlen(buffer)," %3s",intensity.eigen[ilevelF].cgamma[i]);
			  sprintf(buffer + strlen(buffer)," ");
			   for(int i = 1; i <= intensity.molec.nmodes; i++)
					sprintf(buffer + strlen(buffer)," %3i",quantaF[i]);
			   sprintf(buffer + strlen(buffer)," ) <- ( %3s %3i ) ( ",intensity.eigen[ilevelI].cgamma[0],intensity.eigen[ilevelI].krot);

			   for(int i = 1; i <= intensity.molec.nclasses; i++)
					sprintf(buffer + strlen(buffer)," %3s",intensity.eigen[ilevelI].cgamma[i]);
			   sprintf(buffer + strlen(buffer)," ");
			   for(int i = 1; i <= intensity.molec.nmodes; i++)
					sprintf(buffer + strlen(buffer)," %3i",quantaI[i]);
			   sprintf(buffer + strlen(buffer),")  %16.8e %16.8e %16.8e   %6i <- %6i %8i %8i ( ",final_ls,A_einst,absorption_int,intensity.eigen[ilevelF].ilevel+1,intensity.eigen[ilevelI].ilevel+1,0,0);
			   
			   for(int i = 1; i <= intensity.molec.nmodes; i++)
					sprintf(buffer + strlen(buffer)," %3i",normalF[i]);
			  sprintf(buffer + strlen(buffer)," ) <-  ( ");

			   for(int i = 1; i <= intensity.molec.nmodes; i++)
					sprintf(buffer + strlen(buffer)," %3i",normalI[i]);
			  sprintf(buffer + strlen(buffer)," ) ");
			   //printf(" )  %16.9e\n",1.23456789);	
			   for(int idegF=0; idegF < ndegF; idegF++){
				 for(int idegI=0; idegI < ndegI; idegI++){
						sprintf(buffer + strlen(buffer)," %16.9e",line_str[ivec + idegI*no_final_states_gpu + idegF*no_final_states_gpu*intensity.molec.sym_maxdegen]);
						
					}
				}		
			    sprintf(buffer+ strlen(buffer),"\n");
			    fprintf(output_intensity,buffer);
			   // std::cout<<buffer;


			}			
			//return; 
			ilevelF=last_ilevelF+1;
			//Save the new vector_count
			vector_count = vector_idx;
			
			cudaDeviceSynchronize();
			CheckCudaError("Compute ls");
			current_time = GetTimeMs64()-current_time;
			ddot_total_time+= current_time - cur_io_time;
		}

          if(process_id == 0){
          	printf("--------------Process 0: Timing Information---------------------\n");
          	printf("Half_ls (Sans IO): \tCalls: %d\tTotal_time:%.fs\tTime per call:%12.6fs\n",half_ls_count,half_ls_total_time/1000.0,half_ls_total_time/(1000.0*double(half_ls_count)));
          	printf("Final State (Sans IO): \tCalls: %d\tTotal_time:%.fs\tTime per call:%12.6fs\n",ddot_count,ddot_total_time/1000.0,ddot_total_time/(1000.0*double(ddot_count)));
		printf("IO: \tRequests: %d\tTotal_time:%.fs\tTime per call:%12.6fs  Bandwidth: %12.6f GB/s\n",io_calls,io_total_time/1000.0,io_total_time/(1000.0*double(io_calls)), total_memory/(io_total_time*1073741824.0/(1000.0)));
		printf("Completion time without IO: %12.6f hours\n",(half_ls_total_time+ddot_total_time)*no_transitions/(1000.0*3600.0*double(ddot_count)) );
		printf("For a single GPU it would take %f hours to do %i transitions\n",(half_ls_total_time+ddot_total_time+io_total_time)*no_transitions/(1000.0*3600.0*double(ddot_count)),int(no_transitions) );
		printf("For a %i GPUs it would take %f hours to do %i transitions\n",num_devices,(half_ls_total_time+ddot_total_time+io_total_time)*no_transitions/(1000.0*double(num_devices)*3600.0*double(ddot_count)),int(no_transitions));
          }	
		
	

	}
	fclose(output_intensity);
//	printf("Thread =%i done",device_id);
	for(int i=0; i< nJ; i++){
		for(int j = 0; j < intensity.molec.sym_nrepres; j++)
		{
			if(!intensity.isym_do[j]) continue;
			if(eigenvec_unit[i + j*nJ]!=-1)
				close(eigenvec_unit[i + j*nJ]);

		}
	}

	cudaDeviceReset();
	cudaFreeHost(&final_vectors);
	cudaFreeHost(&line_str);

	
	unsigned long int g_half_ls_count = 0;
	unsigned long int g_ddot_count = 0;
	double g_half_ls_total_time = 0.0;
	double g_ddot_total_time = 0.0;
	
	MPI_Reduce(&half_ls_count, &g_half_ls_count, 1,
            MPI_UNSIGNED_LONG, MPI_SUM, 0,
           MPI_COMM_WORLD);
           
	MPI_Reduce(&ddot_count, &g_ddot_count, 1,
            MPI_UNSIGNED_LONG, MPI_SUM, 0,
           MPI_COMM_WORLD);
        MPI_Reduce(&half_ls_total_time, &g_half_ls_total_time, 1,
            MPI_DOUBLE, MPI_SUM, 0,
           MPI_COMM_WORLD);  	
           
        MPI_Reduce(&ddot_total_time, &g_ddot_total_time, 1,
            MPI_DOUBLE, MPI_SUM, 0,
           MPI_COMM_WORLD); 

          if(process_id == 0){
          	printf("--------------Timing Information---------------------\n");
          	printf("Half_ls: \tCalls: %d\tTotal_time:%.fs\tTime per call:%12.6fs\n",g_half_ls_count,g_half_ls_total_time/1000.0,g_half_ls_total_time/(1000.0*double(g_half_ls_count)));
          	printf("Final State: \tCalls: %d\tTotal_time:%.fs\tTime per call:%12.6fs\n",g_ddot_count,g_ddot_total_time/1000.0,g_ddot_total_time/(1000.0*double(g_ddot_count)));
          }	

}

