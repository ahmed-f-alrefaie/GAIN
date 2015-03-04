#include "common.h"
#include "cuda_objects.cuh"
#include "cuda_host.cuh"
#include "dipole_kernals.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "fields.h"
#include "Util.h"
#include "trove_functions.h"
#include "test.cuh"
#include <cstring>
#include <mpi.h>






/* Returns the amount of milliseconds elapsed since the UNIX epoch. Works on both
 * windows and linux. */

FintensityJob test_intensity;



int main(int argc,char** argv)
{
	////////----------THIS IS TESTED ON AN EMPTY 8 GPU NODE ON EMERALD
	bool test = false;

	//Initialise mpi
	MPI_Init(&argc,&argv);

	if (argc>2){
		if (strcmp(argv[1],"-test")==0){
			test = true;
		}	
	}


	//printf("%12.6f\n",three_j(2,3,1,1,-1,0));
	
	//float test = 3.6;
	//int i = 2;
	//test = 3.6*i;
	//printf("%11.2f\n",test); 
	//get_cuda_info(test_intensity);
	//exit(0);
	int process_id;
	//
	//bool test=false;
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
	//setvbuf( stdout, NULL, _IONBF, 0 );


	if(!test){
		char* gpu_env = getenv("NUM_GPUS");
		int num_gpu = 1;	
	
		if(gpu_env!=NULL){
			num_gpu = atoi(gpu_env);
		}

		//int free_gpus = count_free_devices();
		//printf("Found %i free devices\n",free_gpus);
		//exit(0);

		//num_gpu;
		MPI_Comm_size(MPI_COMM_WORLD, &num_gpu);
		printf("Running %i processes \n",num_gpu);
		//dipol_do_intensities(test_intensity);
		//dipole_do_intensities_async(test_intensity,0);
		//Set number of threads
		//for(int i = 0; i < argc; i++)
		//	printf("%s\n",argv[i]);
		if(argc==3) {strcpy(test_intensity.filename,argv[1]); strcpy(test_intensity.output_file,argv[2]);}else{ MPI_Finalize(); exit(0);}
		dipole_initialise_cpu(&test_intensity);
		//exit(0);
		//num_gpu = 1;
		//omp_set_dynamic(0);
		//omp_set_num_threads(num_gpu);
		//Parallel region here
	
		double time = GetTimeMs64();
		
		int last_gpu = -1;

		int device=0; 

		//last_gpu = get_free_device(last_gpu);
		//if(last_gpu==-1){
		//	printf("No free gpu's for thread\n");
		//	exit(0);
		//}
		//else{
		//	device = last_gpu;
		//	printf("Thread %i using gpu number %i\n",process_id,last_gpu);
		//}
				
		dipole_do_intensities_async_omp(test_intensity,device,num_gpu);
				
		
		MPI_Barrier(MPI_COMM_WORLD);
		if(process_id==0){
			time = GetTimeMs64() - time;
			printf("\ndone\n");
			printf("\ndone in %.fs\n",time/1000.0);
		}
		MPI_Barrier(MPI_COMM_WORLD);



	}else{
	//dipole_initialise(&test_intensity);
	if(argc==4) {strcpy(test_intensity.filename,argv[3]);}else{ MPI_Finalize(); exit(0);}
	//dipole_initialise_cpu(&test_intensity);
	benchmark_half_ls(test_intensity,10);
	//dipole_do_intensities(test_intensity);
	}

	
	MPI_Finalize();
	//exit(0);

	return 0;

}
