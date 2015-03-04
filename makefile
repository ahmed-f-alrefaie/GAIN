goal:   main_mpi_1711.x

tarball:
	tar cf main.tar makefile *.cpp
        
checkin:
	ci -l Makefile *.cpp

############################### pathscale ##################################  -ipo -cm -p -g test: -CB  -CA -CS -CV-ipo $(LAPACK)  
##PLAT =
##NVCC = nvcc
##FOR = mpicc
##NVCCFLAGS := --ptxas-options=-v -O3 -gencode arch=compute_20,code=sm_20
##FFLAGS = -O3 -cxxlib -xHost

NVCC = nvcc
FOR = mpicc
ifeq ($(PLAT),KEPLER)
NVCCFLAGS := --ptxas-options=-v -O3 -arch=sm_35 -Xptxas -v -lineinfo -DKEPLER
FFLAGS = -O3 -xHost -openmp -DKEPLER
else
NVCCFLAGS := --ptxas-options=-v -O3 -arch=sm_21 -Xptxas -v -lineinfo
FFLAGS = -O3 -cxxlib -xHost
endif

#-O3 -ipo -xHost -g -traceback
LIBS= -L$(CUDA_HOME)/lib64 -lcudart -lcuda -lcublas
INC = -I$(CUDA_HOME)/include 


###############################################################################

OBJ = trove_functions.o Util.o cuda_host.o dipole_kernals.o fields.o test.o fortfunc.o
#input.o

main_mpi_1711.x:    main.o  $(OBJ) 
	$(FOR) -o main_mpi_1711_$(PLAT).x $(OBJ) $(FFLAGS) main.o $(LIBS) -lifcore -limf

main.o:       main.cu $(OBJ) 
	$(NVCC) -c main.cu $(NVCCFLAGS) $(INC) -I/opt/platform_mpi/include

trove_functions.o: trove_functions.cpp Util.o fortfunc.o
	$(FOR) -c trove_functions.cpp $(FFLAGS)

Util.o:  Util.cpp
	$(FOR) -c Util.cpp $(FFLAGS)

fields.o:  fields.cpp
	$(FOR) -c fields.cpp $(FFLAGS)


dipole_kernals.o:  dipole_kernals.cu
	$(NVCC) -c dipole_kernals.cu $(NVCCFLAGS) 

cuda_host.o:  cuda_host.cu
	$(NVCC) -c cuda_host.cu $(NVCCFLAGS) -I/opt/platform_mpi/include
test.o:  test.cu Util.o trove_functions.o
	$(NVCC) -c test.cu $(NVCCFLAGS)
fortfunc.o: fortfunc.f90 
	ifort -c fortfunc.f90 -O3 -xHost 
#input.o:  input.f90
#       $(FOR) -c input.f90 $(FFLAGS)


clean:
	rm $(OBJ) *.o main.o




