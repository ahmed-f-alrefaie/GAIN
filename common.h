
#include <sys/time.h>
#include <ctime>
#pragma once
typedef unsigned int uint;

//extern int sym_nrepres;
extern int G_VERBOSE;
//extern int sym_degen[4];

extern const char* extFmat_file;
extern const char* j0eigen_filebase;
extern const char* j0eigen_quanta_filebase;
extern const char* j0eigen_vector_gamma_filebase;
extern const char* j0eigen_descr_gamma_filebase;

/*
void verbose_printf(int verbose,const char* fmt, ...){
	//
    if(verbose <= g_verbose){
	
	    va_list args;
	    va_start(args,fmt);
	    vprintf(fmt,args);
	    va_end(args);
	}

}*/
