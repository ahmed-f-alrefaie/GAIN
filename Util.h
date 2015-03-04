#include <cstdio>
#include <string>
#include <sys/time.h>
#include <ctime>
#pragma once

typedef long int int64;
typedef unsigned long int uint64;

void destroy_arr_valid(void** ptr);
inline unsigned int Fortran2D_to_1D(int i, int j,int isize){ return i+ j*isize;};
inline unsigned int Fortran3D_to_1D(int i, int j,int k,int isize,int jsize){ return i+ j*isize + k*isize*jsize ;};
size_t GetFilenameSize(std::string name);
bool fexists(const char *filename);

// trim from start
std::string &ltrim(std::string &s);

// trim from end
std::string &rtrim(std::string &s);

// trim from both ends
std::string &trim(std::string &s);

void ReadFortranRecord(FILE* IO, void* data_ptr);

void assertdouble(double & d1, double & d2, double tol);

int64 GetTimeMs64();
