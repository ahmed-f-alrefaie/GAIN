#include <mpi.h>
#include "fields.h"
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cctype>
#include <iostream>
using namespace std; 

void read_symmetry(const char* symchar, FintensityJob* intensity){
	//Reading symmetry
	printf("Reading symmetry\n");
	if(strcmp("C2V(M)",symchar)==0)
	{
		//printf("C2v(M) sym");
		intensity->molec.sym_type = C2VM_ID;
		intensity->molec.sym_nrepres = 4;
		intensity->molec.sym_degen = new int[4];
		intensity->molec.sym_maxdegen = 1;
		intensity->isym_do = new bool[4];
		intensity->molec.sym_degen[0] = 1;
		intensity->molec.sym_degen[1] = 1;
		intensity->molec.sym_degen[2] = 1;
		intensity->molec.sym_degen[3] = 1;

		intensity->isym_do[0] = true;
		intensity->isym_do[1] = true;
		intensity->isym_do[2] = true;
		intensity->isym_do[3] = true;
		intensity->gns = new double[4];
		intensity->isym_pairs = new int[4];
		intensity->igamma_pair = new int[4];

		intensity->molec.c_sym = new char*[4];
		for(int i = 0; i < 4; i++)
			intensity->molec.c_sym[i] = new char[10];
		strcpy(intensity->molec.c_sym[0],"A1");
		strcpy(intensity->molec.c_sym[1],"A2");
		strcpy(intensity->molec.c_sym[2],"B1");
		strcpy(intensity->molec.c_sym[3],"B2");

	}else if(strcmp("C3V(M)",symchar)==0)
	{
		printf("C3v(M) sym");
		intensity->molec.sym_type = C3VM_ID;
		intensity->molec.sym_nrepres = 3;
		intensity->molec.sym_degen = new int[3];
		intensity->molec.sym_maxdegen = 2;
		intensity->isym_do = new bool[3];
		intensity->molec.sym_degen[0] = 1;
		intensity->molec.sym_degen[1] = 1;
		intensity->molec.sym_degen[2] = 2;

		intensity->isym_do[0] = true;
		intensity->isym_do[1] = true;
		intensity->isym_do[2] = true;
		intensity->gns = new double[3];
		intensity->isym_pairs = new int[3];
		intensity->igamma_pair = new int[3];
		

		intensity->molec.c_sym = new char*[3];
		for(int i = 0; i < 3; i++)
			intensity->molec.c_sym[i] = new char[10];
		strcpy(intensity->molec.c_sym[0],"A1");
		strcpy(intensity->molec.c_sym[1],"A2");
		strcpy(intensity->molec.c_sym[2],"E");

	}else if(strcmp("D3H(M)",symchar)==0)
	{
		printf("D3H(M) sym");
		intensity->molec.sym_type = D3HM_ID;
		intensity->molec.sym_nrepres = 6;
		intensity->molec.sym_degen = new int[6];
		intensity->molec.sym_maxdegen = 2;
		intensity->isym_do = new bool[6];
		intensity->molec.sym_degen[0] = 1;
		intensity->molec.sym_degen[1] = 1;
		intensity->molec.sym_degen[2] = 2;
		intensity->molec.sym_degen[3] = 1;
		intensity->molec.sym_degen[4] = 1;
		intensity->molec.sym_degen[5] = 2;

		intensity->isym_do[0] = true;
		intensity->isym_do[1] = true;
		intensity->isym_do[2] = true;
		intensity->isym_do[3] = true;
		intensity->isym_do[4] = true;
		intensity->isym_do[5] = true;
		intensity->gns = new double[6];

		intensity->isym_pairs = new int[6];
		intensity->igamma_pair = new int[6];
		

		intensity->molec.c_sym = new char*[6];
		for(int i = 0; i < 6; i++)
			intensity->molec.c_sym[i] = new char[10];
		strcpy(intensity->molec.c_sym[0],"A1\'");
		strcpy(intensity->molec.c_sym[1],"A2\'");
		strcpy(intensity->molec.c_sym[2],"E\'");
		strcpy(intensity->molec.c_sym[3],"A1\"");
		strcpy(intensity->molec.c_sym[4],"A2\"");
		strcpy(intensity->molec.c_sym[5],"E\"");

	}else if(strcmp("C2H(M)",symchar)==0)
	{
		printf("C2H(M) sym");
		intensity->molec.sym_type = C2HM_ID;
		intensity->molec.sym_nrepres = 4;
		intensity->molec.sym_degen = new int[4];
		intensity->molec.sym_maxdegen = 1;
		intensity->isym_do = new bool[4];
		intensity->molec.sym_degen[0] = 1;
		intensity->molec.sym_degen[1] = 1;
		intensity->molec.sym_degen[2] = 1;
		intensity->molec.sym_degen[3] = 1;

		intensity->isym_do[0] = true;
		intensity->isym_do[1] = true;
		intensity->isym_do[2] = true;
		intensity->isym_do[3] = true;
		intensity->gns = new double[4];

		intensity->isym_pairs = new int[4];
		intensity->igamma_pair = new int[4];
		

		intensity->molec.c_sym = new char*[4];
		for(int i = 0; i < 4; i++)
			intensity->molec.c_sym[i] = new char[10];
		strcpy(intensity->molec.c_sym[0],"Ag");
		strcpy(intensity->molec.c_sym[1],"Au");
		strcpy(intensity->molec.c_sym[2],"Bg");
		strcpy(intensity->molec.c_sym[3],"Bu");
	}else if(strcmp("D2H(M)",symchar)==0)
	{
		printf("D2H(M) sym");
		intensity->molec.sym_type = D2HM_ID;
		intensity->molec.sym_nrepres = 8;
		intensity->molec.sym_degen = new int[8];
		intensity->molec.sym_maxdegen = 1;
		intensity->isym_do = new bool[8];

		intensity->molec.sym_degen[0] = 1;
		intensity->molec.sym_degen[1] = 1;
		intensity->molec.sym_degen[2] = 1;
		intensity->molec.sym_degen[3] = 1;
		intensity->molec.sym_degen[4] = 1;
		intensity->molec.sym_degen[5] = 1;
		intensity->molec.sym_degen[6] = 1;
		intensity->molec.sym_degen[7] = 1;

		intensity->isym_do[0] = true;
		intensity->isym_do[1] = true;
		intensity->isym_do[2] = true;
		intensity->isym_do[3] = true;
		intensity->isym_do[4] = true;
		intensity->isym_do[5] = true;
		intensity->isym_do[6] = true;
		intensity->isym_do[7] = true;
		intensity->gns = new double[8];

		intensity->isym_pairs = new int[8];
		intensity->igamma_pair = new int[8];
		

		intensity->molec.c_sym = new char*[8];
		for(int i = 0; i < 8; i++)
			intensity->molec.c_sym[i] = new char[10];
		strcpy(intensity->molec.c_sym[0],"Ag");
		strcpy(intensity->molec.c_sym[1],"Au");
		strcpy(intensity->molec.c_sym[2],"B1g");
		strcpy(intensity->molec.c_sym[3],"B1u");
		strcpy(intensity->molec.c_sym[4],"B2g");
		strcpy(intensity->molec.c_sym[5],"B2u");
		strcpy(intensity->molec.c_sym[6],"B3g");
		strcpy(intensity->molec.c_sym[7],"B3u");
	}
	else{
		printf("Symmetry not implemeted\n");
		exit(0);
	}
		
	

};

double readd(){
	char* line_ptr;
	line_ptr=strtok(NULL," ,"); //Get the temperature
	if(line_ptr!=NULL)return strtod(line_ptr,NULL);
	else{
		printf("Read error[ readd]\n");
                exit(0);
		return -1.0;
	}
};	

int readi(){
	char* line_ptr;
	line_ptr=strtok(NULL," ,"); //Get the temperature
	if(line_ptr!=NULL)return strtol(line_ptr,NULL,0);
	else{
		printf("Read error[ readi]\n");
                exit(0);
		return -1.0;
	}
};

char* readc(){
	char* line_ptr;
	line_ptr=strtok(NULL," ,"); //Get the temperature
	return line_ptr;
};

void read_intensities(FintensityJob* intensity,ifstream & ifin)
{
	//
	printf("Intensity Part");
	string line;
	char* line_ptr;
	char linestr[2024];
	while(getline(ifin,line)){

		cout<<line<<endl;

		if(line.find_first_of("(") != -1) line.erase(line.find_first_of("("),line.find_first_of(")")-line.find_first_of("(")+1);

		strcpy (linestr, line.c_str());

		//CAPS ALL OF IT
		for(int i = 0; i < line.length(); i++)
			linestr[i]=toupper(linestr[i]);

		//Begin Reading
		line_ptr=strtok(linestr," ,");
		if(line_ptr!= NULL){
			if(strcmp("END",line_ptr)==0) break;
			else if(strcmp("TEMPERATURE",line_ptr)==0) intensity->temperature = readd();
			else if(strcmp("QSTAT",line_ptr)==0) intensity->q_stat = readd();
			else if(strcmp("ZPE",line_ptr)==0) intensity->ZPE = readd();
			else if(strcmp("J",line_ptr)==0) {
				int start_J = readi();
				int end_J = readi();
				intensity->nJ = abs(end_J-start_J + 1);
				printf("nJ = %i \n",intensity->nJ);
				if(intensity->nJ <2 || start_J > end_J ){ printf("cannot do intensities with configuration J %i,%i\n",start_J,end_J); exit(0);}

				intensity->jvals = new int[intensity->nJ];

				for(int i = start_J; i < end_J+1; i++){
					intensity->jvals[i-start_J]=i;
					printf("%i\n",intensity->jvals[i-start_J]);
				}
				intensity->bset_contr = new TO_bset_contrT[1 + intensity->nJ];
			}
			else if(strcmp("FREQ-WINDOW",line_ptr)==0){intensity->freq_window[0] = readd(); intensity->freq_window[1] = readd();}
			else if(strcmp("ENERGY",line_ptr)==0){
				line_ptr=readc();
				while(line_ptr!=NULL){
		
					if(strcmp("LOW",line_ptr)==0)
					{
						intensity->erange_lower[0] = readd();
						intensity->erange_lower[1] = readd();
					}else if(strcmp("UPPER",line_ptr)==0)
					{
						intensity->erange_upper[0] = readd();
						intensity->erange_upper[1] = readd();
					}
					line_ptr=readc();
				}
			}
			else if(strcmp("GNS",line_ptr)==0){
				for(int i = 0; i < intensity->molec.sym_nrepres; i++)
				{
					intensity->gns[i]=readd();
					if(intensity->gns[i] <=0.00000001){
						intensity->isym_do[i]=false;
					}
					//printf("GNS=%.f\n",intensity->gns[i]);
				}
				//exit(0);
			}
			else if(strcmp("SELECTION",line_ptr)==0){
				for(int i = 0; i < intensity->molec.sym_nrepres; i++)
					intensity->isym_pairs[i]=readi();
			}
			else if(strcmp("THRESH_LINE",line_ptr)==0){
				intensity->thresh_linestrength=readd();
			}
			else if(strcmp("V",line_ptr)==0){
				//exit(0);
				int quanta_no = 0;
				quanta_no = readi()-1;
				line_ptr=readc();
				while(line_ptr!=NULL){
		
					if(strcmp("LOWER",line_ptr)==0)
					{
						intensity->quanta_lower[0][quanta_no] = readi();
						intensity->quanta_lower[1][quanta_no] = readi();
					}else if(strcmp("UPPER",line_ptr)==0)
					{
						intensity->quanta_upper[0][quanta_no] = readi();
						intensity->quanta_upper[1][quanta_no] = readi();
					}
					line_ptr=readc();
				}
			}
			else if(strcmp("SYMMETRY",line_ptr)==0){
				line_ptr=readc();
				if(strcmp("REDUCED",line_ptr)==0)
					intensity->reduced=true;
			}
		}		
	}

};

void read_fields(FintensityJob* intensity){
	//begin reading
	string line;
	char* line_ptr;
	char linestr[2024];
	intensity->quanta_lower=NULL;
	intensity->quanta_upper=NULL;
	intensity->reduced = false;
	intensity->dipole_max = 5000000000;
	ifstream i_file(intensity->filename);
 	if(!i_file){ printf("file %s could not be found",intensity->filename); exit(0);}
	while(getline(i_file,line)){
		cout<<line<<endl;
		strcpy (linestr, line.c_str());
		//CAPS ALL OF IT
		for(int i = 0; i < line.length(); i++)
			linestr[i]=toupper(linestr[i]);
		//Begin Reading
		line_ptr=strtok(linestr," ,");
		//This should contai the first variable to check
		if(line_ptr!= NULL){
			if(strcmp("NMODES",line_ptr)==0){
				line_ptr=strtok(NULL," ,"); //Get the number
				if(line_ptr!=NULL)intensity->molec.nmodes= strtol(line_ptr,NULL,0); else continue;
				//Declare the quanta
				intensity->quanta_lower=new int*[2];
				intensity->quanta_lower[0]=new int[intensity->molec.nmodes];
				intensity->quanta_lower[1]=new int[intensity->molec.nmodes];
				intensity->quanta_upper=new int*[2];
				intensity->quanta_upper[0]=new int[intensity->molec.nmodes];
				intensity->quanta_upper[1]=new int[intensity->molec.nmodes];
				//printf("Nmodes = %i\n",intensity->molec.nmodes);
			}else if(strcmp("MEM",line_ptr)==0){
				unsigned long value  = readd()*1000000000.0;
				intensity->dipole_max = min(value,intensity->dipole_max);

			}else if(strcmp("SYMGROUP",line_ptr)==0){
				line_ptr=strtok(NULL," ");
				read_symmetry(line_ptr,intensity);
			}else if(strcmp("INTENSITY",line_ptr)==0){
				//Do line stuff
				read_intensities(intensity,i_file);
			}
		}
		
	}

	intensity->erange[0] = min(intensity->erange_lower[0],intensity->erange_upper[0]);
	intensity->erange[1] = max(intensity->erange_lower[1],intensity->erange_upper[1]);

	printf("Min: %.f Max: %.f\n",intensity->erange[0],intensity->erange[1]);

	
};




