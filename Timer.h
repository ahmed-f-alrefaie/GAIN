#include "Util.h"
#include "common.h"
#include <string>
class CPUTimer {
public:
  CPUTimer(const char* name) : mStarted(false), mStopped(false), total_time(0.0f),calls(0) {
	sName = name;
  }
  ~CPUTimer() {
  }

  float elapsed() {
    assert(mStopped);
    if (!mStopped) return 0; 
    return (mStop-mStart)/1000.0f;
  }


  void start() { mStart=(float)GetTimeMs64();
				   calls++;
                                   mStarted = true; mStopped = false; }
  void stop()  { assert(mStarted);
                                   mStop=(float)GetTimeMs64(); 
                                   mStarted = false; mStopped = true; 
				   total_time += elapsed();
}


  void printReport(){
	printf("%s\t%8i\t%12.6fs\t%12.6fs\n",sName.c_str(),calls,total_time,total_time/(float)calls);   
  }

private:
  bool mStarted, mStopped;
  float mStart, mStop;
  float total_time;
  uint calls;
  std::string sName;
};


class GPUTimer {
public:
  GPUTimer(const char* name) : mStarted(false), mStopped(false), total_time(0.0f),calls(0) {
    sName = name;
    cudaEventCreate(&mStart);
    cudaEventCreate(&mStop);
  }
  ~GPUTimer() {
    cudaEventDestroy(mStart);
    cudaEventDestroy(mStop);
  }

  float elapsed() {
    assert(mStopped);
    if (!mStopped) return 0; 
    cudaEventSynchronize(mStop);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, mStart, mStop);
    return elapsed/1000.0f;
  }


  void start(cudaStream_t s = 0) { cudaEventRecord(mStart, s); 
				   calls++;
                                   mStarted = true; mStopped = false; }
  void stop(cudaStream_t s = 0)  { assert(mStarted);
                                   cudaEventRecord(mStop, s); 
                                   mStarted = false; mStopped = true; }
  void synchronise_time(){
	total_time += elapsed();
  }

  void printReport(){
	printf("%s\t%8i\t%12.6fs\t%12.6fs\n",sName.c_str(),calls,total_time,total_time/(float)calls);   
  }
private:
  bool mStarted, mStopped;
  float total_time;
  uint calls;
  cudaEvent_t mStart, mStop;
 std::string sName;
};
