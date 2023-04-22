#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <typeinfo>
#include <vector>

using namespace std;
using namespace std::chrono;

#include "systemc_binding.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/sysc_profiler/profiler.h"

struct acc_container {
  // Hardware
  struct systemC_sigs* scs;
  Profile* profile;
  ACCNAME* acc;
  struct stream_dma* sdma;

  // Data
  int size;
  const int8_t* input_A;
  const int8_t* input_B;
  int8_t* output_C;

  // PPU
  int lshift;
  int in1_off;
  int in1_sv;
  int in1_mul;
  int in2_off;
  int in2_sv;
  int in2_mul;
  int out1_off;
  int out1_sv;
  int out1_mul;
  int qa_max;
  int qa_min;

  // Debugging
  int layer = 0;
};

#endif  // ACC_CONTAINER