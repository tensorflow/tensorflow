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

#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/acc_helpers.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/multi_threading.h"

using namespace std;
using namespace std::chrono;

#ifdef ACC_PROFILE
#define prf_start(N) auto start##N = chrono::steady_clock::now();
#define prf_end(N, X)                        \
  auto end##N = chrono::steady_clock::now(); \
  X += end##N - start##N;
#else
#define prf_start(N)
#define prf_end(N, X)
#endif

// Custom struct to save profiled timers
struct toy_times {
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> driver_time;

  void print() {
#ifdef ACC_PROFILE
    cout << "================================================" << endl;
    cout << "driver_time: "
         << chrono::duration_cast<chrono::milliseconds>(driver_time).count()
         << endl;
    cout << "================================================" << endl;
#endif
  }
};

struct acc_container {
  // Hardware
  struct stream_dma* sdma;

  // I/O Data
  const int8_t* input_A;
  const int8_t* input_B;
  int8_t* output_C;
  int length;

  // Post Processing Metadata
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