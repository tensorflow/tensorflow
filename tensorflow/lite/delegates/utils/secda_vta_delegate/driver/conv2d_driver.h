#ifndef ACCEL
#define ACCEL

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <typeinfo>
#include <vector>

// #define VLOG(X)  cout << X
#define VLOG(X)

#define VLOG2(X) cout << X

using namespace std;
using namespace std::chrono;

#define prf_start(N) auto start##N = chrono::steady_clock::now();
#define prf_end(N, X)                        \
  auto end##N = chrono::steady_clock::now(); \
  X += end##N - start##N;

#define del_profile(F, N, X) \
  prf_start(N) F;            \
  prf_end(N, X)

// #ifdef ACC_PROFILE
// #define prf_start(N) auto start##N = chrono::steady_clock::now();
// #define prf_end(N, X)                        \
//   auto end##N = chrono::steady_clock::now(); \
//   X += end##N - start##N;

// #define del_profile(F,N,X) prf_start(N) \
//   F \
//   prf_end(N, X)

// #else
// #define prf_start(N)
// #define prf_end(N, X)
// #endif

struct times {
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> conv_total;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> wpack;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> ipack;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> bpack;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> vta;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> vta_ins;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> vta_mm;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> vta_acc;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> unpack;

  void print() {
    cout << "================================================" << endl;
    cout << "conv_total, "
         << chrono::duration_cast<chrono::milliseconds>(conv_total).count()
         << endl;
    cout << "ipack, "
         << chrono::duration_cast<chrono::milliseconds>(ipack).count() << endl;
    cout << "bpack, "
         << chrono::duration_cast<chrono::milliseconds>(bpack).count() << endl;
    cout << "vta_ins, "
         << chrono::duration_cast<chrono::milliseconds>(vta_ins).count()
         << endl;
    cout << "vta_mm, "
         << chrono::duration_cast<chrono::milliseconds>(vta_mm).count() << endl;
    cout << "vta_acc, "
         << chrono::duration_cast<chrono::milliseconds>(vta_acc).count()
         << endl;
    cout << "unpack, "
         << chrono::duration_cast<chrono::milliseconds>(unpack).count() << endl;

    cout << "================================================" << endl;
  }
};

struct conv2d_driver {
  int layer = 0;
  int vta_count;
  int* acc;
  unsigned long long* opc_mm;
  unsigned int* uop_mm;
  unsigned long long* bias_mm;
  unsigned long long* crf_mm;
  unsigned long long* crx_mm;

  int M;
  int N;
  int K;

  int pN;
  int pM;
  int pK;

  int* bias;
  int* wt_sum;
  int* in_sum;

  int* crf;
  int8_t* crx;
  int ra;
  int rhs_offset = 0;
  int lhs_offset = 0;

  bool flipped;
  int ins_count;
  int w_offset;

  struct times t;

  // conv2d_driver(int8_t** _padded_input,
  //               //   int8_t** _padded_weights,
  //               int8_t** _padded_output, std::vector<int> _wt_sum, int* _crf,
  //               int8_t* _crx) {
  //   padded_input = _padded_input;
  //   //     padded_weights = _padded_weights;
  //   padded_output = _padded_output;
  //   wt_sum = _wt_sum;
  //   crf = _crf;
  //   crx = _crx;
  // }
};

template <typename Integer>
void preload_weights(int8_t* weight_data, int* dims, vector<int>& wt_sum) {
  int width = dims[0];
  int w = ((width + 4 - 1) - ((width + 4 - 1) % 4));
  int depth = dims[1] * dims[2] * dims[3];
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
  int max = width * depth;
  for (int i = 0; i < w / 4; i++) {
    int s0 = 0;
    int s1 = 0;
    int s2 = 0;
    int s3 = 0;

    for (int j = 0; j < d; j++) {
      if (j < depth) {
        int8_t w0 =
            (i * (depth * 4) + j >= max) ? 0 : weight_data[i * (depth * 4) + j];
        int8_t w1 = (i * (depth * 4) + j + depth * 1 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 1];
        int8_t w2 = (i * (depth * 4) + j + depth * 2 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 2];
        int8_t w3 = (i * (depth * 4) + j + depth * 3 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 3];
        int8_t weights[] = {w3, w2, w1, w0};
        s0 += w0;
        s1 += w1;
        s2 += w2;
        s3 += w3;
      }
    }
    wt_sum.push_back(s0);
    wt_sum.push_back(s1);
    wt_sum.push_back(s2);
    wt_sum.push_back(s3);
  }
}

#endif  // TB