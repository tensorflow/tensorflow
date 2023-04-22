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

#include "../vta.h"
#include "systemc_binding.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"

using namespace std;
using namespace std::chrono;

struct acc_container {
  ACCNAME* acc;
  Profile* profile;
  struct sysC_sigs* scs;

  int layer = 0;

  int8_t** padded_input;
  int8_t** padded_weights;
  int8_t** padded_output;

  uint32_t* packed_weights;

  int M;
  int N;
  int K;

  int pN;
  int pM;
  int pK;

  bool save;

  int32_t* bias;
  std::vector<int> wt_sum;
  int* in_sum;

  int* crf;
  int8_t* crx;
  int ra;
  int rhs_offset = 0;
  int lhs_offset = 0;

  bool flipped;
  int vta_count;

  acc_container(int8_t** _padded_input, int8_t** _padded_weights,
                int8_t** _padded_output, std::vector<int> _wt_sum,
                // std::vector<int> _crf, std::vector<int8_t> _crx) {
                int* _crf, int8_t* _crx) {
    padded_input = _padded_input;
    padded_weights = _padded_weights;
    padded_output = _padded_output;
    wt_sum = _wt_sum;
    crf = _crf;
    crx = _crx;
  }
};


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


void create_2d_biases(int sn, int N_dim, int sm, int M_dim, uint32_t* new_bias,
                      int32_t* bias, int32_t* wt_sum, int* in_sum,
                      int32_t rhs_offset, int32_t lhs_offset, int32_t depth) {
  int offdepth = depth * rhs_offset;
  for (int m = 0; m < M_dim; m++) {
    for (int n = 0; n < N_dim; n++) {
      int yt = ((in_sum[sn + n] + offdepth) * lhs_offset);
      int xt = bias[sm + m] + (wt_sum[sm + m] * rhs_offset);
      new_bias[n * M_dim + m] = yt + xt;
    }
  }
}

void create_2d_biases_flipped(int sn, int N_dim, int sm, int M_dim,
                              uint32_t* new_bias, int32_t* bias,
                              int32_t* wt_sum, int* in_sum, int32_t rhs_offset,
                              int32_t lhs_offset, int32_t depth) {
  int offdepth = depth * rhs_offset;
  for (int m = 0; m < M_dim; m++) {
    for (int n = 0; n < N_dim; n++) {
      int yt = ((in_sum[sn + n] + offdepth) * lhs_offset);
      int xt = bias[sm + m] + (wt_sum[sm + m] * rhs_offset);
      new_bias[m * N_dim + n] = yt + xt;
    }
  }
}



#endif  // ACC_CONTAINER