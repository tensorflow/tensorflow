#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <typeinfo>
#include <unistd.h>
#include <vector>

#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/acc_helpers.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/multi_threading.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"

#ifdef ACC_NEON
#include "arm_neon.h"
#endif

using namespace std;
using namespace std::chrono;

#ifdef ACC_PROFILE
#define prf_start(N) auto start##N = chrono::steady_clock::now();
#define prf_end(N, X)                                                          \
  auto end##N = chrono::steady_clock::now();                                   \
  X += end##N - start##N;
#else
#define prf_start(N)
#define prf_end(N, X)
#endif

struct fc_times {
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> conv_total;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> pack;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> bpack;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> acc;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> unpack;

  void print() {
    cout << "================================================" << endl;
    cout << "conv_total, ";
    cout << " pack, ";
    cout << " bpack, ";
    cout << " acc, ";
    cout << " unpack " << endl;
    cout << chrono::duration_cast<chrono::milliseconds>(conv_total).count();
    cout << " , " << chrono::duration_cast<chrono::milliseconds>(pack).count();
    cout << " , " << chrono::duration_cast<chrono::milliseconds>(bpack).count();
    cout << " , " << chrono::duration_cast<chrono::milliseconds>(acc).count();
    cout << " , " << chrono::duration_cast<chrono::milliseconds>(unpack).count()
         << endl;
    cout << "================================================" << endl;
  }
};

struct acc_container {
  // Hardware
  int *acc;
  unsigned long long *insn_mem;
  unsigned long long *bias_mem;

  // Dims
  int M;
  int N;
  int K;

  int pN;
  int pM;
  int pK;

  // Data
  int8_t *padded_input;
  int8_t *padded_weights;
  int8_t *padded_output;

  // PPU
  // bool isBias;
  int *bias;
  int *wt_sum;
  int *in_sum;

  int crf;
  int crx;
  int ra;
  int rhs_offset;
  int lhs_offset;

  // Running Variable
  int start_count;

  // Debugging
  int layer = 0;
  struct fc_times t;
};

//========================//========================//========================//

void precal_sum_load_pad(int8_t *data, int width, int depth, int8_t *shape_data,
                         vector<int> &sums) {
  int w = ((width + 16 - 1) - ((width + 16 - 1) % 16));
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
  int dm = d - 16;
  int i_c = 0;
  for (int i = 0; i < w; i++) {
    int s0 = 0;
    if (i < width) {
#ifndef ACC_NEON
      for (int j = 0; j < d; j++) {
        if (j < depth) {
          int8_t val = data[(i * depth) + j];
          s0 += val;
          shape_data[i_c++] = val;
        } else {
          shape_data[i_c++] = 0;
        }
      }
#else
      int8x16_t tmp0;
      int16x8_t tmp0_1;
      int32x4_t tmp0_2;
      int32x2_t tmp0_3;
      int32x2_t tmp0_4 = vdup_n_s32(0);
      int32_t tmp0_s[2];
      for (int j = 0; j < dm; j += 16) {
        tmp0 = vld1q_s8(data + (i * depth) + j);
        tmp0_1 = vpaddlq_s8(tmp0);
        tmp0_2 = vpaddlq_s16(tmp0_1);
        tmp0_3 = vadd_s32(vget_high_s32(tmp0_2), vget_low_s32(tmp0_2));
        tmp0_4 = vadd_s32(tmp0_4, tmp0_3);
        vst1q_s8(shape_data + i_c, tmp0);
        i_c += 16;
      }
      vst1_s32(tmp0_s, tmp0_4);
      s0 += tmp0_s[0] + tmp0_s[1];
      for (int j = dm; j < d; j++) {
        if (j < depth) {
          int8_t val = data[(i * depth) + j];
          s0 += val;
          shape_data[i_c++] = val;
        } else {
          shape_data[i_c++] = 0;
        }
      }
#endif
    } else {
      for (int j = 0; j < d; j++)
        shape_data[i_c++] = 0;
    }
    sums.push_back(s0);
  }
}

struct precal_sum_load_pad_task : Task {
  precal_sum_load_pad_task(int8_t *_data, int _width, int _depth,
                           int8_t *_shape_data, vector<int> *_sums)
      : data(_data), width(_width), depth(_depth), shape_data(_shape_data),
        sums(_sums) {}

  void Run() override {
    int w = roundUp(width, 16);
    int d = roundUp(depth, 16);
    int dm = d - 16;
    int i_c = 0;
    for (int i = 0; i < w; i++) {
      int s0 = 0;
      if (i < width) {
#ifndef ACC_NEON
        for (int j = 0; j < d; j++) {
          if (j < depth) {
            int8_t val = data[(i * depth) + j];
            s0 += val;
            shape_data[i_c++] = val;
          } else {
            shape_data[i_c++] = 0;
          }
        }
#else
        int8x16_t tmp0;
        int16x8_t tmp0_1;
        int32x4_t tmp0_2;
        int32x2_t tmp0_3;
        int32x2_t tmp0_4 = vdup_n_s32(0);
        int32_t tmp0_s[2];
        for (int j = 0; j < dm; j += 16) {
          tmp0 = vld1q_s8(data + (i * depth) + j);
          tmp0_1 = vpaddlq_s8(tmp0);
          tmp0_2 = vpaddlq_s16(tmp0_1);
          tmp0_3 = vadd_s32(vget_high_s32(tmp0_2), vget_low_s32(tmp0_2));
          tmp0_4 = vadd_s32(tmp0_4, tmp0_3);
          vst1q_s8(shape_data + i_c, tmp0);
          i_c += 16;
        }
        vst1_s32(tmp0_s, tmp0_4);
        s0 += tmp0_s[0] + tmp0_s[1];
        for (int j = dm; j < d; j++) {
          if (j < depth) {
            int8_t val = data[(i * depth) + j];
            s0 += val;
            shape_data[i_c++] = val;
          } else {
            shape_data[i_c++] = 0;
          }
        }
#endif
      } else {
        for (int j = 0; j < d; j++)
          shape_data[i_c++] = 0;
      }
      sums->push_back(s0);
    }
  }

  int8_t *data;
  int width;
  int depth;
  int8_t *shape_data;
  vector<int> *sums;
};

void store_unpad(int8_t *data, int width, int depth, int8_t *shape_data) {
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
  int dm = roundDown(depth, 16);
  int i_c = 0;
  for (int i = 0; i < width; i++) {
#ifndef ACC_NEON
    for (int j = 0; j < depth; j++) {
      int8_t val = data[(i * d) + j];
      shape_data[i_c++] = val;
    }
#else
    int8x16_t tmp0;
    for (int j = 0; j < dm; j += 16) {
      tmp0 = vld1q_s8(data + (i * d) + j);
      vst1q_s8(shape_data + i_c, tmp0);
      i_c += 16;
    }
    for (int j = dm; j < depth; j++) {
      int8_t val = data[(i * d) + j];
      shape_data[i_c++] = val;
    }
#endif
  }
}

struct store_unpad_task : Task {
  store_unpad_task(int _start, int _end, int _depth, int _d, int _i_c,
                   int8_t *_data, int8_t *_shape_data)
      : start(_start), end(_end), depth(_depth), d(_d), i_c(_i_c), data(_data),
        shape_data(_shape_data) {}

  void Run() override {
    int dm = roundDown(depth, 16);
    for (int i = start; i < end; i++) {
#ifndef ACC_NEON
      for (int j = 0; j < depth; j++) {
        int8_t val = data[(i * d) + j];
        shape_data[i_c++] = val;
      }
#else
      int8x16_t tmp0;
      for (int j = 0; j < dm; j += 16) {
        tmp0 = vld1q_s8(data + (i * d) + j);
        vst1q_s8(shape_data + i_c, tmp0);
        i_c += 16;
      }
      for (int j = dm; j < depth; j++) {
        int8_t val = data[(i * d) + j];
        shape_data[i_c++] = val;
      }
#endif
    }
  }

  int start;
  int end;
  int depth;
  int d;
  int i_c;
  int8_t *data;
  int8_t *shape_data;
};

void create_2d_biases(int sn, int N_dim, int sm, int M_dim, int32_t *new_bias,
                      int32_t *bias, int32_t *wt_sum, int *in_sum,
                      int32_t rhs_offset, int32_t lhs_offset, int32_t depth) {
  int offdepth = 0;
  if (-lhs_offset && -rhs_offset)
    offdepth = (-lhs_offset) * depth * (-rhs_offset);

#ifndef ACC_NEON
  for (int m = 0; m < M_dim; m++) {
    for (int n = 0; n < N_dim; n++) {
      int yt = (in_sum[sn + n] * lhs_offset) + offdepth;
      int xt = bias[sm + m] + (wt_sum[sm + m] * rhs_offset);
      new_bias[m * N_dim + n] = yt + xt;
    }
  }
#else
  int32x4_t tmp_offdepth_4 = vdupq_n_s32(offdepth);
  for (int m = 0; m < M_dim; m++) {
    int xt = bias[sm + m] + (wt_sum[sm + m] * rhs_offset);
    int32x4_t tmp_xt = vdupq_n_s32(xt);
    for (int n = 0; n < N_dim; n += 4) {
      int32x4_t tmp_in_sum = vld1q_s32(in_sum + sn + n);
      int32x4_t tmp_lhs_offset_4 = vdupq_n_s32(lhs_offset);
      int32x4_t tmp_yt_mul = vmulq_s32(tmp_in_sum, tmp_lhs_offset_4);
      int32x4_t tmp_yt = vaddq_s32(tmp_yt_mul, tmp_offdepth_4);
      int32x4_t temp_nb = vaddq_s32(tmp_yt, tmp_xt);
      vst1q_s32(new_bias + m * N_dim + n, temp_nb);
    }
  }
#endif
}

#endif // ACC_CONTAINER