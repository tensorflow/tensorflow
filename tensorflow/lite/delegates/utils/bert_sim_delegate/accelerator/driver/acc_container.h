#ifndef ACC_CONTAINER
#define ACC_CONTAINER


#include <vector>
#include "../acc.h"
#include "systemc_binding.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"

struct acc_container {
  // Hardware
  struct sysC_sigs* scs;
  Profile* profile;
  ACCNAME* acc;

  // Dims
  int M;
  int N;
  int K;

  int pN;
  int pM;
  int pK;

  // Data
  int8_t* padded_input;
  int8_t* padded_weights;
  int8_t* padded_output;

  // PPU
  // bool isBias;
  int* bias;
  int* wt_sum;
  int* in_sum;

  int crf;
  int crx;
  int ra;
  int rhs_offset;
  int lhs_offset;

  // Running Variable
  int start_count;

  // Debugging
  int layer = 0;
};
void precal_sum_load_pad(int8_t* data, int width, int depth, int8_t* shape_data,
                         vector<int>& sums) {
  int w = ((width + 16 - 1) - ((width + 16 - 1) % 16));
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
  int max = width * depth;
  int dm = roundDown(depth, 16);
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
      for (int j = 0; j < d; j++) shape_data[i_c++] = 0;
    }
    sums.push_back(s0);
  }
}

void store_unpad(int8_t* data, int width, int depth, int8_t* shape_data) {
  int w = ((width + 16 - 1) - ((width + 16 - 1) % 16));
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

void create_2d_biases(int sn, int N_dim, int sm, int M_dim, int32_t* new_bias,
                      int32_t* bias, int32_t* wt_sum, int* in_sum,
                      int32_t rhs_offset, int32_t lhs_offset, int32_t depth) {
  int offdepth = 0;
  if (-lhs_offset && -rhs_offset)
    offdepth = (-lhs_offset) * depth * (-rhs_offset);
  for (int m = 0; m < M_dim; m++) {
    for (int n = 0; n < N_dim; n++) {
      int yt = (in_sum[sn + n] * lhs_offset) + offdepth;
      int xt = bias[sm + m] + (wt_sum[sm + m] * rhs_offset);
      new_bias[m * N_dim + n] = yt + xt;
      if ((m == 0 && n == 6) || (m == 6 && n == 0)) {
        int k = new_bias[m * N_dim + n];
        int b = 0;
      }
    }
  }
}

#endif  // ACC_CONTAINER