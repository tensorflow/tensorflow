#ifndef VTA_GEMM_DRIVER
#define VTA_GEMM_DRIVER

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <strstream>
#include <typeinfo>

#ifdef ACC_NEON
#include "arm_neon.h"
#endif

#include "acc_container.h"
#include "gen_instructions.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"

namespace tflite_vtasim {

template <typename DST_T, int DST_T_WIDTH, typename SRC_T, int SRC_T_WIDTH>
void unpackBufferBlock(DST_T** dblock, SRC_T* src, int r, int r_b, int r_block,
                       int c, int c_b, int c_block) {
  assert((DST_T_WIDTH * r_block * c_block) % SRC_T_WIDTH == 0);
  int buffer_idx = 0;
  long long int mask = (1ULL << DST_T_WIDTH) - 1;
  int ratio = SRC_T_WIDTH / DST_T_WIDTH;
  for (int i = 0; i < r_b / r_block; i++) {
    for (int j = 0; j < c_b / c_block; j++) {
      for (int k = 0; k < r_block; k++) {
        for (int l = 0; l < c_block; l++) {
          int block_idx = l + k * c_block;
          dblock[r + i * r_block + k][c + j * c_block + l] =
              (src[buffer_idx] >> ((block_idx % ratio) * DST_T_WIDTH)) & mask;
          if (block_idx % ratio == ratio - 1) {
            buffer_idx++;
          }
        }
      }
    }
  }
}

template <typename DST_T, int DST_T_WIDTH, typename SRC_T, int SRC_T_WIDTH>
void packBufferBlock(DST_T* dblock, SRC_T** src, int r, int r_b, int r_block,
                     int c, int c_b, int c_block) {
  assert((SRC_T_WIDTH * r_block * c_block) % DST_T_WIDTH == 0);
  assert(DST_T_WIDTH <= 64);

  int buffer_idx = 0;
  int ratio = DST_T_WIDTH / SRC_T_WIDTH;
  long long int mask = (1ULL << SRC_T_WIDTH) - 1;
  DST_T tmp = 0;
  for (int i = 0; i < r_b / r_block; i++) {
    for (int j = 0; j < c_b / c_block; j++) {
      for (int k = 0; k < r_block; k++) {
        for (int l = 0; l < c_block; l++) {
          int block_idx = l + k * c_block;
          int data_vale = src[r + i * r_block + k][c + j * c_block + l];
          tmp |= (src[r + i * r_block + k][c + j * c_block + l] & mask)
                 << ((block_idx % ratio) * SRC_T_WIDTH);
          if (block_idx % ratio == ratio - 1) {
            dblock[buffer_idx++] = tmp;
            tmp = 0;
          }
        }
      }
    }
  }
}

void BlockIntoVTA(acc_container& drv) {
  int K = drv.pK;
  int M = drv.flipped ? drv.pN : drv.pM;
  int N = drv.flipped ? drv.pM : drv.pN;

  int32_t* wt_sum = &drv.wt_sum[0];
  int32_t* in_sum = drv.in_sum;
  int8_t** padded_output = drv.padded_output;
  uint32_t* weight_buf;
  uint32_t* input_buf;
  uint32_t* crf_buf = (uint32_t*)(drv.crf);
  uint32_t* crx_buf = (uint32_t*)(drv.crx);
  uint32_t* bias_buf = new uint32_t[N * M];
  uint32_t* out_buf = new uint32_t[N * M];

  if (drv.flipped) {
    input_buf = drv.packed_weights;
    weight_buf = new uint32_t[K * M];
    packBufferBlock<uint32_t, 32, int8_t, 8>(weight_buf, drv.padded_input, 0, M,
                                             VTA_BLOCK_OUT, 0, K, VTA_BLOCK_IN);
    create_2d_biases_flipped(0, M, 0, N, bias_buf, drv.bias, wt_sum, in_sum,
                             drv.rhs_offset, drv.lhs_offset, drv.K);
  } else {
    weight_buf = drv.packed_weights;
    input_buf = new uint32_t[N * K];
    packBufferBlock<uint32_t, 32, int8_t, 8>(input_buf, drv.padded_input, 0, N,
                                             VTA_BATCH, 0, K, VTA_BLOCK_IN);
    create_2d_biases(0, N, 0, M, bias_buf, drv.bias, wt_sum, in_sum,
                     drv.rhs_offset, drv.lhs_offset, drv.K);
  }

  blocked_gemm_test_tflite(drv, drv.scs, N, K, M, 0, true, 1, input_buf,
                             weight_buf, bias_buf, out_buf, crf_buf, crx_buf,
                             drv.save, drv.ra, drv.layer, drv.flipped);

  unpackBufferBlock<int8_t, 8, uint32_t, 32>(padded_output, out_buf, 0, N,
                                             VTA_BATCH, 0, M, VTA_BLOCK_OUT);
}

void Entry(acc_container& drv) {
#ifdef DELEGATE_VERBOSE
  cout << "VTA ACC - Layer: " << drv.layer << (drv.flipped ? " Flipped" : "")
       << endl;
  cout << "===========================" << endl;
  cout << "Pre-ACC Info" << endl;
  cout << "padded_K: " << drv.pK << " K: " << drv.K << endl;
  cout << "padded_M: " << drv.pM << " M: " << drv.M << endl;
  cout << "padded_N: " << drv.pN << " N: " << drv.N << endl;
  cout << "===========================" << endl;

  cout << drv.layer;
  cout << " ," << drv.pK;
  cout << " ," << drv.pM;
  cout << " ," << drv.pN;
  cout << " ," << drv.K;
  cout << " ," << drv.M;
  cout << " ," << drv.N << endl;
#endif
  BlockIntoVTA(drv);
}
}  // namespace tflite_vtasim

#endif  // VTA_GEMM_DRIVER
