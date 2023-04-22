#ifndef DATA_HANDLER
#define DATA_HANDLER

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <strstream>
#include <typeinfo>

#include "arm_neon.h"
#include "gen_instructions.h"

namespace tflite_secda {

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

void BlockIntoVTAv2_noW(conv2d_driver& drv) {
  int* bias_mm = (int*)(drv.bias_mm);
  // unsigned int* bias_mm = (unsigned int*)(drv.bias_mm);
  uint32_t* crf_buf = (uint32_t*)(drv.crf);
  uint32_t* crx_buf = (uint32_t*)(drv.crx);
  int32_t* wt_sum = drv.wt_sum;
  int32_t* in_sum = drv.in_sum;
  int K = drv.pK;
  int M = drv.pM;
  int N = drv.pN;

  prf_start(2);
  create_2d_biases(0, N, 0, M, bias_mm, drv.bias, wt_sum, in_sum,
                   drv.rhs_offset, drv.lhs_offset, drv.K);
  prf_end(2, drv.t.bpack);

  prf_start(3);
  blocked_gemm_test_tflitev2(drv, N, K, M, true, crf_buf, crx_buf);
  prf_end(3, drv.t.vta);
}

template <typename Integer>
void Entry(conv2d_driver& drv) {
  VLOG("VTA ACC - Layer: " << drv.layer << endl);
  VLOG("===========================" << endl);
  VLOG("Pre-ACC Info" << endl);
  VLOG("padded_K: " << drv.pK << " K: " << drv.K << endl);
  VLOG("padded_M: " << drv.pM << " M: " << drv.M << endl);
  VLOG("padded_N: " << drv.pN << " N: " << drv.N << endl);
  VLOG("===========================" << endl);
  BlockIntoVTAv2_noW(drv);
}

}  // namespace tflite_secda
#endif
