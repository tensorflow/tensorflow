#ifndef DATA_HANDLER
#define DATA_HANDLER

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <strstream>
#include <typeinfo>

#include "gen_instructions.h"
#include "vta_driver.h"

namespace tflite_secda {

template <typename Integer>
int roundUp(int numToRound, int multiple) {
  if (multiple == 0) return numToRound;
  int remainder = numToRound % multiple;
  if (remainder == 0) return numToRound;
  return numToRound + multiple - remainder;
}

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
          // cout << (r + i * r_block + k) << " " << (c + j * c_block + l) << "
          // "
          //      << data_vale << endl;
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

template <typename PackedResult>
void BlockIntoVTAv2(conv2d_driver& drv) {
  drv.scs->sig_ra_sig.write(drv.ra);

  uint32_t* crf_buf = (uint32_t*)(drv.crf);
  uint32_t* crx_buf = (uint32_t*)(drv.crx);
  uint32_t* weight_bb = new uint32_t[drv.pK * drv.pM];
  packBufferBlock<uint32_t, 32, int8_t, 8>(weight_bb, drv.padded_weights, 0,
                                           drv.pM, VTA_BLOCK_OUT, 0, drv.pK,
                                           VTA_BLOCK_IN);

  uint32_t* input_bb = new uint32_t[drv.pN * drv.pK];
  packBufferBlock<uint32_t, 32, int8_t, 8>(input_bb, drv.padded_input, 0,
                                           drv.pN, VTA_BATCH, 0, drv.pK,
                                           VTA_BLOCK_IN);

  uint32_t* bias_bb = new uint32_t[drv.pN * drv.pM];
  uint32_t* out_buf = new uint32_t[drv.pN * drv.pM];

  create_2d_biases(0, drv.pN, 0, drv.pM, bias_bb, drv.bias, drv.wt_sum, drv.in_sum,
                   drv.rhs_offset, drv.lhs_offset, drv.K);

  blocked_gemm_test_tflitev2(drv,drv.scs, drv.pN, drv.pK, drv.pM, 0, true, 1, input_bb,
                             weight_bb, bias_bb, out_buf, crf_buf, crx_buf,
                             drv.save, drv.ra,drv.layer);

  unpackBufferBlock<int8_t, 8, uint32_t, 32>(drv.padded_output, out_buf, 0, drv.pN,
                                             VTA_BATCH, 0, drv.pM, VTA_BLOCK_OUT);

}

template <typename PackedResult>
void BlockIntoVTA(conv2d_driver& drv) {
  int inp_size = drv.pN / VTA_BATCH * drv.pK / VTA_BLOCK_IN;
  int wgt_size = drv.pK / VTA_BLOCK_IN * drv.pM / VTA_BLOCK_OUT;
  int out_size = drv.pN / VTA_BATCH * drv.pM / VTA_BLOCK_OUT;

  int input_len = VTA_INP_ELEM_BYTES * inp_size / sizeof(unsigned long long);
  int weight_len = VTA_WGT_ELEM_BYTES * wgt_size / sizeof(unsigned long long);
  int bias_len = VTA_ACC_ELEM_BYTES * out_size / sizeof(unsigned long long);
  int out_len = VTA_OUT_ELEM_BYTES * out_size / sizeof(unsigned long long);

  int total_len = input_len + weight_len + bias_len + out_len;


  int k_inc = drv.pK;
  int m_inc = min(256, drv.pM);
  int n_inc = min(256, drv.pN);
  int block = min(8, min(n_inc / 16, min(k_inc / 16, m_inc / 16))) * 16;
  if (m_inc % block != 0 || k_inc % block != 0 || n_inc % block != 0)
    block = 16;

  int ins_count = n_inc / block * m_inc / block * (2 + k_inc / block * 3) + 2;
  if (ins_count > 512) n_inc = min(128, drv.pN);
  ins_count = n_inc / block * m_inc / block * (2 + k_inc / block * 3) + 2;
  if (ins_count > 512) m_inc = min(128, drv.pM);
  ins_count = n_inc / block * m_inc / block * (2 + k_inc / block * 3) + 2;

  if (ins_count > 512) n_inc = min(64, drv.pN);
  ins_count = n_inc / block * m_inc / block * (2 + k_inc / block * 3) + 2;
  if (ins_count > 512) m_inc = min(64, drv.pM);
  ins_count = n_inc / block * m_inc / block * (2 + k_inc / block * 3) + 2;

  if (ins_count > 512) n_inc = min(32, drv.pN);
  ins_count = n_inc / block * m_inc / block * (2 + k_inc / block * 3) + 2;
  if (ins_count > 512) m_inc = min(32, drv.pM);
  ins_count = n_inc / block * m_inc / block * (2 + k_inc / block * 3) + 2;

  if (ins_count > 512) n_inc = min(16, drv.pN);
  ins_count = n_inc / block * m_inc / block * (2 + k_inc / block * 3) + 2;
  if (ins_count > 512) m_inc = min(16, drv.pM);
  ins_count = n_inc / block * m_inc / block * (2 + k_inc / block * 3) + 2;

  int n_inc_r = drv.pN % n_inc;
  if (n_inc_r) {
    block = min(8, min(n_inc_r / 16, min(k_inc / 16, m_inc / 16))) * 16;
    if (m_inc % block != 0 || k_inc % block != 0 || n_inc_r % block != 0)
      block = 16;
    ins_count = n_inc / block * m_inc / block * (2 + k_inc / block * 3) + 2;
    if (ins_count > 512) n_inc = min(16, drv.pN);
  }

  ins_count = n_inc / block * m_inc / block * (2 + k_inc / block * 3) + 2;
  if (ins_count > 512) m_inc = min(128, drv.pM);
  ins_count = n_inc / block * m_inc / block * (2 + k_inc / block * 3) + 2;
  if (ins_count > 512) m_inc = min(64, drv.pM);
  ins_count = n_inc / block * m_inc / block * (2 + k_inc / block * 3) + 2;
  if (ins_count > 512) m_inc = min(32, drv.pM);
  ins_count = n_inc / block * m_inc / block * (2 + k_inc / block * 3) + 2;
  if (ins_count > 512) m_inc = min(16, drv.pM);

  int m_inc_r = drv.pM % m_inc;
  if (m_inc_r) {
    block = min(8, min(n_inc / 16, min(k_inc / 16, m_inc_r / 16))) * 16;
    if (n_inc % block != 0 || k_inc % block != 0 || m_inc_r % block != 0)
      block = 16;
    ins_count = n_inc / block * m_inc / block * (2 + k_inc / block * 3) + 2;
    if (ins_count > 512) m_inc = min(16, drv.pM);
  }

  drv.scs->sig_ra_sig.write(drv.ra);
  int int_block = block;
  for (int k = 0; k < drv.pK; k += k_inc) {  // Common Dim
    int k_b = min(k_inc, drv.pK - k);
    for (int m = 0; m < drv.pM; m += m_inc) {  // Weight Dim
      int m_b = min(m_inc, drv.pM - m);

      uint32_t* crf_buf = (uint32_t*)(drv.crf + m);
      uint32_t* crx_buf = (uint32_t*)(drv.crx + m);

      uint32_t* weight_bb = new uint32_t[k_b * m_b];
      packBufferBlock<uint32_t, 32, int8_t, 8>(weight_bb, drv.padded_weights, m,
                                               m_b, VTA_BLOCK_OUT, k, k_b,
                                               VTA_BLOCK_IN);

      for (int n = 0; n < drv.pN; n += n_inc) {  // Input Dim
        int n_b = min(n_inc, drv.pN - n);

        uint32_t* input_bb = new uint32_t[n_b * k_b];
        packBufferBlock<uint32_t, 32, int8_t, 8>(input_bb, drv.padded_input, n,
                                                 n_b, VTA_BATCH, k, k_b,
                                                 VTA_BLOCK_IN);

        uint32_t* bias_bb = new uint32_t[n_b * m_b];
        uint32_t* out_buf = new uint32_t[n_b * m_b];

        create_2d_biases(n, n_b, m, m_b, bias_bb, drv.bias, drv.wt_sum,
                         drv.in_sum, drv.rhs_offset, drv.lhs_offset, drv.K);

        block = min(8, min(n_b / 16, min(k_b / 16, m_b / 16))) * 16;
        if (m_b % block != 0 || k_b % block != 0 || n_b % block != 0)
          block = 16;

        // for (int p=0;p<n_b * m_b;p++)cout << bias_bb[p] << endl;
        // cout << "===========================" << endl;
        // for (int p=0;p<m_b;p++)cout << crf_buf[p] << endl;
        // cout << "===========================" << endl;
        // for (int p=0;p<m_b;p++)cout << (int) drv.crx[p] << endl;
        // cout << "===========================" << endl;

        blocked_gemm_test_tflite(drv.scs, n_b, k_b, m_b, block, false, 1,
                                 input_bb, weight_bb, bias_bb, out_buf, crf_buf,
                                 crx_buf, drv.save, drv.ra);

        drv.save = false;
        unpackBufferBlock<int8_t, 8, uint32_t, 32>(drv.padded_output, out_buf,
                                                   n, n_b, VTA_BATCH, m, m_b,
                                                   VTA_BLOCK_OUT);
      }
    }
  }
}

template <typename Integer>
void Entry(conv2d_driver& drv) {
  cout << "VTA ACC - Layer: " << drv.layer << endl;
  cout << "===========================" << endl;
  cout << "Pre-ACC Info" << endl;
  cout << "padded_K: " << drv.pK << " K: " << drv.K << endl;
  cout << "padded_M: " << drv.pM << " M: " << drv.M << endl;
  cout << "padded_N: " << drv.pN << " N: " << drv.N << endl;
  cout << "===========================" << endl;
  // BlockIntoVTA<int>(drv);
  BlockIntoVTAv2<int>(drv);
}
}  // namespace tflite_secda
#endif
