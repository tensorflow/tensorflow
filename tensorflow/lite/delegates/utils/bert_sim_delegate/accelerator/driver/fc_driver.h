#ifndef FC_DRIVER
#define FC_DRIVER

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
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"

// FC_Driver for simulated FC-GEMM acccelerator
namespace tflite_bertsim {

void createWeightLoad(uint64_t *insn, int &idx, int wgt_start, int depth,
                      int m_inc) {
  int doffset = wgt_start * (depth / 8);
  int dstride = (depth / 8);
  int x_size = (depth / 8);
  int y_size = m_inc;

  uint64_t p1 = 0;
  uint64_t p2 = 0;
  p1 = dstride;
  p1 = p1 << 16;
  p1 += x_size;
  p1 = p1 << 16;
  p1 += y_size;
  p2 = doffset;
  p2 = p2 << 32;
  p2 += 1;
  insn[idx++] = p2;
  insn[idx++] = p1;
}

void createInputLoad(uint64_t *insn, int &idx, int inp_start, int depth,
                     int n_inc) {
  int doffset = inp_start * (depth / 8);
  int dstride = (depth / 8);
  int x_size = (depth / 8);
  int y_size = n_inc;

  uint64_t p1 = 0;
  uint64_t p2 = 0;
  p1 = dstride;
  p1 = p1 << 16;
  p1 += x_size;
  p1 = p1 << 16;
  p1 += y_size;
  p2 = doffset;
  p2 = p2 << 32;
  p2 += 2;
  insn[idx++] = p2;
  insn[idx++] = p1;
}

void createBiasLoad(uint64_t *insn, int &idx, int bias_start, int stride,
                    int n_inc, int m_inc) {
  int doffset = bias_start / 2;
  int dstride = stride / 2;
  int x_size = n_inc / 2;
  int y_size = m_inc;

  uint64_t p1 = 0;
  uint64_t p2 = 0;
  p1 = dstride;
  p1 = p1 << 16;
  p1 += x_size;
  p1 = p1 << 16;
  p1 += y_size;
  p2 = doffset;
  p2 = p2 << 32;
  p2 += 3;
  insn[idx++] = p2;
  insn[idx++] = p1;
}

void createCompute(uint64_t *insn, int &idx, int out_start, int stride,
                   int inp_block, int wgt_block) {
  int doffset = out_start / 4;
  int dstride = stride / 4;
  int x_size = wgt_block;
  int y_size = inp_block;

  uint64_t p1 = 0;
  uint64_t p2 = 0;
  p1 = dstride;
  p1 = p1 << 16;
  p1 += x_size;
  p1 = p1 << 16;
  p1 += y_size;
  p2 = doffset;
  p2 = p2 << 32;
  p2 += 0;
  insn[idx++] = p2;
  insn[idx++] = p1;
}

void BlockFC(acc_container &drv) {
  int inp_max = INP_SIZE;
  int wgt_max = WGT_SIZE;
  int acc_max = ACC_SIZE;
  int k_inc = drv.pK;
  int m_inc = min((wgt_max), drv.pM);
  int n_inc = min((inp_max), drv.pN);

  while ((n_inc * k_inc > inp_max) && n_inc != 16)
    n_inc -= 16;
  while ((m_inc * k_inc > wgt_max) && m_inc != 16)
    m_inc -= 16;
  while ((n_inc * m_inc > acc_max) && n_inc != 16)
    n_inc -= 16;
  while ((n_inc * m_inc > acc_max) && m_inc != 16)
    m_inc -= 16;

  drv.scs->sig_start_acc = ++drv.start_count;
  drv.scs->sig_crf = drv.crf;
  drv.scs->sig_crx = drv.crx;
  drv.scs->sig_ra = drv.ra;

  int32_t *wt_sum = drv.wt_sum;
  int32_t *in_sum = drv.in_sum;
  int32_t *bias_buf = new int32_t[drv.pN * drv.pM];
  create_2d_biases(0, drv.pN, 0, drv.pM, bias_buf, drv.bias, wt_sum, in_sum,
                   drv.rhs_offset, drv.lhs_offset, drv.K);

  unsigned int insn_count = 0;
  for (int k = 0; k < drv.pK; k += k_inc) { // Common Dim
    int k_b = min(k_inc, drv.pK - k);
    for (int m = 0; m < drv.pM; m += m_inc) { // Weight Dim
      int m_b = min(m_inc, drv.pM - m);
      insn_count += 2;
      for (int n = 0; n < drv.pN; n += n_inc) { // Input Dim
        int n_b = min(n_inc, drv.pN - n);
        insn_count += 6;
      }
    }
  }

  int insn_idx = 0;
  uint64_t *insn =
      static_cast<uint64_t *>(malloc(sizeof(uint64_t) * insn_count));
  for (int k = 0; k < drv.pK; k += k_inc) { // Common Dim
    int k_b = min(k_inc, drv.pK - k);
    for (int m = 0; m < drv.pM; m += m_inc) { // Weight Dim
      int m_b = min(m_inc, drv.pM - m);
      // Load Weight
      createWeightLoad(insn, insn_idx, m, drv.pK, m_b);
      for (int n = 0; n < drv.pN; n += n_inc) { // Input Dim
        int n_b = min(n_inc, drv.pN - n);
        createInputLoad(insn, insn_idx, n, drv.pK, n_b);
        createBiasLoad(insn, insn_idx, drv.pN * m + n, drv.pN, n_b, m_b);
        createCompute(insn, insn_idx, drv.pM * n + m, drv.pM, n_b, m_b);
      }
    }
  }

  // Setting acc control signals
  drv.scs->sig_insn_count = insn_idx / 2;
  drv.scs->sig_insn_addr = 0;
  drv.scs->sig_depth = drv.pK;
  drv.scs->sig_crf = drv.crf;
  drv.scs->sig_crx = drv.crx;
  drv.scs->sig_ra = drv.ra;

  // Move Input data to MMapped DMA buffer to enable accelerator access
  unsigned long long *insn_set = (unsigned long long *)insn;
  drv.scs->insn_mem.burst_write(0, insn_idx, insn_set);
  drv.scs->inp_mem.burst_write(0, drv.pN * drv.pK / 8,
                               (unsigned long long *)&drv.padded_input[0]);
  drv.scs->wgt_mem.burst_write(0, drv.pM * drv.pK / 8,
                               (unsigned long long *)&drv.padded_weights[0]);

  drv.scs->bias_mem.burst_write(0, drv.pN * drv.pM / 2,
                                (unsigned long long *)&bias_buf[0]);

  // Start Accelerator Simulation
  sc_start();
  drv.profile->saveProfile(drv.acc->profiling_vars);

  // Retrive Output data from  MMapped DMA buffer
  unsigned int *out_set = (unsigned int *)drv.padded_output;
  int out_len = drv.pN * drv.pM / 4;
  drv.scs->out_mem.burst_read(0, out_len, out_set);
}

void Entry(acc_container &drv) {
#ifdef DELEGATE_VERBOSE
  cout << "FC ACC - Layer: " << drv.layer << endl;
  cout << "===========================" << endl;
  cout << "Pre-ACC Info" << endl;
  cout << "padded_K: " << drv.pK << " K: " << drv.K << endl;
  cout << "padded_M: " << drv.pM << " M: " << drv.M << endl;
  cout << "padded_N: " << drv.pN << " N: " << drv.N << endl;
  cout << "===========================" << endl;
#endif
  BlockFC(drv);
}

} // namespace tflite_bertsim
#endif // FC_DRIVER
