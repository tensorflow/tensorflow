#ifndef FC_DRIVER
#define FC_DRIVER

#include "acc_container.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/multi_threading.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"

#ifdef ACC_NEON
#include "arm_neon.h"
#endif

#define PAGE_SIZE getpagesize()
#define MM_BL 4194304
#define acc_address 0x43C00000
#define insn_addr 0x16000000
#define in_addr 0x17000000
#define wgt_addr 0x18000000
#define out_addr 0x19000000
#define bias_addr 0x1a000000

#define INP_ACCESS 8
#define WGT_ACCESS 8
#define ACC_ACCESS 2

#define INP_MEMS 4
#define WGT_MEMS 4
#define ACC_MEMS 1

#define INP_DEPTH 4096
#define WGT_DEPTH 8192
#define ACC_DEPTH 8192

#define INP_SIZE (INP_DEPTH * INP_ACCESS * INP_MEMS)
#define WGT_SIZE (WGT_DEPTH * WGT_ACCESS * WGT_MEMS)
#define ACC_SIZE (ACC_DEPTH * ACC_ACCESS * ACC_MEMS)

// FC_Driver for FC-GEMM acccelerator
namespace tflite_bert {

void createWeightLoad(unsigned long long *insn, int &idx, int wgt_start,
                      int depth, int m_inc) {
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

void createInputLoad(unsigned long long *insn, int &idx, int inp_start,
                     int depth, int n_inc) {
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

void createBiasLoad(unsigned long long *insn, int &idx, int bias_start,
                    int stride, int n_inc, int m_inc) {
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

void createCompute(unsigned long long *insn, int &idx, int out_start,
                   int stride, int inp_block, int wgt_block) {
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

  int32_t *wt_sum = drv.wt_sum;
  int32_t *in_sum = drv.in_sum;
  int32_t *bias_buf = (int32_t *)(drv.bias_mem);

  prf_start(0);
  create_2d_biases(0, drv.pN, 0, drv.pM, bias_buf, drv.bias, wt_sum, in_sum,
                   drv.rhs_offset, drv.lhs_offset, drv.K);
  prf_end(0, drv.t.bpack);

  int insn_idx = 0;
  unsigned long long *insn_mem = drv.insn_mem;
  for (int k = 0; k < drv.pK; k += k_inc) { // Common Dim
    int k_b = min(k_inc, drv.pK - k);
    for (int m = 0; m < drv.pM; m += m_inc) { // Weight Dim
      int m_b = min(m_inc, drv.pM - m);
      // Load Weight
      createWeightLoad(insn_mem, insn_idx, m, drv.pK, m_b);
      for (int n = 0; n < drv.pN; n += n_inc) { // Input Dim
        int n_b = min(n_inc, drv.pN - n);
        createInputLoad(insn_mem, insn_idx, n, drv.pK, n_b);
        createBiasLoad(insn_mem, insn_idx, drv.pN * m + n, drv.pN, n_b, m_b);
        createCompute(insn_mem, insn_idx, drv.pM * n + m, drv.pM, n_b, m_b);
      }
    }
  }

  writeMappedReg<int>(drv.acc, 0x5c, drv.crf);
  writeMappedReg<int>(drv.acc, 0x64, drv.crx);
  writeMappedReg<int>(drv.acc, 0x6c, drv.ra);
  writeMappedReg<int>(drv.acc, 0x74, drv.pK);
  writeMappedReg<int>(drv.acc, 0x2c, insn_idx / 2);
  writeMappedReg<int>(drv.acc, 0x14, ++drv.start_count);

  prf_start(1);
  bool done = readMappedReg<int>(drv.acc, 0x1c) == drv.start_count;
  while (!done) {
    done = readMappedReg<int>(drv.acc, 0x1c) == drv.start_count;
  }
  prf_end(1, drv.t.acc);
}

void Entry(acc_container &drv) {
#ifdef DELEGATE_VERBOSE
  VLOG("FC ACC - Layer: " << drv.layer << endl);
  VLOG("===========================" << endl);
  VLOG("Pre-ACC Info" << endl);
  VLOG("padded_K: " << drv.pK << " K: " << drv.K << endl);
  VLOG("padded_M: " << drv.pM << " M: " << drv.M << endl);
  VLOG("padded_N: " << drv.pN << " N: " << drv.N << endl);
  VLOG("===========================" << endl);
#endif
  BlockFC(drv);
}
} // namespace tflite_bert
#endif // FC_DRIVER
