#ifndef GEMM_DRIVER
#define GEMM_DRIVER

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <strstream>
#include <typeinfo>

#include "acc_container.h"
#include "arm_neon.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/acc_helpers.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"

// Pre-Defined Address for Accelerator
#define acc_address 0x43C00000
#define dma_addr0 0x40400000
#define dma_addr1 0x40410000
#define dma_addr2 0x40420000
#define dma_addr3 0x40430000
#define dma_in0 0x16000000
#define dma_in1 0x18000000
#define dma_in2 0x1a000000
#define dma_in3 0x1c000000
#define dma_out0 0x16800000
#define dma_out1 0x18800000
#define dma_out2 0x1a800000
#define dma_out3 0x1c800000
#define DMA_BL 4194304

namespace tflite_secda_sa {

struct Load_LHS_Data_Task : Task {
  Load_LHS_Data_Task(int _start, int _end, size_t _w_dex, INT_DP& _data)
      : start(_start), end(_end), w_dex(_w_dex), data(_data) {}

  void Run() override {
    int inl0 = 3 + start;
    int inl1 = start;
    int inl2 = start;
    int inl3 = start;

#ifndef ACC_NEON
    for (int i = start; i < end; i++) {
      data.W1[inl0++] = data.R1[w_dex + i];
      data.W2[inl1++] = data.R2[w_dex + i];
      data.W3[inl2++] = data.R3[w_dex + i];
      data.W4[inl3++] = data.R4[w_dex + i];
    }
#else
    for (int i = start; i < end; i += 4) {
      vst1q_s32(data.W1 + inl0, vld1q_s32(data.R1 + w_dex + i));
      vst1q_s32(data.W2 + inl1, vld1q_s32(data.R2 + w_dex + i));
      vst1q_s32(data.W3 + inl2, vld1q_s32(data.R3 + w_dex + i));
      vst1q_s32(data.W4 + inl3, vld1q_s32(data.R4 + w_dex + i));
      inl0 += 4;
      inl1 += 4;
      inl2 += 4;
      inl3 += 4;
    }
#endif
  }

  int start;
  int end;
  int w_dex;
  INT_DP data;
};

void Load_RHS_Data(acc_container& drv, int start_col, int cols, int real_depth,
                   int depth) {
  int* in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int* in1 = drv.mdma->dmas[1].dma_get_inbuffer();
  int* in2 = drv.mdma->dmas[2].dma_get_inbuffer();
  int* in3 = drv.mdma->dmas[3].dma_get_inbuffer();

  int inl0 = 0;
  int inl1 = 0;
  int inl2 = 0;
  int inl3 = 0;

  int offdepth = real_depth * drv.rhs_offset;
  int start_dex = (start_col / 4);
  int* p_rhs_sums1 = reinterpret_cast<int*>(&drv.in_sum1[start_dex]);
  int* p_rhs_sums2 = reinterpret_cast<int*>(&drv.in_sum2[start_dex]);
  int* p_rhs_sums3 = reinterpret_cast<int*>(&drv.in_sum3[start_dex]);
  int* p_rhs_sums4 = reinterpret_cast<int*>(&drv.in_sum4[start_dex]);

  int roundedcols = ((cols + 3) - ((cols + 3) % 4));
  int in_sum_length = roundedcols / 4;
  std::uint32_t h = 1;
  uint32_t l = in_sum_length;
  l = l << 16;
  l += roundedcols * depth / 4;
  in0[inl0++] = h;
  in0[inl0++] = 0;
  in0[inl0++] = l;
  in0[inl0++] = drv.ra;

#ifndef ACC_NEON
  for (int c = 0; c < cols; c += 4) {
    for (int i = 0; i < depth / 4; i++) {
      in0[inl0++] = drv.inb_0[i + drv.in_id];
      in1[inl1++] = drv.inb_1[i + drv.in_id];
      in2[inl2++] = drv.inb_2[i + drv.in_id];
      in3[inl3++] = drv.inb_3[i + drv.in_id];
    }
    drv.in_id += depth / 4;
  }
  for (int i = 0; i < in_sum_length; i++) {
    in0[inl0++] = (p_rhs_sums1[i] + offdepth) * drv.lhs_offset;
    in1[inl1++] = (p_rhs_sums2[i] + offdepth) * drv.lhs_offset;
    in2[inl2++] = (p_rhs_sums3[i] + offdepth) * drv.lhs_offset;
    in3[inl3++] = (p_rhs_sums4[i] + offdepth) * drv.lhs_offset;
  }
#else
  int32x4_t tmp0;
  int32x4_t tmp1;
  int32x4_t tmp2;
  int32x4_t tmp3;
  for (int c = 0; c < cols; c += 4) {
    int* inb0 = drv.inb_0;
    int* inb1 = drv.inb_1;
    int* inb2 = drv.inb_2;
    int* inb3 = drv.inb_3;
    for (int i = 0; i < depth / 4; i += 4) {
      tmp0 = vld1q_s32(inb0 + i + drv.in_id);
      tmp1 = vld1q_s32(inb1 + i + drv.in_id);
      tmp2 = vld1q_s32(inb2 + i + drv.in_id);
      tmp3 = vld1q_s32(inb3 + i + drv.in_id);
      vst1q_s32(in0 + inl0, tmp0);
      vst1q_s32(in1 + inl1, tmp1);
      vst1q_s32(in2 + inl2, tmp2);
      vst1q_s32(in3 + inl3, tmp3);

      inl0 += 4;
      inl1 += 4;
      inl2 += 4;
      inl3 += 4;
    }
    drv.in_id += depth / 4;
  }
  int vin_sum_len = roundDown(in_sum_length, 4);
  const int32_t* tmp_lhs_off =
      reinterpret_cast<const int32_t*>(&drv.lhs_offset);
  const int32_t* tmp_offdepth = reinterpret_cast<const int32_t*>(&offdepth);
  int32x4_t vlhs_lhsoffset = vld1q_dup_s32(tmp_lhs_off);
  int32x4_t vlhs_offdepth = vld1q_dup_s32(tmp_offdepth);
  for (int i = 0; i < vin_sum_len; i += 4) {
    vst1q_s32(in0 + inl0,
              vmulq_s32(vaddq_s32(vld1q_s32(p_rhs_sums1 + i), vlhs_offdepth),
                        vlhs_lhsoffset));
    vst1q_s32(in1 + inl1,
              vmulq_s32(vaddq_s32(vld1q_s32(p_rhs_sums2 + i), vlhs_offdepth),
                        vlhs_lhsoffset));
    vst1q_s32(in2 + inl2,
              vmulq_s32(vaddq_s32(vld1q_s32(p_rhs_sums3 + i), vlhs_offdepth),
                        vlhs_lhsoffset));
    vst1q_s32(in3 + inl3,
              vmulq_s32(vaddq_s32(vld1q_s32(p_rhs_sums4 + i), vlhs_offdepth),
                        vlhs_lhsoffset));
    inl0 += 4;
    inl1 += 4;
    inl2 += 4;
    inl3 += 4;
  }
  for (int i = vin_sum_len; i < in_sum_length; i++) {
    in0[inl0++] = (p_rhs_sums1[i] + offdepth) * drv.lhs_offset;
    in1[inl1++] = (p_rhs_sums2[i] + offdepth) * drv.lhs_offset;
    in2[inl2++] = (p_rhs_sums3[i] + offdepth) * drv.lhs_offset;
    in3[inl3++] = (p_rhs_sums4[i] + offdepth) * drv.lhs_offset;
  }
#endif

  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[1].dma_start_send(inl1);
  drv.mdma->dmas[2].dma_start_send(inl2);
  drv.mdma->dmas[3].dma_start_send(inl3);
  drv.mdma->multi_dma_wait_send();
  drv.lhs_start = true;
}

void Load_LHS_Data(acc_container& drv, int free_buf, int8_t* results, int dcs,
                   int start_row, int rows, int start_col, int cols, int depth,
                   int rcols, int rrows) {
  int offset = drv.dfs[0].dbuf_set[free_buf].offset;
  int* in0 = drv.mdma->dmas[0].dma_get_inbuffer() + (offset / 4);
  int* in1 = drv.mdma->dmas[1].dma_get_inbuffer() + (offset / 4);
  int* in2 = drv.mdma->dmas[2].dma_get_inbuffer() + (offset / 4);
  int* in3 = drv.mdma->dmas[3].dma_get_inbuffer() + (offset / 4);

  int inl0 = 0;
  int inl1 = 0;
  int inl2 = 0;
  int inl3 = 0;

  int w_dex = (drv.w_c / 4);
  int data_length = depth * rows;
  int wt_sums_len = rows / 4;

  uint32_t h = 0;
  uint32_t count = rows;
  count = count << 16;
  count += cols;
  uint32_t l = rows * depth / 4;
  l = l << 16;
  l += wt_sums_len;
  h += depth;
  h = h << 8;
  h += 0;
  h = h << 8;
  h += 0;
  h = h << 1;
  h += 1;
  h = h << 1;
  in0[inl0++] = h;
  in0[inl0++] = count;
  in0[inl0++] = l;

  // int task_count = drv.thread_count;
  int task_count = 1;
  if (task_count > 1) {
    int sta = 0;
    int end = data_length / 16;
    int step = roundUp(end / task_count, 4);
    std::vector<Task*> tasks;
    auto* workers_pool = drv.mt_context->workers_pool();
    INT_DP data =
        INT_DP(in0, in1, in2, in3, drv.wb_0, drv.wb_1, drv.wb_2, drv.wb_3);
    for (int i = 0; i < task_count; i++) {
      int c_end = std::min(end, sta + step);
      tasks.push_back(new Load_LHS_Data_Task(sta, c_end, w_dex, data));
      sta += c_end;
    }
    workers_pool->Execute(tasks);
    inl0 += data_length / 16;
    inl1 += data_length / 16;
    inl2 += data_length / 16;
    inl3 += data_length / 16;
  } else {
#ifndef ACC_NEON
    for (int i = 0; i < data_length / 16; i++) {
      in0[inl0++] = drv.wb_0[w_dex + i];
      in1[inl1++] = drv.wb_1[w_dex + i];
      in2[inl2++] = drv.wb_2[w_dex + i];
      in3[inl3++] = drv.wb_3[w_dex + i];
    }
#else
    for (int i = 0; i < data_length / 16; i += 4) {
      vst1q_s32(in0 + inl0, vld1q_s32(drv.wb_0 + w_dex + i));
      vst1q_s32(in1 + inl1, vld1q_s32(drv.wb_1 + w_dex + i));
      vst1q_s32(in2 + inl2, vld1q_s32(drv.wb_2 + w_dex + i));
      vst1q_s32(in3 + inl3, vld1q_s32(drv.wb_3 + w_dex + i));
      inl0 += 4;
      inl1 += 4;
      inl2 += 4;
      inl3 += 4;
    }
#endif
  }

  int b_c = start_row;
  int crf_c = start_row;
  int crx_c = start_row;
  int start_dex = (start_row / 4);
  int* p_lhs_sums1 = reinterpret_cast<int*>(&drv.wt_sum1[start_dex]);
  int* p_lhs_sums2 = reinterpret_cast<int*>(&drv.wt_sum2[start_dex]);
  int* p_lhs_sums3 = reinterpret_cast<int*>(&drv.wt_sum3[start_dex]);
  int* p_lhs_sums4 = reinterpret_cast<int*>(&drv.wt_sum4[start_dex]);
  for (int i = 0; i < wt_sums_len; i++) {
    in0[inl0++] = (p_lhs_sums1[i] * drv.rhs_offset) + drv.bias[b_c++];
    in1[inl1++] = (p_lhs_sums2[i] * drv.rhs_offset) + drv.bias[b_c++];
    in2[inl2++] = (p_lhs_sums3[i] * drv.rhs_offset) + drv.bias[b_c++];
    in3[inl3++] = (p_lhs_sums4[i] * drv.rhs_offset) + drv.bias[b_c++];

    in0[inl0++] = drv.crf[crf_c++];
    in1[inl1++] = drv.crf[crf_c++];
    in2[inl2++] = drv.crf[crf_c++];
    in3[inl3++] = drv.crf[crf_c++];
    int8_t w0 = drv.crx[crx_c++];
    int8_t w1 = drv.crx[crx_c++];
    int8_t w2 = drv.crx[crx_c++];
    int8_t w3 = drv.crx[crx_c++];
    int8_t ex[] = {w0, w1, w2, w3};
    in0[inl0++] = *(int*)(ex);
  }
  drv.w_c += data_length / 4;
  in0[inl0++] = -1;

  int8_t* res_pointer = results + start_row + start_col * dcs;
  drv.st_params[free_buf].dst = reinterpret_cast<int*>(res_pointer);
  drv.st_params[free_buf].dcs = dcs;
  drv.st_params[free_buf].rows = rows;
  drv.st_params[free_buf].cols = cols;
  drv.st_params[free_buf].rcols = rcols;
  drv.st_params[free_buf].rrows = rrows;
  alloc_dbuf(drv.dfs[0], free_buf, drv.dsr.dID, inl0);
  alloc_dbuf(drv.dfs[1], free_buf, drv.dsr.dID, inl1);
  alloc_dbuf(drv.dfs[2], free_buf, drv.dsr.dID, inl2);
  alloc_dbuf(drv.dfs[3], free_buf, drv.dsr.dID, inl3);
  drv.dsr.dID++;
}

void Store_Results_Task(int start, int end, int rows, int dcs, int8_t* base,
                        int8_t* bo0, int8_t* bo1, int8_t* bo2, int8_t* bo3) {
  int r16 = rows - rows % 16;
  int out0 = start * rows * 4;
  int out1 = start * rows * 4;
  int out2 = start * rows * 4;
  int out3 = start * rows * 4;

#ifndef ACC_NEON
  for (int i = start; i < end; i += 16) {
    for (int j = 0; j < r16; j += 16) {
      for (int k = 0; k < 16; k++) {
        base[(i + 0) * dcs + (j) + k] = bo0[out0++];
        base[(i + 1) * dcs + (j) + k] = bo1[out1++];
        base[(i + 2) * dcs + (j) + k] = bo2[out2++];
        base[(i + 3) * dcs + (j) + k] = bo3[out3++];
      }
      for (int k = 0; k < 16; k++) {
        base[(i + 4) * dcs + (j) + k] = bo0[out0++];
        base[(i + 5) * dcs + (j) + k] = bo1[out1++];
        base[(i + 6) * dcs + (j) + k] = bo2[out2++];
        base[(i + 7) * dcs + (j) + k] = bo3[out3++];
      }
      for (int k = 0; k < 16; k++) {
        base[(i + 8) * dcs + (j) + k] = bo0[out0++];
        base[(i + 9) * dcs + (j) + k] = bo1[out1++];
        base[(i + 10) * dcs + (j) + k] = bo2[out2++];
        base[(i + 11) * dcs + (j) + k] = bo3[out3++];
      }
      for (int k = 0; k < 16; k++) {
        base[(i + 12) * dcs + (j) + k] = bo0[out0++];
        base[(i + 13) * dcs + (j) + k] = bo1[out1++];
        base[(i + 14) * dcs + (j) + k] = bo2[out2++];
        base[(i + 15) * dcs + (j) + k] = bo3[out3++];
      }
    }

    for (int j = r16; j < rows; j++) {
      base[(i + 0) * dcs + j] = bo0[out0++];
      base[(i + 1) * dcs + j] = bo1[out1++];
      base[(i + 2) * dcs + j] = bo2[out2++];
      base[(i + 3) * dcs + j] = bo3[out3++];
    }
    for (int j = r16; j < rows; j++) {
      base[(i + 4) * dcs + j] = bo0[out0++];
      base[(i + 5) * dcs + j] = bo1[out1++];
      base[(i + 6) * dcs + j] = bo2[out2++];
      base[(i + 7) * dcs + j] = bo3[out3++];
    }
    for (int j = r16; j < rows; j++) {
      base[(i + 8) * dcs + j] = bo0[out0++];
      base[(i + 9) * dcs + j] = bo1[out1++];
      base[(i + 10) * dcs + j] = bo2[out2++];
      base[(i + 11) * dcs + j] = bo3[out3++];
    }
    for (int j = r16; j < rows; j++) {
      base[(i + 12) * dcs + j] = bo0[out0++];
      base[(i + 13) * dcs + j] = bo1[out1++];
      base[(i + 14) * dcs + j] = bo2[out2++];
      base[(i + 15) * dcs + j] = bo3[out3++];
    }
  }
#else
  for (int i = start; i < end; i += 16) {
    int di0 = i * dcs;
    int di1 = (i + 1) * dcs;
    int di2 = (i + 2) * dcs;
    int di3 = (i + 3) * dcs;
    int di4 = (i + 4) * dcs;
    int di5 = (i + 5) * dcs;
    int di6 = (i + 6) * dcs;
    int di7 = (i + 7) * dcs;
    int di8 = (i + 8) * dcs;
    int di9 = (i + 9) * dcs;
    int di10 = (i + 10) * dcs;
    int di11 = (i + 11) * dcs;
    int di12 = (i + 12) * dcs;
    int di13 = (i + 13) * dcs;
    int di14 = (i + 14) * dcs;
    int di15 = (i + 15) * dcs;

    for (int j = 0; j < r16; j += 16) {
      vst1q_s8(base + di0 + j, vld1q_s8(bo0 + out0));
      vst1q_s8(base + di1 + j, vld1q_s8(bo1 + out1));
      vst1q_s8(base + di2 + j, vld1q_s8(bo2 + out2));
      vst1q_s8(base + di3 + j, vld1q_s8(bo3 + out3));
      vst1q_s8(base + di4 + j, vld1q_s8(bo0 + out0 + 16));
      vst1q_s8(base + di5 + j, vld1q_s8(bo1 + out1 + 16));
      vst1q_s8(base + di6 + j, vld1q_s8(bo2 + out2 + 16));
      vst1q_s8(base + di7 + j, vld1q_s8(bo3 + out3 + 16));
      vst1q_s8(base + di8 + j, vld1q_s8(bo0 + out0 + 32));
      vst1q_s8(base + di9 + j, vld1q_s8(bo1 + out1 + 32));
      vst1q_s8(base + di10 + j, vld1q_s8(bo2 + out2 + 32));
      vst1q_s8(base + di11 + j, vld1q_s8(bo3 + out3 + 32));
      vst1q_s8(base + di12 + j, vld1q_s8(bo0 + out0 + 48));
      vst1q_s8(base + di13 + j, vld1q_s8(bo1 + out1 + 48));
      vst1q_s8(base + di14 + j, vld1q_s8(bo2 + out2 + 48));
      vst1q_s8(base + di15 + j, vld1q_s8(bo3 + out3 + 48));
      out0 += 64;
      out1 += 64;
      out2 += 64;
      out3 += 64;
    }
    for (int j = r16; j < rows; j++) {
      base[di0 + j] = bo0[out0++];
      base[di1 + j] = bo1[out1++];
      base[di2 + j] = bo2[out2++];
      base[di3 + j] = bo3[out3++];
    }
    for (int j = r16; j < rows; j++) {
      base[di4 + j] = bo0[out0++];
      base[di5 + j] = bo1[out1++];
      base[di6 + j] = bo2[out2++];
      base[di7 + j] = bo3[out3++];
    }
    for (int j = r16; j < rows; j++) {
      base[di8 + j] = bo0[out0++];
      base[di9 + j] = bo1[out1++];
      base[di10 + j] = bo2[out2++];
      base[di11 + j] = bo3[out3++];
    }
    for (int j = r16; j < rows; j++) {
      base[di12 + j] = bo0[out0++];
      base[di13 + j] = bo1[out1++];
      base[di14 + j] = bo2[out2++];
      base[di15 + j] = bo3[out3++];
    }
  }
#endif
}

void Store_Results(acc_container& drv) {
  int r_buf = find_dbuf(drv.dfs[0], drv.dsr.rID);
  int offset = drv.dfs[0].dbuf_set[r_buf].offset;
  dealloc_dbuf(drv.dfs[0], r_buf);
  dealloc_dbuf(drv.dfs[1], r_buf);
  dealloc_dbuf(drv.dfs[2], r_buf);
  dealloc_dbuf(drv.dfs[3], r_buf);
  drv.dsr.rID++;

  struct store_params sp = drv.st_params[r_buf];
  int dcs = sp.dcs;
  int rows = sp.rows;
  int cols = sp.cols;
  int rcols = sp.rcols;
  int rrows = sp.rrows;
  int8_t* base = reinterpret_cast<int8_t*>(sp.dst);

  int* o0 = drv.mdma->dmas[0].dma_get_outbuffer() + (offset / 4);
  int* o1 = drv.mdma->dmas[1].dma_get_outbuffer() + (offset / 4);
  int* o2 = drv.mdma->dmas[2].dma_get_outbuffer() + (offset / 4);
  int* o3 = drv.mdma->dmas[3].dma_get_outbuffer() + (offset / 4);
  int8_t* bo0 = reinterpret_cast<int8_t*>(o0);
  int8_t* bo1 = reinterpret_cast<int8_t*>(o1);
  int8_t* bo2 = reinterpret_cast<int8_t*>(o2);
  int8_t* bo3 = reinterpret_cast<int8_t*>(o3);

  int out0 = 0;
  int out1 = 0;
  int out2 = 0;
  int out3 = 0;

  int acc_cols = 16;
  int acc_rows = 16;
  int rem_rows =
      rows - rows % acc_rows;  // Round down to nearest multiple of 16
  int rcolsr = rcols % acc_cols;
  int dcols = rcols - (rcolsr);

  // int r16 = rows - rows % 16;
  // int rcolsr = rcols % 16;
  // int dcols = rcols - (rcolsr);

  // int start = 0;
  // int mid = roundDown(dcols / 2, 16);
  // int end = dcols;
  // secda_threading st;
  // st.add_thread(std::thread(Store_Results_Task<int>, start, mid, rows, dcs,
  //                           base, bo0, bo1, bo2, bo3));
  // st.add_thread(std::thread(Store_Results_Task<int>, mid, end, rows, dcs,
  // base,
  //                           bo0, bo1, bo2, bo3));
  // st.join_threads();
  // out0 = dcols * rows * 4;
  // out1 = dcols * rows * 4;
  // out2 = dcols * rows * 4;
  // out3 = dcols * rows * 4;

#ifndef ACC_NEON
  for (int i = 0; i < dcols; i += acc_cols) {
    for (int j = 0; j < rem_rows; j += acc_rows) {
      for (int k = 0; k < acc_cols; k++) {
        base[(i + 0) * dcs + (j) + k] = bo0[out0++];
        base[(i + 1) * dcs + (j) + k] = bo1[out1++];
        base[(i + 2) * dcs + (j) + k] = bo2[out2++];
        base[(i + 3) * dcs + (j) + k] = bo3[out3++];
      }
      for (int k = 0; k < acc_cols; k++) {
        base[(i + 4) * dcs + (j) + k] = bo0[out0++];
        base[(i + 5) * dcs + (j) + k] = bo1[out1++];
        base[(i + 6) * dcs + (j) + k] = bo2[out2++];
        base[(i + 7) * dcs + (j) + k] = bo3[out3++];
      }

      if (acc_rows == 8) continue;
      for (int k = 0; k < acc_cols; k++) {
        base[(i + 8) * dcs + (j) + k] = bo0[out0++];
        base[(i + 9) * dcs + (j) + k] = bo1[out1++];
        base[(i + 10) * dcs + (j) + k] = bo2[out2++];
        base[(i + 11) * dcs + (j) + k] = bo3[out3++];
      }
      for (int k = 0; k < acc_cols; k++) {
        base[(i + 12) * dcs + (j) + k] = bo0[out0++];
        base[(i + 13) * dcs + (j) + k] = bo1[out1++];
        base[(i + 14) * dcs + (j) + k] = bo2[out2++];
        base[(i + 15) * dcs + (j) + k] = bo3[out3++];
      }
    }

    for (int j = rem_rows; j < rows; j++) {
      base[(i + 0) * dcs + j] = bo0[out0++];
      base[(i + 1) * dcs + j] = bo1[out1++];
      base[(i + 2) * dcs + j] = bo2[out2++];
      base[(i + 3) * dcs + j] = bo3[out3++];
    }
    for (int j = rem_rows; j < rows; j++) {
      base[(i + 4) * dcs + j] = bo0[out0++];
      base[(i + 5) * dcs + j] = bo1[out1++];
      base[(i + 6) * dcs + j] = bo2[out2++];
      base[(i + 7) * dcs + j] = bo3[out3++];
    }

    if (acc_rows == 8) continue;
    for (int j = rem_rows; j < rows; j++) {
      base[(i + 8) * dcs + j] = bo0[out0++];
      base[(i + 9) * dcs + j] = bo1[out1++];
      base[(i + 10) * dcs + j] = bo2[out2++];
      base[(i + 11) * dcs + j] = bo3[out3++];
    }
    for (int j = rem_rows; j < rows; j++) {
      base[(i + 12) * dcs + j] = bo0[out0++];
      base[(i + 13) * dcs + j] = bo1[out1++];
      base[(i + 14) * dcs + j] = bo2[out2++];
      base[(i + 15) * dcs + j] = bo3[out3++];
    }
  }
#else
  for (int i = 0; i < dcols; i += 16) {
    int di0 = i * dcs;
    int di1 = (i + 1) * dcs;
    int di2 = (i + 2) * dcs;
    int di3 = (i + 3) * dcs;
    int di4 = (i + 4) * dcs;
    int di5 = (i + 5) * dcs;
    int di6 = (i + 6) * dcs;
    int di7 = (i + 7) * dcs;
    int di8 = (i + 8) * dcs;
    int di9 = (i + 9) * dcs;
    int di10 = (i + 10) * dcs;
    int di11 = (i + 11) * dcs;
    int di12 = (i + 12) * dcs;
    int di13 = (i + 13) * dcs;
    int di14 = (i + 14) * dcs;
    int di15 = (i + 15) * dcs;

    for (int j = 0; j < r16; j += 16) {
      vst1q_s8(base + di0 + j, vld1q_s8(bo0 + out0));
      vst1q_s8(base + di1 + j, vld1q_s8(bo1 + out1));
      vst1q_s8(base + di2 + j, vld1q_s8(bo2 + out2));
      vst1q_s8(base + di3 + j, vld1q_s8(bo3 + out3));
      vst1q_s8(base + di4 + j, vld1q_s8(bo0 + out0 + 16));
      vst1q_s8(base + di5 + j, vld1q_s8(bo1 + out1 + 16));
      vst1q_s8(base + di6 + j, vld1q_s8(bo2 + out2 + 16));
      vst1q_s8(base + di7 + j, vld1q_s8(bo3 + out3 + 16));
      vst1q_s8(base + di8 + j, vld1q_s8(bo0 + out0 + 32));
      vst1q_s8(base + di9 + j, vld1q_s8(bo1 + out1 + 32));
      vst1q_s8(base + di10 + j, vld1q_s8(bo2 + out2 + 32));
      vst1q_s8(base + di11 + j, vld1q_s8(bo3 + out3 + 32));
      vst1q_s8(base + di12 + j, vld1q_s8(bo0 + out0 + 48));
      vst1q_s8(base + di13 + j, vld1q_s8(bo1 + out1 + 48));
      vst1q_s8(base + di14 + j, vld1q_s8(bo2 + out2 + 48));
      vst1q_s8(base + di15 + j, vld1q_s8(bo3 + out3 + 48));
      out0 += 64;
      out1 += 64;
      out2 += 64;
      out3 += 64;
    }
    for (int j = r16; j < rows; j++) {
      base[di0 + j] = bo0[out0++];
      base[di1 + j] = bo1[out1++];
      base[di2 + j] = bo2[out2++];
      base[di3 + j] = bo3[out3++];
    }
    for (int j = r16; j < rows; j++) {
      base[di4 + j] = bo0[out0++];
      base[di5 + j] = bo1[out1++];
      base[di6 + j] = bo2[out2++];
      base[di7 + j] = bo3[out3++];
    }
    for (int j = r16; j < rows; j++) {
      base[di8 + j] = bo0[out0++];
      base[di9 + j] = bo1[out1++];
      base[di10 + j] = bo2[out2++];
      base[di11 + j] = bo3[out3++];
    }
    for (int j = r16; j < rows; j++) {
      base[di12 + j] = bo0[out0++];
      base[di13 + j] = bo1[out1++];
      base[di14 + j] = bo2[out2++];
      base[di15 + j] = bo3[out3++];
    }
  }
#endif

  for (int j = 0; j < rem_rows; j += acc_rows) {
    for (int i = 0; i < rcolsr; i++) {
      int8_t* bos;
      int outs;
      if (i % 4 == 0) {
        bos = bo0;
        outs = out0;
      }
      if (i % 4 == 1) {
        bos = bo1;
        outs = out1;
      }
      if (i % 4 == 2) {
        bos = bo2;
        outs = out2;
      }
      if (i % 4 == 3) {
        bos = bo3;
        outs = out3;
      }
      for (int k = 0; k < acc_cols; k++)
        base[(dcols + i) * dcs + j + k] = bos[outs++];
      if (i % 4 == 0) out0 = outs;
      if (i % 4 == 1) out1 = outs;
      if (i % 4 == 2) out2 = outs;
      if (i % 4 == 3) out3 = outs;
    }
  }
  for (int i = 0; i < rcolsr; i++) {
    int8_t* bos;
    int outs;
    if (i % 4 == 0) {
      bos = bo0;
      outs = out0;
    }
    if (i % 4 == 1) {
      bos = bo1;
      outs = out1;
    }
    if (i % 4 == 2) {
      bos = bo2;
      outs = out2;
    }
    if (i % 4 == 3) {
      bos = bo3;
      outs = out3;
    }
    for (int j = rem_rows; j < rows; j++)
      base[(dcols + i) * dcs + j] = bos[outs++];
    if (i % 4 == 0) out0 = outs;
    if (i % 4 == 1) out1 = outs;
    if (i % 4 == 2) out2 = outs;
    if (i % 4 == 3) out3 = outs;
  }
}

void DataHandleComputeL1(acc_container& drv, int8_t* results, int dcs,
                         int start_row, int rows, int start_col, int cols,
                         int depth, int rcols, int rrows) {
  int free_buf = 0;
  if (drv.lhs_start) {
    free_buf = check_for_free_dbuf(drv.dfs[0]);
    Load_LHS_Data(drv, free_buf, results, dcs, start_row, rows, start_col, cols,
                  depth, rcols, rrows);
    drv.Set_Results();
    drv.Start_Transfer();
    drv.lhs_start = false;
  } else {
    bool gemm_done = drv.Check_Done();
    free_buf = check_for_free_dbuf(drv.dfs[0]);
    if (free_buf != -1) {
      Load_LHS_Data(drv, free_buf, results, dcs, start_row, rows, start_col,
                    cols, depth, rcols, rrows);
      if (gemm_done) {
        Store_Results(drv);
        drv.Set_Results();
        drv.Start_Transfer();
      }
    } else {
      if (!gemm_done) drv.Recieve_Results();
      Store_Results(drv);
      if (drv.dsr.dID == drv.dsr.sID) {
        free_buf = check_for_free_dbuf(drv.dfs[0]);
        Load_LHS_Data(drv, free_buf, results, dcs, start_row, rows, start_col,
                      cols, depth, rcols, rrows);
        drv.Set_Results();
        drv.Start_Transfer();
      } else {
        drv.Set_Results();
        drv.Start_Transfer();
        free_buf = check_for_free_dbuf(drv.dfs[0]);
        Load_LHS_Data(drv, free_buf, results, dcs, start_row, rows, start_col,
                      cols, depth, rcols, rrows);
      }
    }
  }
}

void DataHandleCompute(acc_container& drv, int dcs, int depth, int t_depth,
                       int bpl2r, int bpl2c, int cols, int8_params dst_params) {
  prf_start(1);
  drv.t.layer_weight_tile = 0;
  drv.t.layer_input_tile = 0;
  int8_t* results = dst_params.data;
  int acc_imax = 4096 * 16;
  int acc_wmax = 8192 * 16;

  int max_rows = acc_imax / t_depth;
  max_rows = max_rows - (max_rows % 4);
  int row_inc = std::min(std::min(bpl2r, max_rows), 2048);
  int max_cols = acc_wmax / t_depth;
  max_cols = max_cols - (max_cols % 4);
  int col_inc = std::min(std::min(bpl2c, max_cols), 2048);

  for (int o = 0; o < bpl2c; o += col_inc) {
    int os = std::min(col_inc, bpl2c - o);
    int rcols = std::min(col_inc, cols - o);
    drv.w_c = 0;
    Load_RHS_Data(drv, o, os, depth, t_depth);

    for (int d = 0; d < t_depth; d += t_depth) {
      int ds = std::min(t_depth, t_depth - d);
      for (int r = 0; r < bpl2r; r += row_inc) {
        int rs = std::min(row_inc, bpl2r - r);
        int rrows = std::min(row_inc, dst_params.rows - r);
        DataHandleComputeL1(drv, results, dcs, r, rs, o, os, ds, rcols, rrows);
        drv.t.layer_input_tile++;
      }
    }

    while (drv.dsr.dID != drv.dsr.rID) {
      drv.Recieve_Results();
      if (drv.dsr.dID != drv.dsr.sID) {
        drv.Set_Results();
        drv.Start_Transfer();
      }
      Store_Results(drv);
    }
    drv.mdma->multi_dma_change_start_4(0);

    drv.t.layer_weight_tile++;
  }
  prf_end(1, drv.t2.sa_acc);
}

void Entry(acc_container& drv, int8_params lhs_params, int8_params rhs_params,
           int8_params dst_params) {
  int depth = lhs_params.depth;
  int rows = dst_params.rows;
  int cols = dst_params.cols;
  int dcs = dst_params.rows;
  int temp_depth = roundUp(depth, 16);
  int temp_cols = roundUp(cols, 1);
  int temp_rows = roundUp(rows, 4);
  drv.dsr.reset();

#ifdef DELEGATE_VERBOSE
  cout << "===========================" << endl;
  cout << "Pre-ACC Info" << endl;
  cout << "temp_depth: " << temp_depth << " depth: " << depth << endl;
  cout << "temp_cols: " << temp_cols << " cols: " << dst_params.cols << endl;
  cout << "temp_rows: " << temp_rows << " rows: " << dst_params.rows << endl;
  cout << "old_dcs: " << temp_rows << " dcs: " << dcs << endl;
  cout << "===========================" << endl;
#endif

  DataHandleCompute(drv, dcs, depth, temp_depth, temp_rows, temp_cols, cols,
                    dst_params);

#ifdef DELEGATE_DEBUG
  ofstream myfile;
  myfile.open("aData/out_" + std::to_string(drv.t.layer) + "_1.csv");
  int8_t* res_pointer = dst_params.data;
  int index = 0;
  for (int c = 0; c < cols; c++) {
    for (int r = 0; r < rows; r++) {
      myfile << (int)res_pointer[index];
      if (r + 1 < rows) myfile << ",";
      index++;
    }
    myfile << endl;
  }
  myfile.close();
#endif
}
}  // namespace tflite_secda_sa
#endif  // GEMM_DRIVER