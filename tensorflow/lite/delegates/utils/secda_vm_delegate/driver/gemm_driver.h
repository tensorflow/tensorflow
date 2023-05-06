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

// #define dma_in0 0x16000000
// #define dma_in1 0x18000000
// #define dma_in2 0x1a000000
// #define dma_in3 0x1c000000
namespace tflite_secda_vm {

struct Load_LHS_Data_Task : Task {
  Load_LHS_Data_Task(int _start, int _end, size_t _w_dex, INT_DP &_data)
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

void Load_Input_Data(acc_container &drv, int start_row, int rows_step,
                     int depth, int rdepth) {
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int *in1 = drv.mdma->dmas[1].dma_get_inbuffer();
  int *in2 = drv.mdma->dmas[2].dma_get_inbuffer();
  int *in3 = drv.mdma->dmas[3].dma_get_inbuffer();

  int inl0 = 0;
  int inl1 = 0;
  int inl2 = 0;
  int inl3 = 0;

  int offdepth = depth * drv.rhs_offset;
  int start_dex = (start_row / 4);
  int *p_rhs_sums1 = reinterpret_cast<int *>(&drv.in_sum1[start_dex]);
  int *p_rhs_sums2 = reinterpret_cast<int *>(&drv.in_sum2[start_dex]);
  int *p_rhs_sums3 = reinterpret_cast<int *>(&drv.in_sum3[start_dex]);
  int *p_rhs_sums4 = reinterpret_cast<int *>(&drv.in_sum4[start_dex]);

  int rrow_steps = ((rows_step + 3) - ((rows_step + 3) % 4));
  int in_sum_length = rrow_steps / 4;
  uint32_t h = 1;
  uint32_t l = in_sum_length;
  l = l << 16;
  l += rrow_steps * rdepth / 4;
  in0[inl0++] = h;
  in0[inl0++] = 0;
  in0[inl0++] = l;
  in0[inl0++] = drv.ra;

#ifndef ACC_NEON
  for (int c = 0; c < rows_step; c += 4) {
    for (int i = 0; i < rdepth / 4; i++) {
      in0[inl0++] = drv.inb_0[i + drv.in_id];
      in1[inl1++] = drv.inb_1[i + drv.in_id];
      in2[inl2++] = drv.inb_2[i + drv.in_id];
      in3[inl3++] = drv.inb_3[i + drv.in_id];
    }
    drv.in_id += rdepth / 4;
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
  for (int r = 0; r < rows_step; r += 4) {
    int *inb0 = drv.inb_0;
    int *inb1 = drv.inb_1;
    int *inb2 = drv.inb_2;
    int *inb3 = drv.inb_3;
    for (int i = 0; i < rdepth / 4; i += 4) {
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
    drv.in_id += rdepth / 4;
  }
  int vin_sum_len = roundDown(in_sum_length, 4);
  const int32_t *tmp_lhs_off =
      reinterpret_cast<const int32_t *>(&drv.lhs_offset);
  const int32_t *tmp_offdepth = reinterpret_cast<const int32_t *>(&offdepth);
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

void Load_Weight_Data(acc_container &drv, int free_buf, int8_t *results,
                      int output_stride, int c, int rcols_step, int r,
                      int rrows_step, int rdepth_step, int rows_step,
                      int cols_step) {
  int offset = drv.dfs[0].dbuf_set[free_buf].offset;
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer() + (offset / 4);
  int *in1 = drv.mdma->dmas[1].dma_get_inbuffer() + (offset / 4);
  int *in2 = drv.mdma->dmas[2].dma_get_inbuffer() + (offset / 4);
  int *in3 = drv.mdma->dmas[3].dma_get_inbuffer() + (offset / 4);

  int inl0 = 0;
  int inl1 = 0;
  int inl2 = 0;
  int inl3 = 0;

  int w_dex = (drv.w_c / 4);
  int data_length = rdepth_step * rcols_step;
  int wt_sums_len = rcols_step / 4;

  uint32_t h = 0;
  uint32_t count = rcols_step;
  count = count << 16;
  count += rrows_step;
  uint32_t l = rcols_step * rdepth_step / 4;
  l = l << 16;
  l += wt_sums_len;
  h += rdepth_step;
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
  // int task_count = 1;
  // if (task_count > 1) {
  //   int sta = 0;
  //   int end = data_length / 16;
  //   int step = roundUp(end / task_count, 4);
  //   std::vector<Task *> tasks;
  //   auto *workers_pool = drv.mt_context->workers_pool();
  //   INT_DP data =
  //       INT_DP(in0, in1, in2, in3, drv.wb_0, drv.wb_1, drv.wb_2, drv.wb_3);
  //   for (int i = 0; i < task_count; i++) {
  //     int c_end = std::min(end, sta + step);
  //     tasks.push_back(new Load_LHS_Data_Task(sta, c_end, w_dex, data));
  //     sta += c_end;
  //   }
  //   workers_pool->Execute(tasks);
  //   inl0 += data_length / 16;
  //   inl1 += data_length / 16;
  //   inl2 += data_length / 16;
  //   inl3 += data_length / 16;
  // } else {
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
  // }

  int b_c = c;
  int crf_c = c;
  int crx_c = c;
  int start_dex = (c / 4);
  int *wsums1 = reinterpret_cast<int *>(&drv.wt_sum1[start_dex]);
  int *wsums2 = reinterpret_cast<int *>(&drv.wt_sum2[start_dex]);
  int *wsums3 = reinterpret_cast<int *>(&drv.wt_sum3[start_dex]);
  int *wsums4 = reinterpret_cast<int *>(&drv.wt_sum4[start_dex]);
  for (int i = 0; i < wt_sums_len; i++) {
    in0[inl0++] = (wsums1[i] * drv.rhs_offset) + drv.bias[b_c++];
    in1[inl1++] = (wsums2[i] * drv.rhs_offset) + drv.bias[b_c++];
    in2[inl2++] = (wsums3[i] * drv.rhs_offset) + drv.bias[b_c++];
    in3[inl3++] = (wsums4[i] * drv.rhs_offset) + drv.bias[b_c++];

    in0[inl0++] = drv.crf[crf_c++];
    in1[inl1++] = drv.crf[crf_c++];
    in2[inl2++] = drv.crf[crf_c++];
    in3[inl3++] = drv.crf[crf_c++];
    int8_t w0 = drv.crx[crx_c++];
    int8_t w1 = drv.crx[crx_c++];
    int8_t w2 = drv.crx[crx_c++];
    int8_t w3 = drv.crx[crx_c++];
    int8_t ex[] = {w0, w1, w2, w3};
    in0[inl0++] = *(int *)(ex);
  }
  drv.w_c += data_length / 4;
  in0[inl0++] = -1;

  int8_t *res_pointer = results + c + r * output_stride;
  drv.st_params[free_buf].dst = reinterpret_cast<int *>(res_pointer);
  drv.st_params[free_buf].dcs = output_stride;
  drv.st_params[free_buf].cols = rcols_step;
  drv.st_params[free_buf].rows = rrows_step;
  drv.st_params[free_buf].rrows = rows_step;
  drv.st_params[free_buf].rcols = cols_step;
  alloc_dbuf(drv.dfs[0], free_buf, drv.dsr.dID, inl0);
  alloc_dbuf(drv.dfs[1], free_buf, drv.dsr.dID, inl1);
  alloc_dbuf(drv.dfs[2], free_buf, drv.dsr.dID, inl2);
  alloc_dbuf(drv.dfs[3], free_buf, drv.dsr.dID, inl3);
  drv.dsr.dID++;
}

void Store_Results(acc_container &drv) {
  int r_buf = find_dbuf(drv.dfs[0], drv.dsr.rID);
  int offset = drv.dfs[0].dbuf_set[r_buf].offset;
  dealloc_dbuf(drv.dfs[0], r_buf);
  dealloc_dbuf(drv.dfs[1], r_buf);
  dealloc_dbuf(drv.dfs[2], r_buf);
  dealloc_dbuf(drv.dfs[3], r_buf);
  drv.dsr.rID++;

  struct store_params sp = drv.st_params[r_buf];
  int output_stride = sp.dcs;
  int rcols_step = sp.cols;
  int rows_step = sp.rrows;
  int cols_step = sp.rcols;
  int8_t *base = reinterpret_cast<int8_t *>(sp.dst);
  int *o0 = drv.mdma->dmas[0].dma_get_outbuffer() + (offset / 4);
  int *o1 = drv.mdma->dmas[1].dma_get_outbuffer() + (offset / 4);
  int *o2 = drv.mdma->dmas[2].dma_get_outbuffer() + (offset / 4);
  int *o3 = drv.mdma->dmas[3].dma_get_outbuffer() + (offset / 4);
  int8_t *bo0 = reinterpret_cast<int8_t *>(o0);
  int8_t *bo1 = reinterpret_cast<int8_t *>(o1);
  int8_t *bo2 = reinterpret_cast<int8_t *>(o2);
  int8_t *bo3 = reinterpret_cast<int8_t *>(o3);
  int out0 = 0;
  int out1 = 0;
  int out2 = 0;
  int out3 = 0;
  int drows = rows_step - (rows_step % 4);
  int colsr = rcols_step - cols_step;
  int unrolled_cols = cols_step - cols_step % 16;

#ifndef ACC_NEON
  for (int i = 0; i < drows; i += 4) {
    for (int j = 0; j < cols_step; j++) {
      base[(i + 0) * output_stride + j] = bo0[out0++];
      base[(i + 1) * output_stride + j] = bo1[out1++];
      base[(i + 2) * output_stride + j] = bo2[out2++];
      base[(i + 3) * output_stride + j] = bo3[out3++];
    }
    out0 += colsr;
    out1 += colsr;
    out2 += colsr;
    out3 += colsr;
  }

#else
  for (int i = 0; i < drows; i += 4) {
    int8x16_t tmp0;
    int8x16_t tmp1;
    int8x16_t tmp2;
    int8x16_t tmp3;
    int di0 = i * output_stride;
    int di1 = (i + 1) * output_stride;
    int di2 = (i + 2) * output_stride;
    int di3 = (i + 3) * output_stride;
    for (int j = 0; j < unrolled_cols; j += 16) {
      tmp0 = vld1q_s8(bo0 + out0);
      tmp1 = vld1q_s8(bo1 + out1);
      tmp2 = vld1q_s8(bo2 + out2);
      tmp3 = vld1q_s8(bo3 + out3);
      vst1q_s8(base + di0 + j, tmp0);
      vst1q_s8(base + di1 + j, tmp1);
      vst1q_s8(base + di2 + j, tmp2);
      vst1q_s8(base + di3 + j, tmp3);
      out0 += 16;
      out1 += 16;
      out2 += 16;
      out3 += 16;
    }
    for (int j = unrolled_cols; j < cols_step; j++) {
      base[di0 + j] = bo0[out0++];
      base[di1 + j] = bo1[out1++];
      base[di2 + j] = bo2[out2++];
      base[di3 + j] = bo3[out3++];
    }
    out0 += colsr;
    out1 += colsr;
    out2 += colsr;
    out3 += colsr;
  }
#endif

  if ((rows_step % 4) == 3) {
    for (int j = 0; j < cols_step; j++) {
      base[(drows + 0) * output_stride + j] = bo0[out0++];
      base[(drows + 1) * output_stride + j] = bo1[out1++];
      base[(drows + 2) * output_stride + j] = bo2[out2++];
    }
    out0 += colsr;
    out1 += colsr;
    out2 += colsr;
  } else if ((rows_step % 4) == 2) {
    for (int j = 0; j < cols_step; j++) {
      base[(drows + 0) * output_stride + j] = bo0[out0++];
      base[(drows + 1) * output_stride + j] = bo1[out1++];
    }
    out0 += colsr;
    out1 += colsr;
  } else if ((rows_step % 4) == 1) {
    for (int j = 0; j < cols_step; j++) {
      base[(drows + 0) * output_stride + j] = bo0[out0++];
    }
    out0 += colsr;
  }
}

void Load_Weight_Compute_Store(acc_container &drv, int8_t *results,
                               int output_stride, int c, int rcols_step, int r,
                               int rrows_step, int rdepth_step, int rows_step,
                               int cols_step) {
  int free_buf = 0;
  if (drv.lhs_start) {
    free_buf = check_for_free_dbuf(drv.dfs[0]);
    Load_Weight_Data(drv, free_buf, results, output_stride, c, rcols_step, r,
                     rrows_step, rdepth_step, rows_step, cols_step);
    drv.Set_Results();
    drv.Start_Transfer();
    drv.lhs_start = false;
  } else {
    bool gemm_done = drv.Check_Done();
    free_buf = check_for_free_dbuf(drv.dfs[0]);
    if (free_buf != -1) {
      Load_Weight_Data(drv, free_buf, results, output_stride, c, rcols_step, r,
                       rrows_step, rdepth_step, rows_step, cols_step);
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
        Load_Weight_Data(drv, free_buf, results, output_stride, c, rcols_step,
                         r, rrows_step, rdepth_step, rows_step, cols_step);
        drv.Set_Results();
        drv.Start_Transfer();
      } else {
        drv.Set_Results();
        drv.Start_Transfer();
        free_buf = check_for_free_dbuf(drv.dfs[0]);
        Load_Weight_Data(drv, free_buf, results, output_stride, c, rcols_step,
                         r, rrows_step, rdepth_step, rows_step, cols_step);
      }
    }
  }
}

void TileGEMM(acc_container &drv, int output_stride, int depth, int rdepth,
              int rows, int rrows, int cols, int rcols, int8_t *results) {
  prf_start(1);
  drv.t.layer_weight_tile = 0;
  drv.t.layer_input_tile = 0;
  int acc_weight_buffer_size = 2048 * 16;
  int acc_input_buffer_size = 8192 * 16;
  int max_cols = acc_weight_buffer_size / rdepth;
  max_cols = max_cols - (max_cols % 4);
  int col_inc = std::min(std::min(rcols, max_cols), 2048);
  int max_rows = acc_input_buffer_size / rdepth;
  max_rows = max_rows - (max_rows % 4);
  int row_inc = std::min(std::min(rrows, max_rows), 2048);

  for (int r = 0; r < rrows; r += row_inc) {
    int rrows_step = std::min(row_inc, rrows - r);
    int rows_step = std::min(row_inc, rows - r);
    drv.w_c = 0;
    // Load Inputs into the accelerator
    Load_Input_Data(drv, r, rrows_step, depth, rdepth);
    for (int c = 0; c < rcols; c += col_inc) {
      int rcols_step = std::min(col_inc, rcols - c);
      int cols_step = std::min(col_inc, cols - c);
      Load_Weight_Compute_Store(drv, results, output_stride, c, rcols_step, r,
                                rrows_step, rdepth, rows_step, cols_step);
      drv.t.layer_weight_tile++;
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
    drv.t.layer_input_tile++;
  }
  prf_end(1, drv.t2.vm_acc);
}

void Entry(acc_container &drv, int8_t *dst) {
  int rows = drv.rows;
  int cols = drv.cols;
  int depth = drv.depth;
  int rrows = roundUp(drv.rows, 2);
  int rcols = roundUp(drv.cols, 4);
  int rdepth = roundUp(drv.depth, 16);
  int output_stride = drv.cols;
  drv.dsr.reset();

#ifdef DELEGATE_VERBOSE
  cerr << "VM" << endl;
  cerr << "===========================" << endl;
  cerr << "Pre-ACC Info" << endl;
  cerr << "rdepth: " << rdepth << " depth: " << depth << endl;
  cerr << "rcols: " << rcols << " cols: " << cols << endl;
  cerr << "rrows: " << rrows << " rows: " << rows << endl;
  cerr << "output_stride: " << output_stride << endl;
  cerr << "===========================" << endl;
#endif

  TileGEMM(drv, output_stride, depth, rdepth, rows, rrows, cols, rcols, dst);

#ifdef DELEGATE_DEBUG
  mkdir("aData", 0777);
  ofstream myfile;
  myfile.open("aData/" + std::to_string(drv.t.layer) + "_out_acc_.csv");
  int8_t *res_pointer = dst;
  int index = 0;
  for (int c = 0; c < cols; c++) {
    myfile << endl;
    for (int r = 0; r < rows; r++) {
      myfile << (int)res_pointer[index] << ",";
      index++;
    }
  }
  myfile.close();
#endif
}
} // namespace tflite_secda_vm
#endif // GEMM_DRIVER