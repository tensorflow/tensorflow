#ifndef GEMM_DRIVER
#define GEMM_DRIVER

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <strstream>
#include <typeinfo>

#include "acc_container.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"

// GEMM_Driver for simulated VM acccelerator
namespace tflite_vmsim {

// Previously called Load_RHS_Data
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

  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[1].dma_start_send(inl1);
  drv.mdma->dmas[2].dma_start_send(inl2);
  drv.mdma->dmas[3].dma_start_send(inl3);
  drv.mdma->multi_dma_wait_send();
  drv.profile->saveProfile(drv.acc->profiling_vars);
}

void Load_LHS_Data(acc_container& drv, int8_t* results, int dcs, int start_row,
                   int rows, int start_col, int cols, int depth, int rcols,
                   int rrows) {
  int* in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int* in1 = drv.mdma->dmas[1].dma_get_inbuffer();
  int* in2 = drv.mdma->dmas[2].dma_get_inbuffer();
  int* in3 = drv.mdma->dmas[3].dma_get_inbuffer();

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

  for (int i = 0; i < data_length / 16; i++) {
    in0[inl0++] = drv.wb_0[w_dex + i];
    in1[inl1++] = drv.wb_1[w_dex + i];
    in2[inl2++] = drv.wb_2[w_dex + i];
    in3[inl3++] = drv.wb_3[w_dex + i];
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
  drv.st_params.dst = reinterpret_cast<int*>(res_pointer);
  drv.st_params.dcs = dcs;
  drv.st_params.rows = rows;
  drv.st_params.cols = cols;
  drv.st_params.rcols = rcols;
  drv.st_params.rrows = rrows;

  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[1].dma_start_send(inl1);
  drv.mdma->dmas[2].dma_start_send(inl2);
  drv.mdma->dmas[3].dma_start_send(inl3);
  drv.mdma->multi_dma_wait_send();
  drv.profile->saveProfile(drv.acc->profiling_vars);
}

void Store_Results(acc_container& drv) {
  struct store_params sp = drv.st_params;
  int dcs = sp.dcs;
  int rows = sp.rows;
  int cols = sp.cols;
  int rcols = sp.rcols;
  int rrows = sp.rrows;
  int8_t* base = reinterpret_cast<int8_t*>(sp.dst);
  int* o0 = drv.mdma->dmas[0].dma_get_outbuffer();
  int* o1 = drv.mdma->dmas[1].dma_get_outbuffer();
  int* o2 = drv.mdma->dmas[2].dma_get_outbuffer();
  int* o3 = drv.mdma->dmas[3].dma_get_outbuffer();
  int8_t* bo0 = reinterpret_cast<int8_t*>(o0);
  int8_t* bo1 = reinterpret_cast<int8_t*>(o1);
  int8_t* bo2 = reinterpret_cast<int8_t*>(o2);
  int8_t* bo3 = reinterpret_cast<int8_t*>(o3);
  int out0 = 0;
  int out1 = 0;
  int out2 = 0;
  int out3 = 0;
  int dcols = rcols - (rcols % 4);
  int rowsdiff = rows - rrows;
  int r16 = rrows - rrows % 16;

  for (int i = 0; i < dcols; i += 4) {
    for (int j = 0; j < rrows; j++) {
      base[(i + 0) * dcs + j] = bo0[out0++];
      int tk = base[(i + 0) * dcs + j];
      base[(i + 1) * dcs + j] = bo1[out1++];
      base[(i + 2) * dcs + j] = bo2[out2++];
      base[(i + 3) * dcs + j] = bo3[out3++];
    }
    out0 += rowsdiff;
    out1 += rowsdiff;
    out2 += rowsdiff;
    out3 += rowsdiff;
  }

  if ((rcols % 4) == 3) {
    for (int j = 0; j < rrows; j++) {
      base[(dcols + 0) * dcs + j] = bo0[out0++];
      base[(dcols + 1) * dcs + j] = bo1[out1++];
      base[(dcols + 2) * dcs + j] = bo2[out2++];
    }
    out0 += rowsdiff;
    out1 += rowsdiff;
    out2 += rowsdiff;
  } else if ((rcols % 4) == 2) {
    for (int j = 0; j < rrows; j++) {
      base[(dcols + 0) * dcs + j] = bo0[out0++];
      base[(dcols + 1) * dcs + j] = bo1[out1++];
    }
    out0 += rowsdiff;
    out1 += rowsdiff;
  } else if ((rcols % 4) == 1) {
    for (int j = 0; j < rrows; j++) {
      base[(dcols + 0) * dcs + j] = bo0[out0++];
    }
    out0 += rowsdiff;
  }
}

void Load_LHS_Compute_Store(acc_container& drv, int8_t* results, int dcs,
                            int start_row, int rows, int start_col, int cols,
                            int depth, int rcols, int rrows) {
  Load_LHS_Data(drv, results, dcs, start_row, rows, start_col, cols, depth,
                rcols, rrows);

  drv.mdma->multi_dma_start_recv();
  drv.mdma->multi_dma_wait_recv();
  drv.profile->saveProfile(drv.acc->profiling_vars);
  Store_Results(drv);
}

void TileGEMM(acc_container& drv, int dcs, int depth, int t_depth, int bpl2r,
              int bpl2c, int cols, int8_params dst_params) {
  drv.t.layer_weight_tile = 0;
  drv.t.layer_input_tile = 0;
  int8_t* results = dst_params.data;
  int acc_imax = 2048 * 16;
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
        Load_LHS_Compute_Store(drv, results, dcs, r, rs, o, os, ds, rcols,
                               rrows);
        drv.t.layer_input_tile++;
      }
    }
    drv.t.layer_weight_tile++;
  }
}

void Entry(acc_container& drv, int8_params lhs_params, int8_params rhs_params,
           int8_params dst_params) {
  int depth = lhs_params.depth;
  int rows = dst_params.rows;
  int cols = dst_params.cols;
  int dcs = dst_params.rows;

  int rounded_depth = roundUp(depth, 16);
  int rounded_cols = roundUp(cols, 2);
  int rounded_rows = roundUp(rows, 4);

#ifdef DELEGATE_VERBOSE
  cout << "VM" << endl;
  cout << "===========================" << endl;
  cout << "Pre-ACC Info" << endl;
  cout << "rounded_depth: " << rounded_depth << " depth: " << depth << endl;
  cout << "rounded_cols: " << rounded_cols << " cols: " << dst_params.cols
       << endl;
  cout << "rounded_rows: " << rounded_rows << " rows: " << dst_params.rows
       << endl;
  cout << "old_dcs: " << rounded_rows << " dcs: " << dcs << endl;
  cout << "===========================" << endl;
#endif

  TileGEMM(drv, dcs, depth, rounded_depth, rounded_rows, rounded_cols, cols,
           dst_params);

#ifdef DELEGATE_DEBUG
  mkdir("aData", 0777);
  ofstream myfile;
  myfile.open("aData/out_vm_" + std::to_string(drv.t.layer) + "_1.csv");
  int8_t* res_pointer = dst_params.data;
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

// dcs is used to calculate correct striding when accessing the output tensor

// Change all code to support the following:
// input =  depth * rows 
// fliter =  cols * depth
// output =  cols * rows

// Remove the need of anything but drv for Entry

}  // namespace tflite_vmsim
#endif  // GEMM_DRIVER
