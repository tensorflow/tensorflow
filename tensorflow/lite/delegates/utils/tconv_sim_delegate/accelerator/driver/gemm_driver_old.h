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

void Col2im_Mapping(int depth, int height, int width, int filter_h,
                    int filter_w, int pad_t, int pad_l, int pad_b, int pad_r,
                    int stride_h, int stride_w, uint32_t* index_map) {
  int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;
  int h_pad = -pad_t;

  int im_dex = 0;
  int col_dex = 0;
  int map_dex = 0;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      im_dex = (h_pad * width + w_pad) * depth;
      for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
        for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            for (int i = 0; i < depth; ++i) {
              index_map[map_dex++] = im_dex;
              col_dex++;
              im_dex++;
            }
          } else {
            col_dex += depth;
            im_dex += depth;
          }
        }
        im_dex += depth * (width - filter_w);
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

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
  uint32_t h = 1;
  uint32_t l = in_sum_length;
  l = l << 16;
  l += roundedcols * depth / 4;
  in0[inl0++] = h;
  in0[inl0++] = 0;
  in0[inl0++] = l;

  for (int c = 0; c < cols; c += 4) {
    for (int i = 0; i < depth / 4; i++) {
      in0[inl0++] = drv.inb_0[i + drv.in_id];
      in1[inl1++] = drv.inb_1[i + drv.in_id];
      in2[inl2++] = drv.inb_2[i + drv.in_id];
      in3[inl3++] = drv.inb_3[i + drv.in_id];
    }
    drv.in_id += depth / 4;
  }

  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[1].dma_start_send(inl1);
  drv.mdma->dmas[2].dma_start_send(inl2);
  drv.mdma->dmas[3].dma_start_send(inl3);
  drv.mdma->multi_dma_wait_send();
  drv.profile->saveProfile(drv.acc->profiling_vars);
}

void Load_LHS_Data(acc_container& drv, int32_t* results, int dcs, int start_row,
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
  int outlen = cols * rows / 4;

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
    in0[inl0++] = (p_lhs_sums1[i] * drv.rhs_offset);
    in1[inl1++] = (p_lhs_sums2[i] * drv.rhs_offset);
    in2[inl2++] = (p_lhs_sums3[i] * drv.rhs_offset);
    in3[inl3++] = (p_lhs_sums4[i] * drv.rhs_offset);
  }
  drv.w_c += data_length / 4;

  for (int r = start_row; r < start_row + rows; r++) {
    cerr << endl;
    for (int c = start_col; c < start_col + cols; c += 4) {
      cerr << (int)drv.dex_map[(c + 0) * drv.rows + r] << ",";
      cerr << (int)drv.dex_map[(c + 1) * drv.rows + r] << ",";
      cerr << (int)drv.dex_map[(c + 2) * drv.rows + r] << ",";
      cerr << (int)drv.dex_map[(c + 3) * drv.rows + r] << ",";
      in0[inl0++] = drv.dex_map[(c + 0) * drv.rows + r];
      in1[inl1++] = drv.dex_map[(c + 1) * drv.rows + r];
      in2[inl2++] = drv.dex_map[(c + 2) * drv.rows + r];
      in3[inl3++] = drv.dex_map[(c + 3) * drv.rows + r];
    }
  }

  // for (int c = start_col; c < cols; c++) {
  //   cerr << endl;
  //   for (int r = start_row; r < start_row + rows; r += 4) {
  //     cerr << (int)drv.dex_map[c * drv.rows + r + 0] << ",";
  //     cerr << (int)drv.dex_map[c * drv.rows + r + 1] << ",";
  //     cerr << (int)drv.dex_map[c * drv.rows + r + 2] << ",";
  //     cerr << (int)drv.dex_map[c * drv.rows + r + 3] << ",";
  //     in0[inl0++] = drv.dex_map[c * drv.rows + r + 0];
  //     in1[inl1++] = drv.dex_map[c * drv.rows + r + 1];
  //     in2[inl2++] = drv.dex_map[c * drv.rows + r + 2];
  //     in3[inl3++] = drv.dex_map[c * drv.rows + r + 3];
  //   }
  // }

  in0[inl0++] = -1;

  int32_t* res_pointer = results + start_row + start_col * dcs;
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
  int32_t* base = reinterpret_cast<int32_t*>(sp.dst);
  int* o0 = drv.mdma->dmas[0].dma_get_outbuffer();
  int* o1 = drv.mdma->dmas[1].dma_get_outbuffer();
  int* o2 = drv.mdma->dmas[2].dma_get_outbuffer();
  int* o3 = drv.mdma->dmas[3].dma_get_outbuffer();
  int32_t* bo0 = reinterpret_cast<int32_t*>(o0);
  int32_t* bo1 = reinterpret_cast<int32_t*>(o1);
  int32_t* bo2 = reinterpret_cast<int32_t*>(o2);
  int32_t* bo3 = reinterpret_cast<int32_t*>(o3);
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

void Store_Results_COL2IM(acc_container& drv) {
  int* in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  in0[inl0++] = -2;
  in0[inl0++] = drv.output_depth;
  in0[inl0++] = drv.output_height * drv.output_width;
  in0[inl0++] = drv.ra;
  for (int c = 0; c < drv.output_depth; c++) {
    in0[inl0++] = drv.bias[c];
    in0[inl0++] = drv.crf[c];
    in0[inl0++] = drv.crx[c];
  }

  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[0].dma_wait_send();
  drv.mdma->multi_dma_start_recv();
  drv.mdma->multi_dma_wait_recv();

  int* o0 = drv.mdma->dmas[0].dma_get_outbuffer();
  int* o1 = drv.mdma->dmas[1].dma_get_outbuffer();
  int* o2 = drv.mdma->dmas[2].dma_get_outbuffer();
  int* o3 = drv.mdma->dmas[3].dma_get_outbuffer();
  int32_t* bo0 = reinterpret_cast<int32_t*>(o0);
  int32_t* bo1 = reinterpret_cast<int32_t*>(o1);
  int32_t* bo2 = reinterpret_cast<int32_t*>(o2);
  int32_t* bo3 = reinterpret_cast<int32_t*>(o3);

  int i = 0;
  for (int c = 0; c < drv.cols; c++) {
    cerr << endl;
    for (int r = 0; r < drv.rows / 4; r++) {
      cerr << (int)bo0[i] << ",";
      cerr << (int)bo1[i] << ",";
      cerr << (int)bo2[i] << ",";
      cerr << (int)bo3[i++] << ",";
    }
  }
}

void Load_LHS_Compute_Store(acc_container& drv, int32_t* results, int dcs,
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
              int bpl2c, int cols, int32_params dst_params) {
  drv.t.layer_weight_tile = 0;
  drv.t.layer_input_tile = 0;
  int32_t* results = dst_params.data;
  int acc_imax = 2048 * 16;
  int acc_wmax = 8192 * 16;

  int max_rows = acc_imax / t_depth;
  max_rows = max_rows - (max_rows % 4);
  int row_inc = std::min(std::min(bpl2r, max_rows), 16);
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
  Store_Results_COL2IM(drv);
}

void Entry(acc_container& drv, int8_params lhs_params, int8_params rhs_params,
           int32_params dst_params) {
  int depth = lhs_params.depth;
  int rows = dst_params.rows;
  int cols = dst_params.cols;
  int dcs = dst_params.rows;

  int temp_depth = roundUp(depth, 16);
  int temp_cols = roundUp(cols, 2);
  int temp_rows = roundUp(rows, 4);

  uint32_t dex_map[rows * cols];
  drv.rows = rows;
  drv.cols = cols;

  Col2im_Mapping(drv.output_depth, drv.output_height, drv.output_width,
                 drv.filter_height, drv.filter_width, drv.padding_top,
                 drv.padding_left, drv.padding_bottom, drv.padding_right,
                 drv.stride_height, drv.stride_width, dex_map);

  drv.dex_map = dex_map;
  int index = 0;
  for (int c = 0; c < cols; c++) {
    cerr << endl;
    for (int r = 0; r < rows; r++) {
      cerr << (int)dex_map[index] << ",";
      index++;
    }
  }

#ifdef DELEGATE_VERBOSE
  cout << "VM_CUSTOM" << endl;
  cout << "===========================" << endl;
  cout << "Pre-ACC Info" << endl;
  cout << "temp_depth: " << temp_depth << " depth: " << depth << endl;
  cout << "temp_cols: " << temp_cols << " cols: " << dst_params.cols << endl;
  cout << "temp_rows: " << temp_rows << " rows: " << dst_params.rows << endl;
  cout << "old_dcs: " << temp_rows << " dcs: " << dcs << endl;
  cout << "===========================" << endl;
#endif

  TileGEMM(drv, dcs, depth, temp_depth, temp_rows, temp_cols, cols, dst_params);

#ifdef DELEGATE_DEBUG
  mkdir("aData", 0777);
  ofstream myfile;
  myfile.open("aData/out_vm_" + std::to_string(drv.t.layer) + "_1.csv");
  int32_t* res_pointer = dst_params.data;
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

}  // namespace tflite_vmsim
#endif  // GEMM_DRIVER
