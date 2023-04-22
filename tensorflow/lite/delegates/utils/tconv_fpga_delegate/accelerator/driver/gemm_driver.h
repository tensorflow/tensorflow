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

// GEMM_Driver for VM acccelerator
namespace tflite_vm_tconv_fpga {

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
            for (int i = 0; i < depth; ++i) {
              index_map[map_dex++] = 32767;
              im_dex++;
            }
          }
        }
        im_dex += depth * (width - filter_w);
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

void pad_dexmap(int rows, int cols, int rrow, int ccols, uint32_t* map,
                uint32_t* pmap) {
  for (int i = 0; i < rrow; i++) {
    for (int j = 0; j < ccols; j++) {
      if (i >= rows || j >= cols)
        pmap[i * ccols + j] = 32767;
      else
        pmap[i * ccols + j] = map[i * cols + j];
    }
  }
}

void Load_Input_Data(acc_container& drv, int start_row, int rows_step,
                     int depth, int rdepth) {
  int* in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int* in1 = drv.mdma->dmas[1].dma_get_inbuffer();
  int* in2 = drv.mdma->dmas[2].dma_get_inbuffer();
  int* in3 = drv.mdma->dmas[3].dma_get_inbuffer();

  int inl0 = 0;
  int inl1 = 0;
  int inl2 = 0;
  int inl3 = 0;

  int rrow_steps = ((rows_step + 3) - ((rows_step + 3) % 4));
  int in_sum_length = rrow_steps / 4;
  uint32_t h = 1;
  uint32_t l = in_sum_length;
  l = l << 16;
  l += rrow_steps * rdepth / 4;
  in0[inl0++] = h;
  in0[inl0++] = 0;
  in0[inl0++] = l;

  for (int c = 0; c < rows_step; c += 4) {
    for (int i = 0; i < rdepth / 4; i++) {
      in0[inl0++] = drv.inb_0[i + drv.in_id];
      in1[inl1++] = drv.inb_1[i + drv.in_id];
      in2[inl2++] = drv.inb_2[i + drv.in_id];
      in3[inl3++] = drv.inb_3[i + drv.in_id];
    }
    drv.in_id += rdepth / 4;
  }

  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[1].dma_start_send(inl1);
  drv.mdma->dmas[2].dma_start_send(inl2);
  drv.mdma->dmas[3].dma_start_send(inl3);
  drv.mdma->multi_dma_wait_send();
}

void Load_Weight_Data(acc_container& drv, int32_t* gemm_dst, int output_stride,
                      int c, int rcols_step, int r, int rrows_step,
                      int rdepth_step, int rows_step, int cols_step) {
  int* in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int* in1 = drv.mdma->dmas[1].dma_get_inbuffer();
  int* in2 = drv.mdma->dmas[2].dma_get_inbuffer();
  int* in3 = drv.mdma->dmas[3].dma_get_inbuffer();

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

  for (int i = 0; i < data_length / 16; i++) {
    in0[inl0++] = drv.wb_0[w_dex + i];
    in1[inl1++] = drv.wb_1[w_dex + i];
    in2[inl2++] = drv.wb_2[w_dex + i];
    in3[inl3++] = drv.wb_3[w_dex + i];
  }

  int b_c = c;
  int crf_c = c;
  int crx_c = c;
  int start_dex = (c / 4);
  int* wsums1 = reinterpret_cast<int*>(&drv.wt_sum1[start_dex]);
  int* wsums2 = reinterpret_cast<int*>(&drv.wt_sum2[start_dex]);
  int* wsums3 = reinterpret_cast<int*>(&drv.wt_sum3[start_dex]);
  int* wsums4 = reinterpret_cast<int*>(&drv.wt_sum4[start_dex]);

  for (int i = 0; i < wt_sums_len; i++) {
    in0[inl0++] = (wsums1[i] * drv.rhs_offset);
    in1[inl1++] = (wsums2[i] * drv.rhs_offset);
    in2[inl2++] = (wsums3[i] * drv.rhs_offset);
    in3[inl3++] = (wsums4[i] * drv.rhs_offset);
  }
  drv.w_c += data_length / 4;

  for (int j = r; j < r + rrows_step; j += 4) {
    for (int i = c; i < c + rcols_step; i++) {
      in0[inl0++] = drv.dex_map[(j + 0) * drv.rcols + i];
      in1[inl1++] = drv.dex_map[(j + 1) * drv.rcols + i];
      in2[inl2++] = drv.dex_map[(j + 2) * drv.rcols + i];
      in3[inl3++] = drv.dex_map[(j + 3) * drv.rcols + i];
    }
  }

  in0[inl0++] = -1;
  int32_t* res_pointer = gemm_dst + c + r * output_stride;
  drv.st_params[0].dst = reinterpret_cast<int*>(res_pointer);
  drv.st_params[0].dcs = output_stride;
  drv.st_params[0].cols = rcols_step;
  drv.st_params[0].rows = rrows_step;
  drv.st_params[0].rrows = rows_step;
  drv.st_params[0].rcols = cols_step;
  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[1].dma_start_send(inl1);
  drv.mdma->dmas[2].dma_start_send(inl2);
  drv.mdma->dmas[3].dma_start_send(inl3);
  drv.mdma->multi_dma_wait_send();
}

void Store_Results_COL2IM(acc_container& drv, int8_t* dst) {
  int* in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int inl0 = 0;
  int out_int8_len = drv.output_depth * drv.output_height * drv.output_width;
  int out_int8_lenr = roundDown(out_int8_len, 4);

  in0[inl0++] = -2;
  in0[inl0++] = drv.output_depth;
  in0[inl0++] = drv.output_height * drv.output_width;
  in0[inl0++] = out_int8_len;
  in0[inl0++] = out_int8_lenr;
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
  int out_dex = 0;
  for (int j = 0; j < out_int8_lenr / 4; j++) {
    dst[out_dex++] = (int8_t)bo0[i];
    dst[out_dex++] = (int8_t)bo1[i];
    dst[out_dex++] = (int8_t)bo2[i];
    dst[out_dex++] = (int8_t)bo3[i++];
  }

  for (int j = 0; j < out_int8_len - out_int8_lenr; j++) {
    dst[out_dex++] = (int8_t)bo0[i++];
  }
}

void Load_Weight_Compute_Store(acc_container& drv, int32_t* gemm_dst,
                               int output_stride, int c, int rcols_step, int r,
                               int rrows_step, int rdepth_step, int rows_step,
                               int cols_step) {
  Load_Weight_Data(drv, gemm_dst, output_stride, c, rcols_step, r, rrows_step,
                   rdepth_step, rows_step, cols_step);

  drv.mdma->multi_dma_start_recv();
  drv.mdma->multi_dma_wait_recv();
  // Store_Results(drv);
}

void TileGEMM(acc_container& drv, int output_stride, int depth, int rdepth,
              int rows, int rrows, int cols, int rcols, int32_t* gemm_dst,
              int8_t* dst) {
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
    Load_Input_Data(drv, r, rrows_step, depth, rdepth);
    for (int d = 0; d < rdepth; d += rdepth) {
      int rdepth_step = std::min(rdepth, rdepth - d);
      for (int c = 0; c < rcols; c += col_inc) {
        int rcols_step = std::min(col_inc, rcols - c);
        int cols_step = std::min(col_inc, cols - c);
        Load_Weight_Compute_Store(drv, gemm_dst, output_stride, c, rcols_step,
                                  r, rrows_step, rdepth_step, rows_step,
                                  cols_step);
        drv.t.layer_weight_tile++;
      }
    }
    drv.t.layer_input_tile++;
  }
  Store_Results_COL2IM(drv, dst);
}

void Entry(acc_container& drv, int32_t* gemm_dst, int8_t* dst) {
  int rows = drv.rows;
  int cols = drv.cols;
  int depth = drv.depth;

  int rrows = roundUp(drv.rows, 4);
  int rcols = roundUp(drv.cols, 4);
  int rdepth = roundUp(drv.depth, 16);
  int output_stride = drv.cols;

  drv.rdepth = rdepth;
  drv.rrows = rrows;
  drv.rcols = rcols;

  uint32_t dex_map[rows * cols] = {};
  uint32_t pdex_map[rrows * rcols] = {};
  Col2im_Mapping(drv.output_depth, drv.output_height, drv.output_width,
                 drv.filter_height, drv.filter_width, drv.padding_top,
                 drv.padding_left, drv.padding_bottom, drv.padding_right,
                 drv.stride_height, drv.stride_width, dex_map);

  pad_dexmap(rows, cols, rrows, rcols, dex_map, pdex_map);
  drv.dex_map = pdex_map;

#ifdef DELEGATE_VERBOSE
  cout << "VM_TCONV" << endl;
  cout << "===========================" << endl;
  cout << "Pre-ACC Info" << endl;
  cout << "temp_depth: " << temp_depth << " depth: " << depth << endl;
  cout << "temp_cols: " << temp_cols << " cols: " << dst_params.cols << endl;
  cout << "temp_rows: " << temp_rows << " rows: " << dst_params.rows << endl;
  cout << "old_dcs: " << temp_rows << " dcs: " << dcs << endl;
  cout << "===========================" << endl;
#endif

  // TileGEMM(drv, dcs, depth, temp_depth, temp_rows, temp_cols, cols,
  // dst_params);
  TileGEMM(drv, output_stride, depth, rdepth, rows, rrows, cols, rcols,
           gemm_dst, dst);

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

}  // namespace tflite_vm_tconv_fpga
#endif  // GEMM_DRIVER
