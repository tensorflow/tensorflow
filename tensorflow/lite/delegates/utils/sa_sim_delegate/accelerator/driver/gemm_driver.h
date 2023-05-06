#ifndef GEMM_DRIVER
#define GEMM_DRIVER

#include "acc_container.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"

#define SA_SIZE_X 16
#define SA_SIZE_Y 16

// GEMM_Driver for simulated SA acccelerator
namespace tflite_sasim {

// Previously called Load_RHS_Data
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

  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[1].dma_start_send(inl1);
  drv.mdma->dmas[2].dma_start_send(inl2);
  drv.mdma->dmas[3].dma_start_send(inl3);
  drv.mdma->multi_dma_wait_send();
  drv.profile->saveProfile(drv.acc->profiling_vars);
}

void Load_Weight_Data(acc_container &drv, int8_t *results, int output_stride,
                      int c, int rcols_step, int r, int rrows_step,
                      int rdepth_step, int rows_step, int cols_step) {
  int *in0 = drv.mdma->dmas[0].dma_get_inbuffer();
  int *in1 = drv.mdma->dmas[1].dma_get_inbuffer();
  int *in2 = drv.mdma->dmas[2].dma_get_inbuffer();
  int *in3 = drv.mdma->dmas[3].dma_get_inbuffer();

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
  drv.st_params.dst = reinterpret_cast<int *>(res_pointer);
  drv.st_params.dcs = output_stride;
  drv.st_params.cols = rcols_step;
  drv.st_params.rows = rrows_step;
  drv.st_params.rrows = rows_step;
  drv.st_params.rcols = cols_step;

  drv.mdma->dmas[0].dma_start_send(inl0);
  drv.mdma->dmas[1].dma_start_send(inl1);
  drv.mdma->dmas[2].dma_start_send(inl2);
  drv.mdma->dmas[3].dma_start_send(inl3);
  drv.mdma->multi_dma_wait_send();
  drv.profile->saveProfile(drv.acc->profiling_vars);
}

void Store_Results(acc_container &drv) {
  struct store_params sp = drv.st_params;
  int output_stride = sp.dcs;
  int rcols_step = sp.cols;
  int rows_step = sp.rrows;
  int8_t *base = reinterpret_cast<int8_t *>(sp.dst);
  int *o0 = drv.mdma->dmas[0].dma_get_outbuffer();
  int *o1 = drv.mdma->dmas[1].dma_get_outbuffer();
  int *o2 = drv.mdma->dmas[2].dma_get_outbuffer();
  int *o3 = drv.mdma->dmas[3].dma_get_outbuffer();
  int8_t *bo0 = reinterpret_cast<int8_t *>(o0);
  int8_t *bo1 = reinterpret_cast<int8_t *>(o1);
  int8_t *bo2 = reinterpret_cast<int8_t *>(o2);
  int8_t *bo3 = reinterpret_cast<int8_t *>(o3);
  int out0 = 0;
  int out1 = 0;
  int out2 = 0;
  int out3 = 0;

  int acc_rows = SA_SIZE_Y;
  int acc_cols = SA_SIZE_X;
  int unrolled_cols = rcols_step - rcols_step % acc_cols;
  int unrolled_rows = rows_step - (rows_step % acc_rows);

  for (int i = 0; i < unrolled_rows; i += acc_rows) {
    for (int j = 0; j < unrolled_cols; j += acc_cols) {
      for (int k = 0; k < acc_cols; k++) {
        base[(i + 0) * output_stride + (j) + k] = bo0[out0++];
        base[(i + 1) * output_stride + (j) + k] = bo1[out1++];
        base[(i + 2) * output_stride + (j) + k] = bo2[out2++];
        base[(i + 3) * output_stride + (j) + k] = bo3[out3++];
      }
      for (int k = 0; k < acc_cols; k++) {
        base[(i + 4) * output_stride + (j) + k] = bo0[out0++];
        base[(i + 5) * output_stride + (j) + k] = bo1[out1++];
        base[(i + 6) * output_stride + (j) + k] = bo2[out2++];
        base[(i + 7) * output_stride + (j) + k] = bo3[out3++];
      }

      if (acc_rows == 8) continue;
      for (int k = 0; k < acc_cols; k++) {
        base[(i + 8) * output_stride + (j) + k] = bo0[out0++];
        base[(i + 9) * output_stride + (j) + k] = bo1[out1++];
        base[(i + 10) * output_stride + (j) + k] = bo2[out2++];
        base[(i + 11) * output_stride + (j) + k] = bo3[out3++];
      }
      for (int k = 0; k < acc_cols; k++) {
        base[(i + 12) * output_stride + (j) + k] = bo0[out0++];
        base[(i + 13) * output_stride + (j) + k] = bo1[out1++];
        base[(i + 14) * output_stride + (j) + k] = bo2[out2++];
        base[(i + 15) * output_stride + (j) + k] = bo3[out3++];
      }
    }

    for (int j = unrolled_cols; j < rcols_step; j++) {
      base[(i + 0) * output_stride + j] = bo0[out0++];
      base[(i + 1) * output_stride + j] = bo1[out1++];
      base[(i + 2) * output_stride + j] = bo2[out2++];
      base[(i + 3) * output_stride + j] = bo3[out3++];
    }
    for (int j = unrolled_cols; j < rcols_step; j++) {
      base[(i + 4) * output_stride + j] = bo0[out0++];
      base[(i + 5) * output_stride + j] = bo1[out1++];
      base[(i + 6) * output_stride + j] = bo2[out2++];
      base[(i + 7) * output_stride + j] = bo3[out3++];
    }

    if (acc_rows == 8) continue;
    for (int j = unrolled_cols; j < rcols_step; j++) {
      base[(i + 8) * output_stride + j] = bo0[out0++];
      base[(i + 9) * output_stride + j] = bo1[out1++];
      base[(i + 10) * output_stride + j] = bo2[out2++];
      base[(i + 11) * output_stride + j] = bo3[out3++];
    }
    for (int j = unrolled_cols; j < rcols_step; j++) {
      base[(i + 12) * output_stride + j] = bo0[out0++];
      base[(i + 13) * output_stride + j] = bo1[out1++];
      base[(i + 14) * output_stride + j] = bo2[out2++];
      base[(i + 15) * output_stride + j] = bo3[out3++];
    }
  }

  for (int j = 0; j < unrolled_cols; j += acc_cols) {
    for (int i = unrolled_rows; i < rows_step; i++) {
      int8_t *bos;
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
        base[(i)*output_stride + j + k] = bos[outs++];
      if (i % 4 == 0) out0 = outs;
      if (i % 4 == 1) out1 = outs;
      if (i % 4 == 2) out2 = outs;
      if (i % 4 == 3) out3 = outs;
    }
  }

  for (int i = unrolled_rows; i < rows_step; i++) {
    int8_t *bos;
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
    for (int j = unrolled_cols; j < rcols_step; j++)
      base[(i)*output_stride + j] = bos[outs++];
    if (i % 4 == 0) out0 = outs;
    if (i % 4 == 1) out1 = outs;
    if (i % 4 == 2) out2 = outs;
    if (i % 4 == 3) out3 = outs;
  }
}

void Load_Weight_Compute_Store(acc_container &drv, int8_t *results,
                               int output_stride, int c, int rcols_step, int r,
                               int rrows_step, int rdepth_step, int rows_step,
                               int cols_step) {
  Load_Weight_Data(drv, results, output_stride, c, rcols_step, r, rrows_step,
                   rdepth_step, rows_step, cols_step);
  drv.mdma->multi_dma_start_recv();
  drv.mdma->multi_dma_wait_recv();
  drv.profile->saveProfile(drv.acc->profiling_vars);
  Store_Results(drv);
}
void TileGEMM(acc_container &drv, int output_stride, int depth, int rdepth,
              int rows, int rrows, int cols, int rcols, int8_t *results) {
  drv.t.layer_weight_tile = 0;
  drv.t.layer_input_tile = 0;
  int acc_weight_buffer_size = 4096 * 16;
  int acc_input_buffer_size = 8192 * 16;
  int max_cols = acc_weight_buffer_size / rdepth;
  max_cols = max_cols - (max_cols % 4);
  int col_inc = std::min(std::min(rcols, max_cols), 4096);
  int max_rows = acc_input_buffer_size / rdepth;
  max_rows = max_rows - (max_rows % 4);
  int row_inc = std::min(std::min(rrows, max_rows), 8192);

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
    drv.t.layer_input_tile++;
  }
}

void Entry(acc_container &drv, int8_t *dst) {
  int rows = drv.rows;
  int cols = drv.cols;
  int depth = drv.depth;

  int rrows = roundUp(drv.rows, 1);
  int rcols = roundUp(drv.cols, 4);
  int rdepth = roundUp(drv.depth, 16);
  int output_stride = drv.cols;

  // #ifdef DELEGATE_VERBOSE
  cerr << "Systolic Array" << endl;
  cerr << "===========================" << endl;
  cerr << "Pre-ACC Info" << endl;
  cerr << "rdepth: " << rdepth << " depth: " << depth << endl;
  cerr << "rcols: " << rcols << " cols: " << cols << endl;
  cerr << "rrows: " << rrows << " rows: " << rows << endl;
  cerr << "output_stride: " << output_stride << endl;
  cerr << "===========================" << endl;
  // #endif

  TileGEMM(drv, output_stride, depth, rdepth, rows, rrows, cols, rcols, dst);

#ifdef DELEGATE_DEBUG
  mkdir("aData", 0777);
  ofstream myfile;
  myfile.open("aData/out_sa_" + std::to_string(drv.t.layer) + "_1.csv");
  int8_t *res_pointer = dst;
  int index = 0;
  for (int r = 0; r < rows; r++) {
    myfile << endl;
    for (int c = 0; c < cols; c++) {
      myfile << (int)res_pointer[index] << ",";
      index++;
    }
  }
  myfile.close();
#endif
}
} // namespace tflite_sasim
#endif // GEMM_DRIVER