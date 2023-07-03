#ifndef MM2IM_UTIL_H_
#define MM2IM_UTIL_H_

#include <iomanip>
#include <iostream>
#include <vector>

#include "acc_container.h"
using namespace std;

int CPU_Quantised_Multiplier(int x, int qm, int8_t shift) {
  int64_t pl;
  int32_t pr;
  int32_t msk;
  int32_t sm;
  if (shift > 0) {
    pl = shift;
    // pl = (1 << shift);
    pr = 0;
    msk = 0;
    sm = 0;
  } else {
    // pl = 1;
    pl = 0;
    pr = -shift;
    msk = (1 << -shift) - 1;
    sm = msk >> 1;
  }
  // int64_t val = x * pl;
  int64_t val = x * (1 << pl);
  if (val > MAX)
    val = MAX; // ALU MIN
  if (val < MIN)
    val = MIN; // ALU MAX
  int64_t val_2 = val * qm;
  int32_t temp_1;
  temp_1 = (val_2 + POS) / DIVMAX;
  if (val_2 < 0)
    temp_1 = (val_2 + NEG) / DIVMAX;
  int32_t val_3 = temp_1;
  val_3 = val_3 >> pr;
  int32_t temp_2 = temp_1 & msk;
  int32_t temp_3 = (temp_1 < 0) & 1;
  int32_t temp_4 = sm + temp_3;
  int32_t temp_5 = ((temp_2 > temp_4) & 1);
  int32_t result_32 = val_3 + temp_5;
  int res = result_32;
  return result_32;
}

void col2im_mapping_v2(int depth, int height, int width, int filter_h,
                       int filter_w, int pad_t, int pad_l, int pad_b, int pad_r,
                       int stride_h, int stride_w, int32_t *index_map) {
  int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;
  int h_pad = -pad_t;
  int im_dex = 0;
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
              im_dex++;
            }
          } else {
            for (int i = 0; i < depth; ++i) {
              index_map[map_dex++] = -1;
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

void col2im_mapping_v3(int depth, int height, int width, int filter_h,
                       int filter_w, int pad_t, int pad_l, int pad_b, int pad_r,
                       int stride_h, int stride_w, int32_t *index_map) {
  int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;
  int h_pad = -pad_t;
  int im_dex = 0;
  int map_dex = 0;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      im_dex = (h_pad * width + w_pad) * depth;
      for (int i = 0; i < depth; ++i) {
        im_dex = (h_pad * width + w_pad) * depth + i;
        for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
          for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
              index_map[map_dex++] = im_dex;
              im_dex += depth;

            } else {
              index_map[map_dex++] = -1;
              im_dex += depth;
            }
          }
          im_dex += depth * (width - filter_w);
        }
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

int ComputeOutSize(string padding, int image_size, int filter_size, int stride,
                   int dilation_rate = 1) {
  int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  if (stride == 0)
    return 0;

  if (padding == "same")
    return (image_size + stride - 1) / stride;
  else if (padding == "valid")
    return (image_size + stride - effective_filter_size) / stride;
  else
    return 0;
}

int compute_padding_with_offset(int stride, int dilation_rate, int in_size,
                                int filter_size, int out_size, int &offset) {
  int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  int total_padding =
      ((out_size - 1) * stride) + effective_filter_size - in_size;
  total_padding = (total_padding > 0) ? total_padding : 0;
  offset = total_padding % 2;
  return (total_padding / 2);
}

void compute_padding_height_width(string padding, int stride_height,
                                  int stride_width, int in_height, int in_width,
                                  int filter_height, int filter_width, int &ph,
                                  int &pw, int &pho, int &pwo, int &out_width,
                                  int &out_height) {
  int dilation_rate_height = 1;
  int dilation_rate_width = 1;
  int offset, p_height, p_width = 0;
  out_width = ComputeOutSize(padding, in_width, filter_width, stride_width,
                             dilation_rate_width);
  out_height = ComputeOutSize(padding, in_height, filter_height, stride_height,
                              dilation_rate_height);
  ph =
      compute_padding_with_offset(stride_height, dilation_rate_height,
                                  in_height, filter_height, out_height, offset);
  int h_offset = offset;

  pw = compute_padding_with_offset(stride_width, dilation_rate_width, in_width,
                                   filter_width, out_width, offset);
  int w_offset = offset;
  pwo = w_offset;
  pho = h_offset;
}

void calParams(int stride_x, int stride_y, int filters, int kernel_size,
               int in1, int in2, int in3, string padding, int &rows, int &cols,
               int &depth, int &out1, int &out2, int &out3, int &pt, int &pb,
               int &pl, int &pr) {
  int ph, pw, pho, pwo;
  out1 = in1 + kernel_size - stride_x;
  out2 = in2 + kernel_size - stride_y;
  out3 = filters;
  rows = filters * kernel_size * kernel_size;
  cols = in1 * in2;
  depth = in3;
  if (padding == "same") {
    out1 = in1 * stride_x;
    out2 = in2 * stride_y;
  } else {
    out1 = in1 + kernel_size - stride_x;
    out2 = in2 + kernel_size - stride_y;
  }
  int iout1, iout2;
  compute_padding_height_width(padding, stride_x, stride_y, out1, out2,
                               kernel_size, kernel_size, ph, pw, pho, pwo,
                               iout1, iout2);
  pt = ph;
  pb = ph + pho;
  pl = pw;
  pr = pw + pwo;
  // print all
  // cerr << "*******************" << endl;
  // cout << "stride_x: " << stride_x << ", ";
  // cout << "stride_y: " << stride_y << endl;
  // cout << "filters: " << filters << ", ";
  // cout << "kernel_size: " << kernel_size << endl;
  // cout << "in1: " << in1 << ", ";
  // cout << "in2: " << in2 << ", ";
  // cout << "in3: " << in3 << ", ";
  // cout << "padding: " << padding << endl;
  // cout << "out1: " << out1 << ", ";
  // cout << "out2: " << out2 << ", ";
  // cout << "out3: " << out3 << ", ";
  // cout << "rows: " << rows << ", ";
  // cout << "cols: " << cols << ", ";
  // cout << "depth: " << depth << endl;
  // cout << "pt: " << pt << ", ";
  // cout << "pb: " << pb << ", ";
  // cout << "pl: " << pl << ", ";
  // cout << "pr: " << pr << endl;
  // cerr << "*******************" << endl;
}

struct mm2im_params {
  int sx, sy, f, ks, ih, iw, ic = 0;
  int rows, cols, depth, o1, o2, o3, pt, pb, pl, pr = 0;
  bool padding_type = false;
  int8_t *input_data;
  int8_t *weight_data;
  int32_t *output_data;
  int32_t *dex_map;
  int32_t *wt_sum;
  int32_t *bias_data;
  int32_t *crf;
  int8_t *crx;
  int ra;
  bool compute = false;

  vector<vector<int>> mm2im_map;
  vector<vector<vector<int>>> o1_map;
  vector<int> o1_starts;
  vector<int> o1_ends;
  vector<int> o1_lengths;

  vector<vector<int>> col_dexs;
  vector<vector<int>> out_dexs;
  // vector<int> cols_indices_starts;
  // vector<int> cols_indices_lens;

  unsigned int output_size;
  unsigned int output_rows;
  unsigned int output_cols;

  mm2im_params(int _sx, int _sy, int _f, int _ks, int _ih, int _iw, int _ic,
               int _rows, int _cols, int _depth, int _o1, int _o2, int _o3,
               int _pt, int _pb, int _pl, int _pr, bool _padding_type)
      : sx(_sx), sy(_sy), f(_f), ks(_ks), ih(_ih), iw(_iw), ic(_ic),
        rows(_rows), cols(_cols), depth(_depth), o1(_o1), o2(_o2), o3(_o3),
        pt(_pt), pb(_pb), pl(_pl), pr(_pr), padding_type(_padding_type) {
    output_size = o1 * o2 * o3;
    output_rows = o1 * o2;
    output_cols = o3;
  }

  void create_MM2IM_map() {
    mm2im_map.clear();
    mm2im_map.resize(output_size);
    for (int j = 0; j < rows * cols; j++) {
      if (dex_map[j] != -1)
        mm2im_map[dex_map[j]].push_back(j);
    }
  }

  void create_col_indices_map() {
    int f_rows = rows / f;
    for (int c = 0; c < cols; c++) {
      vector<int> col_dex_of_row;
      vector<int> out_dex_of_row;
      for (int r = 0; r < f_rows; r++) {
        int o_dex = c * rows + r;
        if (dex_map[o_dex] != -1) {
          col_dex_of_row.push_back(r);
          out_dex_of_row.push_back(dex_map[o_dex] / f);
        }
      }
      col_dexs.push_back(col_dex_of_row);
      out_dexs.push_back(out_dex_of_row);
    }
    int k = 0;
  }

  void MM2IM_o1_map() {
    create_MM2IM_map();
    create_col_indices_map();

    // We create o1_map for each row, map is used by the accelerator
    for (int o_3 = 0; o_3 < output_cols; o_3++) {
      int f_hi = std::numeric_limits<int>::min();
      int f_lo = std::numeric_limits<int>::max();
      for (int o_1 = 0; o_1 < o1; o_1++) {
        int hi = std::numeric_limits<int>::min();
        int lo = std::numeric_limits<int>::max();
        vector<vector<int>> o1_row_map;
        for (int o_2 = 0; o_2 < o2; o_2++) {
          int o_dex = ((o_1 * o2) + o_2) * output_cols + o_3;
          int size = mm2im_map[o_dex].size();
          for (int i = 0; i < mm2im_map[o_dex].size(); i++) {
            int orow = mm2im_map[o_dex][i] % rows;
            int ocol = mm2im_map[o_dex][i] / rows;
            lo = min(lo, ocol);
            hi = max(hi, ocol);
            f_lo = min(f_lo, orow);
            f_hi = max(f_hi, orow);
          }
        }

        for (int o_2 = 0; o_2 < o2; o_2++) {
          vector<int> output_map;
          int o_dex = ((o_1 * o2) + o_2) * output_cols + o_3;
          for (int i = 0; i < mm2im_map[o_dex].size(); i++) {
            int orow = mm2im_map[o_dex][i] % rows;
            int ocol = mm2im_map[o_dex][i] / rows;
            // This is ensures accelerator start at row 0 for each output row
            output_map.push_back(ocol - lo);
            output_map.push_back(orow);
          }
          o1_row_map.push_back(output_map);
        }

        if (o_3 == 0)
          o1_map.push_back(o1_row_map);
        o1_ends.push_back(hi);
        o1_starts.push_back(lo);
        o1_lengths.push_back(hi - lo);
      }
    }
  };

  void MM2IM_compute() {
    create_MM2IM_map();
    unsigned int output_size = o1 * o2 * o3;
    for (int k = 0; k < output_size; k++) {
      int32_t sum = 0;
      for (int j = 0; j < mm2im_map[k].size(); j++) {
        int orow = mm2im_map[k][j] % rows;
        int ocol = mm2im_map[k][j] / rows;
        for (int d = 0; d < depth; d++) {
          int weight_index = orow * depth + d;
          int input_index = ocol * depth + d;
          int weight = weight_data[weight_index];
          int input = input_data[input_index];
          sum += weight * input;
        }
        int offset = wt_sum[orow] * 128;
        // int offset = 0;
        sum += offset;
      }
      int bias = bias_data[k % o3];
      int crf_data = crf[k % o3];
      int crx_data = crx[k % o3];

      int qm_ret =
          ra + CPU_Quantised_Multiplier(sum + bias, crf_data, crx_data);
      if (qm_ret > MAX8)
        qm_ret = MAX8;
      else if (qm_ret < MIN8)
        qm_ret = MIN8;

      output_data[k] = qm_ret;
    }
  };
};

#endif // MM2IM_UTIL_H_