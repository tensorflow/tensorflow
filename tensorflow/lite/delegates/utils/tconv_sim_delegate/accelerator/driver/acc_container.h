#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#include <vector>

#include "../acc.h"
#include "systemc_binding.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"

void Col2im_Mapping_v2(int depth, int height, int width, int filter_h,
                       int filter_w, int pad_t, int pad_l, int pad_b, int pad_r,
                       int stride_h, int stride_w, int32_t* index_map) {
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

struct tconv_params {
  int stride_x, stride_y, filters, kernel_size, in1, in2, in3 = 0;
  int rows, cols, depth, out1, out2, out3, pt, pb, pl, pr = 0;
  bool padding_type = false;
  const int8_t* input_data;
  const int8_t* weight_data;
  int32_t* output_data;
  int32_t* dex_map;
  int32_t* wt_sum;

  tconv_params(int _stride_x, int _stride_y, int _filters, int _kernel_size,
               int _in1, int _in2, int _in3, int _rows, int _cols, int _depth,
               int _out1, int _out2, int _out3, int _pt, int _pb, int _pl,
               int _pr, bool _padding_type)
      : stride_x(_stride_x),
        stride_y(_stride_y),
        filters(_filters),
        kernel_size(_kernel_size),
        in1(_in1),
        in2(_in2),
        in3(_in3),
        rows(_rows),
        cols(_cols),
        depth(_depth),
        out1(_out1),
        out2(_out2),
        out3(_out3),
        pt(_pt),
        pb(_pb),
        pl(_pl),
        pr(_pr),
        padding_type(_padding_type) {}

  void TCONV() {
    vector<vector<int>> im2col;

    // for each output k, find set J where all j is  dex_map[j] == k
    // then for each J get the weight row and input col which is required
    // compute the dot product now for calculate all J in parrel and sums them
    // up and quantize the result and store in output[k]
    unsigned int output_size = out1 * out2 * out3;
    im2col.resize(output_size);
    for (int j = 0; j < rows * cols; j++) {
      if (dex_map[j] != -1) im2col[dex_map[j]].push_back(j);
    }

    for (int j = 0; j < im2col.size(); j++) {
      cerr << "[";
      for (int i = 0; i < im2col[j].size(); i++) {
        cerr << im2col[j][i] << " ";
      }
      cerr << "]" << endl;
    }

    // GEMM + COL2IM
    for (int k = 0; k < output_size; k++) {
      int32_t sum = 0;
      for (int j = 0; j < im2col[k].size(); j++) {
        int orow = im2col[k][j] % rows;
        int ocol = im2col[k][j] / rows;
        cout << "Wrow:" << orow << " Icol:" << ocol << endl;
        for (int d = 0; d < depth; d++) {
          int weight_index = orow * depth + d;
          int input_index = ocol * depth + d;
          int weight = weight_data[weight_index];
          int input = input_data[input_index];
          sum += weight * input;
        }
        int offset  = wt_sum[orow]*128;
        sum += offset;
      }
      cout << "------------------" << endl;
      output_data[k] = sum;
    }
    int q = 0;
    //BIAS ADD  + Qauntize

  };
};

// Used for storing current GEMM info
struct gemm_details {
  int layer = 0;
  int layer_weight_tile = 0;
  int layer_input_tile = 0;
  int layer_print = -1;
  int layer_ww = -1;
  int layer_iw = -1;
  bool profile = false;
};

// Used for tracking output locations
struct store_params {
  int* dst;
  int dcs;
  int rows;
  int cols;
  int rcols;
  int rrows;
};

struct acc_container {
  // Gives driver access to SystemC modules + profiler
  ACCNAME* acc;
  Profile* profile;
  struct multi_dma* mdma;

  // Temporary Weight non-MMapped Padded Buffers
  int* wb_0;
  int* wb_1;
  int* wb_2;
  int* wb_3;

  // Temporary Input non-MMapped Padded Buffers
  int* inb_0;
  int* inb_1;
  int* inb_2;
  int* inb_3;
  int in_id = 0;

  // Driver variables
  struct store_params st_params;
  int w_c = 0;

  // Output Pipeline Metadata
  std::vector<int> wt_sum1;
  std::vector<int> wt_sum2;
  std::vector<int> wt_sum3;
  std::vector<int> wt_sum4;
  int* in_sum1;
  int* in_sum2;
  int* in_sum3;
  int* in_sum4;
  int* bias;
  std::vector<int> crf;
  std::vector<int8_t> crx;
  int ra;
  int rhs_offset = 0;
  int lhs_offset = 0;

  int output_depth = 0;
  int output_height = 0;
  int output_width = 0;

  int filter_height = 0;
  int filter_width = 0;

  int padding_top = 0;
  int padding_left = 0;
  int padding_bottom = 0;
  int padding_right = 0;

  int stride_height = 0;
  int stride_width = 0;

  uint32_t* dex_map;

  int rows = 0;
  int cols = 0;
  int depth = 0;

  int rrows = 0;
  int rcols = 0;
  int rdepth = 0;
  int32_t* gemm_dst;
  int8_t* dst;

  // GEMM Info variable
  struct gemm_details t;

  acc_container(ACCNAME* _acc, int* _wb_0, int* _wb_1, int* _wb_2, int* _wb_3,
                std::vector<int> _wt_sum1, std::vector<int> _wt_sum2,
                std::vector<int> _wt_sum3, std::vector<int> _wt_sum4,
                std::vector<int> _crf, std::vector<int8_t> _crx) {
    acc = _acc;
    wb_0 = _wb_0;
    wb_1 = _wb_1;
    wb_2 = _wb_2;
    wb_3 = _wb_3;
    wt_sum1 = _wt_sum1;
    wt_sum2 = _wt_sum2;
    wt_sum3 = _wt_sum3;
    wt_sum4 = _wt_sum4;
    crf = _crf;
    crx = _crx;
  }
};

void preload_weights(int8_t* weight_data, int* dims, vector<int8_t>& wb0,
                     vector<int8_t>& wb1, vector<int8_t>& wb2,
                     vector<int8_t>& wb3, vector<int>& wt_sum1,
                     vector<int>& wt_sum2, vector<int>& wt_sum3,
                     vector<int>& wt_sum4,vector<int>& wt_sum) {
  int width = dims[1] * dims[2] * dims[0];
  int w = ((width + 4 - 1) - ((width + 4 - 1) % 4));
  int depth = dims[3];
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
  int max = width * depth;
  for (int i = 0; i < w / 4; i++) {
    int s0 = 0;
    int s1 = 0;
    int s2 = 0;
    int s3 = 0;

    for (int j = 0; j < d; j++) {
      if (j < depth) {
        int8_t w0 =
            (i * (depth * 4) + j >= max) ? 0 : weight_data[i * (depth * 4) + j];
        int8_t w1 = (i * (depth * 4) + j + depth * 1 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 1];
        int8_t w2 = (i * (depth * 4) + j + depth * 2 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 2];
        int8_t w3 = (i * (depth * 4) + j + depth * 3 >= max)
                        ? 0
                        : weight_data[i * (depth * 4) + j + depth * 3];
        int8_t weights[] = {w3, w2, w1, w0};
        s0 += w0;
        s1 += w1;
        s2 += w2;
        s3 += w3;
        wb0.push_back(w0);
        wb1.push_back(w1);
        wb2.push_back(w2);
        wb3.push_back(w3);
      } else {
        wb0.push_back(0);
        wb1.push_back(0);
        wb2.push_back(0);
        wb3.push_back(0);
      }
    }
    wt_sum1.push_back(s0);
    wt_sum2.push_back(s1);
    wt_sum3.push_back(s2);
    wt_sum4.push_back(s3);
    wt_sum.push_back(s0);
    wt_sum.push_back(s1);
    wt_sum.push_back(s2);
    wt_sum.push_back(s3);
  }
}

#endif  // ACC_CONTAINER