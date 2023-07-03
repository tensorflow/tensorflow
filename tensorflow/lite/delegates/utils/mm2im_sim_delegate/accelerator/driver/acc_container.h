#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#include <iomanip>
#include <vector>

#ifdef SYSC
#include "../acc.sc.h"
// #include "sysc_profiler/profiler.h"
#include "systemc_binding.h"
#endif

#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"

int nofSteps(int length, int stride, int ks) {
  return int(((length - ks) / stride) + 1);
}

int ceiling(int a, int b) { return (a + b - 1) / b; }

// Used for storing current GEMM info
struct gemm_details {
  int layer = 0;
  bool profile = false;
};

struct acc_container {

#ifdef SYSC
  // Hardware
  ACCNAME *acc;
  struct sysC_sigs *scs;
  Profile *profile;
#else
  int *acc;
#endif
  struct multi_dma *mdma;

  // Padded Buffers
  int *loaded_weights;
  int *loaded_inputs;
  // int *output_data;
  int8_t *output_data;

  int8_t *weights;
  const int8_t *inputs;

  // mm2im map
  vector<vector<int>> mm2im_map;
  vector<vector<vector<int>>> o1_map;
  int *o1_lengths;
  int *o1_starts;
  int *o1_ends;

  vector<vector<int>> col_dexs;
  vector<vector<int>> out_dexs;

  // Output Pipeline Metadata
  int32_t *acc_wt_sum;
  int *crf;
  int8_t *crx;
  int *bias;

  // External Params
  int ra;
  int rhs_offset = 0;
  int lhs_offset = 0;
  int ih = 0;
  int iw = 0;
  int ic = 0;
  int f = 0;
  int ks = 0;
  int o1 = 0;
  int o2 = 0;
  int o3 = 0;
  int sx = 0;
  int sy = 0;
  int pt = 0;
  int pl = 0;
  int pb = 0;
  int pr = 0;
  int rows = 0;
  int cols = 0;
  int depth = 0;

  // GEMM Info variable
  struct gemm_details t;
  bool verb;
  acc_container() {}
};

void preload_weights(int8_t *wgt, int depth, int rows, int *wt_sum,
                     int8_t *loaded_weights) {
  int d = roundUp(depth, 16);
  int max = rows * depth;
  int wt_sum_dex = 0;
  for (int i = 0; i < rows; i++) {
    int s0 = 0;
    for (int j = 0; j < d; j++) {
      if (j < depth) {
        int8_t w0 = wgt[i * depth + j];
        loaded_weights[i * d + j] = w0;
        s0 += w0;
      } else loaded_weights[i * d + j] = 0;
    }
    wt_sum[wt_sum_dex++] = s0;
  }
}

void preload_inputs(const int8_t *inp, int depth, int rows,
                    int8_t *loaded_inputs) {
  int d = roundUp(depth, 16);
  int max = rows * depth;
  int wt_sum_dex = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < d; j++) {
      if (j < depth) {
        int8_t w0 = inp[i * depth + j];
        loaded_inputs[i * d + j] = w0;
      } else loaded_inputs[i * d + j] = 0;
    }
  }
}
void swap_weights(const int8_t *wgt, int8_t *new_wgt, int filters, int ks,
                  int ic) {
  for (int k = 0; k < ks * ks; k++) {
    for (int j = 0; j < filters; j++) {
      for (int i = 0; i < ic; i++) {
        new_wgt[j * ks * ks * ic + k * ic + i] =
            wgt[k * filters * ic + j * ic + i];
      }
    }
  }
}

#endif // ACC_CONTAINER