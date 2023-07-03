#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#include <chrono>
#include <iomanip>
#include <vector>

#include "acc_config.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"

#ifdef ACC_NEON
#include "arm_neon.h"
#endif

using namespace std;
using namespace std::chrono;

#ifdef ACC_PROFILE
#define prf_start(N) auto start##N = chrono::steady_clock::now();
#define prf_end(N, X)                                                          \
  auto end##N = chrono::steady_clock::now();                                   \
  X += end##N - start##N;
#else
#define prf_start(N)
#define prf_end(N, X)
#endif

int nofSteps(int length, int stride, int ks) {
  return int(((length - ks) / stride) + 1);
}

int ceiling(int a, int b) { return (a + b - 1) / b; }

// Used for storing current GEMM info
struct gemm_details {
  int layer = 0;
  bool profile = false;
};

struct mm2im_times {
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> driver_total;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> acc_total;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> tconv_total;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> store;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> load_wgt;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> load_inp;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> load_rowmap;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> load_colmap;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> start_sched;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> load_config;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> handle_rest;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> del_inp;
  void print() {
#ifdef ACC_PROFILE
    cout << "================================================" << endl;
    cout << "driver_total, "
         << chrono::duration_cast<chrono::milliseconds>(driver_total).count()
         << endl;
    cout << "acc_total, "
         << chrono::duration_cast<chrono::milliseconds>(acc_total).count()
         << endl;
    cout << "tconv_total, "
         << chrono::duration_cast<chrono::milliseconds>(tconv_total).count()
         << endl;
    cout << "store, "
         << chrono::duration_cast<chrono::milliseconds>(store).count() << endl;
    cout << "load_wgt, "
         << chrono::duration_cast<chrono::milliseconds>(load_wgt).count()
         << endl;
    cout << "load_inp, "
         << chrono::duration_cast<chrono::milliseconds>(load_inp).count()
         << endl;
    cout << "load_rowmap, "
         << chrono::duration_cast<chrono::milliseconds>(load_rowmap).count()
         << endl;
    cout << "load_colmap, "
         << chrono::duration_cast<chrono::milliseconds>(load_colmap).count()
         << endl;
    cout << "start_sched, "
         << chrono::duration_cast<chrono::milliseconds>(start_sched).count()
         << endl;
    cout << "load_config, "
         << chrono::duration_cast<chrono::milliseconds>(load_config).count()
         << endl;
    cout << "handle_rest, "
         << chrono::duration_cast<chrono::milliseconds>(handle_rest).count()
         << endl;
    cout << "del_inp, "
         << chrono::duration_cast<chrono::milliseconds>(del_inp).count()
         << endl;
    cout << "================================================" << endl;
#endif
  }
};

struct acc_container {

  struct multi_dma *mdma;

  // Padded Buffers
  int *loaded_weights;
  int *loaded_inputs;
  int8_t *output_data;

  const int8_t *weights;
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

  // GEMM Profiling variable
  struct gemm_details t;
  bool verb;
  struct mm2im_times p_t;

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