#ifndef ACC_CONTAINER
#define ACC_CONTAINER

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <typeinfo>
#include <vector>

#include "tensorflow/lite/delegates/utils/secda_tflite/axi_support/axi_api_v2.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/acc_helpers.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/multi_threading.h"

#ifdef ACC_NEON
#include "arm_neon.h"
#endif

using namespace std;
using namespace std::chrono;

#ifdef ACC_PROFILE
#define prf_start(N) auto start##N = chrono::steady_clock::now();
#define prf_end(N, X)                        \
  auto end##N = chrono::steady_clock::now(); \
  X += end##N - start##N;
#else
#define prf_start(N)
#define prf_end(N, X)
#endif

// Used for tracking output locations
struct store_params {
  int* dst;
  int dcs;
  int rows;
  int cols;
  int rcols;
  int rrows;
};

struct sa_times {
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> load_rhs;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> load_lhs;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> sa_acc;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> store;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> ipack;
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> conv_total;
  void print() {
#ifdef ACC_PROFILE
    cout << "================================================" << endl;
    cout << "conv_total, "
         << chrono::duration_cast<chrono::milliseconds>(conv_total).count()
         << endl;
    cout << "ipack, "
         << chrono::duration_cast<chrono::milliseconds>(ipack).count() << endl;
    cout << "sa_acc, "
         << chrono::duration_cast<chrono::milliseconds>(sa_acc).count() << endl;
    cout << "================================================" << endl;
#endif
  }
};

// Used for profiling
struct gemm_details {
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> acctime =
      std::chrono::duration<long long int, std::ratio<1, 1000000000>>(0);
  std::chrono::duration<long long int, std::ratio<1, 1000000000>> convtime =
      std::chrono::duration<long long int, std::ratio<1, 1000000000>>(0);
  int layer = 0;
  int layer_weight_tile = 0;
  int layer_input_tile = 0;
  int layer_print = -1;
  int layer_ww = -1;
  int layer_iw = -1;
  bool profile = false;
};

struct acc_container {
  // DMAs Pointer
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
  struct store_params* st_params;
  MultiThreadContext* mt_context;
  int thread_count;
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

  // Pipeline vars
  struct dma_buffer_set* dfs;
  struct DSR dsr;
  bool lhs_start = false;
  int recv_len;

  // Profiling varaiable
  struct gemm_details t;
  struct sa_times t2;

  acc_container(int* _wb_0, int* _wb_1, int* _wb_2, int* _wb_3,
                std::vector<int> _wt_sum1, std::vector<int> _wt_sum2,
                std::vector<int> _wt_sum3, std::vector<int> _wt_sum4,
                std::vector<int> _crf, std::vector<int8_t> _crx) {
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

  bool Check_Done() { return (mdma->multi_dma_check_recv() == 0); }

  void End_Transfer() { mdma->multi_dma_wait_send(); }

  void Start_Transfer() {
    int s_buf = find_dbuf(dfs[0], dsr.sID);
    mdma->multi_dma_change_start_4(dfs[0].dbuf_set[s_buf].offset);
    mdma->dmas[0].dma_start_send(dfs[0].dbuf_set[s_buf].len);
    mdma->dmas[1].dma_start_send(dfs[1].dbuf_set[s_buf].len);
    mdma->dmas[2].dma_start_send(dfs[2].dbuf_set[s_buf].len);
    mdma->dmas[3].dma_start_send(dfs[3].dbuf_set[s_buf].len);
    End_Transfer();
    dsr.sID++;
  }

  void Set_Results() {
    int s_buf = find_dbuf(dfs[0], dsr.sID);
    mdma->multi_dma_change_end(dfs[0].dbuf_set[s_buf].offset);
    mdma->multi_dma_start_recv(recv_len);
  }

  void Recieve_Results() { mdma->multi_dma_wait_recv_4(); }
};

//========================//========================//========================//

void preload_weights(int8_t* weight_data, int* dims, vector<int8_t>& wb0,
                     vector<int8_t>& wb1, vector<int8_t>& wb2,
                     vector<int8_t>& wb3, vector<int>& wt_sum1,
                     vector<int>& wt_sum2, vector<int>& wt_sum3,
                     vector<int>& wt_sum4) {
  int width = dims[0];
  int w = ((width + 4 - 1) - ((width + 4 - 1) % 4));
  int depth = dims[1] * dims[2] * dims[3];
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
  }
}

void preload_weights(const int8_t* weight_data, int* dims, vector<int8_t>& wb0,
                     vector<int8_t>& wb1, vector<int8_t>& wb2,
                     vector<int8_t>& wb3, vector<int>& wt_sum1,
                     vector<int>& wt_sum2, vector<int>& wt_sum3,
                     vector<int>& wt_sum4) {
  int width = dims[0];
  int w = ((width + 4 - 1) - ((width + 4 - 1) % 4));
  int depth = dims[1] * dims[2] * dims[3];
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
  }
}

void precal_sum_load_pad(const int8_t* data, int width, int depth, int8_t* inb0,
                         int8_t* inb1, int8_t* inb2, int8_t* inb3, int* in_sum1,
                         int* in_sum2, int* in_sum3, int* in_sum4) {
  int w = ((width + 3) - ((width + 3) % 4));
  int d = ((depth + 15) - ((depth + 15) % 16));
  int d2 = depth * 2;
  int d3 = depth * 3;
  int d4 = depth * 4;
  int i_c = 0;
  int sums_curr = 0;

  const int8_t* rhs_d = reinterpret_cast<const int8_t*>(data);
  int dm = 0;
  for (int i = 0; i < w / 4; i++) {
    int id = i * d4;
    int i0 = id;
    int i1 = id + depth;
    int i2 = id + d2;
    int i3 = id + d3;
    int ss0 = 0;
    int ss1 = 0;
    int ss2 = 0;
    int ss3 = 0;

#ifdef ACC_NEON
    dm = d - 16;
    int8x16_t tmp0;
    int8x16_t tmp1;
    int8x16_t tmp2;
    int8x16_t tmp3;

    int32x4_t tmp0_2;
    int32x4_t tmp1_2;
    int32x4_t tmp2_2;
    int32x4_t tmp3_2;

    int32x2_t tmp0_3;
    int32x2_t tmp1_3;
    int32x2_t tmp2_3;
    int32x2_t tmp3_3;
    int32x2_t tmp0_4 = vdup_n_s32(0);
    int32x2_t tmp1_4 = vdup_n_s32(0);
    int32x2_t tmp2_4 = vdup_n_s32(0);
    int32x2_t tmp3_4 = vdup_n_s32(0);

    for (int j = 0; j < dm; j += 16) {
      tmp0 = vld1q_s8(rhs_d + i0 + j);
      tmp1 = vld1q_s8(rhs_d + i1 + j);
      tmp2 = vld1q_s8(rhs_d + i2 + j);
      tmp3 = vld1q_s8(rhs_d + i3 + j);
      tmp0_2 = vpaddlq_s16(vpaddlq_s8(tmp0));
      tmp1_2 = vpaddlq_s16(vpaddlq_s8(tmp1));
      tmp2_2 = vpaddlq_s16(vpaddlq_s8(tmp2));
      tmp3_2 = vpaddlq_s16(vpaddlq_s8(tmp3));

      tmp0_3 = vadd_s32(vget_high_s32(tmp0_2), vget_low_s32(tmp0_2));
      tmp1_3 = vadd_s32(vget_high_s32(tmp1_2), vget_low_s32(tmp1_2));
      tmp2_3 = vadd_s32(vget_high_s32(tmp2_2), vget_low_s32(tmp2_2));
      tmp3_3 = vadd_s32(vget_high_s32(tmp3_2), vget_low_s32(tmp3_2));
      tmp0_4 = vadd_s32(tmp0_4, tmp0_3);
      tmp1_4 = vadd_s32(tmp1_4, tmp1_3);
      tmp2_4 = vadd_s32(tmp2_4, tmp2_3);
      tmp3_4 = vadd_s32(tmp3_4, tmp3_3);
      vst1q_s8(inb0 + i_c, tmp0);
      vst1q_s8(inb1 + i_c, tmp1);
      vst1q_s8(inb2 + i_c, tmp2);
      vst1q_s8(inb3 + i_c, tmp3);
      i_c += 16;
    }
    int32_t tmp0_s[2];
    int32_t tmp1_s[2];
    int32_t tmp2_s[2];
    int32_t tmp3_s[2];
    vst1_s32(tmp0_s, tmp0_4);
    vst1_s32(tmp1_s, tmp1_4);
    vst1_s32(tmp2_s, tmp2_4);
    vst1_s32(tmp3_s, tmp3_4);
    ss0 += tmp0_s[0] + tmp0_s[1];
    ss1 += tmp1_s[0] + tmp1_s[1];
    ss2 += tmp2_s[0] + tmp2_s[1];
    ss3 += tmp3_s[0] + tmp3_s[1];
#endif
    for (int j = dm; j < d; j++) {
      if (j < depth) {
        unsigned char w0 = data[i0 + j];
        unsigned char w1 = data[i1 + j];
        unsigned char w2 = data[i2 + j];
        unsigned char w3 = data[i3 + j];
        ss0 += w0;
        ss1 += w1;
        ss2 += w2;
        ss3 += w3;
        inb0[i_c] = w0;
        inb1[i_c] = w1;
        inb2[i_c] = w2;
        inb3[i_c++] = w3;
      } else {
        inb0[i_c] = 0;
        inb1[i_c] = 0;
        inb2[i_c] = 0;
        inb3[i_c++] = 0;
      }
    }
    in_sum1[sums_curr] = (ss0);
    in_sum2[sums_curr] = (ss1);
    in_sum3[sums_curr] = (ss2);
    in_sum4[sums_curr++] = (ss3);
  }
}

#endif  // ACC_CONTAINER