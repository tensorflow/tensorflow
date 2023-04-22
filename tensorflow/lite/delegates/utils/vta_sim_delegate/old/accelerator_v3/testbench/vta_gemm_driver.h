#ifndef DATA_HANDLER
#define DATA_HANDLER

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <strstream>
#include <typeinfo>

#include "gen_instructions.h"
#include "vta_driver.h"

namespace tflite_secda {

template <typename Integer>
int roundUp(int numToRound, int multiple) {
  if (multiple == 0) return numToRound;
  int remainder = numToRound % multiple;
  if (remainder == 0) return numToRound;
  return numToRound + multiple - remainder;
}

template <typename DST_T, int DST_T_WIDTH, typename SRC_T, int SRC_T_WIDTH>
void unpackBufferBlock(DST_T** dblock, SRC_T* src, int r, int r_b, int r_block,
                       int c, int c_b, int c_block) {
  assert((DST_T_WIDTH * r_block * c_block) % SRC_T_WIDTH == 0);
  int buffer_idx = 0;
  long long int mask = (1ULL << DST_T_WIDTH) - 1;
  int ratio = SRC_T_WIDTH / DST_T_WIDTH;
  for (int i = 0; i < r_b / r_block; i++) {
    for (int j = 0; j < c_b / c_block; j++) {
      for (int k = 0; k < r_block; k++) {
        for (int l = 0; l < c_block; l++) {
          int block_idx = l + k * c_block;
          dblock[r + i * r_block + k][c + j * c_block + l] =
              (src[buffer_idx] >> ((block_idx % ratio) * DST_T_WIDTH)) & mask;
          if (block_idx % ratio == ratio - 1) {
            buffer_idx++;
          }
        }
      }
    }
  }
}

template <typename DST_T, int DST_T_WIDTH, typename SRC_T, int SRC_T_WIDTH>
void packBufferBlock(DST_T* dblock, SRC_T** src, int r, int r_b, int r_block,
                     int c, int c_b, int c_block) {
  assert((SRC_T_WIDTH * r_block * c_block) % DST_T_WIDTH == 0);
  assert(DST_T_WIDTH <= 64);

  int buffer_idx = 0;
  int ratio = DST_T_WIDTH / SRC_T_WIDTH;
  long long int mask = (1ULL << SRC_T_WIDTH) - 1;
  DST_T tmp = 0;
  for (int i = 0; i < r_b / r_block; i++) {
    for (int j = 0; j < c_b / c_block; j++) {
      for (int k = 0; k < r_block; k++) {
        for (int l = 0; l < c_block; l++) {
          int block_idx = l + k * c_block;
          int data_vale = src[r + i * r_block + k][c + j * c_block + l];
          // cout << (r + i * r_block + k) << " " << (c + j * c_block + l) << "
          // "
          //      << data_vale << endl;
          tmp |= (src[r + i * r_block + k][c + j * c_block + l] & mask)
                 << ((block_idx % ratio) * SRC_T_WIDTH);
          if (block_idx % ratio == ratio - 1) {
            dblock[buffer_idx++] = tmp;
            tmp = 0;
          }
        }
      }
    }
  }
}

template <typename PackedResult>
void BlockIntoVTAv3(conv2d_driver& drv) {
  int K = drv.pK;
  int M = drv.pM;
  int N = drv.pN;
  int8_t** padded_input = drv.padded_input;
  int8_t** padded_weights = drv.padded_weights;
  int8_t** padded_output = drv.padded_output;
  int32_t* wt_sum = &drv.wt_sum[0];
  int32_t* in_sum = drv.in_sum;
  if (drv.flipped) {
    M = drv.pN;
    N = drv.pM;
    padded_input = drv.padded_weights;
    padded_weights = drv.padded_input;
  }
  drv.acc->flipped = drv.flipped;
  drv.scs->sig_ra_sig.write(drv.ra);
  uint32_t* crf_buf = (uint32_t*)(drv.crf);
  uint32_t* crx_buf = (uint32_t*)(drv.crx);
  uint32_t* weight_buf = new uint32_t[K * M];
  packBufferBlock<uint32_t, 32, int8_t, 8>(weight_buf, padded_weights, 0, M,
                                           VTA_BLOCK_OUT, 0, K, VTA_BLOCK_IN);

  uint32_t* input_buf = new uint32_t[N * K];
  packBufferBlock<uint32_t, 32, int8_t, 8>(input_buf, padded_input, 0, N,
                                           VTA_BATCH, 0, K, VTA_BLOCK_IN);

  uint32_t* bias_buf = new uint32_t[N * M];
  uint32_t* out_buf = new uint32_t[N * M];

  if (drv.flipped) {
    create_2d_biases_flipped(0, M, 0, N, bias_buf, drv.bias, wt_sum, in_sum,
                             drv.rhs_offset, drv.lhs_offset, drv.K);
  } else {
    create_2d_biases(0, N, 0, M, bias_buf, drv.bias, wt_sum, in_sum,
                     drv.rhs_offset, drv.lhs_offset, drv.K);
  }

  blocked_gemm_test_tflitev3(drv, drv.scs, N, K, M, 0, true, 1, input_buf,
                             weight_buf, bias_buf, out_buf, crf_buf, crx_buf,
                             drv.save, drv.ra, drv.layer, drv.flipped);

  unpackBufferBlock<int8_t, 8, uint32_t, 32>(padded_output, out_buf, 0, N,
                                             VTA_BATCH, 0, M, VTA_BLOCK_OUT);
}

template <typename PackedResult>
void BlockIntoVTAv4(conv2d_driver& drv) {
  int K = drv.pK;
  int M = drv.flipped ? drv.pN : drv.pM;
  int N = drv.flipped ? drv.pM : drv.pN;

  int32_t* wt_sum = &drv.wt_sum[0];
  int32_t* in_sum = drv.in_sum;

  int8_t** padded_output = drv.padded_output;
  uint32_t* weight_buf;
  uint32_t* input_buf;

  uint32_t* crf_buf = (uint32_t*)(drv.crf);
  uint32_t* crx_buf = (uint32_t*)(drv.crx);
  uint32_t* bias_buf = new uint32_t[N * M];
  uint32_t* out_buf = new uint32_t[N * M];

  if (drv.flipped) {
    input_buf = drv.packed_weights;
    weight_buf = new uint32_t[K * M];
    packBufferBlock<uint32_t, 32, int8_t, 8>(weight_buf, drv.padded_input, 0, M,
                                             VTA_BLOCK_OUT, 0, K, VTA_BLOCK_IN);
    create_2d_biases_flipped(0, M, 0, N, bias_buf, drv.bias, wt_sum, in_sum,
                             drv.rhs_offset, drv.lhs_offset, drv.K);
  } else {
    weight_buf = drv.packed_weights;
    input_buf = new uint32_t[N * K];
    packBufferBlock<uint32_t, 32, int8_t, 8>(input_buf, drv.padded_input, 0, N,
                                             VTA_BATCH, 0, K, VTA_BLOCK_IN);
    create_2d_biases(0, N, 0, M, bias_buf, drv.bias, wt_sum, in_sum,
                     drv.rhs_offset, drv.lhs_offset, drv.K);
  }

  // std::string mname = "conv_v1";
  // std::string mw = "";
  // std::string mi = "";
  // std::string mo = "";

  // {
  //   ofstream myfile;
  //   myfile.open("a_Vta/" + mname + mi + "/del_packed_wgt_" +
  //               std::to_string(drv.layer) + ".csv");
  //   int8_t* res_pointer = (int8_t*)weight_buf;
  //   int index = 0;
  //   for (int c = 0; c < M; c++) {
  //     myfile << endl;
  //     for (int r = 0; r < K; r++) {
  //       myfile << (int)res_pointer[index] << ",";
  //       index++;
  //     }
  //   }
  //   myfile.close();
  // }

  // {
  //   ofstream myfile;
  //   myfile.open("a_Vta/" + mname + mi + "/del_packed_in_" +
  //               std::to_string(drv.layer) + ".csv");
  //   int8_t* res_pointer = (int8_t*)input_buf;
  //   int index = 0;
  //   for (int c = 0; c < N; c++) {
  //     myfile << endl;
  //     for (int r = 0; r < K; r++) {
  //       myfile << (int)res_pointer[index] << ",";
  //       index++;
  //     }
  //   }
  //   myfile.close();
  // }

  blocked_gemm_test_tflitev3(drv, drv.scs, N, K, M, 0, true, 1, input_buf,
                             weight_buf, bias_buf, out_buf, crf_buf, crx_buf,
                             drv.save, drv.ra, drv.layer, drv.flipped);

  // {
  //   ofstream myfile;
  //   myfile.open("a_Vta/" + mname + mi + "/del_packed_out_" +
  //               std::to_string(drv.layer) + ".csv");
  //   int8_t* res_pointer = (int8_t*)out_buf;
  //   int index = 0;
  //   for (int c = 0; c < N; c++) {
  //     myfile << endl;
  //     for (int r = 0; r < M; r++) {
  //       myfile << (int)res_pointer[index] << ",";
  //       index++;
  //     }
  //   }
  //   myfile.close();
  // }

  unpackBufferBlock<int8_t, 8, uint32_t, 32>(padded_output, out_buf, 0, N,
                                             VTA_BATCH, 0, M, VTA_BLOCK_OUT);
  // {
  //   ofstream myfile;
  //   myfile.open("a_Vta/" + mname + mi + "/del_unpacked_out_" +
  //               std::to_string(drv.layer) + ".csv");
  //   int8_t** res_pointer = (int8_t**)padded_output;
  //   int index = 0;
  //   for (int c = 0; c < N; c++) {
  //     myfile << endl;
  //     for (int r = 0; r < M; r++) {
  //       myfile << (int)res_pointer[c][r] << ",";
  //       index++;
  //     }
  //   }
  //   myfile.close();
  // }
}

template <typename Integer>
void Entry(conv2d_driver& drv) {
  // cout << "VTA ACC - Layer: " << drv.layer << (drv.flipped?" Flipped":"") <<
  // endl; cout << "===========================" << endl; cout << "Pre-ACC Info"
  // << endl; cout << "padded_K: " << drv.pK << " K: " << drv.K << endl; cout <<
  // "padded_M: " << drv.pM << " M: " << drv.M << endl; cout << "padded_N: " <<
  // drv.pN << " N: " << drv.N << endl; cout << "===========================" <<
  // endl;

  // cout << drv.layer;
  // cout << " ," << drv.pK;
  // cout << " ," << drv.pM;
  // cout << " ," << drv.pN;
  // cout << " ," << drv.K;
  // cout << " ," << drv.M;
  // cout << " ," << drv.N << endl;

  // BlockIntoVTAv2<int>(drv);
  // BlockIntoVTAv3<int>(drv);

  BlockIntoVTAv4<int>(drv);
}
}  // namespace tflite_secda
#endif
