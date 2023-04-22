#ifndef MMHELPER
#define MMHELPER

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// #include "multi_threading.h"
#ifdef ACC_NEON
#include "arm_neon.h"
#endif

template <int r>
int rounddown(int x) {
  return x - x % r;
}

template <int r>
int roundup(int x) {
  return rounddown<r>(x + (r - 1));
}

int rounddown(int x, int r) { return x - x % r; }

int roundup(int x, int r) { return rounddown(x + (r - 1), r); }

void print_matrix(int N_dim, int M_dim, int8_t* matrix) {
  std::cout << "==================================" << std::endl;
  for (int n = 0; n < N_dim; n++) {
    std::cout << "|";
    for (int m = 0; m < M_dim; m++) {
      // std::cout << matrix[r * M_dim  + c];
      printf("%-3d", matrix[n * M_dim + m]);
      if (m + 1 < M_dim) std::cout << ",";
    }
    std::cout << "|" << std::endl;
  }
  std::cout << "==================================" << std::endl;
}

void print_matrix(int N_dim, int M_dim, const int8_t* matrix) {
  std::cout << "==================================" << std::endl;
  for (int n = 0; n < N_dim; n++) {
    std::cout << "|";
    for (int m = 0; m < M_dim; m++) {
      // std::cout << matrix[r * M_dim  + c];
      printf("%-3d", matrix[n * M_dim + m]);
      if (m + 1 < M_dim) std::cout << ",";
    }
    std::cout << "|" << std::endl;
  }
  std::cout << "==================================" << std::endl;
}

void print_matrix(int N_dim, int M_dim, int8_t** matrix) {
  std::cout << "==================================" << std::endl;
  for (int n = 0; n < N_dim; n++) {
    std::cout << "|";
    for (int m = 0; m < M_dim; m++) {
      // std::cout << matrix[r * M_dim  + c];
      printf("%-3d", matrix[n][m]);
      if (m + 1 < M_dim) std::cout << ",";
    }
    std::cout << "|" << std::endl;
  }
  std::cout << "==================================" << std::endl;
}

void print_matrix(int N_dim, int M_dim, int8_t* matrix, std::string header) {
  std::cout << header << std::endl;
  print_matrix(N_dim, M_dim, matrix);
}

void save_matrix(std::string file, int N_dim, int M_dim, int8_t* matrix) {
  std::ofstream outfile;
  outfile.open(file, std::ios_base::app);
  outfile << "==================================" << std::endl;
  for (int n = 0; n < N_dim; n++) {
    outfile << "|";
    for (int m = 0; m < M_dim; m++) {
      outfile << (int)matrix[n * M_dim + m];
      if (m + 1 < M_dim) outfile << ",";
    }
    outfile << "|" << std::endl;
  }
  outfile << "==================================" << std::endl;
}

void compare_matrix(int N_dim, int M_dim, int8_t* A, int8_t* B) {
  bool equal = true;
  for (int n = 0; n < N_dim; n++) {
    for (int m = 0; m < M_dim; m++) {
      if (A[n * M_dim + m] != B[n * M_dim + m]) {
        equal = false;
        break;
      }
    }
    if (!equal) break;
  }
  if (equal)
    std::cout << "A == B" << std::endl;
  else
    std::cout << "A != B" << std::endl;
}

void simpleMM(int N_dim, int M_dim, int K_dim, int8_t* A, int8_t* B,
              int8_t* C) {
  for (int n = 0; n < N_dim; n++) {
    for (int m = 0; m < M_dim; m++) {
      int acc = 0;
      for (int k = 0; k < K_dim; k++) {
        int x = A[n * K_dim + k];
        int y = B[k * M_dim + m];
        acc += x * y;
      }
      C[n * M_dim + m] = acc;
    }
  }
}

void trans_matrix(int N_dim, int pN, int M_dim, const int8_t* A, int8_t** B) {
  // for (int n = 0; n < N_dim; n++)
  //   for (int m = 0; m < M_dim; m++) B[m * pN + n] = A[n * M_dim + m];
  for (int n = 0; n < N_dim; n++)
    for (int m = 0; m < M_dim; m++) B[m][n] = A[n * M_dim + m];
}

void pad_matrix(int N_dim, int M_dim, int tN, int tM, const int8_t* A,
                int8_t* padded_A) {
  int pM = roundup(M_dim, tM);
  for (int n = 0; n < N_dim; n++) {
    for (int m = 0; m < M_dim; m++) {
      padded_A[n * pM + m] = A[n * M_dim + m];
    }
  }
  // int pN = roundup(N_dim, tN);
  // print_matrix(pN, pM, padded_A, "Padded Matrix");
}

void pad_matrix(int N_dim, int M_dim, int tN, int tM, const int8_t* A,
                int8_t** padded_A) {
  int pM = roundup(M_dim, tM);
  for (int n = 0; n < N_dim; n++) {
    for (int m = 0; m < M_dim; m++) {
      padded_A[n][m] = A[n * M_dim + m];
    }
  }
  // int pN = roundup(N_dim, tN);
  // print_matrix(N_dim, M_dim, A);
  // print_matrix(pN, pM, padded_A);
}

void unpad_matrix(int N_dim, int M_dim, int tN, int tM, int8_t* padded_A,
                  int8_t* A) {
  int pM = roundup(M_dim, tM);
  for (int n = 0; n < N_dim; n++) {
    for (int m = 0; m < M_dim; m++) {
      A[n * M_dim + m] = padded_A[n * pM + m];
    }
  }
  // int pN = roundup(N_dim, tN);
  // print_matrix(pN, pM, padded_A, "Padded Matrix");
  // print_matrix(N_dim, M_dim, A, "UnPadded Matrix");
}

void unpad_matrix(int N_dim, int M_dim, int tN, int tM, int8_t** padded_A,
                  int8_t* A) {
  int pM = roundup(M_dim, tM);
  for (int n = 0; n < N_dim; n++) {
    for (int m = 0; m < M_dim; m++) {
      A[n * M_dim + m] = padded_A[n][m];
    }
  }
  // int pN = roundup(N_dim, tN);
  // print_matrix(pN, pM, padded_A, "Padded Matrix");
  // print_matrix(N_dim, M_dim, A, "UnPadded Matrix");
}

void unpadT_matrix(int N_dim, int M_dim, int tN, int tM, int8_t** padded_A,
                   int8_t* A) {
  for (int n = 0; n < N_dim; n++) {
    for (int m = 0; m < M_dim; m++) {
      A[n * M_dim + m] = padded_A[m][n];
    }
  }
}

void padT_matrix(int N_dim, int M_dim, int tN, int tM, const int8_t* A,
                 int8_t** padded_AT) {
  int pN = roundup(N_dim, tN);
  trans_matrix(N_dim, pN, M_dim, A, padded_AT);
  // int pM = roundup(M_dim, tM);
  // print_matrix(pM, pN, padded_AT, "Padded_Transformed Matrix");
}

void create_2d_biases_flipped(int sn, int N_dim, int sm, int M_dim,
                              uint32_t* new_bias, int32_t* bias,
                              int32_t* wt_sum, int* in_sum, int32_t rhs_offset,
                              int32_t lhs_offset, int32_t depth) {
  int offdepth = depth * rhs_offset;
  for (int m = 0; m < M_dim; m++) {
    for (int n = 0; n < N_dim; n++) {
      int yt = ((in_sum[sn + n] + offdepth) * lhs_offset);
      int xt = bias[sm + m] + (wt_sum[sm + m] * rhs_offset);
      new_bias[m * N_dim + n] = yt + xt;
    }
  }
}

// void create_2d_biases(int sn, int N_dim, int sm, int M_dim, uint32_t*
// new_bias,
//                       int32_t* bias, int32_t* wt_sum, int* in_sum,
//                       int32_t rhs_offset, int32_t lhs_offset, int32_t depth)
//                       {
//   int offdepth = depth * rhs_offset;
//   for (int m = 0; m < M_dim; m++) {
//     for (int n = 0; n < N_dim; n++) {
//       int yt = ((in_sum[sn + n] + offdepth) * lhs_offset);
//       int xt = bias[sm + m] + (wt_sum[sm + m] * rhs_offset);
//       new_bias[n * M_dim + m] = yt + xt;
//     }
//   }
// }

void create_2d_biases(int sn, int N_dim, int sm, int M_dim, int32_t* new_bias,
                      int32_t* bias, int32_t* wt_sum, int* in_sum,
                      int32_t rhs_offset, int32_t lhs_offset, int32_t depth) {
  int offdepth = 0;
  if (-lhs_offset && -rhs_offset)
    offdepth = (-lhs_offset) * depth * (-rhs_offset);

#ifndef ACC_NEON
  for (int m = 0; m < M_dim; m++) {
    for (int n = 0; n < N_dim; n++) {
      int yt = (in_sum[sn + n] * lhs_offset) + offdepth;
      int xt = bias[sm + m] + (wt_sum[sm + m] * rhs_offset);
      new_bias[n * M_dim + m] = yt + xt;
    }
  }
#else

  int32x4_t tmp_offdepth_4 = vdupq_n_s32(offdepth);
  int32x4_t tmp_rhs_offset_4 = vdupq_n_s32(rhs_offset);
  for (int n = 0; n < N_dim; n++) {
    int yt = (in_sum[sn + n] * lhs_offset) + offdepth;
    int32x4_t tmp_yt = vdupq_n_s32(yt);
    for (int m = 0; m < M_dim; m += 4) {
      int32x4_t tmp_bias = vld1q_s32(bias + sm + m);
      int32x4_t tmp_wt_sum = vld1q_s32(wt_sum + sm + m);
      int32x4_t tmp_xt_mul = vmulq_s32(tmp_wt_sum, tmp_rhs_offset_4);
      int32x4_t tmp_xt = vaddq_s32(tmp_xt_mul, tmp_bias);
      int32x4_t temp_nb = vaddq_s32(tmp_yt, tmp_xt);
      vst1q_s32(new_bias + n * M_dim + m, temp_nb);
    }
  }

#endif
}

void precal_sum_load_pad(int8_t* data, int width, int depth, int8_t* shape_data,
                         int* sums) {
  int w = ((width + 16 - 1) - ((width + 16 - 1) % 16));
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
  int dm = d - 16;
  int i_c = 0;
  for (int i = 0; i < w; i++) {
    int s0 = 0;
    if (i < width) {
#ifndef ACC_NEON
      for (int j = 0; j < d; j++) {
        if (j < depth) {
          int8_t val = data[(i * depth) + j];
          s0 += val;
          shape_data[i_c++] = val;
        } else {
          shape_data[i_c++] = 0;
        }
      }
#else
      int8x16_t tmp0;
      int16x8_t tmp0_1;
      int32x4_t tmp0_2;
      int32x2_t tmp0_3;
      int32x2_t tmp0_4 = vdup_n_s32(0);
      int32_t tmp0_s[2];
      for (int j = 0; j < dm; j += 16) {
        tmp0 = vld1q_s8(data + (i * depth) + j);
        tmp0_1 = vpaddlq_s8(tmp0);
        tmp0_2 = vpaddlq_s16(tmp0_1);
        tmp0_3 = vadd_s32(vget_high_s32(tmp0_2), vget_low_s32(tmp0_2));
        tmp0_4 = vadd_s32(tmp0_4, tmp0_3);
        vst1q_s8(shape_data + i_c, tmp0);
        i_c += 16;
      }
      vst1_s32(tmp0_s, tmp0_4);
      s0 += tmp0_s[0] + tmp0_s[1];
      for (int j = dm; j < d; j++) {
        if (j < depth) {
          int8_t val = data[(i * depth) + j];
          s0 += val;
          shape_data[i_c++] = val;
        } else {
          shape_data[i_c++] = 0;
        }
      }
#endif
    } else {
      for (int j = 0; j < d; j++) shape_data[i_c++] = 0;
    }
    sums[w] = (s0);
  }
}

void store_unpad(int8_t* data, int width, int depth, int8_t* shape_data) {
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
  int dm = rounddown(depth, 16);
  int i_c = 0;
  for (int i = 0; i < width; i++) {
#ifndef ACC_NEON
    for (int j = 0; j < depth; j++) {
      int8_t val = data[(i * d) + j];
      shape_data[i_c++] = val;
    }
#else
    int8x16_t tmp0;
    for (int j = 0; j < dm; j += 16) {
      tmp0 = vld1q_s8(data + (i * d) + j);
      vst1q_s8(shape_data + i_c, tmp0);
      i_c += 16;
    }
    for (int j = dm; j < depth; j++) {
      int8_t val = data[(i * d) + j];
      shape_data[i_c++] = val;
    }
#endif
  }
}

// void precal_sum_load_pad(int8_t* data, int width, int depth, int8_t* shape_data,
//                          vector<int>& sums) {
//   int w = ((width + 16 - 1) - ((width + 16 - 1) % 16));
//   int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
//   int dm = d - 16;
//   int i_c = 0;
//   for (int i = 0; i < w; i++) {
//     int s0 = 0;
//     if (i < width) {
// #ifndef ACC_NEON
//       for (int j = 0; j < d; j++) {
//         if (j < depth) {
//           int8_t val = data[(i * depth) + j];
//           s0 += val;
//           shape_data[i_c++] = val;
//         } else {
//           shape_data[i_c++] = 0;
//         }
//       }
// #else
//       int8x16_t tmp0;
//       int16x8_t tmp0_1;
//       int32x4_t tmp0_2;
//       int32x2_t tmp0_3;
//       int32x2_t tmp0_4 = vdup_n_s32(0);
//       int32_t tmp0_s[2];
//       for (int j = 0; j < dm; j += 16) {
//         tmp0 = vld1q_s8(data + (i * depth) + j);
//         tmp0_1 = vpaddlq_s8(tmp0);
//         tmp0_2 = vpaddlq_s16(tmp0_1);
//         tmp0_3 = vadd_s32(vget_high_s32(tmp0_2), vget_low_s32(tmp0_2));
//         tmp0_4 = vadd_s32(tmp0_4, tmp0_3);
//         vst1q_s8(shape_data + i_c, tmp0);
//         i_c += 16;
//       }
//       vst1_s32(tmp0_s, tmp0_4);
//       s0 += tmp0_s[0] + tmp0_s[1];
//       for (int j = dm; j < d; j++) {
//         if (j < depth) {
//           int8_t val = data[(i * depth) + j];
//           s0 += val;
//           shape_data[i_c++] = val;
//         } else {
//           shape_data[i_c++] = 0;
//         }
//       }
// #endif
//     } else {
//       for (int j = 0; j < d; j++) shape_data[i_c++] = 0;
//     }
//     // sums.push_back(s0);
//     sums[w] = (s0);
//   }
// }

#endif  // MMHELPER