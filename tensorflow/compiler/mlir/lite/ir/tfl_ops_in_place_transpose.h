/* Copyright 2026 The LiteRT Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_OPS_IN_PLACE_TRANSPOSE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_OPS_IN_PLACE_TRANSPOSE_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace TFL {
namespace detail {

// Sub-byte bit-level read and write primitives.
inline uint8_t GetSubByte(const uint8_t* data, int64_t index, int bit_width) {
  const int64_t bit_offset = index * bit_width;
  const uint8_t mask = (1 << bit_width) - 1;
  return (data[bit_offset / 8] >> (bit_offset % 8)) & mask;
}

inline void SetSubByte(uint8_t* data, int64_t index, uint8_t val,
                       int bit_width) {
  const int64_t bit_offset = index * bit_width;
  const uint8_t mask = (1 << bit_width) - 1;
  const int shift = bit_offset % 8;
  data[bit_offset / 8] =
      (data[bit_offset / 8] & ~(mask << shift)) | ((val & mask) << shift);
}

// In-place 2D square matrix transpose with cache tiling (O(1) scratch memory).
template <typename T>
void InPlaceTransposeSquare2D(T* data, int64_t N) {
  constexpr int64_t kTile = 64;
  for (int64_t i_tile = 0; i_tile < N; i_tile += kTile) {
    int64_t i_end = std::min(i_tile + kTile, N);
    for (int64_t j_tile = i_tile; j_tile < N; j_tile += kTile) {
      int64_t j_end = std::min(j_tile + kTile, N);
      if (i_tile == j_tile) {
        for (int64_t i = i_tile; i < i_end; ++i) {
          for (int64_t j = i + 1; j < j_end; ++j) {
            std::swap(data[i * N + j], data[j * N + i]);
          }
        }
      } else {
        for (int64_t i = i_tile; i < i_end; ++i) {
          for (int64_t j = j_tile; j < j_end; ++j) {
            std::swap(data[i * N + j], data[j * N + i]);
          }
        }
      }
    }
  }
}

// Fast 2D rectangular matrix transpose with cache tiling and transient buffer.
template <typename T>
void FastTiledTransposeRectangular2D(T* data, int64_t R, int64_t C) {
  const int64_t total = R * C;
  if (total <= 2) return;

  std::vector<T> tmp(total);
  T* dst = tmp.data();
  const T* src = data;
  constexpr int64_t kTile = 64;
  for (int64_t r_tile = 0; r_tile < R; r_tile += kTile) {
    int64_t r_end = std::min(r_tile + kTile, R);
    for (int64_t c_tile = 0; c_tile < C; c_tile += kTile) {
      int64_t c_end = std::min(c_tile + kTile, C);
      for (int64_t r = r_tile; r < r_end; ++r) {
        for (int64_t c = c_tile; c < c_end; ++c) {
          dst[c * R + r] = src[r * C + c];
        }
      }
    }
  }
  std::memcpy(data, tmp.data(), total * sizeof(T));
}

template <typename T>
void InPlaceTranspose2D(T* data, int64_t R, int64_t C) {
  if (R <= 1 || C <= 1) return;
  if (R == C) {
    InPlaceTransposeSquare2D(data, R);
  } else {
    FastTiledTransposeRectangular2D(data, R, C);
  }
}

// In-place 2D transpose for sub-byte types (INT4, INT2, INT1).
inline void InPlaceTransposeSubByte2D(uint8_t* data, int64_t R, int64_t C,
                                      int bit_width) {
  if (R <= 1 || C <= 1) return;
  const int64_t total = R * C;
  if (total <= 2) return;

  if (R == C) {
    for (int64_t i = 0; i < R; ++i) {
      for (int64_t j = i + 1; j < C; ++j) {
        int64_t idx1 = i * C + j;
        int64_t idx2 = j * C + i;
        uint8_t v1 = GetSubByte(data, idx1, bit_width);
        uint8_t v2 = GetSubByte(data, idx2, bit_width);
        SetSubByte(data, idx1, v2, bit_width);
        SetSubByte(data, idx2, v1, bit_width);
      }
    }
    return;
  }

  const int64_t total_bytes = (total * bit_width + 7) / 8;
  std::vector<uint8_t> tmp(total_bytes, 0);
  uint8_t* dst = tmp.data();
  const uint8_t* src = data;
  constexpr int64_t kTile = 64;
  for (int64_t r_tile = 0; r_tile < R; r_tile += kTile) {
    int64_t r_end = std::min(r_tile + kTile, R);
    for (int64_t c_tile = 0; c_tile < C; c_tile += kTile) {
      int64_t c_end = std::min(c_tile + kTile, C);
      for (int64_t r = r_tile; r < r_end; ++r) {
        for (int64_t c = c_tile; c < c_end; ++c) {
          uint8_t val = GetSubByte(src, r * C + c, bit_width);
          SetSubByte(dst, c * R + r, val, bit_width);
        }
      }
    }
  }
  std::memcpy(data, tmp.data(), total_bytes);
}

// Generic fallback for arbitrary element byte size.
inline void InPlaceTransposeGeneric2D(char* data, int64_t R, int64_t C,
                                      int element_size) {
  if (R <= 1 || C <= 1) return;
  const int64_t total = R * C;
  if (total <= 2) return;

  if (R == C) {
    llvm::SmallVector<char, 64> temp_buf(element_size);
    for (int64_t i = 0; i < R; ++i) {
      for (int64_t j = i + 1; j < C; ++j) {
        char* p1 = data + (i * C + j) * element_size;
        char* p2 = data + (j * C + i) * element_size;
        std::memcpy(temp_buf.data(), p1, element_size);
        std::memcpy(p1, p2, element_size);
        std::memcpy(p2, temp_buf.data(), element_size);
      }
    }
    return;
  }

  std::vector<char> tmp(total * element_size);
  char* dst = tmp.data();
  const char* src = data;
  constexpr int64_t kTile = 64;
  for (int64_t r_tile = 0; r_tile < R; r_tile += kTile) {
    int64_t r_end = std::min(r_tile + kTile, R);
    for (int64_t c_tile = 0; c_tile < C; c_tile += kTile) {
      int64_t c_end = std::min(c_tile + kTile, C);
      for (int64_t r = r_tile; r < r_end; ++r) {
        for (int64_t c = c_tile; c < c_end; ++c) {
          std::memcpy(dst + (c * R + r) * element_size,
                      src + (r * C + c) * element_size, element_size);
        }
      }
    }
  }
  std::memcpy(data, tmp.data(), total * element_size);
}

// Arbitrary N-D tensor transpose.
template <typename T>
void InPlaceTransposeNDImpl(T* data, llvm::ArrayRef<int64_t> input_shape,
                            llvm::ArrayRef<int64_t> perms) {
  const int rank = input_shape.size();
  if (rank <= 1) return;

  int64_t total_elements = 1;
  for (int64_t d : input_shape) total_elements *= d;
  if (total_elements <= 2) return;

  // Compute input strides
  llvm::SmallVector<int64_t, 8> in_strides(rank, 1);
  for (int i = rank - 2; i >= 0; --i) {
    in_strides[i] = in_strides[i + 1] * input_shape[i + 1];
  }

  // Compute output shape and output strides
  llvm::SmallVector<int64_t, 8> out_shape(rank);
  for (int i = 0; i < rank; ++i) {
    out_shape[i] = input_shape[perms[i]];
  }
  llvm::SmallVector<int64_t, 8> out_strides(rank, 1);
  for (int i = rank - 2; i >= 0; --i) {
    out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
  }

  // Map input dimension k to output stride
  llvm::SmallVector<int64_t, 8> out_strides_of_in_dim(rank);
  for (int out_dim = 0; out_dim < rank; ++out_dim) {
    int in_dim = perms[out_dim];
    out_strides_of_in_dim[in_dim] = out_strides[out_dim];
  }

  std::vector<T> tmp(total_elements);
  for (int64_t x = 0; x < total_elements; ++x) {
    int64_t y = 0;
    for (int k = 0; k < rank; ++k) {
      int64_t coord = (x / in_strides[k]) % input_shape[k];
      y += coord * out_strides_of_in_dim[k];
    }
    tmp[y] = data[x];
  }
  std::memcpy(data, tmp.data(), total_elements * sizeof(T));
}

// Out-of-place 2D fast cache-tiled matrix transpose.
template <typename T>
void FastTiledTranspose2DOutPlace(const T* src, T* dst, int64_t R, int64_t C) {
  constexpr int64_t kTile = 64;
  for (int64_t r_tile = 0; r_tile < R; r_tile += kTile) {
    int64_t r_end = std::min(r_tile + kTile, R);
    for (int64_t c_tile = 0; c_tile < C; c_tile += kTile) {
      int64_t c_end = std::min(c_tile + kTile, C);
      for (int64_t r = r_tile; r < r_end; ++r) {
        for (int64_t c = c_tile; c < c_end; ++c) {
          dst[c * R + r] = src[r * C + c];
        }
      }
    }
  }
}

// Out-of-place sub-byte 2D fast cache-tiled matrix transpose.
inline void FastTiledTransposeSubByte2DOutPlace(const uint8_t* src,
                                                uint8_t* dst, int64_t R,
                                                int64_t C, int bit_width) {
  constexpr int64_t kTile = 64;
  for (int64_t r_tile = 0; r_tile < R; r_tile += kTile) {
    int64_t r_end = std::min(r_tile + kTile, R);
    for (int64_t c_tile = 0; c_tile < C; c_tile += kTile) {
      int64_t c_end = std::min(c_tile + kTile, C);
      for (int64_t r = r_tile; r < r_end; ++r) {
        for (int64_t c = c_tile; c < c_end; ++c) {
          uint8_t val = GetSubByte(src, r * C + c, bit_width);
          SetSubByte(dst, c * R + r, val, bit_width);
        }
      }
    }
  }
}

// Out-of-place arbitrary N-D tensor transpose.
template <typename T>
void TransposeNDOutPlace(const T* src, T* dst,
                         llvm::ArrayRef<int64_t> input_shape,
                         llvm::ArrayRef<int64_t> perms) {
  const int rank = input_shape.size();
  int64_t total_elements = 1;
  for (int64_t d : input_shape) total_elements *= d;

  llvm::SmallVector<int64_t, 8> in_strides(rank, 1);
  for (int i = rank - 2; i >= 0; --i) {
    in_strides[i] = in_strides[i + 1] * input_shape[i + 1];
  }
  llvm::SmallVector<int64_t, 8> out_shape(rank);
  for (int i = 0; i < rank; ++i) {
    out_shape[i] = input_shape[perms[i]];
  }
  llvm::SmallVector<int64_t, 8> out_strides(rank, 1);
  for (int i = rank - 2; i >= 0; --i) {
    out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
  }
  llvm::SmallVector<int64_t, 8> out_strides_of_in_dim(rank);
  for (int out_dim = 0; out_dim < rank; ++out_dim) {
    int in_dim = perms[out_dim];
    out_strides_of_in_dim[in_dim] = out_strides[out_dim];
  }

  for (int64_t x = 0; x < total_elements; ++x) {
    int64_t y = 0;
    for (int k = 0; k < rank; ++k) {
      int64_t coord = (x / in_strides[k]) % input_shape[k];
      y += coord * out_strides_of_in_dim[k];
    }
    dst[y] = src[x];
  }
}

// Out-of-place arbitrary N-D tensor transpose for sub-byte types.
inline void TransposeSubByteNDOutPlace(const uint8_t* src, uint8_t* dst,
                                       llvm::ArrayRef<int64_t> input_shape,
                                       llvm::ArrayRef<int64_t> perms,
                                       int bit_width) {
  const int rank = input_shape.size();
  int64_t total_elements = 1;
  for (int64_t d : input_shape) total_elements *= d;

  llvm::SmallVector<int64_t, 8> in_strides(rank, 1);
  for (int i = rank - 2; i >= 0; --i) {
    in_strides[i] = in_strides[i + 1] * input_shape[i + 1];
  }
  llvm::SmallVector<int64_t, 8> out_shape(rank);
  for (int i = 0; i < rank; ++i) {
    out_shape[i] = input_shape[perms[i]];
  }
  llvm::SmallVector<int64_t, 8> out_strides(rank, 1);
  for (int i = rank - 2; i >= 0; --i) {
    out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
  }
  llvm::SmallVector<int64_t, 8> out_strides_of_in_dim(rank);
  for (int out_dim = 0; out_dim < rank; ++out_dim) {
    int in_dim = perms[out_dim];
    out_strides_of_in_dim[in_dim] = out_strides[out_dim];
  }

  for (int64_t x = 0; x < total_elements; ++x) {
    int64_t y = 0;
    for (int k = 0; k < rank; ++k) {
      int64_t coord = (x / in_strides[k]) % input_shape[k];
      y += coord * out_strides_of_in_dim[k];
    }
    uint8_t val = GetSubByte(src, x, bit_width);
    SetSubByte(dst, y, val, bit_width);
  }
}

inline void InPlaceTransposeSubByteND(uint8_t* data,
                                      llvm::ArrayRef<int64_t> input_shape,
                                      llvm::ArrayRef<int64_t> perms,
                                      int bit_width) {
  int64_t total_elements = 1;
  for (int64_t d : input_shape) total_elements *= d;
  if (total_elements <= 1) return;

  const int64_t total_bytes = (total_elements * bit_width + 7) / 8;
  std::vector<uint8_t> tmp(total_bytes, 0);
  TransposeSubByteNDOutPlace(data, tmp.data(), input_shape, perms, bit_width);
  std::memcpy(data, tmp.data(), total_bytes);
}

// Out-of-place arbitrary N-D tensor transpose for generic element size.
inline void TransposeGenericNDOutPlace(const char* src, char* dst,
                                       llvm::ArrayRef<int64_t> input_shape,
                                       llvm::ArrayRef<int64_t> perms,
                                       int element_size) {
  const int rank = input_shape.size();
  int64_t total_elements = 1;
  for (int64_t d : input_shape) total_elements *= d;

  llvm::SmallVector<int64_t, 8> in_strides(rank, 1);
  for (int i = rank - 2; i >= 0; --i) {
    in_strides[i] = in_strides[i + 1] * input_shape[i + 1];
  }
  llvm::SmallVector<int64_t, 8> out_shape(rank);
  for (int i = 0; i < rank; ++i) {
    out_shape[i] = input_shape[perms[i]];
  }
  llvm::SmallVector<int64_t, 8> out_strides(rank, 1);
  for (int i = rank - 2; i >= 0; --i) {
    out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
  }
  llvm::SmallVector<int64_t, 8> out_strides_of_in_dim(rank);
  for (int out_dim = 0; out_dim < rank; ++out_dim) {
    int in_dim = perms[out_dim];
    out_strides_of_in_dim[in_dim] = out_strides[out_dim];
  }

  for (int64_t x = 0; x < total_elements; ++x) {
    int64_t y = 0;
    for (int k = 0; k < rank; ++k) {
      int64_t coord = (x / in_strides[k]) % input_shape[k];
      y += coord * out_strides_of_in_dim[k];
    }
    std::memcpy(dst + y * element_size, src + x * element_size, element_size);
  }
}

inline void InPlaceTransposeGenericNDImpl(char* data,
                                          llvm::ArrayRef<int64_t> input_shape,
                                          llvm::ArrayRef<int64_t> perms,
                                          int element_size) {
  int64_t total_elements = 1;
  for (int64_t d : input_shape) total_elements *= d;
  if (total_elements <= 1) return;

  std::vector<char> tmp(total_elements * element_size);
  TransposeGenericNDOutPlace(data, tmp.data(), input_shape, perms,
                             element_size);
  std::memcpy(data, tmp.data(), total_elements * element_size);
}

inline bool IsValidPermutation(llvm::ArrayRef<int64_t> perms, size_t rank) {
  if (perms.size() != rank) return false;
  llvm::SmallVector<bool, 8> seen(rank, false);
  for (int64_t p : perms) {
    if (p < 0 || static_cast<size_t>(p) >= rank || seen[p]) {
      return false;
    }
    seen[p] = true;
  }
  return true;
}

}  // namespace detail

// Transposes a buffer from src to dst based on bit width and rank.
// Returns true on success, false if input arguments or configuration are
// invalid.
inline bool TransposeBuffer(const void* src, void* dst,
                            llvm::ArrayRef<int64_t> input_shape,
                            llvm::ArrayRef<int64_t> perms, int bit_width) {
  if (input_shape.empty() || perms.empty() || src == nullptr ||
      dst == nullptr || bit_width <= 0) {
    return false;
  }
  if (!detail::IsValidPermutation(perms, input_shape.size())) {
    return false;
  }
  for (int64_t d : input_shape) {
    if (d < 0) return false;
    if (d == 0) return true;
  }

  bool is_identity = true;
  for (size_t i = 0; i < perms.size(); ++i) {
    if (perms[i] != static_cast<int64_t>(i)) {
      is_identity = false;
      break;
    }
  }
  int64_t total_elements = 1;
  for (int64_t d : input_shape) total_elements *= d;
  int64_t total_bytes = (total_elements * bit_width + 7) / 8;

  if (is_identity) {
    if (src != dst) {
      std::memcpy(dst, src, total_bytes);
    }
    return true;
  }

  // Sub-byte elements (bit_width < 8)
  if (bit_width < 8) {
    if ((8 % bit_width) != 0) return false;

    // 2D matrix transpose [1, 0]
    if (input_shape.size() == 2 && perms.size() == 2 && perms[0] == 1 &&
        perms[1] == 0) {
      detail::FastTiledTransposeSubByte2DOutPlace(
          reinterpret_cast<const uint8_t*>(src),
          reinterpret_cast<uint8_t*>(dst), input_shape[0], input_shape[1],
          bit_width);
      return true;
    }

    // 3D batch transpose [0, 2, 1] when batches align on byte boundary
    if (input_shape.size() == 3 && perms.size() == 3 && perms[0] == 0 &&
        perms[1] == 2 && perms[2] == 1 &&
        ((input_shape[1] * input_shape[2] * bit_width) % 8 == 0)) {
      int64_t B = input_shape[0];
      int64_t R = input_shape[1];
      int64_t C = input_shape[2];
      int64_t batch_bytes = (R * C * bit_width) / 8;
      const uint8_t* src_u8 = reinterpret_cast<const uint8_t*>(src);
      uint8_t* dst_u8 = reinterpret_cast<uint8_t*>(dst);
      for (int64_t b = 0; b < B; ++b) {
        detail::FastTiledTransposeSubByte2DOutPlace(src_u8 + b * batch_bytes,
                                                    dst_u8 + b * batch_bytes, R,
                                                    C, bit_width);
      }
      return true;
    }

    // General N-D sub-byte
    detail::TransposeSubByteNDOutPlace(reinterpret_cast<const uint8_t*>(src),
                                       reinterpret_cast<uint8_t*>(dst),
                                       input_shape, perms, bit_width);
    return true;
  }

  if ((bit_width % 8) != 0) return false;
  const int element_size = bit_width / 8;

  // 2D matrix transpose [1, 0]
  if (input_shape.size() == 2 && perms.size() == 2 && perms[0] == 1 &&
      perms[1] == 0) {
    int64_t R = input_shape[0];
    int64_t C = input_shape[1];
    switch (element_size) {
      case 1:
        detail::FastTiledTranspose2DOutPlace(
            reinterpret_cast<const uint8_t*>(src),
            reinterpret_cast<uint8_t*>(dst), R, C);
        return true;
      case 2:
        detail::FastTiledTranspose2DOutPlace(
            reinterpret_cast<const uint16_t*>(src),
            reinterpret_cast<uint16_t*>(dst), R, C);
        return true;
      case 4:
        detail::FastTiledTranspose2DOutPlace(
            reinterpret_cast<const uint32_t*>(src),
            reinterpret_cast<uint32_t*>(dst), R, C);
        return true;
      case 8:
        detail::FastTiledTranspose2DOutPlace(
            reinterpret_cast<const uint64_t*>(src),
            reinterpret_cast<uint64_t*>(dst), R, C);
        return true;
      default:
        detail::TransposeGenericNDOutPlace(reinterpret_cast<const char*>(src),
                                           reinterpret_cast<char*>(dst), {R, C},
                                           {1, 0}, element_size);
        return true;
    }
  }

  // 3D batch transpose [0, 2, 1]
  if (input_shape.size() == 3 && perms.size() == 3 && perms[0] == 0 &&
      perms[1] == 2 && perms[2] == 1) {
    int64_t B = input_shape[0];
    int64_t R = input_shape[1];
    int64_t C = input_shape[2];
    int64_t batch_bytes = R * C * element_size;
    const char* src_ptr = reinterpret_cast<const char*>(src);
    char* dst_ptr = reinterpret_cast<char*>(dst);
    for (int64_t b = 0; b < B; ++b) {
      const void* b_src = src_ptr + b * batch_bytes;
      void* b_dst = dst_ptr + b * batch_bytes;
      switch (element_size) {
        case 1:
          detail::FastTiledTranspose2DOutPlace(
              reinterpret_cast<const uint8_t*>(b_src),
              reinterpret_cast<uint8_t*>(b_dst), R, C);
          break;
        case 2:
          detail::FastTiledTranspose2DOutPlace(
              reinterpret_cast<const uint16_t*>(b_src),
              reinterpret_cast<uint16_t*>(b_dst), R, C);
          break;
        case 4:
          detail::FastTiledTranspose2DOutPlace(
              reinterpret_cast<const uint32_t*>(b_src),
              reinterpret_cast<uint32_t*>(b_dst), R, C);
          break;
        case 8:
          detail::FastTiledTranspose2DOutPlace(
              reinterpret_cast<const uint64_t*>(b_src),
              reinterpret_cast<uint64_t*>(b_dst), R, C);
          break;
        default:
          detail::TransposeGenericNDOutPlace(
              reinterpret_cast<const char*>(b_src),
              reinterpret_cast<char*>(b_dst), {R, C}, {1, 0}, element_size);
          break;
      }
    }
    return true;
  }

  // General N-D
  switch (element_size) {
    case 1:
      detail::TransposeNDOutPlace(reinterpret_cast<const uint8_t*>(src),
                                  reinterpret_cast<uint8_t*>(dst), input_shape,
                                  perms);
      return true;
    case 2:
      detail::TransposeNDOutPlace(reinterpret_cast<const uint16_t*>(src),
                                  reinterpret_cast<uint16_t*>(dst), input_shape,
                                  perms);
      return true;
    case 4:
      detail::TransposeNDOutPlace(reinterpret_cast<const uint32_t*>(src),
                                  reinterpret_cast<uint32_t*>(dst), input_shape,
                                  perms);
      return true;
    case 8:
      detail::TransposeNDOutPlace(reinterpret_cast<const uint64_t*>(src),
                                  reinterpret_cast<uint64_t*>(dst), input_shape,
                                  perms);
      return true;
    default:
      detail::TransposeGenericNDOutPlace(reinterpret_cast<const char*>(src),
                                         reinterpret_cast<char*>(dst),
                                         input_shape, perms, element_size);
      return true;
  }
}

// Dispatches in-place tensor transposition based on bit width and rank.
// Returns true on success, false if input arguments or configuration are
// invalid.
inline bool InPlaceTranspose(void* raw_data,
                             llvm::ArrayRef<int64_t> input_shape,
                             llvm::ArrayRef<int64_t> perms, int bit_width) {
  if (input_shape.empty() || perms.empty() || raw_data == nullptr ||
      bit_width <= 0) {
    return false;
  }
  if (!detail::IsValidPermutation(perms, input_shape.size())) {
    return false;
  }
  for (int64_t d : input_shape) {
    if (d < 0) return false;
    if (d == 0) return true;
  }

  // Check for identity permutation
  bool is_identity = true;
  for (size_t i = 0; i < perms.size(); ++i) {
    if (perms[i] != static_cast<int64_t>(i)) {
      is_identity = false;
      break;
    }
  }
  if (is_identity) return true;

  // Sub-byte elements (bit_width < 8)
  if (bit_width < 8) {
    if ((8 % bit_width) != 0) return false;
    if (input_shape.size() == 2 && perms.size() == 2 && perms[0] == 1 &&
        perms[1] == 0) {
      detail::InPlaceTransposeSubByte2D(reinterpret_cast<uint8_t*>(raw_data),
                                        input_shape[0], input_shape[1],
                                        bit_width);
      return true;
    }

    // 3D with leading batch dimension unchanged [0, 2, 1]
    if (input_shape.size() == 3 && perms.size() == 3 && perms[0] == 0 &&
        perms[1] == 2 && perms[2] == 1 &&
        ((input_shape[1] * input_shape[2] * bit_width) % 8 == 0)) {
      int64_t B = input_shape[0];
      int64_t R = input_shape[1];
      int64_t C = input_shape[2];
      int64_t batch_bytes = (R * C * bit_width) / 8;
      uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(raw_data);
      for (int64_t b = 0; b < B; ++b) {
        detail::InPlaceTransposeSubByte2D(byte_ptr + b * batch_bytes, R, C,
                                          bit_width);
      }
      return true;
    }

    // General N-D sub-byte in-place transposition
    detail::InPlaceTransposeSubByteND(reinterpret_cast<uint8_t*>(raw_data),
                                      input_shape, perms, bit_width);
    return true;
  }

  if ((bit_width % 8) != 0) return false;
  const int element_size = bit_width / 8;

  // If rank is 2 and perms is [1, 0]
  if (input_shape.size() == 2 && perms.size() == 2 && perms[0] == 1 &&
      perms[1] == 0) {
    int64_t R = input_shape[0];
    int64_t C = input_shape[1];
    switch (element_size) {
      case 1:
        detail::InPlaceTranspose2D(reinterpret_cast<uint8_t*>(raw_data), R, C);
        return true;
      case 2:
        detail::InPlaceTranspose2D(reinterpret_cast<uint16_t*>(raw_data), R, C);
        return true;
      case 4:
        detail::InPlaceTranspose2D(reinterpret_cast<uint32_t*>(raw_data), R, C);
        return true;
      case 8:
        detail::InPlaceTranspose2D(reinterpret_cast<uint64_t*>(raw_data), R, C);
        return true;
      default:
        detail::InPlaceTransposeGeneric2D(reinterpret_cast<char*>(raw_data), R,
                                          C, element_size);
        return true;
    }
  }

  // If 3D with leading batch dimension unchanged [0, 2, 1]
  if (input_shape.size() == 3 && perms.size() == 3 && perms[0] == 0 &&
      perms[1] == 2 && perms[2] == 1) {
    int64_t B = input_shape[0];
    int64_t R = input_shape[1];
    int64_t C = input_shape[2];
    int64_t batch_bytes = R * C * element_size;
    char* byte_ptr = reinterpret_cast<char*>(raw_data);
    for (int64_t b = 0; b < B; ++b) {
      void* batch_data = byte_ptr + b * batch_bytes;
      switch (element_size) {
        case 1:
          detail::InPlaceTranspose2D(reinterpret_cast<uint8_t*>(batch_data), R,
                                     C);
          break;
        case 2:
          detail::InPlaceTranspose2D(reinterpret_cast<uint16_t*>(batch_data), R,
                                     C);
          break;
        case 4:
          detail::InPlaceTranspose2D(reinterpret_cast<uint32_t*>(batch_data), R,
                                     C);
          break;
        case 8:
          detail::InPlaceTranspose2D(reinterpret_cast<uint64_t*>(batch_data), R,
                                     C);
          break;
        default:
          detail::InPlaceTransposeGeneric2D(reinterpret_cast<char*>(batch_data),
                                            R, C, element_size);
          break;
      }
    }
    return true;
  }

  // General N-D in-place transposition
  switch (element_size) {
    case 1:
      detail::InPlaceTransposeNDImpl(reinterpret_cast<uint8_t*>(raw_data),
                                     input_shape, perms);
      return true;
    case 2:
      detail::InPlaceTransposeNDImpl(reinterpret_cast<uint16_t*>(raw_data),
                                     input_shape, perms);
      return true;
    case 4:
      detail::InPlaceTransposeNDImpl(reinterpret_cast<uint32_t*>(raw_data),
                                     input_shape, perms);
      return true;
    case 8:
      detail::InPlaceTransposeNDImpl(reinterpret_cast<uint64_t*>(raw_data),
                                     input_shape, perms);
      return true;
    default:
      detail::InPlaceTransposeGenericNDImpl(reinterpret_cast<char*>(raw_data),
                                            input_shape, perms, element_size);
      return true;
  }
}

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_OPS_IN_PLACE_TRANSPOSE_H_
