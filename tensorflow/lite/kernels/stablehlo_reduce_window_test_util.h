/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_KERNELS_STABLEHLO_REDUCE_WINDOW_TEST_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_STABLEHLO_REDUCE_WINDOW_TEST_UTIL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"

namespace tflite {
namespace reduce_window {
namespace reference {

constexpr int kMaxDims = 6;

// Holds a buffer and the shape associated to a tensor.
template <class T>
struct Tensor {
  std::vector<int64_t> shape;
  std::vector<T> data;

  // Builds a tensor using the given shape and fill it with the given initial
  // value.
  static Tensor<T> FromShape(std::vector<int64_t> shape,
                             const T init_value = 0) {
    Tensor tensor{std::move(shape)};
    tensor.data.resize(tensor.size(), init_value);
    return tensor;
  }

  // Builds a tensor using the given shape and fill it with incrementing values
  // starting from 1.
  template <class I>
  static Tensor<T> iota(std::initializer_list<I> shape) {
    Tensor<T> tensor;
    tensor.shape.assign(shape.begin(), shape.end());
    tensor.data.resize(absl::c_accumulate(shape, 1, std::multiplies<>()));
    absl::c_iota(tensor.data, 1);
    return tensor;
  }

  // Returns the number of values in the tensor.
  int64_t size() const {
    return absl::c_accumulate(shape, 1, std::multiplies<>());
  }

  // Computes the strides for each valid dimension in the tensor.
  //
  // The returned vector always has a `kMaxDims` size.
  std::vector<int64_t> Strides() const {
    std::vector<int64_t> strides(kMaxDims, 0);
    if (!shape.empty()) {
      strides[shape.size() - 1] = 1;
      for (size_t i = shape.size() - 1; i > 0; --i) {
        strides[i - 1] = shape[i] * strides[i];
      }
    }
    return strides;
  }
};

// Returns a new vector resized to `kMaxDims` with `val` a a default value.
inline std::vector<int64_t> ExtendToMaxDim(std::vector<int64_t> vec,
                                           int64_t val = 0) {
  vec.resize(kMaxDims, val);
  return vec;
}

inline std::vector<int64_t> DilateShape(std::vector<int64_t> shape,
                                        const std::vector<int64_t> dilations) {
  for (size_t i = 0; i < shape.size(); ++i) {
    shape[i] = (shape[i] - 1) * dilations[i] + 1;
  }
  if (absl::c_any_of(shape, [](auto s) { return s <= 0; })) {
    absl::c_fill(shape, 0);
  }
  return shape;
}

template <class T>
Tensor<T> Dilate(const Tensor<T>& input, const std::vector<int64_t>& dilations,
                 const T padding_value) {
  Tensor<T> output =
      Tensor<T>::FromShape(DilateShape(input.shape, dilations), padding_value);

  if (absl::c_all_of(output.shape, [](auto s) { return s == 0; })) {
    return output;
  }

  const std::vector<int64_t> strides = input.Strides();
  const std::vector<int64_t> output_strides = output.Strides();
  const std::vector<int64_t> safe_dilations = ExtendToMaxDim(dilations);
  const std::vector<int64_t> safe_input_shape = ExtendToMaxDim(input.shape);

  int a = 0;
  do {
    int b = 0;
    do {
      int c = 0;
      do {
        int d = 0;
        do {
          int e = 0;
          do {
            int f = 0;
            do {
              const int i_idx = a * strides[0] + b * strides[1] +
                                c * strides[2] + d * strides[3] +
                                e * strides[4] + f * strides[5];
              const int o_idx = a * safe_dilations[0] * output_strides[0] +
                                b * safe_dilations[1] * output_strides[1] +
                                c * safe_dilations[2] * output_strides[2] +
                                d * safe_dilations[3] * output_strides[3] +
                                e * safe_dilations[4] * output_strides[4] +
                                f * safe_dilations[5] * output_strides[5];
              output.data[o_idx] = input.data[i_idx];
            } while (++f < safe_input_shape[5]);
          } while (++e < safe_input_shape[4]);
        } while (++d < safe_input_shape[3]);
      } while (++c < safe_input_shape[2]);
    } while (++b < safe_input_shape[1]);
  } while (++a < safe_input_shape[0]);

  return output;
}

inline std::vector<int64_t> PadCropShape(std::vector<int64_t> shape,
                                         const std::vector<int64_t> padding) {
  for (size_t i = 0; i < shape.size(); ++i) {
    shape[i] = shape[i] + padding[2 * i] + padding[2 * i + 1];
  }
  if (absl::c_any_of(shape, [](auto s) { return s <= 0; })) {
    absl::c_fill(shape, 0);
  }
  return shape;
}

// Pads the input tensor.
//
// `Pad` and `Crop` share the same pad/crop specification. The positive values
// specify padding and the negative values specify cropping.
template <class T>
Tensor<T> Pad(const Tensor<T>& input, const std::vector<int64_t>& padding,
              const T padding_value) {
  // Keep only positive values in the padding.
  std::vector<int64_t> safe_padding(kMaxDims * 2, 0);
  absl::c_transform(padding, safe_padding.begin(),
                    [](int64_t p) { return std::max<int64_t>(p, 0); });

  Tensor<T> output = Tensor<T>::FromShape(
      PadCropShape(input.shape, safe_padding), padding_value);

  if (absl::c_all_of(output.shape, [](auto s) { return s == 0; })) {
    return output;
  }

  const std::vector<int64_t> strides = input.Strides();
  const std::vector<int64_t> output_strides = output.Strides();
  const std::vector<int64_t> safe_input_shape = ExtendToMaxDim(input.shape);

  int a = 0;
  do {
    int b = 0;
    do {
      int c = 0;
      do {
        int d = 0;
        do {
          int e = 0;
          do {
            int f = 0;
            do {
              const int i_idx = a * strides[0] + b * strides[1] +
                                c * strides[2] + d * strides[3] +
                                e * strides[4] + f * strides[5];
              const int o_idx = (a + safe_padding[0]) * output_strides[0] +
                                (b + safe_padding[2]) * output_strides[1] +
                                (c + safe_padding[4]) * output_strides[2] +
                                (d + safe_padding[6]) * output_strides[3] +
                                (e + safe_padding[8]) * output_strides[4] +
                                (f + safe_padding[10]) * output_strides[5];
              output.data[o_idx] = input.data[i_idx];
            } while (++f < safe_input_shape[5]);
          } while (++e < safe_input_shape[4]);
        } while (++d < safe_input_shape[3]);
      } while (++c < safe_input_shape[2]);
    } while (++b < safe_input_shape[1]);
  } while (++a < safe_input_shape[0]);

  return output;
}

// Crops the input tensor.
//
// Only negative values are taken into account for cropping.
template <class T>
Tensor<T> Crop(const Tensor<T>& input, const std::vector<int64_t>& cropping) {
  // Keep only negative values in the cropping.
  std::vector<int64_t> safe_cropping(kMaxDims * 2, 0);
  absl::c_transform(cropping, safe_cropping.begin(),
                    [](int64_t p) { return std::min<int64_t>(p, 0); });

  Tensor<T> output =
      Tensor<T>::FromShape(PadCropShape(input.shape, safe_cropping));

  if (absl::c_all_of(output.shape, [](auto s) { return s == 0; })) {
    return output;
  }

  const std::vector<int64_t> strides = input.Strides();
  const std::vector<int64_t> output_strides = output.Strides();
  const std::vector<int64_t> safe_output_shape = ExtendToMaxDim(output.shape);

  int a = 0;
  do {
    int b = 0;
    do {
      int c = 0;
      do {
        int d = 0;
        do {
          int e = 0;
          do {
            int f = 0;
            do {
              const int i_idx = (a - safe_cropping[0]) * strides[0] +
                                (b - safe_cropping[2]) * strides[1] +
                                (c - safe_cropping[4]) * strides[2] +
                                (d - safe_cropping[6]) * strides[3] +
                                (e - safe_cropping[8]) * strides[4] +
                                (f - safe_cropping[10]) * strides[5];
              const int o_idx = a * output_strides[0] + b * output_strides[1] +
                                c * output_strides[2] + d * output_strides[3] +
                                e * output_strides[4] + f * output_strides[5];
              output.data[o_idx] = input.data[i_idx];
            } while (++f < safe_output_shape[5]);
          } while (++e < safe_output_shape[4]);
        } while (++d < safe_output_shape[3]);
      } while (++c < safe_output_shape[2]);
    } while (++b < safe_output_shape[1]);
  } while (++a < safe_output_shape[0]);

  return output;
}

// Gathers the elements that are visible through the given window spec in a new
// tensor.
template <class T>
Tensor<T> WindowCopy(const Tensor<T>& input,
                     const std::vector<int64_t>& window_dimensions,
                     const std::vector<int64_t>& window_dilations,
                     const std::vector<int64_t>& window_offset) {
  Tensor<T> output = Tensor<T>::FromShape(window_dimensions);

  const std::vector<int64_t> safe_window_dimensions =
      ExtendToMaxDim(window_dimensions);
  const std::vector<int64_t> safe_window_dilations =
      ExtendToMaxDim(window_dilations, 1);
  const std::vector<int64_t> safe_window_offset = ExtendToMaxDim(window_offset);

  const std::vector<int64_t> strides = input.Strides();
  const std::vector<int64_t> output_strides = output.Strides();

  int a = 0;
  do {
    int b = 0;
    do {
      int c = 0;
      do {
        int d = 0;
        do {
          int e = 0;
          do {
            int f = 0;
            do {
              const int i_idx =
                  (a * safe_window_dilations[0] + safe_window_offset[0]) *
                      strides[0] +
                  (b * safe_window_dilations[1] + safe_window_offset[1]) *
                      strides[1] +
                  (c * safe_window_dilations[2] + safe_window_offset[2]) *
                      strides[2] +
                  (d * safe_window_dilations[3] + safe_window_offset[3]) *
                      strides[3] +
                  (e * safe_window_dilations[4] + safe_window_offset[4]) *
                      strides[4] +
                  (f * safe_window_dilations[5] + safe_window_offset[5]) *
                      strides[5];
              const int o_idx = a * output_strides[0] + b * output_strides[1] +
                                c * output_strides[2] + d * output_strides[3] +
                                e * output_strides[4] + f * output_strides[5];
              output.data[o_idx] = input.data[i_idx];
            } while (++f < safe_window_dimensions[5]);
          } while (++e < safe_window_dimensions[4]);
        } while (++d < safe_window_dimensions[3]);
      } while (++c < safe_window_dimensions[2]);
    } while (++b < safe_window_dimensions[1]);
  } while (++a < safe_window_dimensions[0]);

  return output;
}

inline std::vector<int64_t> ReduceWindowShape(
    std::vector<int64_t> shape, const std::vector<int64_t>& base_dilations,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& window_dimensions,
    const std::vector<int64_t>& window_dilations,
    const std::vector<int64_t>& window_strides) {
  const std::vector<int64_t> base_shape =
      PadCropShape(DilateShape(shape, base_dilations), padding);
  const std::vector<int64_t> dilated_window_dimensions =
      DilateShape(window_dimensions, window_dilations);
  shape.assign(base_shape.size(), 0);
  for (int i = 0; i < base_shape.size(); ++i) {
    if (base_shape[i] >= dilated_window_dimensions[i]) {
      shape[i] =
          (base_shape[i] - dilated_window_dimensions[i]) / window_strides[i] +
          1;
    }
  }
  return shape;
}

template <class T, class F>
Tensor<T> ReduceWindow(const Tensor<T>& input,
                       const std::vector<int64_t>& base_dilations,
                       const std::vector<int64_t>& padding, const T& init_value,
                       const std::vector<int64_t>& window_dimensions,
                       const std::vector<int64_t>& window_dilations,
                       const std::vector<int64_t>& window_strides, F&& body) {
  Tensor<T> output = Tensor<T>::FromShape(
      ReduceWindowShape(input.shape, base_dilations, padding, window_dimensions,
                        window_dilations, window_strides),
      init_value);

  if (output.data.empty()) {
    return output;
  }

  const std::vector<int64_t> safe_output_shape = ExtendToMaxDim(output.shape);
  const std::vector<int64_t> safe_window_strides =
      ExtendToMaxDim(window_strides);
  const std::vector<int64_t> output_strides = output.Strides();

  const Tensor<T> dilated = Dilate<T>(input, base_dilations, init_value);
  const Tensor<T> padded = Pad<T>(dilated, padding, init_value);
  const Tensor<T> base = Crop<T>(padded, padding);

  std::vector<int64_t> output_offsets(6, 0);
  std::vector<int64_t> window_offsets(6, 0);
  do {
    output_offsets[1] = 0;
    window_offsets[1] = 0;
    do {
      output_offsets[2] = 0;
      window_offsets[2] = 0;
      do {
        output_offsets[3] = 0;
        window_offsets[3] = 0;
        do {
          output_offsets[4] = 0;
          window_offsets[4] = 0;
          do {
            output_offsets[5] = 0;
            window_offsets[5] = 0;
            do {
              const int64_t o_idx = output_offsets[0] * output_strides[0] +
                                    output_offsets[1] * output_strides[1] +
                                    output_offsets[2] * output_strides[2] +
                                    output_offsets[3] * output_strides[3] +
                                    output_offsets[4] * output_strides[4] +
                                    output_offsets[5] * output_strides[5];
              const Tensor<T> window = WindowCopy(
                  base, window_dimensions, window_dilations, window_offsets);
              if (window.data.empty()) {
                output.data[o_idx] = init_value;
              } else {
                output.data[o_idx] = std::accumulate(
                    window.data.begin(), window.data.end(), init_value, body);
              }
              window_offsets[5] += safe_window_strides[5];
            } while (++output_offsets[5] < safe_output_shape[5]);
            window_offsets[4] += safe_window_strides[4];
          } while (++output_offsets[4] < safe_output_shape[4]);
          window_offsets[3] += safe_window_strides[3];
        } while (++output_offsets[3] < safe_output_shape[3]);
        window_offsets[2] += safe_window_strides[2];
      } while (++output_offsets[2] < safe_output_shape[2]);
      window_offsets[1] += safe_window_strides[1];
    } while (++output_offsets[1] < safe_output_shape[1]);
    window_offsets[0] += safe_window_strides[0];
  } while (++output_offsets[0] < safe_output_shape[0]);
  return output;
}

}  // namespace reference
}  // namespace reduce_window
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_STABLEHLO_REDUCE_WINDOW_TEST_UTIL_H_
