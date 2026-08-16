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
#include <initializer_list>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/absl_check.h"
#include "tensorflow/lite/util.h"

namespace tflite::reduce_window::reference {

inline constexpr int kMaxDims = 6;

// Holds a buffer and the shape associated with a tensor.
template <class T>
struct Tensor {
  std::vector<int64_t> shape;
  std::vector<T> data;

  Tensor() = default;
  explicit Tensor(std::vector<int64_t> shape) : shape(std::move(shape)) {
    for (const int64_t dim : this->shape) {
      ABSL_CHECK_GE(dim, 0);
    }
  }
  Tensor(std::vector<int64_t> shape, std::vector<T> data)
      : shape(std::move(shape)), data(std::move(data)) {
    for (const int64_t dim : this->shape) {
      ABSL_CHECK_GE(dim, 0);
    }
  }

  // Builds a tensor using the given shape and fills it with the given initial
  // value.
  static Tensor<T> FromShape(std::vector<int64_t> shape,
                             const T init_value = T{}) {
    Tensor tensor(std::move(shape));
    tensor.data.resize(tensor.size(), init_value);
    return tensor;
  }

  // Builds a tensor using the given shape and fills it with incrementing values
  // starting from 1.
  static Tensor<T> Iota(std::vector<int64_t> shape) {
    Tensor<T> tensor(std::move(shape));
    tensor.data.resize(tensor.size());
    absl::c_iota(tensor.data, 1);
    return tensor;
  }

  // Alias for compatibility with existing tests.
  template <class I>
  static Tensor<T> iota(std::initializer_list<I> shape) {
    return Iota(std::vector<int64_t>(shape.begin(), shape.end()));
  }

  // Returns the number of values in the tensor.
  int64_t size() const {
    tflite::CheckedInt<int64_t> total_size = 1;
    for (const int64_t dim : shape) {
      ABSL_CHECK_GE(dim, 0);
      total_size *= dim;
    }
    ABSL_CHECK(!total_size.Overflow());
    return total_size.Value();
  }

  // Computes the strides for each valid dimension in the tensor.
  //
  // The returned vector always has a `kMaxDims` size.
  std::vector<int64_t> Strides() const {
    ABSL_CHECK_LE(shape.size(), kMaxDims);
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

// Returns a new vector resized to `kMaxDims` with `val` as the default value.
inline std::vector<int64_t> ExtendToMaxDims(std::vector<int64_t> vec,
                                            int64_t val) {
  ABSL_CHECK_LE(vec.size(), kMaxDims);
  vec.resize(kMaxDims, val);
  return vec;
}

inline std::vector<int64_t> DilateShape(std::vector<int64_t> shape,
                                        const std::vector<int64_t>& dilations) {
  ABSL_CHECK_GE(dilations.size(), shape.size());
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

  if (input.data.empty() || output.data.empty()) {
    return output;
  }

  const std::vector<int64_t> strides = input.Strides();
  const std::vector<int64_t> output_strides = output.Strides();
  const std::vector<int64_t> safe_dilations = ExtendToMaxDims(dilations, 0);
  const std::vector<int64_t> safe_input_shape = ExtendToMaxDims(input.shape, 0);

  int64_t a = 0;
  do {
    int64_t b = 0;
    do {
      int64_t c = 0;
      do {
        int64_t d = 0;
        do {
          int64_t e = 0;
          do {
            int64_t f = 0;
            do {
              const int64_t i_idx = a * strides[0] + b * strides[1] +
                                    c * strides[2] + d * strides[3] +
                                    e * strides[4] + f * strides[5];
              const int64_t o_idx = a * safe_dilations[0] * output_strides[0] +
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
                                         const std::vector<int64_t>& padding) {
  ABSL_CHECK_GE(padding.size(), 2 * shape.size());
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
  ABSL_CHECK_LE(padding.size(), kMaxDims * 2);
  ABSL_CHECK_GE(padding.size(), 2 * input.shape.size());

  // Keep only positive values in the padding.
  std::vector<int64_t> safe_padding(kMaxDims * 2, 0);
  absl::c_transform(padding, safe_padding.begin(),
                    [](int64_t p) { return std::max<int64_t>(p, 0); });

  Tensor<T> output = Tensor<T>::FromShape(
      PadCropShape(input.shape, safe_padding), padding_value);

  if (input.data.empty() || output.data.empty()) {
    return output;
  }

  const std::vector<int64_t> strides = input.Strides();
  const std::vector<int64_t> output_strides = output.Strides();
  const std::vector<int64_t> safe_input_shape = ExtendToMaxDims(input.shape, 0);

  int64_t a = 0;
  do {
    int64_t b = 0;
    do {
      int64_t c = 0;
      do {
        int64_t d = 0;
        do {
          int64_t e = 0;
          do {
            int64_t f = 0;
            do {
              const int64_t i_idx = a * strides[0] + b * strides[1] +
                                    c * strides[2] + d * strides[3] +
                                    e * strides[4] + f * strides[5];
              const int64_t o_idx = (a + safe_padding[0]) * output_strides[0] +
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
  ABSL_CHECK_LE(cropping.size(), kMaxDims * 2);
  ABSL_CHECK_GE(cropping.size(), 2 * input.shape.size());

  // Keep only negative values in the cropping.
  std::vector<int64_t> safe_cropping(kMaxDims * 2, 0);
  absl::c_transform(cropping, safe_cropping.begin(),
                    [](int64_t p) { return std::min<int64_t>(p, 0); });

  Tensor<T> output =
      Tensor<T>::FromShape(PadCropShape(input.shape, safe_cropping));

  if (input.data.empty() || output.data.empty()) {
    return output;
  }

  const std::vector<int64_t> strides = input.Strides();
  const std::vector<int64_t> output_strides = output.Strides();
  const std::vector<int64_t> safe_output_shape =
      ExtendToMaxDims(output.shape, 0);

  int64_t a = 0;
  do {
    int64_t b = 0;
    do {
      int64_t c = 0;
      do {
        int64_t d = 0;
        do {
          int64_t e = 0;
          do {
            int64_t f = 0;
            do {
              const int64_t i_idx = (a - safe_cropping[0]) * strides[0] +
                                    (b - safe_cropping[2]) * strides[1] +
                                    (c - safe_cropping[4]) * strides[2] +
                                    (d - safe_cropping[6]) * strides[3] +
                                    (e - safe_cropping[8]) * strides[4] +
                                    (f - safe_cropping[10]) * strides[5];
              const int64_t o_idx =
                  a * output_strides[0] + b * output_strides[1] +
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

  if (input.data.empty() || output.data.empty()) {
    return output;
  }

  const std::vector<int64_t> safe_window_dimensions =
      ExtendToMaxDims(window_dimensions, 0);
  const std::vector<int64_t> safe_window_dilations =
      ExtendToMaxDims(window_dilations, 1);
  const std::vector<int64_t> safe_window_offset =
      ExtendToMaxDims(window_offset, 0);

  const std::vector<int64_t> strides = input.Strides();
  const std::vector<int64_t> output_strides = output.Strides();

  int64_t a = 0;
  do {
    int64_t b = 0;
    do {
      int64_t c = 0;
      do {
        int64_t d = 0;
        do {
          int64_t e = 0;
          do {
            int64_t f = 0;
            do {
              const int64_t i_idx =
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
              const int64_t o_idx =
                  a * output_strides[0] + b * output_strides[1] +
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
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& base_dilations,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& window_dimensions,
    const std::vector<int64_t>& window_dilations,
    const std::vector<int64_t>& window_strides) {
  ABSL_CHECK_GE(base_dilations.size(), shape.size());
  ABSL_CHECK_GE(padding.size(), 2 * shape.size());
  ABSL_CHECK_GE(window_dimensions.size(), shape.size());
  ABSL_CHECK_GE(window_dilations.size(), shape.size());
  ABSL_CHECK_GE(window_strides.size(), shape.size());

  const std::vector<int64_t> base_shape =
      PadCropShape(DilateShape(shape, base_dilations), padding);
  const std::vector<int64_t> dilated_window_dimensions =
      DilateShape(window_dimensions, window_dilations);

  std::vector<int64_t> out_shape(base_shape.size(), 0);
  for (size_t i = 0; i < base_shape.size(); ++i) {
    ABSL_CHECK_GT(window_strides[i], 0);
    if (base_shape[i] >= dilated_window_dimensions[i]) {
      out_shape[i] =
          (base_shape[i] - dilated_window_dimensions[i]) / window_strides[i] +
          1;
    }
  }
  return out_shape;
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

  if (input.data.empty() || output.data.empty()) {
    return output;
  }

  const std::vector<int64_t> safe_output_shape =
      ExtendToMaxDims(output.shape, 0);
  const std::vector<int64_t> safe_window_strides =
      ExtendToMaxDims(window_strides, 0);
  const std::vector<int64_t> output_strides = output.Strides();

  const Tensor<T> dilated = Dilate(input, base_dilations, init_value);
  const Tensor<T> padded = Pad(dilated, padding, init_value);
  const Tensor<T> base = Crop(padded, padding);

  std::vector<int64_t> window_offsets(kMaxDims, 0);
  int64_t a = 0;
  do {
    window_offsets[0] = a * safe_window_strides[0];
    int64_t b = 0;
    do {
      window_offsets[1] = b * safe_window_strides[1];
      int64_t c = 0;
      do {
        window_offsets[2] = c * safe_window_strides[2];
        int64_t d = 0;
        do {
          window_offsets[3] = d * safe_window_strides[3];
          int64_t e = 0;
          do {
            window_offsets[4] = e * safe_window_strides[4];
            int64_t f = 0;
            do {
              window_offsets[5] = f * safe_window_strides[5];
              const int64_t o_idx =
                  a * output_strides[0] + b * output_strides[1] +
                  c * output_strides[2] + d * output_strides[3] +
                  e * output_strides[4] + f * output_strides[5];
              const Tensor<T> window = WindowCopy(
                  base, window_dimensions, window_dilations, window_offsets);
              output.data[o_idx] =
                  absl::c_accumulate(window.data, init_value, body);
            } while (++f < safe_output_shape[5]);
          } while (++e < safe_output_shape[4]);
        } while (++d < safe_output_shape[3]);
      } while (++c < safe_output_shape[2]);
    } while (++b < safe_output_shape[1]);
  } while (++a < safe_output_shape[0]);

  return output;
}

}  // namespace tflite::reduce_window::reference

#endif  // TENSORFLOW_LITE_KERNELS_STABLEHLO_REDUCE_WINDOW_TEST_UTIL_H_
