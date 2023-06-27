/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_SLICE_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_SLICE_TESTER_H_

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class SliceTester {
 public:
  SliceTester() = default;
  SliceTester(const SliceTester&) = delete;
  SliceTester& operator=(const SliceTester&) = delete;

  inline SliceTester& InputShape(const std::vector<int32_t>& shape) {
    for (const auto dim : shape) {
      EXPECT_GT(dim, 0);
    }
    input_shape_ = std::vector<int32_t>(shape);
    return *this;
  }

  inline const std::vector<int32_t>& InputShape() const { return input_shape_; }

  inline SliceTester& Offsets(const std::vector<int32_t>& offsets) {
    offsets_ = std::vector<int32_t>(offsets);
    offsets_int64_.resize(offsets.size());
    std::copy(offsets_.begin(), offsets_.end(), offsets_int64_.begin());
    return *this;
  }

  inline const std::vector<int32_t> Offsets() const { return offsets_; }

  inline const void* OffsetsData() const {
    return UseInt64OffsetsAndSize() ? (void*)offsets_int64_.data()
                                    : (void*)offsets_.data();
  }

  inline const size_t OffsetsSizeInBytes() const {
    if (use_int64_offsets_and_sizes_) {
      return Offsets().size() * sizeof(int64_t);
    } else {
      return Offsets().size() * sizeof(int32_t);
    }
  }

  inline SliceTester& Sizes(const std::vector<int32_t>& sizes) {
    sizes_ = std::vector<int32_t>(sizes);
    sizes_int64_.resize(sizes.size());
    std::copy(sizes_.begin(), sizes_.end(), sizes_int64_.begin());
    output_shape_ = std::vector<int32_t>(sizes);
    for (size_t i = 0; i < sizes.size(); i++) {
      if (output_shape_[i] < 0) {
        output_shape_[i] = input_shape_[i] - offsets_[i];
      }
    }
    return *this;
  }

  inline const std::vector<int32_t> Sizes() const { return sizes_; }

  inline const void* SizesData() const {
    return UseInt64OffsetsAndSize() ? (void*)sizes_int64_.data()
                                    : (void*)sizes_.data();
  }

  inline const size_t SizesSizeInBytes() const {
    if (use_int64_offsets_and_sizes_) {
      return Sizes().size() * sizeof(int64_t);
    } else {
      return Sizes().size() * sizeof(int32_t);
    }
  }

  std::vector<int32_t> OutputShape() const { return output_shape_; }

  inline SliceTester& UseInt64OffsetsAndSize(bool use_int64) {
    use_int64_offsets_and_sizes_ = use_int64;
    return *this;
  }

  inline bool UseInt64OffsetsAndSize() const {
    return use_int64_offsets_and_sizes_;
  }

  void Test(TensorType tensor_type, TfLiteDelegate* delegate) const;
  template <typename T>
  void Test(Interpreter* default_interpreter,
            Interpreter* delegate_interpreter) const;

 private:
  std::vector<char> CreateTfLiteModel(TensorType tensor_type) const;

  std::vector<int32_t> input_shape_;
  std::vector<int32_t> offsets_;
  std::vector<int64_t> offsets_int64_;
  std::vector<int32_t> sizes_;
  std::vector<int64_t> sizes_int64_;
  std::vector<int32_t> output_shape_;
  bool use_int64_offsets_and_sizes_;
};

int32_t ComputeSize(const std::vector<int32_t>& shape);

std::vector<int32_t> RandomOffsets(std::mt19937& rng,
                                   const std::vector<int32_t>& dims);

std::vector<int32_t> RandomSizes(std::mt19937& rng,
                                 const std::vector<int32_t>& dims,
                                 const std::vector<int32_t>& offsets);

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_SLICE_TESTER_H_
