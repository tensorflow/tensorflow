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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_STRIDED_SLICE_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_STRIDED_SLICE_TESTER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

class StridedSliceTest : public ::testing::Test {
 public:
  StridedSliceTest() {
    xnnpack_delegate_.reset(TfLiteXNNPackDelegateCreate(nullptr));
    std::random_device random_device;
    rng_ = std::mt19937(random_device());
    shape_rng_ = std::bind(shape_dist_, std::ref(rng_));
  }

 protected:
  inline int32_t RandomShape() { return shape_rng_(); }
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate_{nullptr, TfLiteXNNPackDelegateDelete};
  std::mt19937 rng_;
  std::uniform_int_distribution<int32_t> shape_dist_{2, 5};
  std::function<int32_t(void)> shape_rng_;
};

// Type alias for better test names in output.
using SignedQuantizedStridedSliceTest = StridedSliceTest;
using UnsignedQuantizedStridedSliceTest = StridedSliceTest;

int32_t ComputeSize(const std::vector<int32_t>& shape);

template <typename T, typename RNG>
static T RandomElement(const std::vector<T>& v, RNG rng) {
  return v[std::uniform_int_distribution<int>(0, v.size() - 1)(rng)];
}

class StridedSliceTester {
 public:
  StridedSliceTester() = default;
  StridedSliceTester(const StridedSliceTester&) = delete;
  StridedSliceTester& operator=(const StridedSliceTester&) = delete;

  inline StridedSliceTester& InputShape(const std::vector<int32_t>& shape) {
    input_shape_ = shape;
    return *this;
  }

  inline const std::vector<int32_t>& InputShape() const { return input_shape_; }

  inline StridedSliceTester& RandomBegins(std::mt19937& rng) {
    // Begin can be any number from -dims[i] to dims[i]. If it is dims[i], we
    // set begin_mask, to have it be interpreted as 0.
    begins_ = std::vector<int32_t>(InputShape().size());
    for (size_t i = 0; i < begins_.size(); i++) {
      begins_[i] = std::uniform_int_distribution<int32_t>(-InputShape()[i],
                                                          InputShape()[i])(rng);
      if (begins_[i] == InputShape()[i]) {
        begin_mask_ |= 1 << i;
      }
    }
    return *this;
  }

  inline StridedSliceTester& Begins(const std::vector<int32_t>& begins) {
    begins_ = begins;
    return *this;
  }

  inline const std::vector<int32_t> Begins() const { return begins_; }

  // Get the begin at dimension i, taking into account negative values and
  // begin_mask.
  inline const int32_t Begin(int i) const {
    const int32_t begin = Begins()[i];
    if ((BeginMask() & (1 << i)) != 0) {
      return 0;
    } else if (begin < 0) {
      return InputShape()[i] + begin;
    } else {
      return begin;
    }
  }

  inline StridedSliceTester& RandomEnds(std::mt19937& rng) {
    ends_ = std::vector<int32_t>(InputShape().size());
    for (size_t i = 0; i < ends_.size(); i++) {
      // Valid sizes are [1, max_size] for a slice.
      const size_t max_size = InputShape()[i] - Begin(i);
      // Choose between positive indices and negative indices:
      // 1. Positive end offsets range from [Begin(i) + 1, Begin(i) + max_size].
      // 2. Negative end offsets range from [-max_size, -1]
      // 3. Special case for -max_size, which we set end_mask, which represents
      // end offset of InputShape()[i].
      std::vector<int32_t> valid_ends(max_size * 2);
      std::iota(valid_ends.begin(), valid_ends.begin() + max_size,
                Begin(i) + 1);
      std::iota(valid_ends.begin() + max_size, valid_ends.end(), -max_size);
      ends_[i] = RandomElement(valid_ends, rng);
      if (ends_[i] == -max_size) {
        end_mask_ |= 1 << i;
      }
    }
    return *this;
  }

  inline StridedSliceTester& Ends(const std::vector<int32_t>& ends) {
    ends_ = ends;
    return *this;
  }

  inline const std::vector<int32_t> Ends() const { return ends_; }

  // Get the end at dimension i, taking into account negative values and
  // end_mask.
  inline const int32_t End(int i) const {
    const int32_t end = Ends()[i];
    if ((EndMask() & (1 << i)) != 0) {
      return InputShape()[i];
    } else if (end < 0) {
      return InputShape()[i] + end;
    } else {
      return end;
    }
  }

  inline const std::vector<int32_t> Strides() const {
    return std::vector<int32_t>(InputShape().size(), 1);
  }

  std::vector<int32_t> OutputShape() const {
    auto output_shape = std::vector<int32_t>(InputShape().size());
    for (size_t i = 0; i < output_shape.size(); i++) {
      output_shape[i] = End(i) - Begin(i);
    }
    return output_shape;
  }

  uint32_t BeginMask() const { return begin_mask_; }
  uint32_t EndMask() const { return end_mask_; }
  uint32_t EllipsisMask() const { return 0; }
  uint32_t NewAxisMask() const { return 0; }
  uint32_t ShrinkAxisMask() const { return 0; }

  void Test(TensorType tensor_type, TfLiteDelegate* delegate) const;
  template <typename T>
  void Test(Interpreter* default_interpreter,
            Interpreter* delegate_interpreter) const;

 private:
  std::vector<char> CreateTfLiteModel(TensorType tensor_type) const;

  std::vector<int32_t> input_shape_;
  std::vector<int32_t> begins_;
  std::vector<int32_t> ends_;
  std::vector<int32_t> strides_;
  uint32_t begin_mask_ = 0;
  uint32_t end_mask_ = 0;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_STRIDED_SLICE_TESTER_H_
