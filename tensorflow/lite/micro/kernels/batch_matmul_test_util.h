/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_BATCH_MATMUL_TEST_UTIL_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_BATCH_MATMUL_TEST_UTIL_H_

#include <cstdint>
#include <initializer_list>
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_utils_common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

constexpr int kMaxDims = RuntimeShape::kMaxSmallSize;
constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 2;
constexpr float kTolerance = 1e-5;

enum TestInputIndex {
  kInputIndex0,
  kInputIndex1,
};
constexpr size_t kMaxInputs = TestInputIndex::kInputIndex1 + 1;

struct TestTensorData {
  TestTensorData(const TfLiteType datum_type,
                 const std::initializer_list<int>& datum_list = {},
                 float datum_min = 0.0f, float datum_max = 0.0f,
                 float datum_scale = 0.0f, int32_t datum_zero_point = 0)
      : type(datum_type),
        shape(datum_list),
        minimum(datum_min),
        maximum(datum_max),
        scale(datum_scale),
        zero_point(datum_zero_point) {}
  const TfLiteType type;
  const std::initializer_list<int>& shape;
  const float minimum;
  const float maximum;
  const float scale;
  const int32_t zero_point;
};

template <typename T, size_t D>
struct ElementArray {
  ElementArray() : scale(0.0f), zero_point(0) {}

  template <typename TA>
  explicit ElementArray(const TA (&a)[D]) : ElementArray() {
    for (size_t i = 0; i < D; i++) {
      data[i] = static_cast<T>(a[i]);
    }
  }

  T data[D];
  TestInputIndex index;

  // quantization parameters
  float scale;
  int32_t zero_point;
};

template <typename T, size_t D>
struct ElementArrayNear : ElementArray<T, D> {
  template <typename TA>
  explicit ElementArrayNear(const TA (&a)[D],
                            const float tolerance_param = kTolerance)
      : ElementArray<T, D>(a), tolerance(tolerance_param) {}

  const float tolerance;
};

template <size_t D>
inline ElementArrayNear<float, D> ElementsAreArray(const double (&a)[D]) {
  return ElementArrayNear<float, D>(a);
}

template <size_t D>
inline ElementArray<int, D> ElementsAreArray(const int (&a)[D]) {
  return ElementArray<int, D>(a);
}

template <typename T, size_t D>
inline const ElementArrayNear<T, D>& ElementsAreArray(
    const ElementArrayNear<T, D>& a) {
  return a;
}

template <typename T, size_t D>
inline ElementArrayNear<float, D> ArrayFloatNear(
    const T (&a)[D], const float tolerance = kTolerance) {
  return ElementArrayNear<float, D>(a, tolerance);
}

template <size_t D>
void ExpectThat(const TfLiteIntArray& actual,
                const ElementArray<int, D>& expected) {
  TF_LITE_MICRO_EXPECT_EQ(actual.size, static_cast<int>(D));
  for (int i = 0; i < actual.size; i++) {
    TF_LITE_MICRO_EXPECT_EQ(actual.data[i], expected.data[i]);
  }
}

template <typename T, size_t D>
void ExpectThat(const ElementArray<T, D>& actual,
                const ElementArrayNear<T, D>& expected) {
  for (size_t i = 0; i < D; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(actual.data[i], expected.data[i],
                              expected.tolerance);
  }
}

template <typename T1, typename T2, size_t D>
void ExpectThat(const ElementArray<T1, D>& actual,
                const ElementArray<T2, D>& expected) {
  for (size_t i = 0; i < D; i++) {
    TF_LITE_MICRO_EXPECT_EQ(actual.data[i], static_cast<T1>(expected.data[i]));
  }
}

template <typename T1, typename T2, size_t D>
void ExpectThat(const ElementArray<T1, D>& actual,
                const ElementArrayNear<T2, D>& expected) {
  for (size_t i = 0; i < D; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(static_cast<T2>(actual.data[i]), expected.data[i],
                              expected.tolerance);
  }
}

inline void IntArrayCopy(const TfLiteIntArray& from, TfLiteIntArray* to) {
  if (from.size > 0) {
    for (int i = 0; i < from.size; i++) {
      to->data[i] = from.data[i];
    }
    to->size = from.size;
  }
}

inline void IntArrayCopy(const std::initializer_list<int>& from,
                         TfLiteIntArray* to) {
  if (from.size() > 0) {
    for (size_t i = 0; i < from.size(); i++) {
      to->data[i] = from.begin()[i];
    }
    to->size = from.size();
  }
}

template <typename TIN1, typename TIN2, typename TOUT, size_t IN1, size_t IN2,
          size_t OUT>
class TestOpModel {
 public:
  explicit TestOpModel(const TfLiteRegistration& registration)
      : registration_(registration) {
    TfLiteIntArray* dims = IntArrayFromInts(dims_output_);
    dims->size = 1;
    dims->data[0] = OUT;

    dims = IntArrayFromInts(dims_inputs_[kInputIndex0]);
    dims->size = 1;
    dims->data[0] = IN1;
    data_input0_.index = kInputIndex0;

    dims = IntArrayFromInts(dims_inputs_[kInputIndex1]);
    dims->size = 1;
    dims->data[0] = IN2;
    data_input1_.index = kInputIndex1;
  }

  void AddInput(const TestTensorData& datum, const TestInputIndex index) {
    TF_LITE_MICRO_EXPECT_LE(datum.shape.size(), kMaxDims);
    TfLiteIntArray& dims = GetInputShape(index);
    IntArrayCopy(datum.shape, &dims);
    TF_LITE_MICRO_EXPECT_EQ(ElementCount(dims),
                            static_cast<int>(GetInputSize(index)));
  }

  template <typename T, size_t D>
  void AddInput(const TestTensorData& datum, ElementArray<T, D>* const input) {
    TF_LITE_MICRO_EXPECT_LE(datum.shape.size(), kMaxDims);
    TestInputIndex index = input->index;
    TfLiteIntArray& dims = GetInputShape(index);
    IntArrayCopy(datum.shape, &dims);
    TF_LITE_MICRO_EXPECT_EQ(ElementCount(dims), static_cast<int>(D));

    const bool quantizable =
        (datum.type == kTfLiteInt8) &&
        (datum.minimum != 0.0f || datum.maximum != 0.0f || datum.scale != 0.0f);
    if (quantizable) {
      if (datum.scale != 0.0f) {
        input->scale = datum.scale;
        input->zero_point = datum.zero_point;
      } else {
        input->scale = ScaleFromMinMax<int8_t>(datum.minimum, datum.maximum);
        input->zero_point =
            ZeroPointFromMinMax<int8_t>(datum.minimum, datum.maximum);
      }
    }
  }

  void AddOutput(const TestTensorData& datum) {
    TF_LITE_MICRO_EXPECT_LE(datum.shape.size(), kMaxDims);
    TfLiteIntArray& dims = GetOutputShape();
    IntArrayCopy(datum.shape, &dims);
    TF_LITE_MICRO_EXPECT_EQ(ElementCount(dims), static_cast<int>(OUT));

    const bool quantizable =
        (datum.type == kTfLiteInt8) &&
        (datum.minimum != 0.0f || datum.maximum != 0.0f || datum.scale != 0.0f);
    if (quantizable) {
      if (datum.scale != 0.0f) {
        data_output_.scale = datum.scale;
        data_output_.zero_point = datum.zero_point;
      } else {
        data_output_.scale =
            ScaleFromMinMax<int8_t>(datum.minimum, datum.maximum);
        data_output_.zero_point =
            ZeroPointFromMinMax<int8_t>(datum.minimum, datum.maximum);
      }
    }
  }

  ElementArray<TOUT, OUT>& GetOutput() { return data_output_; }
  TfLiteIntArray& GetOutputShape() { return *IntArrayFromInts(dims_output_); }
  ElementArray<TIN1, IN1>& GetInput0() { return data_input0_; }
  ElementArray<TIN2, IN2>& GetInput1() { return data_input1_; }
  TfLiteIntArray& GetInputShape(const TestInputIndex index) {
    return *IntArrayFromInts(dims_inputs_[index]);
  }
  size_t GetInputSize(const TestInputIndex index) {
    if (index == kInputIndex0) {
      return IN1;
    } else {  // (index == kInputIndex1)
      return IN2;
    }
  }

  template <typename T, size_t D>
  void PopulateTensor(ElementArray<T, D>* const input,
                      const std::initializer_list<float>& list) {
    TF_LITE_MICRO_EXPECT_EQ(list.size(), D);

    auto iter = list.begin();
    for (size_t i = 0; i < list.size(); i++) {
      input->data[i] = static_cast<T>(iter[i]);
    }
  }

  template <typename T, size_t D>
  void QuantizeAndPopulate(ElementArray<T, D>* const input,
                           const std::initializer_list<float>& list) {
    TF_LITE_MICRO_EXPECT_EQ(list.size(), D);

    Quantize(list.begin(), input->data, D, input->scale, input->zero_point);
  }

  template <typename T, size_t D>
  void SignedSymmetricQuantizeAndPopulate(
      ElementArray<T, D>* const input,
      const std::initializer_list<float>& list) {
    TF_LITE_MICRO_EXPECT_EQ(list.size(), D);

    float min, max, scaling_factor;
    tensor_utils::SymmetricQuantizeFloats(list.begin(), static_cast<int>(D),
                                          input->data, &min, &max,
                                          &scaling_factor);
    input->scale = scaling_factor;
    input->zero_point = 0;
  }

  template <typename T>
  ElementArray<float, OUT> GetDequantizedOutput() {
    ElementArray<float, OUT> result;
    auto& output = this->GetOutput();
    Dequantize<T>(output.data, OUT, output.scale, output.zero_point,
                  result.data);

    return result;
  }

  template <typename T>
  ElementArray<T, OUT>& GetOutput() {
    return data_output_;
  }

 protected:
  void DoInvoke(const void* params, TfLiteTensor* tensors,
                const int tensors_count) {
    int kInputArrayData[] = {kMaxInputs, kInputTensor1, kInputTensor2};
    TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
    int kOutputArrayData[] = {1, kOutputTensor};
    TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

    micro::KernelRunner runner(registration_, tensors, tensors_count,
                               inputs_array, outputs_array,
                               const_cast<void*>(params));

    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

    // The output tensor dims will have moved to a location in the
    // memory arena.  Copy the tensor dims back into <dims_output_>
    TfLiteIntArray* dims = IntArrayFromInts(dims_output_);
    IntArrayCopy(*tensors[kOutputTensor].dims, dims);
  }

 private:
  int dims_inputs_[kMaxInputs][kMaxDims + 1];  // TfLiteIntArray[kMaxInputs]
  int dims_output_[kMaxDims + 1];              // TfLiteIntArray
  ElementArray<TIN1, IN1> data_input0_;
  ElementArray<TIN2, IN2> data_input1_;
  ElementArray<TOUT, OUT> data_output_;
  const TfLiteRegistration registration_;
};

}  // namespace testing
}  // namespace tflite

using tflite::testing::ArrayFloatNear;
using tflite::testing::ElementsAreArray;
using tflite::testing::ExpectThat;

#define EXPECT_THAT(a, b) ExpectThat(a, b)

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_BATCH_MATMUL_TEST_UTIL_H_
