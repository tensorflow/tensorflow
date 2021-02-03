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
#include <cstddef>
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
namespace {

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
        min(datum_min),
        max(datum_max),
        scale(datum_scale),
        zero_point(datum_zero_point) {}
  const TfLiteType type;
  const std::initializer_list<int>& shape;
  const float min;
  const float max;
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
  TestOpModel() {
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
        (datum.min != 0.0f || datum.max != 0.0f || datum.scale != 0.0f);
    if (quantizable) {
      if (datum.scale != 0.0f) {
        input->scale = datum.scale;
        input->zero_point = datum.zero_point;
      } else {
        input->scale = ScaleFromMinMax<int8_t>(datum.min, datum.max);
        input->zero_point = ZeroPointFromMinMax<int8_t>(datum.min, datum.max);
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
        (datum.min != 0.0f || datum.max != 0.0f || datum.scale != 0.0f);
    if (quantizable) {
      if (datum.scale != 0.0f) {
        data_output_.scale = datum.scale;
        data_output_.zero_point = datum.zero_point;
      } else {
        data_output_.scale = ScaleFromMinMax<int8_t>(datum.min, datum.max);
        data_output_.zero_point =
            ZeroPointFromMinMax<int8_t>(datum.min, datum.max);
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
  void DoInvoke(const TfLiteBatchMatMulParams& params, TfLiteTensor* tensors,
                const int tensors_count) {
    constexpr int kInputArrayData[] = {kMaxInputs, kInputTensor1,
                                       kInputTensor2};
    TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
    constexpr int kOutputArrayData[] = {1, kOutputTensor};
    TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

    const TfLiteRegistration registration = tflite::Register_BATCH_MATMUL();
    micro::KernelRunner runner(
        registration, tensors, tensors_count, inputs_array, outputs_array,
        static_cast<void*>(const_cast<TfLiteBatchMatMulParams*>(&params)));

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
};

template <typename T, size_t IN1, size_t IN2, size_t OUT>
class BatchMatMulOpModel : public TestOpModel<T, T, T, IN1, IN2, OUT> {
 public:
  BatchMatMulOpModel(const TestTensorData& lhs, const TestTensorData& rhs,
                     bool adj_x = false, bool adj_y = false)
      : TestOpModel<T, T, T, IN1, IN2, OUT>(), adj_x_(adj_x), adj_y_(adj_y) {
    this->AddInput(lhs, kInputIndex0);
    this->AddInput(rhs, kInputIndex1);
  }

  inline ElementArray<T, IN1>* lhs() { return &this->GetInput0(); }
  inline ElementArray<T, IN2>* rhs() { return &this->GetInput1(); }

  void Invoke() {
    TfLiteTensor tensors[] = {
        CreateTensor(lhs()->data, &this->GetInputShape(kInputIndex0)),
        CreateTensor(rhs()->data, &this->GetInputShape(kInputIndex1)),
        CreateTensor(this->GetOutput().data, &this->GetOutputShape()),
    };
    constexpr int tensors_count = std::extent<decltype(tensors)>::value;

    TfLiteBatchMatMulParams params;
    params.adj_x = adj_x_;
    params.adj_y = adj_y_;
    params.asymmetric_quantize_inputs = false;
    this->DoInvoke(params, tensors, tensors_count);
  }

 private:
  bool adj_x_;
  bool adj_y_;
};

template <typename T, size_t IN1, size_t IN2, size_t OUT>
class QuantizedBatchMatMulOpModel : public TestOpModel<T, T, T, IN1, IN2, OUT> {
 public:
  QuantizedBatchMatMulOpModel(int units, int batches, const TestTensorData& lhs,
                              const TestTensorData& output = {kTfLiteInt8},
                              bool adj_x = false, bool adj_y = false)
      : TestOpModel<T, T, T, IN1, IN2, OUT>(), adj_x_(adj_x), adj_y_(adj_y) {
    int input_size = ElementCount(this->GetInputShape(kInputIndex0)) / batches;

    this->AddInput(lhs, &this->GetInput0());
    this->AddInput({lhs.type,
                    {input_size, units},
                    0,
                    0,
                    this->GetInput0().scale,
                    this->GetInput0().zero_point},
                   &this->GetInput1());
    this->AddOutput(output);
  }

  template <typename TRHS>
  void SetWeights(const std::initializer_list<float>& data) {
    this->template QuantizeAndPopulate<TRHS>(rhs(), data);
  }

  template <typename TLHS>
  void SetInput(const std::initializer_list<float>& data) {
    this->template QuantizeAndPopulate<TLHS>(lhs(), data);
  }

  inline ElementArray<T, IN1>* lhs() { return &this->GetInput0(); }
  inline ElementArray<T, IN2>* rhs() { return &this->GetInput1(); }

  void Invoke() {
    TfLiteTensor tensors[] = {
        CreateQuantizedTensor(lhs()->data, &this->GetInputShape(kInputIndex0),
                              lhs()->scale, lhs()->zero_point),
        CreateQuantizedTensor(rhs()->data, &this->GetInputShape(kInputIndex1),
                              rhs()->scale, rhs()->zero_point),
        CreateQuantizedTensor(this->GetOutput().data, &this->GetOutputShape(),
                              this->GetOutput().scale,
                              this->GetOutput().zero_point),
    };
    constexpr int tensors_count = std::extent<decltype(tensors)>::value;

    TfLiteBatchMatMulParams params;
    params.adj_x = adj_x_;
    params.adj_y = adj_y_;
    params.asymmetric_quantize_inputs = false;
    this->DoInvoke(params, tensors, tensors_count);
  }

 private:
  bool adj_x_;
  bool adj_y_;
};

template <typename T1, typename T2, size_t IN1, size_t IN2, size_t OUT>
class HybridBatchMatMulOpModel : public TestOpModel<T1, T2, T1, IN1, IN2, OUT> {
 public:
  HybridBatchMatMulOpModel(int units, int batches, const TestTensorData& lhs,
                           const TestTensorData& rhs,
                           const TestTensorData& output = {kTfLiteFloat32},
                           bool asymmetric_quantize_inputs = true)
      : TestOpModel<T1, T2, T1, IN1, IN2, OUT>(),
        asymmetric_quantize_inputs_(asymmetric_quantize_inputs) {
    this->AddInput(lhs, &this->GetInput0());
    this->AddInput(rhs, &this->GetInput1());
  }

  void SetSignedWeights(const std::initializer_list<float>& data) {
    this->SignedSymmetricQuantizeAndPopulate(rhs(), data);
  }

  void SetInput(const std::initializer_list<float>& data) {
    this->PopulateTensor(lhs(), data);
  }

  inline ElementArray<T1, IN1>* lhs() { return &this->GetInput0(); }
  inline ElementArray<T2, IN2>* rhs() { return &this->GetInput1(); }

  void Invoke() {
    TfLiteTensor tensors[] = {
        CreateTensor(lhs()->data, &this->GetInputShape(kInputIndex0)),
        CreateQuantizedTensor(rhs()->data, &this->GetInputShape(kInputIndex1),
                              rhs()->scale, rhs()->zero_point),
        CreateTensor(this->GetOutput().data, &this->GetOutputShape()),
    };
    constexpr int tensors_count = std::extent<decltype(tensors)>::value;

    TfLiteBatchMatMulParams params;
    params.adj_x = false;
    params.adj_y = false;
    params.asymmetric_quantize_inputs = asymmetric_quantize_inputs_;
    this->DoInvoke(params, tensors, tensors_count);
  }

 private:
  bool asymmetric_quantize_inputs_;
};

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::ArrayFloatNear;
using tflite::testing::BatchMatMulOpModel;
using tflite::testing::ElementsAreArray;
using tflite::testing::ExpectThat;
using tflite::testing::HybridBatchMatMulOpModel;
using tflite::testing::QuantizedBatchMatMulOpModel;

TF_LITE_MICRO_TESTS_BEGIN

#define EXPECT_THAT(a, b) ExpectThat(a, b)

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Simple) {
  BatchMatMulOpModel<float, 6, 12, 8> model({kTfLiteFloat32, {1, 2, 3}},
                                            {kTfLiteFloat32, {1, 3, 4}});
  model.PopulateTensor<float>(model.lhs(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  model.Invoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4}));
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_SimpleRHSAdjoint) {
  BatchMatMulOpModel<float, 6, 12, 8> model(
      {kTfLiteFloat32, {1, 2, 3}}, {kTfLiteFloat32, {1, 4, 3}}, false, true);
  model.PopulateTensor<float>(model.lhs(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 11, 15, 8, 12, 16, 9, 13, 17, 10, 14, 18});
  model.Invoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4}));
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_SimpleLHSAdjoint) {
  BatchMatMulOpModel<float, 6, 12, 8> model(
      {kTfLiteFloat32, {1, 3, 2}}, {kTfLiteFloat32, {1, 3, 4}}, true, false);
  model.PopulateTensor<float>(model.lhs(), {1, 4, 2, 5, 3, 6});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  model.Invoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4}));
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_BatchSizeTwo) {
  BatchMatMulOpModel<float, 12, 24, 16> model({kTfLiteFloat32, {2, 2, 3}},
                                              {kTfLiteFloat32, {2, 3, 4}});
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});
  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218., 560., 584.,
                        608., 632., 767., 800., 833., 866.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 4}));
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Broadcast) {
  BatchMatMulOpModel<float, 12, 12, 16> model({kTfLiteFloat32, {2, 2, 3}},
                                              {kTfLiteFloat32, {3, 4}});
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});

  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218., 272., 296.,
                        320., 344., 371., 404., 437., 470.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 4}));
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_BroadcastLHSAdjoint) {
  BatchMatMulOpModel<float, 12, 12, 16> model(
      {kTfLiteFloat32, {2, 3, 2}}, {kTfLiteFloat32, {3, 4}}, true, false);
  model.PopulateTensor<float>(model.lhs(),
                              {1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});

  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({74., 80., 86., 92., 173., 188., 203., 218., 272., 296.,
                        320., 344., 371., 404., 437., 470.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 4}));
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Broadcast2) {
  BatchMatMulOpModel<float, 12, 24, 72> model({kTfLiteFloat32, {2, 1, 3, 2}},
                                              {kTfLiteFloat32, {3, 2, 4}});
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});

  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101.,
                        112., 123., 134., 53.,  56.,  59.,  62.,  121., 128.,
                        135., 142., 189., 200., 211., 222., 77.,  80.,  83.,
                        86.,  177., 184., 191., 198., 277., 288., 299., 310.,
                        137., 152., 167., 182., 173., 192., 211., 230., 209.,
                        232., 255., 278., 257., 272., 287., 302., 325., 344.,
                        363., 382., 393., 416., 439., 462., 377., 392., 407.,
                        422., 477., 496., 515., 534., 577., 600., 623., 646.}));

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3, 3, 4}));
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Broadcast2LHSAdjoint) {
  BatchMatMulOpModel<float, 12, 24, 72> model(
      {kTfLiteFloat32, {2, 1, 2, 3}}, {kTfLiteFloat32, {3, 2, 4}}, true, false);
  model.PopulateTensor<float>(model.lhs(),
                              {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});

  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101.,
                        112., 123., 134., 53.,  56.,  59.,  62.,  121., 128.,
                        135., 142., 189., 200., 211., 222., 77.,  80.,  83.,
                        86.,  177., 184., 191., 198., 277., 288., 299., 310.,
                        137., 152., 167., 182., 173., 192., 211., 230., 209.,
                        232., 255., 278., 257., 272., 287., 302., 325., 344.,
                        363., 382., 393., 416., 439., 462., 377., 392., 407.,
                        422., 477., 496., 515., 534., 577., 600., 623., 646.}));

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3, 3, 4}));
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Broadcast2RHSAdjoint) {
  BatchMatMulOpModel<float, 12, 24, 72> model(
      {kTfLiteFloat32, {2, 1, 3, 2}}, {kTfLiteFloat32, {3, 4, 2}}, false, true);
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  11, 8,  12, 9,  13, 10, 14, 15, 19, 16, 20,
                               17, 21, 18, 22, 23, 27, 24, 28, 25, 29, 26, 30});
  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101.,
                        112., 123., 134., 53.,  56.,  59.,  62.,  121., 128.,
                        135., 142., 189., 200., 211., 222., 77.,  80.,  83.,
                        86.,  177., 184., 191., 198., 277., 288., 299., 310.,
                        137., 152., 167., 182., 173., 192., 211., 230., 209.,
                        232., 255., 278., 257., 272., 287., 302., 325., 344.,
                        363., 382., 393., 416., 439., 462., 377., 392., 407.,
                        422., 477., 496., 515., 534., 577., 600., 623., 646.}));

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3, 3, 4}));
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_Broadcast2BothAdjoint) {
  BatchMatMulOpModel<float, 12, 24, 72> model(
      {kTfLiteFloat32, {2, 1, 2, 3}}, {kTfLiteFloat32, {3, 4, 2}}, true, true);
  model.PopulateTensor<float>(model.lhs(),
                              {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  11, 8,  12, 9,  13, 10, 14, 15, 19, 16, 20,
                               17, 21, 18, 22, 23, 27, 24, 28, 25, 29, 26, 30});
  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({29.,  32.,  35.,  38.,  65.,  72.,  79.,  86.,  101.,
                        112., 123., 134., 53.,  56.,  59.,  62.,  121., 128.,
                        135., 142., 189., 200., 211., 222., 77.,  80.,  83.,
                        86.,  177., 184., 191., 198., 277., 288., 299., 310.,
                        137., 152., 167., 182., 173., 192., 211., 230., 209.,
                        232., 255., 278., 257., 272., 287., 302., 325., 344.,
                        363., 382., 393., 416., 439., 462., 377., 392., 407.,
                        422., 477., 496., 515., 534., 577., 600., 623., 646.}));

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3, 3, 4}));
}

TF_LITE_MICRO_TEST(BatchMatMulOpTestFloat32Test_BroadcastFromRHS) {
  BatchMatMulOpModel<float, 20, 30, 24> model({kTfLiteFloat32, {4, 5}},
                                              {kTfLiteFloat32, {3, 1, 5, 2}});
  model.PopulateTensor<float>(
      model.lhs(),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  model.PopulateTensor<float>(
      model.rhs(),
      {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
       22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36});

  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({185., 200., 460.,  500.,  735.,  800.,  1010., 1100.,
                        335., 350., 860.,  900.,  1385., 1450., 1910., 2000.,
                        485., 500., 1260., 1300., 2035., 2100., 2810., 2900.}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 1, 4, 2}));
}

TF_LITE_MICRO_TEST(HybridAsymmetricBatchMatMulOpTestSimpleTestQuantizedInt8) {
  HybridBatchMatMulOpModel<float, int8_t, 20, 30, 6> m(
      /*units=*/3, /*batches=*/2,
      /*lhs=*/{kTfLiteFloat32, {2, 10}},
      /*rhs=*/{kTfLiteInt8, {10, 3}, 0, 0, 10.0 / 127.0, 0});

  m.SetSignedWeights({
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput({
      11, 12, 13, 14, 15, 16, 17, 18,  -19, -20,  // batch 1, 0
      11, 12, 13, 14, 15, 16, 17, -18, 19,  -20,  // batch 1, 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     196,
                                     196,
                                     196,
                                     246,
                                     246,
                                     246,
                                 },
                                 /*max_abs_error=*/0.64f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
}

TF_LITE_MICRO_TEST(
    HybridAsymmetricBatchMatMulOpTestQuantizedInt8BroadcastWeights) {
  HybridBatchMatMulOpModel<float, int8_t, 40, 30, 12> m(
      /*units=*/3, /*batches=*/2,
      /*lhs=*/{kTfLiteFloat32, {2, 2, 10}},
      /*rhs=*/{kTfLiteInt8, {10, 3}, 0, 0, 10.0 / 127.0, 0});

  m.SetSignedWeights({
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput({
      1,  2,  3,  4,  5,  6,  7,  8,   -9,  -10,  // batch 0, 0
      1,  2,  3,  4,  5,  6,  7,  -8,  9,   -10,  // batch 0, 1
      11, 12, 13, 14, 15, 16, 17, 18,  -19, -20,  // batch 1, 0
      11, 12, 13, 14, 15, 16, 17, -18, 19,  -20,  // batch 1, 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, 24, 24,     //
                                     58, 58, 58,     //
                                     196, 196, 196,  //
                                     246, 246, 246,  //
                                 },
                                 /*max_abs_error=*/1.3f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 3}));
}

TF_LITE_MICRO_TEST(
    HybridAsymmetricBatchMatMulOpTestQuantizedInt8BroadcastBigWeights) {
  HybridBatchMatMulOpModel<float, int8_t, 40, 90, 36> m(
      /*units=*/9, /*batches=*/2,
      /*lhs=*/{kTfLiteFloat32, {2, 2, 10}},
      /*rhs=*/{kTfLiteInt8, {10, 9}, 0, 0, 10.0 / 127.0, 0});

  m.SetSignedWeights({
      1, 1, 1, 17, 17, 17, 26, 26, 26, 2,  2,  2,  18, 18, 18, 27, 27, 27,
      3, 3, 3, 19, 19, 19, 28, 28, 28, 4,  4,  4,  20, 20, 20, 29, 29, 29,
      5, 5, 5, 21, 21, 21, 30, 30, 30, 6,  6,  6,  22, 22, 22, 31, 31, 31,
      7, 7, 7, 23, 23, 23, 32, 32, 32, 8,  8,  8,  24, 24, 24, 33, 33, 33,
      9, 9, 9, 25, 25, 25, 34, 34, 34, 10, 10, 10, 26, 26, 26, 35, 35, 35,
  });

  m.SetInput({
      1,  2,  3,  4,  5,  6,  7,  8,   -9,  -10,  // batch 0, 0
      1,  2,  3,  4,  5,  6,  7,  -8,  9,   -10,  // batch 0, 1
      11, 12, 13, 14, 15, 16, 17, 18,  -19, -20,  // batch 1, 0
      11, 12, 13, 14, 15, 16, 17, -18, 19,  -20,  // batch 1, 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      23,  23,  23,  295,  295,  295,  449,  449,  449,   //
                      60,  60,  60,  364,  364,  364,  533,  533,  533,   //
                      195, 195, 195, 1429, 1429, 1429, 2124, 2124, 2124,  //
                      250, 250, 250, 1512, 1512, 1512, 2213, 2213, 2213   //
                  },
                  /*max_abs_error=*/1.3f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 9}));
}

TF_LITE_MICRO_TEST(
    HybridAsymmetricBatchMatMulOpTestQuantizedInt8BroadcastInputs) {
  HybridBatchMatMulOpModel<float, int8_t, 20, 60, 12> m(
      /*units=*/3, /*batches=*/2,
      /*lhs=*/{kTfLiteFloat32, {2, 10}},
      /*rhs=*/{kTfLiteInt8, {2, 10, 3}, 0, 0, 10.0 / 127.0, 0});

  m.SetSignedWeights({
      1, -3, 1, 2, -2, 2, 3, -1, 3, 4,  0, 4, 5, 1, 5, 6, 2, 6,  7,  3,
      7, 8,  4, 8, 9,  5, 9, 10, 6, 10, 1, 1, 1, 2, 2, 2, 3, 3,  3,  4,
      4, 4,  5, 5, 5,  6, 6, 6,  7, 7,  7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // batch 0, 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // batch 0, 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, -45, 24,  //
                                     58, -18, 58,  //
                                     24, 24, 24,   //
                                     58, 58, 58,   //
                                 },
                                 /*max_abs_error=*/0.64f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 3}));
}

TF_LITE_MICRO_TEST(HybridSymmetricBatchMatMulOpTestSimpleTestQuantizedInt8) {
  HybridBatchMatMulOpModel<float, int8_t, 20, 30, 6> m(
      /*units=*/3, /*batches=*/2,
      /*lhs=*/{kTfLiteFloat32, {2, 10}},
      /*rhs=*/{kTfLiteInt8, {10, 3}, 0, 0, 10.0 / 127.0, 0},
      /*output=*/{kTfLiteFloat32}, /*asymmetric_quantize_inputs=*/false);

  m.SetSignedWeights({
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput({
      11, 12, 13, 14, 15, 16, 17, 18,  -19, -20,  // batch 1, 0
      11, 12, 13, 14, 15, 16, 17, -18, 19,  -20,  // batch 1, 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     194,
                                     194,
                                     194,
                                     248,
                                     248,
                                     248,
                                 },
                                 /*max_abs_error=*/0.64f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
}

TF_LITE_MICRO_TEST(
    HybridSymmetricBatchMatMulOpTestQuantizedInt8BroadcastWeights) {
  HybridBatchMatMulOpModel<float, int8_t, 40, 30, 12> m(
      /*units=*/3, /*batches=*/2,
      /*lhs=*/{kTfLiteFloat32, {2, 2, 10}},
      /*rhs=*/{kTfLiteInt8, {10, 3}, 0, 0, 10.0 / 127.0, 0},
      /*output=*/{kTfLiteFloat32}, /*asymmetric_quantize_inputs=*/false);

  m.SetSignedWeights({
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput({
      1,  2,  3,  4,  5,  6,  7,  8,   -9,  -10,  // batch 0, 0
      1,  2,  3,  4,  5,  6,  7,  -8,  9,   -10,  // batch 0, 1
      11, 12, 13, 14, 15, 16, 17, 18,  -19, -20,  // batch 1, 0
      11, 12, 13, 14, 15, 16, 17, -18, 19,  -20,  // batch 1, 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, 24, 24,     //
                                     56, 56, 56,     //
                                     194, 194, 194,  //
                                     248, 248, 248,  //
                                 },
                                 /*max_abs_error=*/1.3f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 3}));
}

TF_LITE_MICRO_TEST(
    HybridSymmetricBatchMatMulOpTestQuantizedInt8BroadcastBigWeights) {
  HybridBatchMatMulOpModel<float, int8_t, 40, 90, 36> m(
      /*units=*/9, /*batches=*/2,
      /*lhs=*/{kTfLiteFloat32, {2, 2, 10}},
      /*rhs=*/{kTfLiteInt8, {10, 9}, 0, 0, 10.0 / 127.0, 0}, {kTfLiteFloat32},
      false);

  m.SetSignedWeights({
      1, 1, 1, 17, 17, 17, 26, 26, 26, 2,  2,  2,  18, 18, 18, 27, 27, 27,
      3, 3, 3, 19, 19, 19, 28, 28, 28, 4,  4,  4,  20, 20, 20, 29, 29, 29,
      5, 5, 5, 21, 21, 21, 30, 30, 30, 6,  6,  6,  22, 22, 22, 31, 31, 31,
      7, 7, 7, 23, 23, 23, 32, 32, 32, 8,  8,  8,  24, 24, 24, 33, 33, 33,
      9, 9, 9, 25, 25, 25, 34, 34, 34, 10, 10, 10, 26, 26, 26, 35, 35, 35,
  });

  m.SetInput({
      1,  2,  3,  4,  5,  6,  7,  8,   -9,  -10,  // batch 0, 0
      1,  2,  3,  4,  5,  6,  7,  -8,  9,   -10,  // batch 0, 1
      11, 12, 13, 14, 15, 16, 17, 18,  -19, -20,  // batch 1, 0
      11, 12, 13, 14, 15, 16, 17, -18, 19,  -20,  // batch 1, 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      23,  23,  23,  296,  296,  296,  451,  451,  451,   //
                      58,  58,  58,  362,  362,  362,  529,  529,  529,   //
                      193, 193, 193, 1424, 1424, 1424, 2118, 2118, 2118,  //
                      253, 253, 253, 1519, 1519, 1519, 2223, 2223, 2223   //
                  },
                  /*max_abs_error=*/1.3f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 9}));
}

TF_LITE_MICRO_TEST(
    HybridSymmetricBatchMatMulOpTestQuantizedInt8BroadcastInputs) {
  HybridBatchMatMulOpModel<float, int8_t, 20, 60, 12> m(
      /*units=*/3, /*batches=*/2,
      /*lhs=*/{kTfLiteFloat32, {2, 10}},
      /*rhs=*/{kTfLiteInt8, {2, 10, 3}, 0, 0, 10.0 / 127.0, 0},
      {kTfLiteFloat32}, false);

  m.SetSignedWeights({
      1, -3, 1, 2, -2, 2, 3, -1, 3, 4,  0, 4, 5, 1, 5, 6, 2, 6,  7,  3,
      7, 8,  4, 8, 9,  5, 9, 10, 6, 10, 1, 1, 1, 2, 2, 2, 3, 3,  3,  4,
      4, 4,  5, 5, 5,  6, 6, 6,  7, 7,  7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // batch 0, 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // batch 0, 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {
                                     24, -45, 24,  //
                                     56, -19, 56,  //
                                     24, 24, 24,   //
                                     56, 56, 56,   //
                                 },
                                 /*max_abs_error=*/0.64f)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 3}));
}

TF_LITE_MICRO_TEST(QuantizedBatchMatMulOpTestSimpleTestQuantizedInt8) {
  QuantizedBatchMatMulOpModel<int8_t, 20, 30, 6> m(
      /*units=*/3, /*batches*/ 2,
      /*lhs=*/{kTfLiteInt8, {2, 10}, -63.5, 64},
      /*output=*/{kTfLiteInt8, {}, -127, 128});

  m.SetWeights<int8_t>({
      1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5,  5,  5,
      6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
  });

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  m.Invoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({23, 23, 23, 57, 57, 57})));
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({22, 22, 22, 56, 56, 56}));
}

TF_LITE_MICRO_TESTS_END
