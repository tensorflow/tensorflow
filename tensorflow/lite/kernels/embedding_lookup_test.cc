/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License
for the specific language governing permissions and limitations under the
License.
==============================================================================*/
// Unit test for TFLite Lookup op.

#include <stdint.h>

#include <functional>
#include <initializer_list>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

float kTestTolerance = 7.41e-03;

using ::testing::ElementsAreArray;

class BaseEmbeddingLookupOpModel : public SingleOpModel {
 public:
  BaseEmbeddingLookupOpModel(
      std::initializer_list<int> index_shape,
      std::initializer_list<int> weight_shape,
      TensorType weight_type = TensorType_FLOAT32,
      TensorType output_type = TensorType_FLOAT32,
      const std::vector<float>& per_channel_quantization_scales = {}) {
    input_ = AddInput(TensorType_INT32);
    if (per_channel_quantization_scales.empty()) {
      weight_ = AddInput(weight_type);
    } else {
      std::vector<int64_t> per_channel_quantization_offsets(
          per_channel_quantization_scales.size(), 0);
      weight_ = AddInput({weight_type, weight_shape, 0, 0, 0, 0, true,
                          per_channel_quantization_scales,
                          per_channel_quantization_offsets, 0});
    }
    output_ = AddOutput(output_type);
    SetBuiltinOp(BuiltinOperator_EMBEDDING_LOOKUP, BuiltinOptions_NONE, 0);
    BuildInterpreter({index_shape, weight_shape});
  }

  void SetInput(std::initializer_list<int> data) {
    PopulateTensor(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input_;
  int weight_;
  int output_;
};

class EmbeddingLookupOpModel : public BaseEmbeddingLookupOpModel {
 public:
  using BaseEmbeddingLookupOpModel::BaseEmbeddingLookupOpModel;

  template <typename T>
  void Set3DWeightMatrix(const std::function<T(int, int, int)>& function) {
    TfLiteTensor* tensor = interpreter_->tensor(weight_);
    int rows = tensor->dims->data[0];
    int columns = tensor->dims->data[1];
    int features = tensor->dims->data[2];
    T* data = GetTensorData<T>(tensor);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        for (int k = 0; k < features; k++) {
          data[(i * columns + j) * features + k] = function(i, j, k);
        }
      }
    }
  }

  template <typename T>
  void Set2DWeightMatrix(const std::function<T(int, int)>& function) {
    TfLiteTensor* tensor = interpreter_->tensor(weight_);
    int64_t rows = tensor->dims->data[0];
    int64_t columns = tensor->dims->data[1];
    T* data = GetTensorData<T>(tensor);
    for (int64_t i = 0; i < rows; i++) {
      for (int64_t j = 0; j < columns; j++) {
        data[i * columns + j] = function(i, j);
      }
    }
  }
};

class HybridEmbeddingLookupOpModel : public BaseEmbeddingLookupOpModel {
 public:
  HybridEmbeddingLookupOpModel(std::initializer_list<int> index_shape,
                               std::initializer_list<int> weight_shape,
                               TensorType type)
      : BaseEmbeddingLookupOpModel(index_shape, weight_shape, type) {}

  void SetWeight(std::initializer_list<float> data) {
    SymmetricQuantizeAndPopulate(weight_, data);
  }

  void SetSignedWeight(std::initializer_list<float> data) {
    SignedSymmetricQuantizeAndPopulate(weight_, data);
  }
};

class PerAxisHybridEmbeddingLookupOpModel : public BaseEmbeddingLookupOpModel {
 public:
  PerAxisHybridEmbeddingLookupOpModel(
      std::initializer_list<int> index_shape,
      std::initializer_list<int> weight_shape,
      const std::vector<float>& per_channel_quantization_scales,
      TensorType type)
      : BaseEmbeddingLookupOpModel(index_shape, weight_shape, type,
                                   TensorType_FLOAT32,
                                   per_channel_quantization_scales) {}

  void SetSignedWeight(std::initializer_list<float> data) {
    PerChannelSymmetricQuantizeAndPopulate(weight_, data);
  }
};

// TODO(ahentz): write more tests that exercise the details of the op, such as
// lookup errors and variable input shapes.
TEST(EmbeddingLookupOpTest, SimpleTest) {
  EmbeddingLookupOpModel m({3}, {3, 2, 4});
  m.SetInput({1, 0, 2});
  m.Set3DWeightMatrix<float>(
      [](int i, int j, int k) -> float { return i + j / 10.0f + k / 100.0f; });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({
                  1.00, 1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                  0.00, 0.01, 0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                  2.00, 2.01, 2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
              })));
}

#if !defined(MEMORY_SANITIZER) && !defined(GOOGLE_UNSUPPORTED_OS_LOONIX) && \
    defined(__LP64__)
TEST(EmbeddingLookupOpTest, LargeTableTest) {
  EmbeddingLookupOpModel m({1}, {256000, 9216});
  // Choose a value specifically designed to overflow int32.max
  m.SetInput({235248});
  m.Set2DWeightMatrix<float>(
      [](int i, int j) -> float { return j + i / 100.; });

  // This will cause a lookup at index 235248 in a buffer where every row
  // has 9216 entries * 4 bytes per entry, which will overflow unless
  // the Op is using a 64-bit offset for address calculation.
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  std::vector<float> exp(9216);

  for (int s = 0; s < exp.size(); s++) {
    exp[s] = static_cast<float>(s) + 2352.48f;
  }
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear(exp)));
}
#endif

TEST(HybridEmbeddingLookupHybridOpTest, Simple2DTestUint8) {
  HybridEmbeddingLookupOpModel m({3}, {3, 8}, TensorType_UINT8);
  m.SetInput({1, 0, 2});
  m.SetWeight({
      0.00, 0.01, 0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, 1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01, 2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, 1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01, 0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01, 2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  kTestTolerance)));
}

TEST(HybridEmbeddingLookupHybridOpTest, Simple3DTestUint8) {
  HybridEmbeddingLookupOpModel m({3}, {3, 2, 4}, TensorType_UINT8);
  m.SetInput({1, 0, 2});
  m.SetWeight({
      0.00, 0.01, 0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, 1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01, 2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, 1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01, 0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01, 2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  kTestTolerance)));
}

TEST(HybridEmbeddingLookupHybridOpTest, Simple4DTestUint8) {
  HybridEmbeddingLookupOpModel m({3}, {3, 2, 2, 2}, TensorType_UINT8);
  m.SetInput({1, 0, 2});
  m.SetWeight({
      0.00, 0.01, 0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, 1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01, 2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, 1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01, 0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01, 2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  kTestTolerance)));
}

TEST(HybridEmbeddingLookupHybridOpTest, Simple2DTestInt8) {
  HybridEmbeddingLookupOpModel m({3}, {3, 8}, TensorType_INT8);
  m.SetInput({1, 0, 2});
  m.SetSignedWeight({
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  kTestTolerance)));
}

TEST(HybridEmbeddingLookupHybridOpTest, Simple3DTestInt8) {
  HybridEmbeddingLookupOpModel m({3}, {3, 2, 4}, TensorType_INT8);
  m.SetInput({1, 0, 2});
  m.SetSignedWeight({
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  kTestTolerance)));
}

TEST(HybridEmbeddingLookupHybridOpTest, Simple4DTestInt8) {
  HybridEmbeddingLookupOpModel m({3}, {3, 2, 2, 2}, TensorType_INT8);
  m.SetInput({1, 0, 2});
  m.SetSignedWeight({
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  kTestTolerance)));
}

TEST(EmbeddingLookupHybridOpTest, Simple3DTestQuantized) {
  EmbeddingLookupOpModel m({3}, {3, 2, 4}, TensorType_UINT8, TensorType_INT8);
  m.SetInput({1, 0, 2});
  m.Set3DWeightMatrix<uint8_t>(
      [](int i, int j, int k) -> uint8_t { return 100 * i + 10 * j + k; });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({
                  100, 101, 102, 103, 110, 111, 112, 113,  // Row 1
                  0,   1,   2,   3,   10,  11,  12,  13,   // Row 0
                  200, 201, 202, 203, 210, 211, 212, 213,  // Row 2
              }));
}

TEST(PerAxisHybridEmbeddingLookupHybridOpTest, PerAxisSimple2DTestInt8) {
  PerAxisHybridEmbeddingLookupOpModel m(
      {3}, {3, 8}, {0.00102, 0.0089, 0.016772}, TensorType_INT8);
  m.SetInput({1, 0, 2});
  m.SetSignedWeight({
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  kTestTolerance)));
}

TEST(PerAxisHybridEmbeddingLookupHybridOpTest, PerAxisSimple3DTestInt8) {
  PerAxisHybridEmbeddingLookupOpModel m(
      {3}, {3, 2, 4}, {0.00102, 0.0089, 0.016772}, TensorType_INT8);
  m.SetInput({1, 0, 2});
  m.SetSignedWeight({
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  kTestTolerance)));
}

TEST(PerAxisHybridEmbeddingLookupHybridOpTest, PerAxisSimple4DTestInt8) {
  PerAxisHybridEmbeddingLookupOpModel m(
      {3}, {3, 2, 2, 2}, {0.00102, 0.0089, 0.016772}, TensorType_INT8);
  m.SetInput({1, 0, 2});
  m.SetSignedWeight({
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  kTestTolerance)));
}

TEST(PerAxisHybridEmbeddingLookupHybridOpTest, PerAxisSimple2DTestInt4) {
  PerAxisHybridEmbeddingLookupOpModel m({3}, {3, 8}, {0.001, 0.02, 0.3},
                                        TensorType_INT4);
  m.SetInput({1, 0, 2});
  m.SetSignedWeight({
      0.00, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,  // Row 0
      0.02, -0.02, 0.04,  0.06,  0.08,  -0.04, -0.08, -0.06,  // Row 1
      0.3,  0.6,   0.9,   1.2,   1.5,   -0.3,  -0.6,  -0.9,   // Row 2
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(
      m.GetOutput<float>(),
      ElementsAreArray(ArrayFloatNear(
          {
              0.02, -0.02, 0.04,  0.06,  0.08,  -0.04, -0.08, -0.06,  // Row 1
              0.00, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,  // Row 0
              0.3,  0.6,   0.9,   1.2,   1.5,   -0.3,  -0.6,  -0.9,   // Row 2
          },
          kTestTolerance)));
}

TEST(PerAxisHybridEmbeddingLookupHybridOpTest, PerAxisSimple3DTestInt4) {
  PerAxisHybridEmbeddingLookupOpModel m({3}, {3, 2, 4}, {0.001, 0.02, 0.3},
                                        TensorType_INT4);
  m.SetInput({1, 0, 2});
  m.SetSignedWeight({
      0.00, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,  // Row 0
      0.02, -0.02, 0.04,  0.06,  0.08,  -0.04, -0.08, -0.06,  // Row 1
      0.3,  0.6,   0.9,   1.2,   1.5,   -0.3,  -0.6,  -0.9,   // Row 2
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(
      m.GetOutput<float>(),
      ElementsAreArray(ArrayFloatNear(
          {
              0.02, -0.02, 0.04,  0.06,  0.08,  -0.04, -0.08, -0.06,  // Row 1
              0.00, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,  // Row 0
              0.3,  0.6,   0.9,   1.2,   1.5,   -0.3,  -0.6,  -0.9,   // Row 2
          },
          kTestTolerance)));
}

TEST(PerAxisHybridEmbeddingLookupHybridOpTest, PerAxisSimple4DTestInt4) {
  PerAxisHybridEmbeddingLookupOpModel m({3}, {3, 2, 2, 2}, {0.001, 0.02, 0.3},
                                        TensorType_INT4);
  m.SetInput({1, 0, 2});
  m.SetSignedWeight({
      0.00, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,  // Row 0
      0.02, -0.02, 0.04,  0.06,  0.08,  -0.04, -0.08, -0.06,  // Row 1
      0.3,  0.6,   0.9,   1.2,   1.5,   -0.3,  -0.6,  -0.9,   // Row 2
  });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(
      m.GetOutput<float>(),
      ElementsAreArray(ArrayFloatNear(
          {
              0.02, -0.02, 0.04,  0.06,  0.08,  -0.04, -0.08, -0.06,  // Row 1
              0.00, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,  // Row 0
              0.3,  0.6,   0.9,   1.2,   1.5,   -0.3,  -0.6,  -0.9,   // Row 2
          },
          kTestTolerance)));
}

}  // namespace
}  // namespace tflite
