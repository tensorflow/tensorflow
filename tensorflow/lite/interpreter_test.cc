/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/interpreter.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <map>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/external_cpu_backend_context.h"
#include "tensorflow/lite/interpreter_test_util.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/util.h"

namespace tflite {

namespace ops {
namespace builtin {
TfLiteRegistration* Register_PADV2();
TfLiteRegistration* Register_NEG();
}  // namespace builtin
}  // namespace ops

namespace {

using ::testing::IsEmpty;

// Make an interpreter that has no tensors and no nodes
TEST(BasicInterpreter, ZeroInterpreter) {
  testing::internal::CaptureStderr();

  Interpreter interpreter;

#ifndef NDEBUG
  const char* kExpectedLog = "INFO: Initialized TensorFlow Lite runtime";
#else
  const char* kExpectedLog = "";
#endif
  EXPECT_THAT(testing::internal::GetCapturedStderr(),
              testing::HasSubstr(kExpectedLog));

  interpreter.SetInputs({});
  interpreter.SetOutputs({});
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);

  // Creating a new interpreter should not redundantly log runtime init.
  testing::internal::CaptureStderr();
  Interpreter interpreter2;
  EXPECT_THAT(testing::internal::GetCapturedStderr(), IsEmpty());
}

// Test various error conditions.
TEST(BasicInterpreter, InvokeInvalidModel) {
  Interpreter interpreter;
  ASSERT_NE(interpreter.Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
}

TEST(BasicInterpreter, TestAllocateTensorsResetVariableTensorsFloatAndHyrbid) {
  Interpreter interpreter;
  int tensor_index;
  ASSERT_EQ(interpreter.AddTensors(1, &tensor_index), kTfLiteOk);
  constexpr int kTensorSize = 16;
  TfLiteQuantizationParams quant;
  interpreter.SetTensorParametersReadWrite(tensor_index, kTfLiteFloat32, "",
                                           {kTensorSize}, quant,
                                           /*is_variable=*/true);
  interpreter.SetVariables({tensor_index});
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  TfLiteTensor* tensor = interpreter.tensor(tensor_index);
  // Ensure that variable tensors are reset to zero.
  for (int i = 0; i < kTensorSize; ++i) {
    ASSERT_EQ(tensor->data.f[i], 0.0f);
  }
}

TEST(BasicInterpreter, TestAllocateTensorsResetVariableTensorsInt8) {
  Interpreter interpreter;
  int tensor_index;
  ASSERT_EQ(interpreter.AddTensors(1, &tensor_index), kTfLiteOk);
  constexpr int kTensorSize = 16;
  TfLiteQuantizationParams quant;
  quant.scale = 0.15;
  quant.zero_point = -3;
  interpreter.SetTensorParametersReadWrite(tensor_index, kTfLiteInt8, "",
                                           {kTensorSize}, quant,
                                           /*is_variable=*/true);
  interpreter.SetVariables({tensor_index});
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  TfLiteTensor* tensor = interpreter.tensor(tensor_index);
  // Ensure that variable tensors are reset to zero point.
  for (int i = 0; i < kTensorSize; ++i) {
    ASSERT_EQ(tensor->data.int8[i], -3);
  }
}

// Test size accessor functions.
TEST(BasicInterpreter, TestSizeFunctions) {
  Interpreter interpreter;
  int base_index;
  ASSERT_EQ(interpreter.nodes_size(), 0);
  ASSERT_EQ(interpreter.tensors_size(), 0);
  ASSERT_EQ(interpreter.AddTensors(2, &base_index), kTfLiteOk);
  ASSERT_EQ(interpreter.tensors_size(), 2);
  ASSERT_EQ(base_index, 0);
  ASSERT_EQ(interpreter.AddTensors(3, &base_index), kTfLiteOk);
  ASSERT_EQ(interpreter.tensors_size(), 5);
  ASSERT_EQ(interpreter.AddTensors(1), kTfLiteOk);
  ASSERT_EQ(interpreter.tensors_size(), 6);
  ASSERT_EQ(base_index, 2);
}

// Test if invalid indices make a model inconsistent (and conversely if
// valid indices keep a model consistent).
TEST(BasicInterpreter, InconsistentModel) {
  // Invalid inputs
  {
    Interpreter interpreter;
    ASSERT_NE(interpreter.SetInputs({5}), kTfLiteOk);
    ASSERT_NE(interpreter.AllocateTensors(), kTfLiteOk);
    ASSERT_NE(interpreter.Invoke(), kTfLiteOk);
    ASSERT_EQ(interpreter.inputs(), std::vector<int>());
  }
  // Invalid outputs
  {
    Interpreter interpreter;
    ASSERT_NE(interpreter.SetOutputs({5}), kTfLiteOk);
    ASSERT_NE(interpreter.AllocateTensors(), kTfLiteOk);
    ASSERT_NE(interpreter.Invoke(), kTfLiteOk);
    ASSERT_EQ(interpreter.outputs(), std::vector<int>());
  }
  // Invalid node inputs
  {
    Interpreter interpreter;
    TfLiteRegistration registration = {nullptr, nullptr, nullptr, nullptr};
    ASSERT_NE(interpreter.AddNodeWithParameters({3}, {0}, nullptr, 0, nullptr,
                                                &registration),
              kTfLiteOk);
    ASSERT_NE(interpreter.AllocateTensors(), kTfLiteOk);
    ASSERT_NE(interpreter.Invoke(), kTfLiteOk);
  }
  // Valid inputs and outputs and a node with valid inputs and outputs
  {
    Interpreter interpreter;
    ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
    TfLiteRegistration registration = {nullptr, nullptr, nullptr, nullptr};
    ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
    ASSERT_EQ(interpreter.SetOutputs({0}), kTfLiteOk);
    ASSERT_EQ(interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr,
                                                &registration),
              kTfLiteOk);
  }
}

// Make an interpreter that has one tensor but no ops
TEST(BasicInterpreter, CheckAllocate) {
  struct {
    TfLiteType type;
    size_t size;
  } cases[] = {
      {kTfLiteFloat32, sizeof(float)},         {kTfLiteInt32, sizeof(int32_t)},
      {kTfLiteUInt32, sizeof(uint32_t)},       {kTfLiteUInt8, sizeof(uint8_t)},
      {kTfLiteInt64, sizeof(int64_t)},         {kTfLiteInt16, sizeof(int16_t)},
      {kTfLiteFloat16, sizeof(TfLiteFloat16)},
  };

  for (auto test : cases) {
    Interpreter interpreter;
    ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
    interpreter.SetInputs({0, 1});
    interpreter.SetOutputs({});
    TfLiteQuantizationParams quant;

    interpreter.SetTensorParametersReadWrite(0, test.type, "", {3}, quant);
    interpreter.SetTensorParametersReadWrite(1, test.type, "", {4}, quant);
    ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
    ASSERT_EQ(interpreter.tensor(0)->bytes, 3 * test.size);
    ASSERT_NE(interpreter.tensor(0)->data.raw, nullptr);
    ASSERT_EQ(interpreter.tensor(1)->bytes, 4 * test.size);
    ASSERT_NE(interpreter.tensor(1)->data.raw, nullptr);
  }
}

TEST(BasicInterpreter, CheckQuantization) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
  interpreter.SetInputs({0, 1});
  interpreter.SetOutputs({});
  TfLiteType tensor_type = kTfLiteInt8;
  const uint8_t int8s[] = {3, 4};
  float scale = 0.5f;
  int32_t zero_point = 12;

  TfLiteQuantization rw_quantization;
  rw_quantization.type = kTfLiteAffineQuantization;
  auto* rw_affine_quantization = static_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  rw_affine_quantization->scale = TfLiteFloatArrayCreate(1);
  rw_affine_quantization->zero_point = TfLiteIntArrayCreate(1);
  rw_affine_quantization->scale->data[0] = scale;
  rw_affine_quantization->zero_point->data[0] = zero_point;
  rw_quantization.params = rw_affine_quantization;

  TfLiteQuantization ro_quantization;
  ro_quantization.type = kTfLiteAffineQuantization;
  auto* ro_affine_quantization = static_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  ro_affine_quantization->scale = TfLiteFloatArrayCreate(1);
  ro_affine_quantization->zero_point = TfLiteIntArrayCreate(1);
  ro_affine_quantization->scale->data[0] = scale;
  ro_affine_quantization->zero_point->data[0] = zero_point;
  ro_quantization.params = ro_affine_quantization;

  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(0, tensor_type, "", {3},
                                                     rw_quantization),
            kTfLiteOk);
  ASSERT_EQ(interpreter.SetTensorParametersReadOnly(
                1, tensor_type, "", {2}, ro_quantization,
                reinterpret_cast<const char*>(int8s), 2),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  // Check that the legacy scale and zero_point are set correctly.
  ASSERT_EQ(interpreter.tensor(0)->params.scale, scale);
  ASSERT_EQ(interpreter.tensor(0)->params.zero_point, zero_point);
  ASSERT_EQ(interpreter.tensor(0)->quantization.type, rw_quantization.type);
  ASSERT_EQ(interpreter.tensor(1)->params.scale, scale);
  ASSERT_EQ(interpreter.tensor(1)->params.zero_point, zero_point);
  ASSERT_EQ(interpreter.tensor(1)->quantization.type, ro_quantization.type);
}

TEST(BasicInterpreter, CheckResize) {
  const float floats[] = {-3., -4.};
  const int32_t int32s[] = {-3, -4};
  const uint32_t uint32s[] = {3, 4};
  const uint8_t uint8s[] = {3, 4};
  const int64_t int64s[] = {6, -7};
  const int16_t int16s[] = {8, -9};
  const Eigen::half float16s[] = {Eigen::half_impl::float_to_half_rtne(-3.f),
                                  Eigen::half_impl::float_to_half_rtne(-4.f)};

  struct {
    TfLiteType type;
    size_t size;
    const char* array;
  } cases[] = {
      {kTfLiteFloat32, sizeof(float), reinterpret_cast<const char*>(floats)},
      {kTfLiteInt32, sizeof(int32_t), reinterpret_cast<const char*>(int32s)},
      {kTfLiteUInt32, sizeof(uint32_t), reinterpret_cast<const char*>(uint32s)},
      {kTfLiteUInt8, sizeof(uint8_t), reinterpret_cast<const char*>(uint8s)},
      {kTfLiteInt64, sizeof(int64_t), reinterpret_cast<const char*>(int64s)},
      {kTfLiteInt16, sizeof(int16_t), reinterpret_cast<const char*>(int16s)},
      {kTfLiteFloat16, sizeof(TfLiteFloat16),
       reinterpret_cast<const char*>(float16s)},
  };

  for (auto test : cases) {
    Interpreter interpreter;

    ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
    interpreter.SetInputs({0, 1});
    interpreter.SetOutputs({});
    TfLiteQuantizationParams quant;

    ASSERT_EQ(
        interpreter.SetTensorParametersReadWrite(0, test.type, "", {3}, quant),
        kTfLiteOk);
    ASSERT_EQ(interpreter.SetTensorParametersReadOnly(
                  1, test.type, "", {2}, quant, test.array, 2 * test.size),
              kTfLiteOk);
    ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
    ASSERT_EQ(interpreter.ResizeInputTensor(0, {1, 2}), kTfLiteOk);
    // Resizing a mmapped tensor is not allowed and should produce error.
    ASSERT_NE(interpreter.ResizeInputTensor(1, {3}), kTfLiteOk);
    // Set the tensor to be mmapped but with a buffer size that is insufficient
    // to match the dimensionality.
    ASSERT_NE(interpreter.SetTensorParametersReadOnly(
                  1, test.type, "", {2}, quant, test.array, 1 * test.size),
              kTfLiteOk);
    // Allocating should work since we should have our last correct array
    // values in place.
    ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  }
}

TEST(BasicInterpreter, CheckAlignment) {
  struct {
    TfLiteType type;
  } cases[] = {{kTfLiteFloat32}, {kTfLiteInt32}, {kTfLiteUInt32},
               {kTfLiteUInt8},   {kTfLiteInt64}, {kTfLiteInt16},
               {kTfLiteFloat16}};

  for (auto test : cases) {
    Interpreter interpreter;

    ASSERT_EQ(interpreter.AddTensors(4), kTfLiteOk);

    for (int i = 0; i < 4; i++) {
      TfLiteQuantizationParams quant;
      interpreter.SetTensorParametersReadWrite(i, test.type, "", {2 * i + 1},
                                               quant);
    }
    interpreter.AllocateTensors();
    for (int i = 0; i < 4; i++) {
      const TfLiteTensor& tensor = *interpreter.tensor(i);
      ASSERT_EQ(reinterpret_cast<intptr_t>(tensor.data.raw) % 4, 0);
    }
  }
}

TEST(BasicInterpreter, CheckArenaAllocation) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(10), kTfLiteOk);

  TfLiteQuantizationParams quant;
  TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

  std::vector<int> sizes{2048, 4096, 1023, 2047, 1021,
                         2047, 1023, 2046, 0,    2048};
  for (size_t i = 0; i < sizes.size(); ++i) {
    interpreter.SetTensorParametersReadWrite(static_cast<int>(i), kTfLiteUInt8,
                                             "", {sizes[i]}, quant);
  }
  interpreter.SetInputs({0, 1});
  interpreter.SetOutputs({9, 4});
  interpreter.AddNodeWithParameters({0, 1}, {2, 3}, nullptr, 0, nullptr, &reg);
  interpreter.AddNodeWithParameters({2, 1}, {4, 5}, nullptr, 0, nullptr, &reg);
  interpreter.AddNodeWithParameters({4, 3}, {6, 7}, nullptr, 0, nullptr, &reg);
  interpreter.AddNodeWithParameters({6, 5}, {8}, nullptr, 0, nullptr, &reg);
  interpreter.AddNodeWithParameters({8, 7}, {9}, nullptr, 0, nullptr, &reg);

  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  ASSERT_LT(interpreter.tensor(0)->data.raw, interpreter.tensor(1)->data.raw);
  ASSERT_LT(interpreter.tensor(1)->data.raw, interpreter.tensor(3)->data.raw);
  ASSERT_EQ(interpreter.tensor(3)->data.raw, interpreter.tensor(9)->data.raw);
  ASSERT_LT(interpreter.tensor(3)->data.raw, interpreter.tensor(5)->data.raw);
  ASSERT_LT(interpreter.tensor(5)->data.raw, interpreter.tensor(2)->data.raw);
  ASSERT_EQ(interpreter.tensor(2)->data.raw, interpreter.tensor(7)->data.raw);
  ASSERT_LT(interpreter.tensor(2)->data.raw, interpreter.tensor(4)->data.raw);
  // #4 is the one with the largest pointer.
  ASSERT_EQ(interpreter.tensor(8)->data.raw, nullptr);
}

TEST(BasicInterpreter, BufferAccess) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(1), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);

  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(
                0, kTfLiteFloat32, "", {3}, TfLiteQuantizationParams()),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  // Verify we get a valid pointer.
  ASSERT_NE(interpreter.typed_tensor<float>(0), nullptr);
  // Verify incorrect pointer is not returned.
  ASSERT_EQ(interpreter.typed_tensor<int>(0), nullptr);
  // Verify that raw c interface ptr matches safe interface.
  ASSERT_EQ(interpreter.typed_tensor<float>(0), interpreter.tensor(0)->data.f);
}

TEST(BasicInterpreter, NoOpInterpreter) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(1), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({0}), kTfLiteOk);

  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(
                0, kTfLiteFloat32, "", {3}, TfLiteQuantizationParams()),
            kTfLiteOk);

  ASSERT_EQ(interpreter.ResizeInputTensor(interpreter.inputs()[0], {1, 2, 3}),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
}

TEST(BasicInterpreter, RedundantAllocateTensors) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(1), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);

  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(
                0, kTfLiteFloat32, "", {3}, TfLiteQuantizationParams()),
            kTfLiteOk);

  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  const auto data_raw = interpreter.tensor(0)->data.raw;
  ASSERT_NE(data_raw, nullptr);

  // A redundant allocation request should have no impact.
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.tensor(0)->data.raw, data_raw);
}

TEST(BasicInterpreter, RedundantAllocateTensorsWithDynamicInputs) {
  Interpreter interpreter;
  TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};
  ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
  interpreter.SetInputs({0});
  interpreter.SetOutputs({1});
  interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr, &reg);

  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(
                0, kTfLiteFloat32, "", {3}, TfLiteQuantizationParams()),
            kTfLiteOk);
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(
                1, kTfLiteFloat32, "", {3}, TfLiteQuantizationParams()),
            kTfLiteOk);

  // Configure the input tensor as dynamic.
  interpreter.tensor(0)->data.raw = nullptr;
  interpreter.tensor(0)->allocation_type = kTfLiteDynamic;

  ASSERT_EQ(interpreter.ResizeInputTensor(interpreter.inputs()[0], {1, 2, 3}),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_NE(interpreter.tensor(1)->data.raw, nullptr);

  // Reset the output tensor's buffer.
  interpreter.tensor(1)->data.raw = nullptr;

  // A redundant allocation request should be honored, as the input tensor
  // was marked dynamic.
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_NE(interpreter.tensor(1)->data.raw, nullptr);
}

TEST(BasicInterpreter, ResizingTensors) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(1), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({0}), kTfLiteOk);

  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(
                0, kTfLiteFloat32, "", {3}, TfLiteQuantizationParams()),
            kTfLiteOk);

  int t = interpreter.inputs()[0];
  TfLiteTensor* tensor = interpreter.tensor(t);

  ASSERT_EQ(interpreter.ResizeInputTensor(t, {1, 2, 3}), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 6 * sizeof(float));
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  tensor->data.f[5] = 0.123f;

  // Changing from kTfLiteArenaRw to kTfLiteDynamic is quite complicate: we need
  // to unset data.raw, otherwise Realloc will try to free that memory.
  tensor->data.raw = nullptr;
  tensor->allocation_type = kTfLiteDynamic;

  ASSERT_EQ(interpreter.ResizeInputTensor(t, {1, 2, 4}), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 8 * sizeof(float));
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter.ResizeInputTensor(t, {}), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 1 * sizeof(float));
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter.ResizeInputTensor(t, {0}), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 0);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter.ResizeInputTensor(t, {1, 2, 0}), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 0);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  // TODO(ahentz): We shouldn't have to force reallocation, but
  // ResizeInputTensor doesn't realloc dynamic tensors. Also note that
  // TfLiteTensorRealloc(tensor->bytes, tensor) is a no-op.
  TfLiteTensorRealloc(9 * sizeof(float), tensor);
  tensor->data.f[7] = 0.123f;

  ASSERT_EQ(interpreter.ResizeInputTensor(t, {2, 2, 4}), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 16 * sizeof(float));
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  // TODO(ahentz): We shouldn't have to force reallocation, but
  // ResizeInputTensor doesn't realloc dynamic tensors. Also note that
  // TfLiteTensorRealloc(tensor->bytes, tensor) is a no-op.
  TfLiteTensorRealloc(17 * sizeof(float), tensor);
  tensor->data.f[15] = 0.123f;
}

TEST(BasicInterpreter, NoopResizingTensors) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(1), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({0}), kTfLiteOk);

  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(
                0, kTfLiteFloat32, "", {3}, TfLiteQuantizationParams()),
            kTfLiteOk);

  int t = interpreter.inputs()[0];
  TfLiteTensor* tensor = interpreter.tensor(t);

  ASSERT_EQ(interpreter.ResizeInputTensor(t, {1, 2, 3}), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 6 * sizeof(float));
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  tensor->data.f[5] = 0.123f;

  // Resizing to the same size should not trigger re-allocation.
  ASSERT_EQ(interpreter.ResizeInputTensor(t, {1, 2, 3}), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 6 * sizeof(float));
  ASSERT_NE(tensor->data.raw, nullptr);
  ASSERT_EQ(tensor->data.f[5], 0.123f);

  // Explicitly allocating should be a no-op, as no resize was performed.
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 6 * sizeof(float));
  ASSERT_NE(tensor->data.raw, nullptr);
  ASSERT_EQ(tensor->data.f[5], 0.123f);
}

TEST(BasicInterpreter, ResizingTensorsStrictInvalid) {
  // Tests ResizeInputTensorStrict where `dims_signature` is not specified.
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(1), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({0}), kTfLiteOk);

  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(
                0, kTfLiteFloat32, "", {1, 1, 3}, TfLiteQuantizationParams()),
            kTfLiteOk);

  int t = interpreter.inputs()[0];
  TfLiteTensor* tensor = interpreter.tensor(t);

  ASSERT_EQ(interpreter.ResizeInputTensorStrict(t, {1, 1, 3}), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 3 * sizeof(float));
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  // Invalid becuase `dims_signature` is not specified.
  ASSERT_EQ(interpreter.ResizeInputTensorStrict(t, {1, 2, 3}), kTfLiteError);
  EXPECT_EQ(tensor->bytes, 3 * sizeof(float));
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  // Assert that ResizeInputTensor works for this value.
  ASSERT_EQ(interpreter.ResizeInputTensor(t, {1, 2, 3}), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 6 * sizeof(float));
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
}

TEST(BasicInterpreter, ResizingTensorsStrict) {
  // Tests ResizeInputTensorStrict where `dims_signature` is specified.
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(1), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({0}), kTfLiteOk);

  std::vector<int> dims_signature = {-1, -1, 3};
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(
                0, kTfLiteFloat32, "", {1, 1, 3}, TfLiteQuantizationParams(),
                false, &dims_signature),
            kTfLiteOk);

  int t = interpreter.inputs()[0];
  TfLiteTensor* tensor = interpreter.tensor(t);

  ASSERT_EQ(interpreter.ResizeInputTensorStrict(t, {1, 2, 3}), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 6 * sizeof(float));
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter.ResizeInputTensorStrict(t, {1, 2, 4}), kTfLiteError);
  EXPECT_EQ(tensor->bytes, 6 * sizeof(float));
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  // Assert that ResizeInputTensor works for this value.
  ASSERT_EQ(interpreter.ResizeInputTensor(t, {1, 2, 4}), kTfLiteOk);
  EXPECT_EQ(tensor->bytes, 8 * sizeof(float));
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
}

// Simple op that does input = output.
TfLiteRegistration GetPassthroughOpRegistration() {
  TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};
  reg.init = [](TfLiteContext* context, const char*, size_t) -> void* {
    auto* first_new_tensor = new int;
    context->AddTensors(context, 2, first_new_tensor);
    return first_new_tensor;
  };
  reg.free = [](TfLiteContext* context, void* buffer) {
    delete static_cast<int*>(buffer);
  };
  reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    auto* first_new_tensor = static_cast<int*>(node->user_data);

    const TfLiteTensor* tensor0;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &tensor0));
    TfLiteTensor* tensor1;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &tensor1));

    TfLiteIntArray* newSize = TfLiteIntArrayCopy(tensor0->dims);
    TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, tensor1, newSize));

    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(2);
    for (int i = 0; i < 2; ++i) {
      node->temporaries->data[i] = *(first_new_tensor) + i;
    }

    auto setup_temporary = [&](int id) {
      TfLiteTensor* tmp = &context->tensors[id];
      tmp->type = kTfLiteFloat32;
      tmp->allocation_type = kTfLiteArenaRw;
      return context->ResizeTensor(context, tmp,
                                   TfLiteIntArrayCopy(tensor0->dims));
    };
    TF_LITE_ENSURE_STATUS(setup_temporary(node->temporaries->data[0]));
    TF_LITE_ENSURE_STATUS(setup_temporary(node->temporaries->data[1]));

    return kTfLiteOk;
  };
  reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* a0;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &a0));

    auto populate = [&](int id) {
      TfLiteTensor* t = &context->tensors[id];
      int num = a0->dims->data[0];
      for (int i = 0; i < num; i++) {
        t->data.f[i] = a0->data.f[i];
      }
    };

    populate(node->outputs->data[0]);
    populate(node->temporaries->data[0]);
    populate(node->temporaries->data[1]);
    return kTfLiteOk;
  };

  return reg;
}

TEST(BasicInterpreter, OneOpInterpreter) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({1}), kTfLiteOk);

  TfLiteQuantizationParams quantized;
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(0, kTfLiteFloat32, "in1",
                                                     {3}, quantized),
            kTfLiteOk);
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(1, kTfLiteFloat32, "out0",
                                                     {3}, quantized),
            kTfLiteOk);

  ASSERT_EQ(interpreter.GetInputName(0), "in1");
  ASSERT_EQ(interpreter.GetOutputName(0), "out0");

  TfLiteRegistration reg = GetPassthroughOpRegistration();

  ASSERT_EQ(
      interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr, &reg),
      kTfLiteOk);
  ASSERT_EQ(interpreter.ResizeInputTensor(0, {3}), kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
}

TEST(BasicInterpreter, ReleaseNonPersistentMemory) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({1}), kTfLiteOk);

  TfLiteQuantizationParams quantized;
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(0, kTfLiteFloat32, "in1",
                                                     {3}, quantized),
            kTfLiteOk);
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(1, kTfLiteFloat32, "out0",
                                                     {3}, quantized),
            kTfLiteOk);

  TfLiteRegistration reg = GetPassthroughOpRegistration();

  ASSERT_EQ(
      interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr, &reg),
      kTfLiteOk);
  ASSERT_EQ(interpreter.ResizeInputTensor(0, {3}), kTfLiteOk);

  // AllocateTensors() hasn't been called yet, so this should be a no-op.
  ASSERT_EQ(interpreter.ReleaseNonPersistentMemory(), kTfLiteOk);

  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);

  ASSERT_EQ(interpreter.ReleaseNonPersistentMemory(), kTfLiteOk);
  // Invoke() now fails because non-persistent arenas have been released.
  ASSERT_NE(interpreter.Invoke(), kTfLiteOk);

  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);

  // ResizeInputTensors just after ReleaseNonPersistentMemory should also need
  // AllocateTensors, without causing any unexpected crashes.
  ASSERT_EQ(interpreter.ReleaseNonPersistentMemory(), kTfLiteOk);
  ASSERT_EQ(interpreter.ResizeInputTensor(0, {4}), kTfLiteOk);
  ASSERT_NE(interpreter.Invoke(), kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
}

// Forcefully divides tensor allocation in three steps: one before invocation
// and two more at invocation time. This happens because we use string tensors
// and their sizes can't be determined until invocation time.
TEST(BasicInterpreter, ThreeStepAllocate) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(5), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({4}), kTfLiteOk);

  TfLiteQuantizationParams quantized;

  // String tensor with one string of length 3
  union {
    char raw_bytes[15];
    struct {
      int32_t num_strs;
      int32_t offsets[2];
      char str_data[3];
    } tensor_data;
  } data;
  data.tensor_data = {1, {12, 15}, {'A', 'B', 'C'}};

  // Read only string tensor.
  ASSERT_EQ(interpreter.SetTensorParametersReadOnly(0, kTfLiteString, "", {1},
                                                    quantized, data.raw_bytes,
                                                    sizeof(data.raw_bytes)),
            kTfLiteOk);
  // Read-write string tensor.
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(1, kTfLiteString, "", {1},
                                                     quantized),
            kTfLiteOk);
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(2, kTfLiteInt32, "", {1},
                                                     quantized),
            kTfLiteOk);
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(3, kTfLiteString, "", {1},
                                                     quantized),
            kTfLiteOk);
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(4, kTfLiteInt32, "", {1},
                                                     quantized),
            kTfLiteOk);

  // String-in String-out node.
  TfLiteRegistration reg_copy = {nullptr, nullptr, nullptr, nullptr};
  reg_copy.invoke = [](TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
    DynamicBuffer buf;
    StringRef str_ref = GetString(input, 0);
    buf.AddString(str_ref);
    buf.WriteToTensorAsVector(output);
    return kTfLiteOk;
  };

  // String-in Int-out node.
  TfLiteRegistration reg_len = {nullptr, nullptr, nullptr, nullptr};
  reg_len.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
    TfLiteIntArray* outputSize = TfLiteIntArrayCreate(1);
    outputSize->data[0] = 1;
    return context->ResizeTensor(context, output, outputSize);
  };
  reg_len.invoke = [](TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* a0;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &a0));
    TfLiteTensor* a1;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &a1));
    a1->data.i32[0] = a0->bytes;
    return kTfLiteOk;
  };

  ASSERT_EQ(interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr,
                                              &reg_copy),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AddNodeWithParameters({1}, {2}, nullptr, 0, nullptr,
                                              &reg_len),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AddNodeWithParameters({0}, {3}, nullptr, 0, nullptr,
                                              &reg_copy),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AddNodeWithParameters({3}, {4}, nullptr, 0, nullptr,
                                              &reg_len),
            kTfLiteOk);

  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);

  ASSERT_EQ(interpreter.tensor(0)->bytes, 15);
  ASSERT_NE(interpreter.tensor(0)->data.raw, nullptr);
  ASSERT_EQ(interpreter.tensor(1)->bytes, 15);
  ASSERT_NE(interpreter.tensor(1)->data.raw, nullptr);
  ASSERT_EQ(interpreter.tensor(3)->bytes, 15);
  ASSERT_NE(interpreter.tensor(4)->data.raw, nullptr);
  ASSERT_EQ(interpreter.tensor(2)->bytes, 4);
  ASSERT_EQ(interpreter.tensor(2)->data.i32[0], 15);
  ASSERT_EQ(interpreter.tensor(4)->bytes, 4);
  ASSERT_EQ(interpreter.tensor(4)->data.i32[0], 15);
}

TEST(BasicInterpreter, AllocateTwice) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({1}), kTfLiteOk);

  TfLiteQuantizationParams quantized;
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {3},
                                                     quantized),
            kTfLiteOk);
  ASSERT_EQ(interpreter.SetTensorParametersReadWrite(1, kTfLiteFloat32, "", {3},
                                                     quantized),
            kTfLiteOk);

  TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};
  reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* tensor0;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &tensor0));
    TfLiteTensor* tensor1;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &tensor1));
    TfLiteIntArray* newSize = TfLiteIntArrayCopy(tensor0->dims);
    return context->ResizeTensor(context, tensor1, newSize);
  };
  reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
    const TfLiteTensor* a0;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &a0));
    TfLiteTensor* a1;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &a1));
    int num = a0->dims->data[0];
    for (int i = 0; i < num; i++) {
      a1->data.f[i] = a0->data.f[i];
    }
    return kTfLiteOk;
  };
  ASSERT_EQ(
      interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr, &reg),
      kTfLiteOk);
  ASSERT_EQ(interpreter.ResizeInputTensor(0, {3}), kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
  char* old_tensor0_ptr = interpreter.tensor(0)->data.raw;
  char* old_tensor1_ptr = interpreter.tensor(1)->data.raw;

  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
  ASSERT_EQ(old_tensor0_ptr, interpreter.tensor(0)->data.raw);
  ASSERT_EQ(old_tensor1_ptr, interpreter.tensor(1)->data.raw);
}

TEST(BasicInterpreter, TestNullErrorReporter) {
  TestErrorReporter reporter;
  Interpreter interpreter;
}

TEST(BasicInterpreter, TestCustomErrorReporter) {
  TestErrorReporter reporter;
  Interpreter interpreter(&reporter);
  ASSERT_NE(interpreter.Invoke(), kTfLiteOk);
  ASSERT_EQ(reporter.error_messages(),
            "Invoke called on model that is not ready.");
  ASSERT_EQ(reporter.num_calls(), 1);
}

TEST(BasicInterpreter, TestOverflow) {
  TestErrorReporter reporter;
  Interpreter interpreter(&reporter);
  TfLiteQuantizationParams quantized;

  ASSERT_EQ(interpreter.AddTensors(1), kTfLiteOk);
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({0}), kTfLiteOk);
  // Overflow testing is pointer word size dependent.
  if (sizeof(size_t) == 8) {
    // #bits for bytecount = 30 + 30 + 2 = 62 < 64
    ASSERT_EQ(interpreter.SetTensorParametersReadWrite(
                  0, kTfLiteFloat32, "in1", {1 << 30, 1 << 30}, quantized),
              kTfLiteOk);
    // #bits for element count = 30 + 30 + 2 = 62 < 64 (no overflow)
    // #bits for byte count = 30 + 30 + 2 + 2 = 64 == 64 (overflow)
    ASSERT_NE(
        interpreter.SetTensorParametersReadWrite(
            0, kTfLiteFloat32, "in1", {1 << 30, 1 << 30, 1 << 2}, quantized),
        kTfLiteOk);
    EXPECT_THAT(
        reporter.error_messages(),
        testing::EndsWith("BytesRequired number of bytes overflowed.\n"));
    // #bits for element count = 30 + 30 + 2 + 4 = 66 > 64 (overflow).
    // #bits for byte count = 30 + 30 + 2 + 4 + 2 = 68 > 64 (overflow).
    reporter.Reset();
    ASSERT_NE(interpreter.SetTensorParametersReadWrite(
                  0, kTfLiteFloat32, "in1", {1 << 30, 1 << 30, 1 << 2, 1 << 4},
                  quantized),
              kTfLiteOk);
    EXPECT_THAT(
        reporter.error_messages(),
        testing::EndsWith("BytesRequired number of elements overflowed.\n"));

  } else if (sizeof(size_t) == 4) {
    // #bits for bytecount = 14 + 14 + 2 = 30 < 32
    ASSERT_EQ(interpreter.SetTensorParametersReadWrite(
                  0, kTfLiteFloat32, "in1", {1 << 14, 1 << 14}, quantized),
              kTfLiteOk);
    // #bits for element count = 14 + 14 + 3 = 31 < 32 (no overflow).
    // #bits for byte count = 14 + 14 + 3 + 2 = 33 > 32 (overflow).
    ASSERT_NE(
        interpreter.SetTensorParametersReadWrite(
            0, kTfLiteFloat32, "in1", {1 << 14, 1 << 14, 1 << 3}, quantized),
        kTfLiteOk);
    EXPECT_THAT(
        reporter.error_messages(),
        testing::EndsWith("BytesRequired number of bytes overflowed.\n"));
    // #bits for element count = 14 + 14 + 4 = 32 == 32 (overflow).
    // byte count also overflows, but we don't get to that check.
    reporter.Reset();
    ASSERT_NE(
        interpreter.SetTensorParametersReadWrite(
            0, kTfLiteFloat32, "in1", {1 << 14, 1 << 14, 1 << 4}, quantized),
        kTfLiteOk);
    EXPECT_THAT(
        reporter.error_messages(),
        testing::EndsWith("BytesRequired number of elements overflowed.\n"));
  } else {
    // This test failing means that we are using a non 32/64 bit architecture.
    ASSERT_TRUE(false);
  }
}

TEST(BasicInterpreter, TestUnsupportedDelegateFunctions) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
  TfLiteRegistration registration = {nullptr, nullptr, nullptr, nullptr};
  // These functions are only supported inside Delegate's Prepare function.
  // The test verifies that these functions returns `kTfLiteError`, but not
  // `kTfLiteOk` or just crashes.
  registration.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    {
      TfLiteIntArray* execution_plan;
      EXPECT_EQ(context->GetExecutionPlan(context, &execution_plan),
                kTfLiteError);
    }
    {
      TfLiteNode* node;
      TfLiteRegistration* registration;
      EXPECT_EQ(
          context->GetNodeAndRegistration(context, 0, &node, &registration),
          kTfLiteError);
    }
    {
      TfLiteRegistration delegate_registration = {nullptr, nullptr, nullptr,
                                                  nullptr};
      TfLiteIntArray nodes_to_replace;
      nodes_to_replace.size = 0;
      EXPECT_EQ(context->ReplaceNodeSubsetsWithDelegateKernels(
                    context, delegate_registration, &nodes_to_replace, nullptr),
                kTfLiteError);
    }
    return kTfLiteError;
  };
  ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.SetOutputs({0}), kTfLiteOk);
  ASSERT_EQ(interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr,
                                              &registration),
            kTfLiteOk);
  EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteError);
}

TEST(BasicInterpreter, DynamicTensorsResizeDescendants) {
  // Assemble a graph with a node that has dynamically sized output (via the
  // pad op), followed by a node with a standard element-wise op (negate).
  Interpreter interpreter;
  interpreter.AddTensors(4);
  interpreter.SetInputs({0, 1});
  interpreter.SetOutputs({3});
  TfLiteQuantizationParams quant;
  interpreter.SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {2, 2, 1, 1},
                                           quant);
  interpreter.SetTensorParametersReadWrite(1, kTfLiteInt32, "", {4, 2}, quant);
  interpreter.SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {}, quant);
  interpreter.SetTensorParametersReadWrite(3, kTfLiteFloat32, "", {}, quant);

  TfLiteRegistration* pad_op = tflite::ops::builtin::Register_PADV2();
  TfLiteRegistration* neg_op = tflite::ops::builtin::Register_NEG();
  interpreter.AddNodeWithParameters({0, 1}, {2}, nullptr, 0, nullptr, pad_op);
  interpreter.AddNodeWithParameters({2}, {3}, nullptr, 0, nullptr, neg_op);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  // Configure [[2,2],[4,4]] padding and execute the graph.
  interpreter.typed_tensor<int>(1)[0] = 2;
  interpreter.typed_tensor<int>(1)[1] = 2;
  interpreter.typed_tensor<int>(1)[2] = 2;
  interpreter.typed_tensor<int>(1)[3] = 2;
  interpreter.typed_tensor<int>(1)[4] = 0;
  interpreter.typed_tensor<int>(1)[5] = 0;
  interpreter.typed_tensor<int>(1)[6] = 0;
  interpreter.typed_tensor<int>(1)[7] = 0;
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);

  // Both the output and intermediate tensor sizes should reflect the output
  // from the dynamic pad operation.
  ASSERT_EQ(interpreter.tensor(2)->bytes, sizeof(float) * 6 * 6);
  ASSERT_EQ(interpreter.tensor(3)->bytes, sizeof(float) * 6 * 6);

  // Now configure [[4,4],[6,6]] padding and execute the graph.
  interpreter.typed_tensor<int>(1)[0] = 4;
  interpreter.typed_tensor<int>(1)[1] = 4;
  interpreter.typed_tensor<int>(1)[2] = 6;
  interpreter.typed_tensor<int>(1)[3] = 6;
  interpreter.typed_tensor<int>(1)[4] = 0;
  interpreter.typed_tensor<int>(1)[5] = 0;
  interpreter.typed_tensor<int>(1)[6] = 0;
  interpreter.typed_tensor<int>(1)[7] = 0;
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);

  // Again, the output and intermediate tensor sizes should reflect the *new*
  // resize from the latest pad operation.
  ASSERT_EQ(interpreter.tensor(2)->bytes, sizeof(float) * 10 * 14);
  ASSERT_EQ(interpreter.tensor(3)->bytes, sizeof(float) * 10 * 14);
}

TEST(InterpreterTensorsCapacityTest, TestWithinHeadroom) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(Interpreter::kTensorsReservedCapacity),
            kTfLiteOk);
  TfLiteRegistration registration = {nullptr, nullptr, nullptr, nullptr};
  registration.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    TfLiteTensor* first_tensor = context->tensors;

    int new_tensor_index;
    context->AddTensors(context, Interpreter::kTensorsCapacityHeadroom,
                        &new_tensor_index);
    EXPECT_EQ(first_tensor, context->tensors);
    return kTfLiteOk;
  };
  ASSERT_EQ(interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr,
                                              &registration),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
}

TEST(InterpreterTensorsCapacityTest, TestExceedHeadroom) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(Interpreter::kTensorsReservedCapacity),
            kTfLiteOk);
  TfLiteRegistration registration = {nullptr, nullptr, nullptr, nullptr};
  registration.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    TfLiteTensor* first_tensor = context->tensors;

    int new_tensor_index;
    // Add enough tensors to trigger buffer re-allocation.
    context->AddTensors(
        context,
        (context->tensors_size + Interpreter::kTensorsCapacityHeadroom + 1) * 2,
        &new_tensor_index);
    EXPECT_NE(first_tensor, context->tensors);
    return kTfLiteOk;
  };
  ASSERT_EQ(interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr,
                                              &registration),
            kTfLiteOk);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
}

struct TestExternalContext : public TfLiteExternalContext {
  static constexpr TfLiteExternalContextType kType = kTfLiteGemmLowpContext;

  static TestExternalContext* Get(TfLiteContext* context) {
    return reinterpret_cast<TestExternalContext*>(
        context->GetExternalContext(context, kType));
  }

  static void Set(TfLiteContext* context, TestExternalContext* value) {
    context->SetExternalContext(context, kType, value);
  }

  int num_refreshes = 0;
};

TEST_F(InterpreterTest, GetSetResetExternalContexts) {
  auto* context = GetInterpreterContext();

  TestExternalContext external_context;
  external_context.Refresh = [](TfLiteContext* context) {
    auto* ptr = TestExternalContext::Get(context);
    if (ptr != nullptr) {
      ++ptr->num_refreshes;
    }
    return kTfLiteOk;
  };

  EXPECT_EQ(TestExternalContext::Get(context), nullptr);
  ASSERT_EQ(interpreter_.SetNumThreads(4), kTfLiteOk);

  TestExternalContext::Set(context, &external_context);
  EXPECT_EQ(TestExternalContext::Get(context), &external_context);
  ASSERT_EQ(interpreter_.SetNumThreads(4), kTfLiteOk);
  ASSERT_EQ(interpreter_.SetNumThreads(5), kTfLiteOk);
  EXPECT_EQ(external_context.num_refreshes, 2);

  // Reset refresh count to 0
  external_context.num_refreshes = 0;
  // Below should not call external context refresh
  ASSERT_EQ(interpreter_.SetNumThreads(-2), kTfLiteError);
  EXPECT_EQ(external_context.num_refreshes, 0);

  ASSERT_EQ(interpreter_.SetNumThreads(-1), kTfLiteOk);
  EXPECT_EQ(external_context.num_refreshes, 1);

  TestExternalContext::Set(context, nullptr);
  EXPECT_EQ(TestExternalContext::Get(context), nullptr);
  ASSERT_EQ(interpreter_.SetNumThreads(4), kTfLiteOk);
}

TEST_F(InterpreterTest, SetNumThreadsSucceedsWithZero) {
  ASSERT_EQ(interpreter_.SetNumThreads(0), kTfLiteOk);
  // num_threads == 0 has the same effect as num_threads == 1.
  EXPECT_EQ(interpreter_.subgraph(0)->context()->recommended_num_threads, 1);
}

struct TestCpuBackendContext : public TfLiteInternalBackendContext {
  // Count the number of calls to ClearCaches for the backend context.
  void ClearCaches() override { ++num_calls; }
  void SetMaxNumThreads(int num_threads) override {}
  int num_calls = 0;
};

TEST_F(InterpreterTest, ExternalBackendContextClearsCachesOnDelete) {
  ExternalCpuBackendContext external_cpu_context;
  TestCpuBackendContext* cpu_backend_context = new TestCpuBackendContext();
  external_cpu_context.set_internal_backend_context(
      std::unique_ptr<TfLiteInternalBackendContext>(cpu_backend_context));

  {
    // Create an interpreter with an external Cpu backend context and ensure
    // it goes out of scope.
    Interpreter interpreter;
    interpreter.SetExternalContext(kTfLiteCpuBackendContext,
                                   &external_cpu_context);
    EXPECT_EQ(cpu_backend_context->num_calls, 0);
  }
  EXPECT_EQ(cpu_backend_context->num_calls, 1);
}

// Test fixture that allows playing with execution plans. It creates a two
// node graph that can be executed in either [0,1] order or [1,0] order.
// The CopyOp records when it is invoked in the class member run_order_
// so we can test whether the execution plan was honored.
class TestExecutionPlan : public InterpreterTest {
  // Encapsulates the node ids and provides them to a C primitive data type
  // Allocatable with placement new, but never destructed, so make sure this
  // doesn't own any heap allocated data. This is then is used as op local
  // data to allow access to the test fixture data.
  class CallReporting {
   public:
    CallReporting(int node_id, std::vector<int>* run_order)
        : node_id_(node_id), run_order_(run_order) {}

    void Record() { run_order_->push_back(node_id_); }

   private:
    // The node id for this particular node
    int node_id_;
    // A pointer to the global run-order
    std::vector<int>* run_order_;
  };

  // Build a kernel registration for an op that copies its one input
  // to an output
  TfLiteRegistration CopyOpRegistration() {
    TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      // Set output size to input size
      const TfLiteTensor* tensor0;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &tensor0));
      TfLiteTensor* tensor1;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &tensor1));
      TfLiteIntArray* newSize = TfLiteIntArrayCopy(tensor0->dims);
      return context->ResizeTensor(context, tensor1, newSize);
    };

    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
      CallReporting* call_reporting =
          static_cast<CallReporting*>(node->builtin_data);
      // Copy input data to output data.
      const TfLiteTensor* a0;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &a0));
      TfLiteTensor* a1;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &a1));
      int num = a0->dims->data[0];
      for (int i = 0; i < num; i++) {
        a1->data.f[i] = a0->data.f[i];
      }
      call_reporting->Record();
      return kTfLiteOk;
    };
    return reg;
  }

  // Adds a copy node going from tensor `input` to output tensor `output`.
  // Note, input is used as the node_id. Inject run_order as op accessible
  // data. Note: this is a little strange of a way to do this, but it is
  // using op functionality to avoid static global variables.
  void MakeCopyNode(int input, int output) {
    // Ownership of call_reporting is taken by interpreter (malloc is used due
    // to nodes being a C99 interface so free() is used).
    TfLiteRegistration copy_op = CopyOpRegistration();
    CallReporting* call_reporting_1 =
        static_cast<CallReporting*>(malloc(sizeof(CallReporting)));
    new (call_reporting_1) CallReporting(input, &run_order_);
    ASSERT_EQ(interpreter_.AddNodeWithParameters(
                  {0}, {2}, nullptr, 0, static_cast<void*>(call_reporting_1),
                  &copy_op),
              kTfLiteOk);
    ASSERT_EQ(interpreter_.ResizeInputTensor(input, {3}), kTfLiteOk);
  }

  void SetUp() final {
    // Add two inputs and two outputs that don't depend on each other
    ASSERT_EQ(interpreter_.AddTensors(4), kTfLiteOk);
    interpreter_.SetInputs({0, 1});
    interpreter_.SetOutputs({2, 3});
    TfLiteQuantizationParams quantized;
    for (int tensor_index = 0; tensor_index < 4; tensor_index++) {
      ASSERT_EQ(interpreter_.SetTensorParametersReadWrite(
                    tensor_index, kTfLiteFloat32, "", {3}, quantized),
                kTfLiteOk);
    }

    // Define two copy functions that also use the user_data to report that
    // they were called.
    // i.e. tensor[2] = copy(tensor[0]); tensor[3] = copy(tensor[1]);
    // thus we can reorder the two nodes arbitrary and still satisfy dependency
    // order.
    MakeCopyNode(0, 2);
    MakeCopyNode(1, 3);

    ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);
  }

 protected:
  // list of node_ids that were run
  std::vector<int> run_order_;
};

TEST_F(TestExecutionPlan, DefaultExecutionPlan) {
  // Check default order
  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);
  ASSERT_EQ(run_order_, std::vector<int>({0, 1}));
}

TEST_F(TestExecutionPlan, ReversedExecutionPlan) {
  // Check reversed order
  SetExecutionPlan({1, 0});
  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);
  ASSERT_EQ(run_order_, std::vector<int>({1, 0}));
}

TEST_F(TestExecutionPlan, SubsetExecutionPlan) {
  // Check running only node index 1
  SetExecutionPlan({1});
  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);
  ASSERT_EQ(run_order_, std::vector<int>({1}));
}

TEST_F(TestExecutionPlan, NullExecutionPlan) {
  // Check nothing executed.
  SetExecutionPlan({});
  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);
  ASSERT_EQ(run_order_, std::vector<int>());
}

TEST(TestDelegateOwnership, ProperlyDisposed) {
  struct TfLiteInterpreterOwnedDelegate : public TfLiteDelegate {
    TfLiteInterpreterOwnedDelegate(bool* destroyed, bool* prepared)
        : destroyed(destroyed), prepared(prepared) {
      flags = kTfLiteDelegateFlagsNone;
      Prepare = [](TfLiteContext*, TfLiteDelegate* delegate) -> TfLiteStatus {
        *static_cast<TfLiteInterpreterOwnedDelegate*>(delegate)->prepared =
            true;
        return kTfLiteOk;
      };
    }
    ~TfLiteInterpreterOwnedDelegate() { *destroyed = true; }

    bool* destroyed;
    bool* prepared;
  };

  // Construct a delegate with flags for indicating preparation/destruction.
  bool destroyed = false;
  bool prepared = false;
  std::unique_ptr<TfLiteInterpreterOwnedDelegate> delegate(
      new TfLiteInterpreterOwnedDelegate(&destroyed, &prepared));
  {
    // Create an interpreter and assemble a simple graph.
    Interpreter interpreter;
    TfLiteRegistration registration = {nullptr, nullptr, nullptr, nullptr};
    ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
    ASSERT_EQ(interpreter.SetInputs({0}), kTfLiteOk);
    ASSERT_EQ(interpreter.SetOutputs({1}), kTfLiteOk);
    ASSERT_EQ(interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr,
                                                &registration),
              kTfLiteOk);

    // Pass delegate ownership to that interpreter.
    ASSERT_EQ(InterpreterTest::ModifyGraphWithDelegate(&interpreter,
                                                       std::move(delegate)),
              kTfLiteOk);

    // The delegate should be prepared as normal, and should be preserved.
    EXPECT_TRUE(prepared);
    EXPECT_FALSE(destroyed);

    // Interpreter interaction should not impact the delegate's validity.
    interpreter.AllocateTensors();
    interpreter.Invoke();
    EXPECT_FALSE(destroyed);
  }

  // Only after the interpreter is destroyed should the delegate be destroyed.
  EXPECT_TRUE(destroyed);
}

// CancellationData contains the data required to cancel a call to Invoke().
struct CancellationData {
  bool is_cancelled = false;
};

// Indicates whether Invoke() has been cancelled based on the value of the
// CancellationData object passed in.
bool CheckCancellation(void* data) {
  CancellationData* cancellation_data =
      static_cast<struct CancellationData*>(data);
  return cancellation_data->is_cancelled;
}

static struct CancellationData cancellation_data_;

// Test fixture to test cancellation within the Interpreter.
class CancellationTest : public ::testing::Test {
 public:
  TfLiteStatus Invoke() { return interpreter_.Invoke(); }
  void Cancel() { cancellation_data_.is_cancelled = true; }

  // Adds an CancelOp with input tensor `input` and output tensor `output`.
  void MakeCancelNode(int input, int output) {
    TfLiteRegistration op = CancelOpRegistration();
    ASSERT_EQ(interpreter_.AddNodeWithParameters({input}, {output}, nullptr, 0,
                                                 nullptr, &op),
              kTfLiteOk);
    ASSERT_EQ(interpreter_.ResizeInputTensor(input, {3}), kTfLiteOk);
  }

  // Adds an OkOp with input tensor `input` and output tensor `output`.
  void MakeOkNode(int input, int output) {
    TfLiteRegistration op = OkOpRegistration();
    ASSERT_EQ(interpreter_.AddNodeWithParameters({input}, {output}, nullptr, 0,
                                                 nullptr, &op),
              kTfLiteOk);
    ASSERT_EQ(interpreter_.ResizeInputTensor(input, {3}), kTfLiteOk);
  }

  Interpreter interpreter_;

 private:
  // Build the kernel registration for an op that cancels the operation.
  TfLiteRegistration CancelOpRegistration() {
    TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

    // Set output size to the input size in CancelOp::Prepare(). Code exists to
    // have a framework in Prepare. The input and output tensors are not used.
    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      const TfLiteTensor* in_tensor;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &in_tensor));
      TfLiteTensor* out_tensor;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &out_tensor));
      TfLiteIntArray* new_size = TfLiteIntArrayCopy(in_tensor->dims);
      return context->ResizeTensor(context, out_tensor, new_size);
    };

    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
      cancellation_data_.is_cancelled = true;
      return kTfLiteOk;
    };
    return reg;
  }

  // Build the kernel registration for an op that returns kTfLiteOk.
  TfLiteRegistration OkOpRegistration() {
    TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

    // Set output size to the input size in OkOp::Prepare(). Code exists to have
    // a framework in Prepare. The input and output tensors are not used.
    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      const TfLiteTensor* in_tensor;
      TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &in_tensor));
      TfLiteTensor* out_tensor;
      TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &out_tensor));
      TfLiteIntArray* new_size = TfLiteIntArrayCopy(in_tensor->dims);
      return context->ResizeTensor(context, out_tensor, new_size);
    };

    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
      return kTfLiteOk;
    };
    return reg;
  }

  void SetUp() final {
    cancellation_data_.is_cancelled = false;

    // Set up the interpreter. Create the input and output tensors.
    int num_tensors = 3;
    ASSERT_EQ(interpreter_.AddTensors(num_tensors), kTfLiteOk);
    interpreter_.SetInputs({0});
    interpreter_.SetOutputs({2});
    TfLiteQuantizationParams quantized;
    for (int tensor_index = 0; tensor_index < num_tensors; tensor_index++) {
      ASSERT_EQ(interpreter_.SetTensorParametersReadWrite(
                    tensor_index, kTfLiteFloat32, "", {3}, quantized),
                kTfLiteOk);
    }
    interpreter_.SetCancellationFunction(&cancellation_data_,
                                         &CheckCancellation);
  }
};

TEST_F(CancellationTest, CancelBeforeInvoke) {
  // Cancel prior to calling Invoke.
  CancellationTest::MakeOkNode(1, 2);
  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  CancellationTest::Cancel();
  TfLiteStatus invoke_error_code = CancellationTest::Invoke();
  ASSERT_EQ(invoke_error_code, kTfLiteError);
}

TEST_F(CancellationTest, CancelDuringInvoke) {
  // Tests a model which sets the cancel in order to test cancellation works
  // between ops.
  //
  // The first op will set the cancellation bit to true. The second op returns
  // `kTfLiteOk` if executed.
  CancellationTest::MakeCancelNode(0, 1);
  CancellationTest::MakeOkNode(1, 2);
  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  TfLiteStatus invoke_error_code = CancellationTest::Invoke();
  ASSERT_EQ(invoke_error_code, kTfLiteError);
}

// Tests functionality related to custom memory allocations in TFLite.
class TestCustomAllocation : public ::testing::Test {
 protected:
  void SetUp() override {
    // Simple model with two custom ops that add 2 float tensors each.
    interpreter_.reset(new Interpreter);
    interpreter_->AddTensors(7);
    interpreter_->SetInputs({0, 1});
    interpreter_->SetOutputs({3, 4, 6});
    TfLiteQuantizationParams quant;
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(3, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(4, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(5, kTfLiteFloat32, "", {3},
                                               quant, /*is_variable=*/true);
    interpreter_->SetTensorParametersReadWrite(6, kTfLiteFloat32, "", {3},
                                               quant);
    auto* add_reg = ops::builtin::Register_ADD();
    TfLiteAddParams* builtin_data0 =
        reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
    TfLiteAddParams* builtin_data1 =
        reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
    TfLiteAddParams* builtin_data2 =
        reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
    TfLiteAddParams* builtin_data3 =
        reinterpret_cast<TfLiteAddParams*>(malloc(sizeof(TfLiteAddParams)));
    builtin_data0->activation = kTfLiteActNone;
    builtin_data1->activation = kTfLiteActNone;
    builtin_data2->activation = kTfLiteActNone;
    builtin_data3->activation = kTfLiteActNone;
    interpreter_->AddNodeWithParameters({0, 0}, {2}, nullptr, 0, builtin_data0,
                                        add_reg);
    interpreter_->AddNodeWithParameters({1, 1}, {3}, nullptr, 0, builtin_data1,
                                        add_reg);
    interpreter_->AddNodeWithParameters({2, 1}, {4}, nullptr, 0, builtin_data2,
                                        add_reg);
    interpreter_->AddNodeWithParameters({0, 5}, {6}, nullptr, 0, builtin_data3,
                                        add_reg);
    interpreter_->SetVariables({5});
  }

  void AssignCustomAllocForTensor(int tensor_idx, int required_alignment) {
    const TfLiteTensor* tensor = interpreter_->tensor(tensor_idx);
    auto tensor_alloc = NewCustomAlloc(tensor->bytes, required_alignment);
    ASSERT_EQ(
        interpreter_->SetCustomAllocationForTensor(tensor_idx, tensor_alloc),
        kTfLiteOk);
  }

  void VerifyInvoke() {
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    std::vector<float> variable = {0.0f, 1.0f, 2.0f};
    std::vector<float> expected_output = {2.0f, 4.0f, 6.0f};

    // typed_tensor<...> should work irrespective of custom alloc, since it
    // accesses output_tensor.data.
    memcpy(interpreter_->typed_tensor<float>(interpreter_->variables()[0]),
           variable.data(), 3 * sizeof(float));
    memcpy(interpreter_->typed_tensor<float>(0), input.data(),
           3 * sizeof(float));
    memcpy(interpreter_->typed_tensor<float>(1), input.data(),
           3 * sizeof(float));
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
    TfLiteTensor* output_tensor =
        interpreter_->tensor(interpreter_->outputs()[0]);
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(output_tensor->data.f[i], expected_output[i]) << i;
    }
  }

  // Actual initialized allocation is more than num_bytes, to account for
  // required_allocation.
  TfLiteCustomAllocation NewCustomAlloc(size_t num_bytes,
                                        int required_alignment) {
    // Extra memory to ensure alignment.
    char* new_alloc = new char[num_bytes + required_alignment];
    char* new_underlying_buffer_aligned_ptr = reinterpret_cast<char*>(
        AlignTo(required_alignment, reinterpret_cast<intptr_t>(new_alloc)));
    custom_alloc_buffers_.emplace_back(new_alloc);

    return TfLiteCustomAllocation(
        {new_underlying_buffer_aligned_ptr, num_bytes});
  }

  intptr_t AlignTo(size_t alignment, intptr_t offset) {
    return offset % alignment == 0 ? offset
                                   : offset + (alignment - offset % alignment);
  }

  void TearDown() override {
    interpreter_.reset();
    custom_alloc_buffers_.clear();
  }

 protected:
  TfLiteAddParams add_params_;
  std::unique_ptr<Interpreter> interpreter_;
  std::vector<std::unique_ptr<char[]>> custom_alloc_buffers_;
};

TEST_F(TestCustomAllocation, InvalidAlignment) {
  const TfLiteTensor* input_tensor =
      interpreter_->tensor(interpreter_->inputs()[0]);
  intptr_t dummy_ptr = kDefaultTensorAlignment - 1;
  TfLiteCustomAllocation input_alloc{reinterpret_cast<void*>(dummy_ptr),
                                     input_tensor->bytes};
  ASSERT_EQ(interpreter_->SetCustomAllocationForTensor(
                interpreter_->inputs()[0], input_alloc),
            kTfLiteError);

  // Allocate tensors & Invoke should still work.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  VerifyInvoke();
}

TEST_F(TestCustomAllocation, InvalidAlignment_SkipCheck) {
  const TfLiteTensor* input_tensor =
      interpreter_->tensor(interpreter_->inputs()[0]);
  const int required_alignment = kDefaultTensorAlignment - 1;
  auto tensor_alloc = NewCustomAlloc(input_tensor->bytes, required_alignment);
  ASSERT_EQ(interpreter_->SetCustomAllocationForTensor(
                interpreter_->inputs()[0], tensor_alloc,
                /**flags**/ kTfLiteCustomAllocationFlagsSkipAlignCheck),
            kTfLiteOk);

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
}

TEST_F(TestCustomAllocation, InsufficientBytes) {
  auto input_alloc = NewCustomAlloc(4, kDefaultTensorAlignment);

  // Setting the custom alloc works, but AllocateTensors doesn't.
  ASSERT_EQ(interpreter_->SetCustomAllocationForTensor(
                interpreter_->inputs()[0], input_alloc),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteError);
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteError);
}

TEST_F(TestCustomAllocation, CustomInputAlloc) {
  // Set custom allocation for one input tensor.
  AssignCustomAllocForTensor(interpreter_->inputs()[0],
                             /*required_alignment=*/kDefaultTensorAlignment);

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  VerifyInvoke();
}

TEST_F(TestCustomAllocation, CustomInputAlloc_MultipleAssigns) {
  // Set custom allocation for one input tensor.
  AssignCustomAllocForTensor(interpreter_->inputs()[0],
                             /*required_alignment=*/kDefaultTensorAlignment);

  AssignCustomAllocForTensor(interpreter_->inputs()[0],
                             /*required_alignment=*/kDefaultTensorAlignment);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  VerifyInvoke();

  AssignCustomAllocForTensor(interpreter_->inputs()[0],
                             /*required_alignment=*/kDefaultTensorAlignment);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  VerifyInvoke();
}

TEST_F(TestCustomAllocation, CustomInputAlloc_AllocateTensorsBefore) {
  // Allocate tensors.
  // Allocating now will cause TFLite to reserve some extra memory, but nothing
  // should break.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  AssignCustomAllocForTensor(interpreter_->inputs()[0],
                             /*required_alignment=*/kDefaultTensorAlignment);

  VerifyInvoke();
}

TEST_F(TestCustomAllocation, CustomInputAndOutputAllocs) {
  // Set custom allocations for all IO tensors.
  AssignCustomAllocForTensor(interpreter_->inputs()[0],
                             /*required_alignment=*/kDefaultTensorAlignment);
  AssignCustomAllocForTensor(interpreter_->inputs()[1],
                             /*required_alignment=*/kDefaultTensorAlignment);
  AssignCustomAllocForTensor(interpreter_->outputs()[0],
                             /*required_alignment=*/kDefaultTensorAlignment);
  AssignCustomAllocForTensor(interpreter_->outputs()[1],
                             /*required_alignment=*/kDefaultTensorAlignment);

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  VerifyInvoke();
}

// Ensure that custom allocs work for tensors on persistent arena as well.
TEST_F(TestCustomAllocation, CustomAlloc_VariableTensor) {
  // Set custom allocation for one input tensor.
  AssignCustomAllocForTensor(interpreter_->variables()[0],
                             /*required_alignment=*/kDefaultTensorAlignment);

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  VerifyInvoke();

  AssignCustomAllocForTensor(interpreter_->variables()[0],
                             /*required_alignment=*/kDefaultTensorAlignment);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  std::vector<float> input = {2.0f, 3.0f, 4.0f};
  std::vector<float> variable = {1.0f, 2.0f, 3.0f};
  std::vector<float> expected_output = {3.0f, 5.0f, 7.0f};
  memcpy(interpreter_->typed_tensor<float>(interpreter_->variables()[0]),
         variable.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 3 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 3 * sizeof(float));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  // expected_output = input + variable
  TfLiteTensor* output_tensor =
      interpreter_->tensor(interpreter_->outputs()[2]);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(output_tensor->data.f[i], expected_output[i]) << i;
  }
}

TEST_F(TestCustomAllocation, ResizeInputsWithoutEnoughMemory) {
  // Set custom allocations for all input tensors.
  AssignCustomAllocForTensor(interpreter_->inputs()[0],
                             /*required_alignment=*/kDefaultTensorAlignment);
  AssignCustomAllocForTensor(interpreter_->inputs()[1],
                             /*required_alignment=*/kDefaultTensorAlignment);

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  // Now resize tensors to double the size.
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {2, 3}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {2, 3}),
            kTfLiteOk);

  // Since the custom memory previously allocated isn't enough,
  // AllocateTensors() will fail.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteError);
  // Interpreter should no longer be in invokable state, so expect failure.
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteError);
}

TEST_F(TestCustomAllocation, ResizeInputsWithEnoughMemory) {
  // Set custom allocations for all input tensors, with double the required
  // memory.
  const TfLiteTensor* input0_tensor =
      interpreter_->tensor(interpreter_->inputs()[0]);
  auto input0_alloc =
      NewCustomAlloc(2 * input0_tensor->bytes, kDefaultTensorAlignment);
  ASSERT_EQ(interpreter_->SetCustomAllocationForTensor(
                interpreter_->inputs()[0], input0_alloc),
            kTfLiteOk);
  const TfLiteTensor* input1_tensor =
      interpreter_->tensor(interpreter_->inputs()[1]);
  auto input1_alloc =
      NewCustomAlloc(2 * input1_tensor->bytes, kDefaultTensorAlignment);
  ASSERT_EQ(interpreter_->SetCustomAllocationForTensor(
                interpreter_->inputs()[1], input1_alloc),
            kTfLiteOk);

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  // Now resize tensors to double the size.
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {6, 1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {6, 1}),
            kTfLiteOk);

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> expected_output = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
  TfLiteTensor* tensor = interpreter_->tensor(interpreter_->outputs()[0]);
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), 6 * sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), 6 * sizeof(float));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(tensor->data.f[i], expected_output[i]) << i;
  }

  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {3, 1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {3, 1}),
            kTfLiteOk);

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  VerifyInvoke();
}

// Verify typical use-cases where tensors are resized & custom allocs need to be
// set for every Invoke().
TEST_F(TestCustomAllocation, ResizeAndAllocateForEveryInvoke) {
  // First assign exactly sized allocs for all IO tensors.
  AssignCustomAllocForTensor(interpreter_->inputs()[0],
                             /*required_alignment=*/kDefaultTensorAlignment);
  AssignCustomAllocForTensor(interpreter_->inputs()[1],
                             /*required_alignment=*/kDefaultTensorAlignment);
  AssignCustomAllocForTensor(interpreter_->outputs()[0],
                             /*required_alignment=*/kDefaultTensorAlignment);
  AssignCustomAllocForTensor(interpreter_->outputs()[1],
                             /*required_alignment=*/kDefaultTensorAlignment);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  // Now resize inputs to a smaller: {3, 1} to {1, 1}.
  // Total alloc sized required now: 1 float == 4 bytes.
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[0], {1, 1}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(interpreter_->inputs()[1], {1, 1}),
            kTfLiteOk);
  // Assign allocs for all I/O tensors.
  // Even though the smaller output tensor sizes have not been propagated yet,
  // custom allocation works because verification of allocs happens after
  // preparing all ops & tensors.
  auto input0_alloc =
      NewCustomAlloc(/**num_bytes=**/ 4, kDefaultTensorAlignment);
  ASSERT_EQ(interpreter_->SetCustomAllocationForTensor(
                interpreter_->inputs()[0], input0_alloc),
            kTfLiteOk);
  auto input1_alloc =
      NewCustomAlloc(/**num_bytes=**/ 4, kDefaultTensorAlignment);
  ASSERT_EQ(interpreter_->SetCustomAllocationForTensor(
                interpreter_->inputs()[1], input1_alloc),
            kTfLiteOk);
  auto output0_alloc =
      NewCustomAlloc(/**num_bytes=**/ 4, kDefaultTensorAlignment);
  ASSERT_EQ(interpreter_->SetCustomAllocationForTensor(
                interpreter_->outputs()[0], output0_alloc),
            kTfLiteOk);
  auto output1_alloc =
      NewCustomAlloc(/**num_bytes=**/ 4, kDefaultTensorAlignment);
  ASSERT_EQ(interpreter_->SetCustomAllocationForTensor(
                interpreter_->outputs()[1], output1_alloc),
            kTfLiteOk);
  // AllocateTensors works.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  std::vector<float> input = {2.0f};
  std::vector<float> expected_output = {4.0f};
  TfLiteTensor* tensor = interpreter_->tensor(interpreter_->outputs()[0]);
  memcpy(interpreter_->typed_tensor<float>(0), input.data(), sizeof(float));
  memcpy(interpreter_->typed_tensor<float>(1), input.data(), sizeof(float));
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);
  EXPECT_EQ(tensor->data.f[0], expected_output[0]);
}

// Tests related to lazy delegate providers that are primarily used for applying
// TfLite delegates by default.
class TestLazyDelegateProvider : public InterpreterTest {
 protected:
  struct DummyLazyDelegateProvider : public TfLiteDelegate {
    explicit DummyLazyDelegateProvider(int64_t support_flags) {
      data_ = static_cast<void*>(this);
      flags = support_flags;
      Prepare = [](TfLiteContext*, TfLiteDelegate* delegate) -> TfLiteStatus {
        return kTfLiteOk;
      };
    }
  };

  void InitWithLazyDelegate(int64_t delegate_flags,
                            bool create_dyanmic_tensor = false,
                            bool return_error = false) {
    TfLiteRegistration reg = {nullptr};
    if (return_error) {
      reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
        return kTfLiteError;
      };
    }
    ASSERT_EQ(interpreter_.AddTensors(2), kTfLiteOk);
    interpreter_.SetInputs({0});
    interpreter_.SetOutputs({1});
    interpreter_.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr, &reg);

    Interpreter::TfLiteDelegatePtr delegate(
        new DummyLazyDelegateProvider(delegate_flags),
        [](TfLiteDelegate* delegate) {
          auto* dummy =
              static_cast<DummyLazyDelegateProvider*>(delegate->data_);
          delete dummy;
        });
    mutable_lazy_delegate_providers()->push_back(std::move(delegate));

    if (create_dyanmic_tensor) {
      // Mark the output as dynamic tensor.
      interpreter_.tensor(1)->data.raw = nullptr;
      interpreter_.tensor(1)->allocation_type = kTfLiteDynamic;
    }
  }
};

TEST_F(TestLazyDelegateProvider, ApplicationSuccess) {
  InitWithLazyDelegate(kTfLiteDelegateFlagsNone);
  EXPECT_EQ(kTfLiteOk, interpreter_.AllocateTensors());
  // We clear Interpreter::lazy_delegate_providers_ after they are tried out.
  EXPECT_TRUE(mutable_lazy_delegate_providers()->empty());
  EXPECT_TRUE(HasDelegates());
}

TEST_F(TestLazyDelegateProvider, ApplicationFailure) {
  InitWithLazyDelegate(kTfLiteDelegateFlagsNone,
                       false /* create_dyanmic_tensor */,
                       true /* return_error */);
  EXPECT_EQ(kTfLiteError, interpreter_.AllocateTensors());
  // We clear Interpreter::lazy_delegate_providers_ after they are tried out.
  EXPECT_TRUE(mutable_lazy_delegate_providers()->empty());
  EXPECT_FALSE(HasDelegates());
}

TEST_F(TestLazyDelegateProvider, ApplicationSkipped) {
  InitWithLazyDelegate(kTfLiteDelegateFlagsNone,
                       true /* create_dyanmic_tensor */);
  EXPECT_EQ(kTfLiteOk, interpreter_.AllocateTensors());
  EXPECT_TRUE(mutable_lazy_delegate_providers()->empty());
  // As the delegate doesn't allow dynamic tensor, the delegate won't be applied
  // and the interpreter doesn't have any delegate applied.
  EXPECT_FALSE(HasDelegates());
}

TEST_F(InterpreterTest, SingleSignature_get_signatures) {
  const char kMethodName[] = "test_method";
  const char kSignatureDefKey[] = "test_key";
  BuildSignature(kMethodName, kSignatureDefKey, {{"Input1", 0}, {"Input2", 1}},
                 {{"Output1", 5}});
  auto results = interpreter_.signature_def_names();
  ASSERT_EQ(1, results.size());
  EXPECT_EQ(kMethodName, *results[0]);
}

TEST_F(InterpreterTest, SingleSignature_get_inputs) {
  const char kMethodName[] = "test_method";
  const char kSignatureDefKey[] = "test_key";
  const std::map<std::string, uint32_t> inputs = {{"Input1", 0}, {"Input2", 1}};
  const std::map<std::string, uint32_t> outputs = {{"Output1", 5}};
  BuildSignature(kMethodName, kSignatureDefKey, inputs, outputs);
  EXPECT_THAT(interpreter_.signature_inputs(kMethodName), testing::Eq(inputs));
  EXPECT_THAT(interpreter_.signature_outputs(kMethodName),
              testing::Eq(outputs));
}

TEST_F(InterpreterTest, SingleSignature_validate_get_tensor) {
  const char kMethodName[] = "test_method";
  const char kSignatureDefKey[] = "test_key";
  const std::map<std::string, uint32_t> inputs = {{"Input1", 0}, {"Input2", 1}};
  const std::map<std::string, uint32_t> outputs = {{"Output1", 5}};

  BuildSignature(kMethodName, kSignatureDefKey, inputs, outputs);
  ASSERT_EQ(interpreter_.AddTensors(6), kTfLiteOk);
  ASSERT_EQ(interpreter_.SetInputs({0, 1}), kTfLiteOk);
  ASSERT_EQ(interpreter_.SetOutputs({5}), kTfLiteOk);
  ASSERT_EQ(interpreter_.SetTensorParametersReadWrite(
                0, kTfLiteFloat32, "", {3}, TfLiteQuantizationParams()),
            kTfLiteOk);
  ASSERT_EQ(interpreter_.SetTensorParametersReadWrite(
                1, kTfLiteFloat32, "", {3}, TfLiteQuantizationParams()),
            kTfLiteOk);
  ASSERT_EQ(interpreter_.ResizeInputTensor(interpreter_.inputs()[0], {1, 2, 3}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_.ResizeInputTensor(interpreter_.inputs()[1], {1, 2, 3}),
            kTfLiteOk);
  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

  EXPECT_TRUE(interpreter_.input_tensor_by_signature_name(
                  "Input1", kMethodName) != nullptr);
  EXPECT_TRUE(interpreter_.input_tensor_by_signature_name(
                  "Input2", kMethodName) != nullptr);
  EXPECT_TRUE(interpreter_.output_tensor_by_signature_name(
                  "Output1", kMethodName) != nullptr);

  // Invalid tensor
  EXPECT_EQ(interpreter_.input_tensor_by_signature_name("Input3", kMethodName),
            nullptr);
  EXPECT_EQ(interpreter_.output_tensor_by_signature_name("Input3", kMethodName),
            nullptr);
  // Invalid method
  EXPECT_EQ(
      interpreter_.input_tensor_by_signature_name("Input1", "InvalidMethod"),
      nullptr);
  EXPECT_EQ(
      interpreter_.output_tensor_by_signature_name("Output1", "InvalidMethod"),
      nullptr);
}

}  // namespace
}  // namespace tflite
