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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {

// InterpreterTest is a friend of Interpreter, so it can access context_.
class InterpreterTest : public ::testing::Test {
 public:
  template <typename Delegate>
  static TfLiteStatus ModifyGraphWithDelegate(
      Interpreter* interpreter, std::unique_ptr<Delegate> delegate) {
    Interpreter::TfLiteDelegatePtr tflite_delegate(
        delegate.release(), [](TfLiteDelegate* delegate) {
          delete reinterpret_cast<Delegate*>(delegate);
        });
    return interpreter->ModifyGraphWithDelegate(std::move(tflite_delegate));
  }

 protected:
  TfLiteContext* GetInterpreterContext() { return interpreter_.context_; }

  Interpreter interpreter_;
};

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
  EXPECT_THAT(testing::internal::GetCapturedStderr(),
              testing::HasSubstr("INFO: Initialized TensorFlow Lite runtime"));

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

TEST(BasicInterpreter, TestAllocateTensorsResetVariableTensors) {
  Interpreter interpreter;
  int tensor_index;
  ASSERT_EQ(interpreter.AddTensors(1, &tensor_index), kTfLiteOk);
  constexpr int kTensorSize = 16;
  TfLiteQuantizationParams quant;
  interpreter.SetTensorParametersReadWrite(tensor_index, kTfLiteFloat32, "",
                                           {kTensorSize}, quant, true);
  interpreter.SetVariables({tensor_index});
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  TfLiteTensor* tensor = interpreter.tensor(tensor_index);
  // Ensure that variable tensors are reset to zero.
  for (int i = 0; i < kTensorSize; ++i) {
    ASSERT_EQ(tensor->data.f[i], 0.0f);
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
      {kTfLiteFloat32, sizeof(float)}, {kTfLiteInt32, sizeof(int32_t)},
      {kTfLiteUInt8, sizeof(uint8_t)}, {kTfLiteInt64, sizeof(int64_t)},
      {kTfLiteInt16, sizeof(int16_t)}, {kTfLiteFloat16, sizeof(TfLiteFloat16)},
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
  auto* rw_affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  rw_affine_quantization->scale = TfLiteFloatArrayCreate(1);
  rw_affine_quantization->zero_point = TfLiteIntArrayCreate(1);
  rw_affine_quantization->scale->data[0] = scale;
  rw_affine_quantization->zero_point->data[0] = zero_point;
  rw_quantization.params = rw_affine_quantization;

  TfLiteQuantization ro_quantization;
  ro_quantization.type = kTfLiteAffineQuantization;
  auto* ro_affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
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
  } cases[] = {{kTfLiteFloat32}, {kTfLiteInt32}, {kTfLiteUInt8},
               {kTfLiteInt64},   {kTfLiteInt16}, {kTfLiteFloat16}};

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
  ASSERT_LT(interpreter.tensor(1)->data.raw, interpreter.tensor(2)->data.raw);
  ASSERT_LT(interpreter.tensor(2)->data.raw, interpreter.tensor(3)->data.raw);
  ASSERT_LT(interpreter.tensor(3)->data.raw, interpreter.tensor(4)->data.raw);
  ASSERT_LT(interpreter.tensor(4)->data.raw, interpreter.tensor(5)->data.raw);
  ASSERT_LT(interpreter.tensor(5)->data.raw, interpreter.tensor(7)->data.raw);
  ASSERT_EQ(interpreter.tensor(6)->data.raw, interpreter.tensor(2)->data.raw);
  // #7 is the one with the largest pointer.
  ASSERT_EQ(interpreter.tensor(8)->data.raw, nullptr);
  ASSERT_EQ(interpreter.tensor(9)->data.raw, interpreter.tensor(5)->data.raw);
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

  TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};
  reg.init = [](TfLiteContext* context, const char*, size_t) -> void* {
    auto* first_new_tensor = new int;
    context->AddTensors(context, 2, first_new_tensor);
    return first_new_tensor;
  };
  reg.free = [](TfLiteContext* context, void* buffer) {
    delete reinterpret_cast<int*>(buffer);
  };
  reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    auto* first_new_tensor = reinterpret_cast<int*>(node->user_data);

    TfLiteTensor* tensor0 = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* tensor1 = &context->tensors[node->outputs->data[0]];

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
    TfLiteTensor* a0 = &context->tensors[node->inputs->data[0]];

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
  ASSERT_EQ(
      interpreter.AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr, &reg),
      kTfLiteOk);
  ASSERT_EQ(interpreter.ResizeInputTensor(0, {3}), kTfLiteOk);
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
  char data[] = {1, 0, 0, 0, 12, 0, 0, 0, 15, 0, 0, 0, 'A', 'B', 'C'};
  // Read only string tensor.
  ASSERT_EQ(interpreter.SetTensorParametersReadOnly(0, kTfLiteString, "", {1},
                                                    quantized, data, 15),
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
    TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
    DynamicBuffer buf;
    StringRef str_ref = GetString(input, 0);
    buf.AddString(str_ref);
    buf.WriteToTensorAsVector(output);
    return kTfLiteOk;
  };

  // String-in Int-out node.
  TfLiteRegistration reg_len = {nullptr, nullptr, nullptr, nullptr};
  reg_len.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
    TfLiteIntArray* outputSize = TfLiteIntArrayCreate(1);
    outputSize->data[0] = 1;
    return context->ResizeTensor(context, output, outputSize);
  };
  reg_len.invoke = [](TfLiteContext* context, TfLiteNode* node) {
    TfLiteTensor* a0 = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* a1 = &context->tensors[node->outputs->data[0]];
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
    TfLiteTensor* tensor0 = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* tensor1 = &context->tensors[node->outputs->data[0]];
    TfLiteIntArray* newSize = TfLiteIntArrayCopy(tensor0->dims);
    return context->ResizeTensor(context, tensor1, newSize);
  };
  reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
    TfLiteTensor* a0 = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* a1 = &context->tensors[node->outputs->data[0]];
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

TEST(BasicInterpreter, TestUseNNAPI) {
  TestErrorReporter reporter;
  Interpreter interpreter(&reporter);
  interpreter.UseNNAPI(true);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
  interpreter.UseNNAPI(false);
  ASSERT_EQ(reporter.error_messages(),
            "Attempting to disable NNAPI delegate after it's applied.");
}

TEST(BasicInterpreter, TestUnsupportedDelegateFunctions) {
  Interpreter interpreter;
  ASSERT_EQ(interpreter.AddTensors(2), kTfLiteOk);
  TfLiteRegistration registration = {
      .init = nullptr, .free = nullptr, .prepare = nullptr, .invoke = nullptr};
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
    context->AddTensors(context, Interpreter::kTensorsCapacityHeadroom + 1,
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
  static const TfLiteExternalContextType kType = kTfLiteGemmLowpContext;

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
  interpreter_.SetNumThreads(4);

  TestExternalContext::Set(context, &external_context);
  EXPECT_EQ(TestExternalContext::Get(context), &external_context);
  interpreter_.SetNumThreads(4);
  interpreter_.SetNumThreads(5);
  EXPECT_EQ(external_context.num_refreshes, 2);

  TestExternalContext::Set(context, nullptr);
  EXPECT_EQ(TestExternalContext::Get(context), nullptr);
  interpreter_.SetNumThreads(4);
}

// Test fixture that allows playing with execution plans. It creates a two
// node graph that can be executed in either [0,1] order or [1,0] order.
// The CopyOp records when it is invoked in the class member run_order_
// so we can test whether the execution plan was honored.
class TestExecutionPlan : public ::testing::Test {
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
      TfLiteTensor* tensor0 = &context->tensors[node->inputs->data[0]];
      TfLiteTensor* tensor1 = &context->tensors[node->outputs->data[0]];
      TfLiteIntArray* newSize = TfLiteIntArrayCopy(tensor0->dims);
      return context->ResizeTensor(context, tensor1, newSize);
    };

    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
      CallReporting* call_reporting =
          reinterpret_cast<CallReporting*>(node->builtin_data);
      // Copy input data to output data.
      TfLiteTensor* a0 = &context->tensors[node->inputs->data[0]];
      TfLiteTensor* a1 = &context->tensors[node->outputs->data[0]];
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
        reinterpret_cast<CallReporting*>(malloc(sizeof(CallReporting)));
    new (call_reporting_1) CallReporting(input, &run_order_);
    ASSERT_EQ(interpreter_.AddNodeWithParameters(
                  {0}, {2}, nullptr, 0,
                  reinterpret_cast<void*>(call_reporting_1), &copy_op),
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
  Interpreter interpreter_;

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
  interpreter_.SetExecutionPlan({1, 0});
  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);
  ASSERT_EQ(run_order_, std::vector<int>({1, 0}));
}

TEST_F(TestExecutionPlan, SubsetExecutionPlan) {
  // Check running only node index 1
  interpreter_.SetExecutionPlan({1});
  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);
  ASSERT_EQ(run_order_, std::vector<int>({1}));
}

TEST_F(TestExecutionPlan, NullExecutionPlan) {
  // Check nothing executed.
  interpreter_.SetExecutionPlan({});
  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);
  ASSERT_EQ(run_order_, std::vector<int>());
}

// Build a kernel registration for an op that copies its one input
// to an output
TfLiteRegistration AddOpRegistration() {
  TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

  reg.custom_name = "my_add";
  reg.builtin_code = tflite::BuiltinOperator_CUSTOM;

  reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
    // Set output size to input size
    TfLiteTensor* input1 = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* input2 = &context->tensors[node->inputs->data[1]];
    TfLiteTensor* output = &context->tensors[node->outputs->data[0]];

    TF_LITE_ENSURE_EQ(context, input1->dims->size, input2->dims->size);
    for (int i = 0; i < input1->dims->size; ++i) {
      TF_LITE_ENSURE_EQ(context, input1->dims->data[i], input2->dims->data[i]);
    }

    TF_LITE_ENSURE_STATUS(context->ResizeTensor(
        context, output, TfLiteIntArrayCopy(input1->dims)));
    return kTfLiteOk;
  };

  reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
    // Copy input data to output data.
    TfLiteTensor* a0 = &context->tensors[node->inputs->data[0]];
    TfLiteTensor* a1 = &context->tensors[node->inputs->data[1]];
    TfLiteTensor* out = &context->tensors[node->outputs->data[0]];
    int num = a0->dims->data[0];
    for (int i = 0; i < num; i++) {
      out->data.f[i] = a0->data.f[i] + a1->data.f[i];
    }
    return kTfLiteOk;
  };
  return reg;
}

class TestDelegate : public ::testing::Test {
 protected:
  void SetUp() override {
    interpreter_.reset(new Interpreter);
    interpreter_->AddTensors(5);
    interpreter_->SetInputs({0, 1});
    interpreter_->SetOutputs({3, 4});
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
    TfLiteRegistration reg = AddOpRegistration();
    interpreter_->AddNodeWithParameters({0, 0}, {2}, nullptr, 0, nullptr, &reg);
    interpreter_->AddNodeWithParameters({1, 1}, {3}, nullptr, 0, nullptr, &reg);
    interpreter_->AddNodeWithParameters({2, 1}, {4}, nullptr, 0, nullptr, &reg);
  }

  void TearDown() override {
    // Interpreter relies on delegate to free the resources properly. Thus
    // the life cycle of delegate must be longer than interpreter.
    interpreter_.reset();
    delegate_.reset();
  }

  TfLiteBufferHandle last_allocated_handle_ = kTfLiteNullBufferHandle;

  TfLiteBufferHandle AllocateBufferHandle() { return ++last_allocated_handle_; }

 protected:
  class SimpleDelegate {
   public:
    // Create a simple implementation of a TfLiteDelegate. We use the C++ class
    // SimpleDelegate and it can produce a handle TfLiteDelegate that is
    // value-copyable and compatible with TfLite.
    explicit SimpleDelegate(const std::vector<int>& nodes) : nodes_(nodes) {
      delegate_.Prepare = [](TfLiteContext* context,
                             TfLiteDelegate* delegate) -> TfLiteStatus {
        auto* simple = reinterpret_cast<SimpleDelegate*>(delegate->data_);
        TfLiteIntArray* nodes_to_separate =
            TfLiteIntArrayCreate(simple->nodes_.size());
        // Mark nodes that we want in TfLiteIntArray* structure.
        int index = 0;
        for (auto node_index : simple->nodes_) {
          nodes_to_separate->data[index++] = node_index;
          // make sure node is added
          TfLiteNode* node;
          TfLiteRegistration* reg;
          context->GetNodeAndRegistration(context, node_index, &node, &reg);
          TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_CUSTOM);
          TFLITE_CHECK_EQ(strcmp(reg->custom_name, "my_add"), 0);
        }
        // Check that all nodes are available
        TfLiteIntArray* execution_plan;
        TF_LITE_ENSURE_STATUS(
            context->GetExecutionPlan(context, &execution_plan));
        for (int exec_index = 0; exec_index < execution_plan->size;
             exec_index++) {
          int node_index = execution_plan->data[exec_index];
          // Check that we are an identity map to start.
          TFLITE_CHECK_EQ(exec_index, node_index);
          TfLiteNode* node;
          TfLiteRegistration* reg;
          context->GetNodeAndRegistration(context, node_index, &node, &reg);
          TFLITE_CHECK_EQ(reg->builtin_code, tflite::BuiltinOperator_CUSTOM);
          TFLITE_CHECK_EQ(strcmp(reg->custom_name, "my_add"), 0);
        }

        context->ReplaceNodeSubsetsWithDelegateKernels(
            context, FakeFusedRegistration(), nodes_to_separate, delegate);
        TfLiteIntArrayFree(nodes_to_separate);
        return kTfLiteOk;
      };
      delegate_.CopyToBufferHandle = [](TfLiteContext* context,
                                        TfLiteDelegate* delegate,
                                        TfLiteBufferHandle buffer_handle,
                                        TfLiteTensor* tensor) -> TfLiteStatus {
        // TODO(ycling): Implement tests to test buffer copying logic.
        return kTfLiteOk;
      };
      delegate_.CopyFromBufferHandle =
          [](TfLiteContext* context, TfLiteDelegate* delegate,
             TfLiteBufferHandle buffer_handle,
             TfLiteTensor* output) -> TfLiteStatus {
        TFLITE_CHECK_GE(buffer_handle, -1);
        TFLITE_CHECK_EQ(output->buffer_handle, buffer_handle);
        const float floats[] = {6., 6., 6.};
        int num = output->dims->data[0];
        for (int i = 0; i < num; i++) {
          output->data.f[i] = floats[i];
        }
        return kTfLiteOk;
      };

      delegate_.FreeBufferHandle =
          [](TfLiteContext* context, TfLiteDelegate* delegate,
             TfLiteBufferHandle* handle) { *handle = kTfLiteNullBufferHandle; };
      // Store type-punned data SimpleDelegate structure.
      delegate_.data_ = reinterpret_cast<void*>(this);
      delegate_.flags = kTfLiteDelegateFlagsNone;
    }

    static TfLiteRegistration FakeFusedRegistration() {
      TfLiteRegistration reg = {nullptr};
      reg.custom_name = "fake_fused_op";

      reg.invoke = [](TfLiteContext* context,
                      TfLiteNode* node) -> TfLiteStatus {
        // Copy input data to output data.
        TfLiteTensor* a0 = &context->tensors[node->inputs->data[0]];
        TfLiteTensor* a1 = &context->tensors[node->inputs->data[1]];
        TfLiteTensor* out = &context->tensors[node->outputs->data[0]];
        int num = a0->dims->data[0];
        for (int i = 0; i < num; i++) {
          out->data.f[i] = a0->data.f[i] + a1->data.f[i];
        }
        // Make the data stale so that CopyFromBufferHandle can be invoked
        out->data_is_stale = true;
        return kTfLiteOk;
      };
      return reg;
    }

    TfLiteDelegate* get_tf_lite_delegate() { return &delegate_; }

   private:
    std::vector<int> nodes_;
    TfLiteDelegate delegate_;
  };
  std::unique_ptr<Interpreter> interpreter_;
  std::unique_ptr<SimpleDelegate> delegate_;
};

TEST_F(TestDelegate, BasicDelegate) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate());

  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  int node = interpreter_->execution_plan()[0];
  const auto* node_and_reg = interpreter_->node_and_registration(node);
  EXPECT_EQ(node_and_reg->second.custom_name,
            SimpleDelegate::FakeFusedRegistration().custom_name);

  const TfLiteDelegateParams* params =
      reinterpret_cast<const TfLiteDelegateParams*>(
          node_and_reg->first.builtin_data);
  ASSERT_EQ(params->nodes_to_replace->size, 3);
  EXPECT_EQ(params->nodes_to_replace->data[0], 0);
  EXPECT_EQ(params->nodes_to_replace->data[1], 1);
  EXPECT_EQ(params->nodes_to_replace->data[2], 2);

  ASSERT_EQ(params->input_tensors->size, 2);
  EXPECT_EQ(params->input_tensors->data[0], 0);
  EXPECT_EQ(params->input_tensors->data[1], 1);

  ASSERT_EQ(params->output_tensors->size, 2);
  EXPECT_EQ(params->output_tensors->data[0], 3);
  EXPECT_EQ(params->output_tensors->data[1], 4);
}

TEST_F(TestDelegate, StaticDelegateMakesGraphImmutable) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);

  // As the delegate doesn't support dynamic resizing, further graph mutation is
  // prohibited.
  ASSERT_NE(interpreter_->ResizeInputTensor(0, {0}), kTfLiteOk);
  ASSERT_NE(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);

  // Deliberately try to set tensor params with quantization while immutable,
  // ensuring quantization is properly freed.
  TfLiteQuantization quant = {};
  quant.type = kTfLiteAffineQuantization;
  auto quant_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  quant_params->scale = nullptr;
  quant_params->zero_point = nullptr;
  quant_params->quantized_dimension = 0;
  quant.params = quant_params;
  ASSERT_NE(interpreter_->SetTensorParametersReadWrite(0, kTfLiteInt8, "", {3},
                                                       quant),
            kTfLiteOk);
}

TEST_F(TestDelegate, ComplexDelegate) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({1, 2}));
  interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate());

  ASSERT_EQ(interpreter_->execution_plan().size(), 2);
  // 0th should be a non-delegated original op
  ASSERT_EQ(interpreter_->execution_plan()[0], 0);
  // 1st should be a new macro op (3) which didn't exist)
  ASSERT_EQ(interpreter_->execution_plan()[1], 3);
  const auto* node_and_reg = interpreter_->node_and_registration(3);
  ASSERT_EQ(node_and_reg->second.custom_name,
            SimpleDelegate::FakeFusedRegistration().custom_name);
}

TEST_F(TestDelegate, SetBufferHandleToInput) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  TfLiteDelegate* delegate = delegate_->get_tf_lite_delegate();
  interpreter_->ModifyGraphWithDelegate(delegate);

  constexpr int kOutputTensorIndex = 0;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  ASSERT_EQ(tensor->delegate, nullptr);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  TfLiteBufferHandle handle = AllocateBufferHandle();
  TfLiteStatus status =
      interpreter_->SetBufferHandle(kOutputTensorIndex, handle, delegate);
  ASSERT_EQ(status, kTfLiteOk);
  EXPECT_EQ(tensor->delegate, delegate);
  EXPECT_EQ(tensor->buffer_handle, handle);
}

TEST_F(TestDelegate, SetBufferHandleToOutput) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  TfLiteDelegate* delegate = delegate_->get_tf_lite_delegate();
  interpreter_->ModifyGraphWithDelegate(delegate);

  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  // Before setting the buffer handle, the tensor's `delegate` is already set
  // because it will be written by the delegate.
  ASSERT_EQ(tensor->delegate, delegate);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  TfLiteBufferHandle handle = AllocateBufferHandle();
  TfLiteStatus status =
      interpreter_->SetBufferHandle(kOutputTensorIndex, handle, delegate);
  ASSERT_EQ(status, kTfLiteOk);
  EXPECT_EQ(tensor->delegate, delegate);
  EXPECT_EQ(tensor->buffer_handle, handle);
}

TEST_F(TestDelegate, SetInvalidHandleToTensor) {
  interpreter_->Invoke();
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  TfLiteDelegate* delegate = delegate_->get_tf_lite_delegate();
  interpreter_->ModifyGraphWithDelegate(delegate);

  SimpleDelegate another_simple_delegate({0, 1, 2});

  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  // Before setting the buffer handle, the tensor's `delegate` is already set
  // because it will be written by the delegate.
  ASSERT_EQ(tensor->delegate, delegate);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  TfLiteBufferHandle handle = AllocateBufferHandle();
  TfLiteStatus status = interpreter_->SetBufferHandle(
      kOutputTensorIndex, handle,
      another_simple_delegate.get_tf_lite_delegate());
  // Setting a buffer handle to a tensor with another delegate will fail.
  ASSERT_EQ(status, kTfLiteError);
  EXPECT_EQ(tensor->delegate, delegate);
  EXPECT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);
}

TEST_F(TestDelegate, ResizeInputWithNonDynamicDelegateShouldFail) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 2}), kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(1, {1, 2}), kTfLiteOk);
  ASSERT_EQ(
      interpreter_->ModifyGraphWithDelegate(delegate_->get_tf_lite_delegate()),
      kTfLiteOk);
  ASSERT_EQ(interpreter_->ResizeInputTensor(0, {1, 2}), kTfLiteError);
}

TEST_F(TestDelegate, TestCopyFromBufferInvoke) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  TfLiteDelegate* delegate = delegate_->get_tf_lite_delegate();
  interpreter_->ModifyGraphWithDelegate(delegate);

  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  std::vector<float> floats = {1.0f, 2.0f, 3.0f};
  memcpy(interpreter_->typed_tensor<float>(0), floats.data(),
         floats.size() * sizeof(float));

  memcpy(interpreter_->typed_tensor<float>(1), floats.data(),
         floats.size() * sizeof(float));

  // Before setting the buffer handle, the tensor's `delegate` is already set
  // because it will be written by the delegate.
  ASSERT_EQ(tensor->delegate, delegate);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  // Called Invoke without setting the buffer will not call the CopyFromBuffer
  interpreter_->Invoke();
  std::vector<float> res = {2.0f, 4.0f, 6.0f};
  for (int i = 0; i < tensor->dims->data[0]; ++i) {
    ASSERT_EQ(tensor->data.f[i], res[i]);
  }
}

TEST_F(TestDelegate, TestCopyFromBuffer) {
  delegate_ = std::unique_ptr<SimpleDelegate>(new SimpleDelegate({0, 1, 2}));
  TfLiteDelegate* delegate = delegate_->get_tf_lite_delegate();
  interpreter_->ModifyGraphWithDelegate(delegate);

  constexpr int kOutputTensorIndex = 3;
  TfLiteTensor* tensor = interpreter_->tensor(kOutputTensorIndex);
  std::vector<float> floats = {1.0f, 2.0f, 3.0f};
  memcpy(interpreter_->typed_tensor<float>(0), floats.data(),
         floats.size() * sizeof(float));

  memcpy(interpreter_->typed_tensor<float>(1), floats.data(),
         floats.size() * sizeof(float));

  // Before setting the buffer handle, the tensor's `delegate` is already set
  // because it will be written by the delegate.
  ASSERT_EQ(tensor->delegate, delegate);
  ASSERT_EQ(tensor->buffer_handle, kTfLiteNullBufferHandle);

  TfLiteBufferHandle handle = AllocateBufferHandle();
  TfLiteStatus status =
      interpreter_->SetBufferHandle(kOutputTensorIndex, handle, delegate);
  interpreter_->Invoke();
  ASSERT_EQ(status, kTfLiteOk);
  EXPECT_EQ(tensor->delegate, delegate);
  EXPECT_EQ(tensor->buffer_handle, handle);
  for (int i = 0; i < tensor->dims->data[0]; ++i) {
    ASSERT_EQ(tensor->data.f[i], 6.0f);
  }
}

class TestDelegateWithDynamicTensors : public ::testing::Test {
 protected:
  void SetUp() override {
    interpreter_.reset(new Interpreter);

    interpreter_->AddTensors(2);
    interpreter_->SetInputs({0});
    interpreter_->SetOutputs({1});
    TfLiteQuantizationParams quant;
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {3},
                                               quant);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteFloat32, "", {3},
                                               quant);
    TfLiteRegistration reg = DynamicCopyOpRegistration();
    interpreter_->AddNodeWithParameters({0}, {1}, nullptr, 0, nullptr, &reg);

    delegate_.Prepare = [](TfLiteContext* context,
                           TfLiteDelegate* delegate) -> TfLiteStatus {
      // In this test, the delegate replaces all the nodes if this function is
      // called.
      TfLiteIntArray* execution_plan;
      TF_LITE_ENSURE_STATUS(
          context->GetExecutionPlan(context, &execution_plan));
      context->ReplaceNodeSubsetsWithDelegateKernels(
          context, DelegateRegistration(), execution_plan, delegate);
      return kTfLiteOk;
    };
    delegate_.flags = kTfLiteDelegateFlagsNone;
  }

  static TfLiteRegistration DynamicCopyOpRegistration() {
    TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};

    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
      SetTensorToDynamic(output);
      return kTfLiteOk;
    };

    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
      // Not implemented since this isn't required in testing.
      return kTfLiteOk;
    };
    return reg;
  }

  static TfLiteRegistration DelegateRegistration() {
    TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};
    return reg;
  }

  std::unique_ptr<Interpreter> interpreter_;
  TfLiteDelegate delegate_;
};

TEST_F(TestDelegateWithDynamicTensors, DisallowDynamicTensors) {
  interpreter_->ModifyGraphWithDelegate(&delegate_);

  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  // The interpreter should not call delegate's `Prepare` when dynamic tensors
  // exist. So the node ID isn't changed.
  ASSERT_EQ(interpreter_->execution_plan()[0], 0);
}

TEST_F(TestDelegateWithDynamicTensors, AllowDynamicTensors) {
  delegate_.flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  interpreter_->ModifyGraphWithDelegate(&delegate_);

  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  // The node should be replaced because dynamic tensors are allowed. Therefore
  // only node ID in the execution plan is changed from 0 to 1.
  ASSERT_EQ(interpreter_->execution_plan()[0], 1);
}

TEST_F(TestDelegateWithDynamicTensors, ModifyGraphAfterAllocate) {
  // Trigger allocation *before* delegate application.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  delegate_.flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  ASSERT_EQ(interpreter_->ModifyGraphWithDelegate(&delegate_), kTfLiteOk);
  ASSERT_EQ(interpreter_->execution_plan().size(), 1);
  ASSERT_EQ(interpreter_->execution_plan()[0], 1);

  // Allocation should still succeed.
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
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
      TfLiteTensor* in_tensor = &context->tensors[node->inputs->data[0]];
      TfLiteTensor* out_tensor = &context->tensors[node->outputs->data[0]];
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
      TfLiteTensor* in_tensor = &context->tensors[node->inputs->data[0]];
      TfLiteTensor* out_tensor = &context->tensors[node->outputs->data[0]];
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

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
