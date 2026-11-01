/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/kernel_util.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace {
using ::testing::ElementsAre;

struct TestContext : public TfLiteContext {
  string error;
};

void ReportError(TfLiteContext* context, const char* format, ...) {
  TestContext* c = static_cast<TestContext*>(context);
  const size_t kBufferSize = 1024;
  char temp_buffer[kBufferSize];

  va_list args;
  va_start(args, format);
  vsnprintf(temp_buffer, kBufferSize, format, args);
  va_end(args);

  c->error = temp_buffer;
}

class TestWithTfLiteContext : public ::testing::Test {
 public:
  TestWithTfLiteContext() { context_.ReportError = ReportError; }

  // `allocation_type` and `type` are not relavant for most of these tests,
  // so we provide a simpler wrapper to construct tensors.
  TensorUniquePtr BuildTfLiteTensorForTest(std::initializer_list<int> dims) {
    return BuildTfLiteTensor(kTfLiteInt32, dims, kTfLiteDynamic);
  }

 protected:
  TestContext context_;
};

TfLiteStatus CheckIntermediatesCustomOpPrepare(TfLiteContext* context,
                                               TfLiteNode* node) {
  TfLiteTensor* intermediate = nullptr;
  TF_LITE_ENSURE_OK(context,
                    GetIntermediatesSafe(context, node, 0, &intermediate));
  TF_LITE_ENSURE(context, intermediate != nullptr);
  volatile size_t observed_bytes = intermediate->bytes;
  (void)observed_bytes;
  const TfLiteTensor* nonsafe = GetIntermediates(context, node, 0);
  TF_LITE_ENSURE(context, nonsafe == intermediate);
  return kTfLiteOk;
}

std::vector<uint8_t> BuildCustomIntermediatesModelBuffer(
    int32_t intermediate_tensor_index) {
  flatbuffers::FlatBufferBuilder fbb;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {CreateBuffer(fbb)};
  std::vector<flatbuffers::Offset<Tensor>> tensors = {
      CreateTensor(fbb, fbb.CreateVector<int32_t>({2}), TensorType_FLOAT32, 0,
                   fbb.CreateString("in")),
      CreateTensor(fbb, fbb.CreateVector<int32_t>({2}), TensorType_FLOAT32, 0,
                   fbb.CreateString("mid")),
      CreateTensor(fbb, fbb.CreateVector<int32_t>({2}), TensorType_FLOAT32, 0,
                   fbb.CreateString("out")),
  };
  std::vector<flatbuffers::Offset<OperatorCode>> opcodes = {
      CreateOperatorCodeDirect(fbb, BuiltinOperator_CUSTOM,
                               "test_intermediates_op"),
  };
  std::vector<flatbuffers::Offset<Operator>> operators = {
      CreateOperator(fbb, /*opcode_index=*/0, fbb.CreateVector<int32_t>({0}),
                     fbb.CreateVector<int32_t>({2}), BuiltinOptions_NONE, 0, 0,
                     CustomOptionsFormat_FLEXBUFFERS, 0,
                     fbb.CreateVector<int32_t>({intermediate_tensor_index})),
  };
  std::vector<flatbuffers::Offset<SubGraph>> subgraphs = {
      CreateSubGraph(fbb, fbb.CreateVector(tensors),
                     fbb.CreateVector<int32_t>({0}),
                     fbb.CreateVector<int32_t>({2}),
                     fbb.CreateVector(operators), fbb.CreateString("main")),
  };
  FinishModelBuffer(
      fbb, CreateModel(fbb, 3, fbb.CreateVector(opcodes),
                       fbb.CreateVector(subgraphs),
                       fbb.CreateString("custom_intermediates_test"),
                       fbb.CreateVector(buffers)));
  return std::vector<uint8_t>(fbb.GetBufferPointer(),
                              fbb.GetBufferPointer() + fbb.GetSize());
}

std::vector<uint8_t> BuildQuantizedBuiltinLstmModelBuffer(
    int32_t first_intermediate_index) {
  flatbuffers::FlatBufferBuilder fbb;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {CreateBuffer(fbb)};
  std::vector<flatbuffers::Offset<Tensor>> tensors;

  auto add_tensor = [&](const std::vector<int32_t>& shape, TensorType type,
                        float scale, int64_t zp,
                        const std::vector<uint8_t>& raw_bytes,
                        bool is_variable) -> int32_t {
    uint32_t buffer_idx = 0;
    if (!raw_bytes.empty()) {
      buffer_idx = static_cast<uint32_t>(buffers.size());
      buffers.push_back(CreateBuffer(fbb, fbb.CreateVector(raw_bytes)));
    }
    auto quant = CreateQuantizationParameters(
        fbb, 0, 0, fbb.CreateVector<float>({scale}),
        fbb.CreateVector<int64_t>({zp}));
    int32_t idx = static_cast<int32_t>(tensors.size());
    std::string name = "t" + std::to_string(idx);
    tensors.push_back(CreateTensor(fbb, fbb.CreateVector(shape), type,
                                   buffer_idx, fbb.CreateString(name), quant,
                                   is_variable));
    return idx;
  };

  const std::vector<uint8_t> w4 = {4, 252, 4, 252};
  const int32_t bias_vals[2] = {10, -10};
  std::vector<uint8_t> b8(sizeof(bias_vals));
  memcpy(b8.data(), bias_vals, sizeof(bias_vals));
  const int16_t ln_vals[2] = {256, 256};
  std::vector<uint8_t> ln4(sizeof(ln_vals));
  memcpy(ln4.data(), ln_vals, sizeof(ln_vals));

  const int32_t t_in = add_tensor({1, 2}, TensorType_INT8, 0.01f, 0, {}, false);
  const int32_t t_i2f =
      add_tensor({2, 2}, TensorType_INT8, 0.01f, 0, w4, false);
  const int32_t t_i2c =
      add_tensor({2, 2}, TensorType_INT8, 0.01f, 0, w4, false);
  const int32_t t_i2o =
      add_tensor({2, 2}, TensorType_INT8, 0.01f, 0, w4, false);
  const int32_t t_r2f =
      add_tensor({2, 2}, TensorType_INT8, 0.01f, 0, w4, false);
  const int32_t t_r2c =
      add_tensor({2, 2}, TensorType_INT8, 0.01f, 0, w4, false);
  const int32_t t_r2o =
      add_tensor({2, 2}, TensorType_INT8, 0.01f, 0, w4, false);
  const int32_t t_fb =
      add_tensor({2}, TensorType_INT32, 0.0001f, 0, b8, false);
  const int32_t t_cb =
      add_tensor({2}, TensorType_INT32, 0.0001f, 0, b8, false);
  const int32_t t_ob =
      add_tensor({2}, TensorType_INT32, 0.0001f, 0, b8, false);
  const int32_t t_out_state =
      add_tensor({1, 2}, TensorType_INT8, 0.01f, 0, {}, true);
  const int32_t t_cell_state =
      add_tensor({1, 2}, TensorType_INT16, 1.0f / 32768.0f, 0, {}, true);
  const int32_t t_ln_f =
      add_tensor({2}, TensorType_INT16, 0.01f, 0, ln4, false);
  const int32_t t_ln_c =
      add_tensor({2}, TensorType_INT16, 0.01f, 0, ln4, false);
  const int32_t t_ln_o =
      add_tensor({2}, TensorType_INT16, 0.01f, 0, ln4, false);
  const int32_t t_out =
      add_tensor({1, 2}, TensorType_INT8, 0.01f, 0, {}, false);
  const int32_t t_mid0 =
      add_tensor({1, 2}, TensorType_INT16, 0.007f, 0, {}, false);
  const int32_t t_mid1 =
      add_tensor({1, 2}, TensorType_INT16, 0.007f, 0, {}, false);
  const int32_t t_mid2 =
      add_tensor({1, 2}, TensorType_INT16, 0.007f, 0, {}, false);
  const int32_t t_mid3 =
      add_tensor({1, 2}, TensorType_INT16, 0.007f, 0, {}, false);
  const int32_t t_mid4 =
      add_tensor({1, 2}, TensorType_INT16, 0.007f, 0, {}, false);

  const std::vector<int32_t> op_inputs = {
      t_in, -1, t_i2f, t_i2c, t_i2o, -1, t_r2f,       t_r2c,
      t_r2o, -1, -1,   -1,    -1,    t_fb, t_cb,      t_ob,
      -1,   -1, t_out_state,  t_cell_state, -1, t_ln_f, t_ln_c,
      t_ln_o};
  const std::vector<int32_t> op_outputs = {t_out};
  const std::vector<int32_t> op_intermediates = {
      first_intermediate_index == -999 ? t_mid0 : first_intermediate_index,
      t_mid1, t_mid2, t_mid3, t_mid4};

  std::vector<flatbuffers::Offset<OperatorCode>> opcodes = {
      CreateOperatorCode(fbb, static_cast<int8_t>(BuiltinOperator_LSTM), 0,
                         /*version=*/4, BuiltinOperator_LSTM),
  };
  auto lstm_opts = CreateLSTMOptions(
      fbb, ActivationFunctionType_TANH, /*cell_clip=*/0.0f,
      /*proj_clip=*/0.0f, LSTMKernelType_FULL,
      /*asymmetric_quantize_inputs=*/false);
  std::vector<flatbuffers::Offset<Operator>> operators = {
      CreateOperator(fbb, /*opcode_index=*/0, fbb.CreateVector(op_inputs),
                     fbb.CreateVector(op_outputs), BuiltinOptions_LSTMOptions,
                     lstm_opts.Union(), /*custom_options=*/0,
                     CustomOptionsFormat_FLEXBUFFERS, 0,
                     fbb.CreateVector(op_intermediates)),
  };
  std::vector<flatbuffers::Offset<SubGraph>> subgraphs = {
      CreateSubGraph(fbb, fbb.CreateVector(tensors),
                     fbb.CreateVector<int32_t>({t_in}),
                     fbb.CreateVector<int32_t>({t_out}),
                     fbb.CreateVector(operators), fbb.CreateString("main")),
  };
  FinishModelBuffer(fbb, CreateModel(fbb, 3, fbb.CreateVector(opcodes),
                                     fbb.CreateVector(subgraphs),
                                     fbb.CreateString("lstm_intermediates"),
                                     fbb.CreateVector(buffers)));
  return std::vector<uint8_t>(fbb.GetBufferPointer(),
                              fbb.GetBufferPointer() + fbb.GetSize());
}

class TensorIndexingBoundsTest : public ::testing::Test {
 protected:
  TensorIndexingBoundsTest() {
    memset(static_cast<TfLiteContext*>(&context_), 0, sizeof(TfLiteContext));
    context_.ReportError = ReportError;
    memset(&tensors_[0], 0, sizeof(tensors_));
    tensors_[0].bytes = 16;
    tensors_[0].is_variable = false;
    tensors_[1].bytes = 32;
    tensors_[1].is_variable = true;
    context_.tensors = tensors_;
    context_.tensors_size = 2;
    memset(&node_, 0, sizeof(node_));
  }

  TestContext context_;
  TfLiteTensor tensors_[2];
  TfLiteNode node_;
};

TEST_F(TensorIndexingBoundsTest, ValidTensorLookupsSucceed) {
  IntArrayUniquePtr inputs = BuildTfLiteArray({0, 1, kTfLiteOptionalTensor});
  IntArrayUniquePtr outputs = BuildTfLiteArray({1, kTfLiteOptionalTensor});
  IntArrayUniquePtr temporaries = BuildTfLiteArray({0, kTfLiteOptionalTensor});
  IntArrayUniquePtr intermediates =
      BuildTfLiteArray({1, kTfLiteOptionalTensor});
  node_.inputs = inputs.get();
  node_.outputs = outputs.get();
  node_.temporaries = temporaries.get();
  node_.intermediates = intermediates.get();

  EXPECT_EQ(GetInput(&context_, &node_, 0), &tensors_[0]);
  EXPECT_EQ(GetInput(&context_, &node_, 1), &tensors_[1]);
  EXPECT_EQ(GetInput(&context_, &node_, 2), nullptr);
  EXPECT_EQ(GetInput(&context_, &node_, -1), nullptr);
  EXPECT_EQ(GetInput(&context_, &node_, 3), nullptr);
  EXPECT_EQ(GetOptionalInputTensor(&context_, &node_, 0), &tensors_[0]);
  EXPECT_EQ(GetOptionalInputTensor(&context_, &node_, 2), nullptr);
  EXPECT_EQ(GetVariableInput(&context_, &node_, 0), nullptr);
  EXPECT_EQ(GetVariableInput(&context_, &node_, 1), &tensors_[1]);
  EXPECT_EQ(GetVariableInput(&context_, &node_, 2), nullptr);

  const TfLiteTensor* in_safe = nullptr;
  ASSERT_EQ(GetInputSafe(&context_, &node_, 0, &in_safe), kTfLiteOk);
  EXPECT_EQ(in_safe, &tensors_[0]);
  EXPECT_EQ(GetInputSafe(&context_, &node_, 2, &in_safe), kTfLiteError);

  EXPECT_EQ(GetOutput(&context_, &node_, 0), &tensors_[1]);
  EXPECT_EQ(GetOutput(&context_, &node_, 1), nullptr);
  TfLiteTensor* out_safe = nullptr;
  ASSERT_EQ(GetOutputSafe(&context_, &node_, 0, &out_safe), kTfLiteOk);
  EXPECT_EQ(out_safe, &tensors_[1]);
  EXPECT_EQ(GetOutputSafe(&context_, &node_, 1, &out_safe), kTfLiteError);

  EXPECT_EQ(GetTemporary(&context_, &node_, 0), &tensors_[0]);
  EXPECT_EQ(GetTemporary(&context_, &node_, 1), nullptr);
  TfLiteTensor* tmp_safe = nullptr;
  ASSERT_EQ(GetTemporarySafe(&context_, &node_, 0, &tmp_safe), kTfLiteOk);
  EXPECT_EQ(tmp_safe, &tensors_[0]);
  EXPECT_EQ(GetTemporarySafe(&context_, &node_, 1, &tmp_safe), kTfLiteError);

  EXPECT_EQ(GetIntermediates(&context_, &node_, 0), &tensors_[1]);
  EXPECT_EQ(GetIntermediates(&context_, &node_, 1), nullptr);
  TfLiteTensor* mid_safe = nullptr;
  ASSERT_EQ(GetIntermediatesSafe(&context_, &node_, 0, &mid_safe), kTfLiteOk);
  EXPECT_EQ(mid_safe, &tensors_[1]);
  EXPECT_EQ(GetIntermediatesSafe(&context_, &node_, 1, &mid_safe),
            kTfLiteError);

  // Also verify custom GetTensor callback mode when context.tensors == nullptr.
  context_.tensors = nullptr;
  context_.impl_ = tensors_;
  context_.GetTensor = [](const TfLiteContext* ctx,
                          int idx) -> TfLiteTensor* {
    return &(static_cast<TfLiteTensor*>(ctx->impl_)[idx]);
  };
  ASSERT_EQ(GetInputSafe(&context_, &node_, 1, &in_safe), kTfLiteOk);
  EXPECT_EQ(in_safe, &tensors_[1]);
  EXPECT_EQ(GetInput(&context_, &node_, 0), &tensors_[0]);
}

TEST_F(TensorIndexingBoundsTest, ValidFlatBufferIntermediatesModelsSucceed) {
  const std::vector<uint8_t> custom_buf =
      BuildCustomIntermediatesModelBuffer(/*intermediate_tensor_index=*/1);
  auto custom_model = FlatBufferModel::VerifyAndBuildFromBuffer(
      reinterpret_cast<const char*>(custom_buf.data()), custom_buf.size());
  ASSERT_NE(custom_model, nullptr);
  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates custom_resolver;
  TfLiteRegistration reg = {};
  reg.prepare = CheckIntermediatesCustomOpPrepare;
  reg.invoke = [](TfLiteContext*, TfLiteNode*) { return kTfLiteOk; };
  custom_resolver.AddCustom("test_intermediates_op", &reg);
  std::unique_ptr<Interpreter> custom_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(*custom_model, custom_resolver)(&custom_interpreter),
      kTfLiteOk);
  ASSERT_NE(custom_interpreter, nullptr);
  ASSERT_EQ(custom_interpreter->AllocateTensors(), kTfLiteOk);
  EXPECT_EQ(custom_interpreter->Invoke(), kTfLiteOk);

  const std::vector<uint8_t> lstm_buf =
      BuildQuantizedBuiltinLstmModelBuffer(/*first_intermediate_index=*/-999);
  auto lstm_model = FlatBufferModel::VerifyAndBuildFromBuffer(
      reinterpret_cast<const char*>(lstm_buf.data()), lstm_buf.size());
  ASSERT_NE(lstm_model, nullptr);
  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates lstm_resolver;
  std::unique_ptr<Interpreter> lstm_interpreter;
  ASSERT_EQ(InterpreterBuilder(*lstm_model, lstm_resolver)(&lstm_interpreter),
            kTfLiteOk);
  ASSERT_NE(lstm_interpreter, nullptr);
  ASSERT_EQ(lstm_interpreter->AllocateTensors(), kTfLiteOk);
  EXPECT_EQ(lstm_interpreter->Invoke(), kTfLiteOk);
}

TEST_F(TensorIndexingBoundsTest,
       SafeHelpersRejectNegativeNonOptionalTensorIndices) {
  for (int bad_index : {-2, std::numeric_limits<int>::min()}) {
    IntArrayUniquePtr arr = BuildTfLiteArray({bad_index});
    node_.inputs = arr.get();
    node_.outputs = arr.get();
    node_.temporaries = arr.get();
    node_.intermediates = arr.get();

    const TfLiteTensor* in_tensor = &tensors_[0];
    TfLiteStatus status = GetInputSafe(&context_, &node_, 0, &in_tensor);
    if (status == kTfLiteOk && in_tensor != nullptr) {
      volatile size_t observed = in_tensor->bytes;
      (void)observed;
    }
    EXPECT_EQ(status, kTfLiteError);
    EXPECT_EQ(in_tensor, nullptr);

    TfLiteTensor* out_tensor = &tensors_[0];
    status = GetOutputSafe(&context_, &node_, 0, &out_tensor);
    if (status == kTfLiteOk && out_tensor != nullptr) {
      volatile size_t observed = out_tensor->bytes;
      (void)observed;
    }
    EXPECT_EQ(status, kTfLiteError);
    EXPECT_EQ(out_tensor, nullptr);

    TfLiteTensor* tmp_tensor = &tensors_[0];
    status = GetTemporarySafe(&context_, &node_, 0, &tmp_tensor);
    if (status == kTfLiteOk && tmp_tensor != nullptr) {
      volatile size_t observed = tmp_tensor->bytes;
      (void)observed;
    }
    EXPECT_EQ(status, kTfLiteError);
    EXPECT_EQ(tmp_tensor, nullptr);

    TfLiteTensor* mid_tensor = &tensors_[0];
    status = GetIntermediatesSafe(&context_, &node_, 0, &mid_tensor);
    if (status == kTfLiteOk && mid_tensor != nullptr) {
      volatile size_t observed = mid_tensor->bytes;
      (void)observed;
    }
    EXPECT_EQ(status, kTfLiteError);
    EXPECT_EQ(mid_tensor, nullptr);
  }
}

TEST_F(TensorIndexingBoundsTest,
       SafeHelpersRejectTensorIndicesAtOrAboveTensorsSize) {
  for (int bad_index : {2, 99, std::numeric_limits<int>::max()}) {
    IntArrayUniquePtr arr = BuildTfLiteArray({bad_index});
    node_.inputs = arr.get();
    node_.outputs = arr.get();
    node_.temporaries = arr.get();
    node_.intermediates = arr.get();

    const TfLiteTensor* in_tensor = &tensors_[0];
    TfLiteStatus status = GetInputSafe(&context_, &node_, 0, &in_tensor);
    if (status == kTfLiteOk && in_tensor != nullptr) {
      volatile size_t observed = in_tensor->bytes;
      (void)observed;
    }
    EXPECT_EQ(status, kTfLiteError);
    EXPECT_EQ(in_tensor, nullptr);

    TfLiteTensor* out_tensor = &tensors_[0];
    status = GetOutputSafe(&context_, &node_, 0, &out_tensor);
    if (status == kTfLiteOk && out_tensor != nullptr) {
      volatile size_t observed = out_tensor->bytes;
      (void)observed;
    }
    EXPECT_EQ(status, kTfLiteError);
    EXPECT_EQ(out_tensor, nullptr);

    TfLiteTensor* tmp_tensor = &tensors_[0];
    status = GetTemporarySafe(&context_, &node_, 0, &tmp_tensor);
    if (status == kTfLiteOk && tmp_tensor != nullptr) {
      volatile size_t observed = tmp_tensor->bytes;
      (void)observed;
    }
    EXPECT_EQ(status, kTfLiteError);
    EXPECT_EQ(tmp_tensor, nullptr);

    TfLiteTensor* mid_tensor = &tensors_[0];
    status = GetIntermediatesSafe(&context_, &node_, 0, &mid_tensor);
    if (status == kTfLiteOk && mid_tensor != nullptr) {
      volatile size_t observed = mid_tensor->bytes;
      (void)observed;
    }
    EXPECT_EQ(status, kTfLiteError);
    EXPECT_EQ(mid_tensor, nullptr);
  }

  // Zero-tensor context must reject subgraph tensor index 0.
  context_.tensors_size = 0;
  IntArrayUniquePtr zero_arr = BuildTfLiteArray({0});
  node_.intermediates = zero_arr.get();
  TfLiteTensor* mid_tensor = &tensors_[0];
  EXPECT_EQ(GetIntermediatesSafe(&context_, &node_, 0, &mid_tensor),
            kTfLiteError);
  EXPECT_EQ(mid_tensor, nullptr);
}

TEST_F(TensorIndexingBoundsTest,
       NonSafeHelpersRejectOutOfBoundsTensorIndices) {
  for (int bad_index :
       {-2, std::numeric_limits<int>::min(), 2, 99,
        std::numeric_limits<int>::max()}) {
    IntArrayUniquePtr arr = BuildTfLiteArray({bad_index});
    node_.inputs = arr.get();
    node_.outputs = arr.get();
    node_.temporaries = arr.get();
    node_.intermediates = arr.get();

    const TfLiteTensor* in_tensor = GetInput(&context_, &node_, 0);
    if (in_tensor != nullptr) {
      volatile size_t observed = in_tensor->bytes;
      (void)observed;
    }
    EXPECT_EQ(in_tensor, nullptr);

    const TfLiteTensor* opt_in = GetOptionalInputTensor(&context_, &node_, 0);
    if (opt_in != nullptr) {
      volatile size_t observed = opt_in->bytes;
      (void)observed;
    }
    EXPECT_EQ(opt_in, nullptr);

    TfLiteTensor* var_in = GetVariableInput(&context_, &node_, 0);
    if (var_in != nullptr) {
      volatile size_t observed = var_in->bytes;
      (void)observed;
    }
    EXPECT_EQ(var_in, nullptr);

    TfLiteTensor* out_tensor = GetOutput(&context_, &node_, 0);
    if (out_tensor != nullptr) {
      volatile size_t observed = out_tensor->bytes;
      (void)observed;
    }
    EXPECT_EQ(out_tensor, nullptr);

    TfLiteTensor* tmp_tensor = GetTemporary(&context_, &node_, 0);
    if (tmp_tensor != nullptr) {
      volatile size_t observed = tmp_tensor->bytes;
      (void)observed;
    }
    EXPECT_EQ(tmp_tensor, nullptr);

    const TfLiteTensor* mid_tensor = GetIntermediates(&context_, &node_, 0);
    if (mid_tensor != nullptr) {
      volatile size_t observed = mid_tensor->bytes;
      (void)observed;
    }
    EXPECT_EQ(mid_tensor, nullptr);
  }
}

TEST_F(TensorIndexingBoundsTest,
       HelpersRejectNullNodeArraysContextAndOutputPointers) {
  // node.{inputs,outputs,temporaries,intermediates} are all nullptr initially.
  EXPECT_EQ(GetInput(&context_, &node_, 0), nullptr);
  EXPECT_EQ(GetOptionalInputTensor(&context_, &node_, 0), nullptr);
  EXPECT_EQ(GetVariableInput(&context_, &node_, 0), nullptr);
  EXPECT_EQ(GetOutput(&context_, &node_, 0), nullptr);
  EXPECT_EQ(GetTemporary(&context_, &node_, 0), nullptr);
  EXPECT_EQ(GetIntermediates(&context_, &node_, 0), nullptr);

  const TfLiteTensor* in_safe = &tensors_[0];
  EXPECT_EQ(GetInputSafe(&context_, &node_, 0, &in_safe), kTfLiteError);
  EXPECT_EQ(in_safe, nullptr);
  TfLiteTensor* out_safe = &tensors_[0];
  EXPECT_EQ(GetOutputSafe(&context_, &node_, 0, &out_safe), kTfLiteError);
  EXPECT_EQ(out_safe, nullptr);
  TfLiteTensor* tmp_safe = &tensors_[0];
  EXPECT_EQ(GetTemporarySafe(&context_, &node_, 0, &tmp_safe), kTfLiteError);
  EXPECT_EQ(tmp_safe, nullptr);
  TfLiteTensor* mid_safe = &tensors_[0];
  EXPECT_EQ(GetIntermediatesSafe(&context_, &node_, 0, &mid_safe),
            kTfLiteError);
  EXPECT_EQ(mid_safe, nullptr);

  IntArrayUniquePtr valid_arr = BuildTfLiteArray({0});
  node_.inputs = valid_arr.get();
  node_.outputs = valid_arr.get();
  node_.temporaries = valid_arr.get();
  node_.intermediates = valid_arr.get();

  // Null context, null node, null output pointer, and null ReportError.
  EXPECT_EQ(GetInput(nullptr, &node_, 0), nullptr);
  EXPECT_EQ(GetInput(&context_, nullptr, 0), nullptr);
  EXPECT_EQ(GetInputSafe(nullptr, &node_, 0, &in_safe), kTfLiteError);
  EXPECT_EQ(GetInputSafe(&context_, nullptr, 0, &in_safe), kTfLiteError);
  EXPECT_EQ(GetInputSafe(&context_, &node_, 0, nullptr), kTfLiteError);

  context_.ReportError = nullptr;
  context_.tensors = nullptr;
  context_.GetTensor = nullptr;
  out_safe = &tensors_[0];
  EXPECT_EQ(GetOutputSafe(&context_, &node_, 0, &out_safe), kTfLiteError);
  EXPECT_EQ(out_safe, nullptr);
  EXPECT_EQ(GetOutput(&context_, &node_, 0), nullptr);
}

TEST_F(TensorIndexingBoundsTest,
       FlatBufferCustomOpRejectsOutOfBoundsIntermediates) {
  for (int32_t bad_intermediate : {-2, 3, 99}) {
    const std::vector<uint8_t> buf =
        BuildCustomIntermediatesModelBuffer(bad_intermediate);
    auto model = FlatBufferModel::VerifyAndBuildFromBuffer(
        reinterpret_cast<const char*>(buf.data()), buf.size());
    ASSERT_NE(model, nullptr);
    ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
    TfLiteRegistration reg = {};
    reg.prepare = CheckIntermediatesCustomOpPrepare;
    reg.invoke = [](TfLiteContext*, TfLiteNode*) { return kTfLiteOk; };
    resolver.AddCustom("test_intermediates_op", &reg);
    std::unique_ptr<Interpreter> interpreter;
    ASSERT_EQ(InterpreterBuilder(*model, resolver)(&interpreter), kTfLiteOk);
    ASSERT_NE(interpreter, nullptr);
    EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteError);
  }
}

TEST_F(TensorIndexingBoundsTest,
       FlatBufferBuiltinLstmSurvivesOutOfBoundsLayerNormIntermediates) {
  for (int32_t bad_intermediate : {-2, 999}) {
    const std::vector<uint8_t> buf =
        BuildQuantizedBuiltinLstmModelBuffer(bad_intermediate);
    auto model = FlatBufferModel::VerifyAndBuildFromBuffer(
        reinterpret_cast<const char*>(buf.data()), buf.size());
    ASSERT_NE(model, nullptr);
    ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
    std::unique_ptr<Interpreter> interpreter;
    ASSERT_EQ(InterpreterBuilder(*model, resolver)(&interpreter), kTfLiteOk);
    ASSERT_NE(interpreter, nullptr);
    // On unpatched builds, BuiltinOperator_LSTM Prepare() calls
    // GetIntermediatesSafe() on slot 0 and dereferences &context->tensors[idx]
    // out of bounds under ASan.
    (void)interpreter->AllocateTensors();
  }
}

class HaveSameShapeTest : public TestWithTfLiteContext {};

TEST_F(HaveSameShapeTest, NullPointerIsSameShape) {
  TensorUniquePtr t1 = BuildTfLiteTensor();
  t1->dims = nullptr;

  TensorUniquePtr t2 = BuildTfLiteTensor();
  t2->dims = nullptr;

  EXPECT_TRUE(HaveSameShapes(t1.get(), t2.get()));
}

TEST_F(HaveSameShapeTest, NotSameShapeFalse) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({2, 3});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({3});

  EXPECT_FALSE(HaveSameShapes(t1.get(), t2.get()));
}

TEST_F(HaveSameShapeTest, EmptyShapeEqualTrue) {
  TensorUniquePtr t1 = BuildTfLiteTensor();
  TensorUniquePtr t2 = BuildTfLiteTensor();

  EXPECT_TRUE(HaveSameShapes(t1.get(), t2.get()));
}

class BroadcastShapeTest : public TestWithTfLiteContext {};

TEST_F(BroadcastShapeTest, IncompatibleDimNullptr) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({1, 3});

  TfLiteIntArray* output = nullptr;

  EXPECT_NE(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, t1.get(), t2.get(), &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(context_.error,
            "Given shapes, [1,2] and [1,3], are not broadcastable.");
}

TEST_F(BroadcastShapeTest, IncompatibleDimWithZeroNullptr) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 0});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({1, 3});

  TfLiteIntArray* output = nullptr;

  EXPECT_NE(kTfLiteOk,
            CalculateShapeForBroadcast(&context_, t1.get(), t2.get(), &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(context_.error,
            "Given shapes, [1,0] and [1,3], are not broadcastable.");
}

TEST_F(BroadcastShapeTest, BroadCastSecondDimension) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 1});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({1, 3});

  TfLiteIntArray* raw_output;
  auto status =
      CalculateShapeForBroadcast(&context_, t1.get(), t2.get(), &raw_output);
  ASSERT_EQ(kTfLiteOk, status);
  IntArrayUniquePtr output(raw_output);

  EXPECT_THAT(output.get(), DimsAre({1, 3}));
}

TEST_F(BroadcastShapeTest, ScalarAnd2dBroadcastsTo2d) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  &raw_output));

  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({1, 2}));
}

TEST_F(BroadcastShapeTest, DifferentRankBroadcastsToHigherRank) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({3, 1, 2});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  &raw_output));
  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({3, 1, 2}));
}

TEST_F(BroadcastShapeTest, ZeroDimDifferentRankBroadcastsToHigherRank) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({3, 0, 2});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  &raw_output));
  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({3, 0, 2}));
}

TEST_F(BroadcastShapeTest, ZeroDimSameRankBroadcastsToHigherRank) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({3, 0, 1});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  &raw_output));

  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({3, 0, 2}));
}

TEST_F(BroadcastShapeTest, IncompatibleDimOnThreeTensorsNullptr) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({1, 3});
  TensorUniquePtr t3 = BuildTfLiteTensorForTest({1, 4});

  TfLiteIntArray* raw_output = nullptr;
  EXPECT_NE(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  t3.get(), &raw_output));
  EXPECT_EQ(raw_output, nullptr);
  EXPECT_EQ(context_.error,
            "Given shapes, [1,2], [1,3] and [1,4], are not broadcastable.");
}

TEST_F(BroadcastShapeTest, IncompatibleDimWithZeroOnThreeTensorsNullptr) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 1});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({1, 3});
  TensorUniquePtr t3 = BuildTfLiteTensorForTest({1, 0});

  TfLiteIntArray* raw_output = nullptr;
  EXPECT_NE(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  t3.get(), &raw_output));
  EXPECT_EQ(raw_output, nullptr);
  EXPECT_EQ(context_.error,
            "Given shapes, [1,1], [1,3] and [1,0], are not broadcastable.");
}

TEST_F(BroadcastShapeTest, ThreeTensorsBroadcastToLarger2ndDim) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 1});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({1, 1});
  TensorUniquePtr t3 = BuildTfLiteTensorForTest({1, 3});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  t3.get(), &raw_output));
  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({1, 3}));
}

TEST_F(BroadcastShapeTest, TwoScalarsBroadcastTo2d) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({});
  TensorUniquePtr t3 = BuildTfLiteTensorForTest({});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  t3.get(), &raw_output));
  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({1, 2}));
}

TEST_F(BroadcastShapeTest, DifferentSizesOnThreeTensorsBroadcastToLargerRank) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({3, 1, 1});
  TensorUniquePtr t3 = BuildTfLiteTensorForTest({3, 1});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  t3.get(), &raw_output));
  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({3, 3, 2}));
}

TEST_F(BroadcastShapeTest,
       DifferentSizesOnThreeTensors4dBroadcastToLargerRank) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({3, 4});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({1, 3, 1});
  TensorUniquePtr t3 = BuildTfLiteTensorForTest({1, 2, 1, 1});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  t3.get(), &raw_output));
  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({1, 2, 3, 4}));
}

TEST_F(BroadcastShapeTest, ZeroOnThreeTensorsBroadcastToLargerRank) {
  TensorUniquePtr t1 = BuildTfLiteTensorForTest({1, 2});
  TensorUniquePtr t2 = BuildTfLiteTensorForTest({3, 1, 1});
  TensorUniquePtr t3 = BuildTfLiteTensorForTest({0, 1});

  TfLiteIntArray* raw_output;
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, t1.get(), t2.get(),
                                                  t3.get(), &raw_output));
  IntArrayUniquePtr output(raw_output);
  EXPECT_THAT(output.get(), DimsAre({3, 0, 2}));
}

TEST(GetShapeDebugStringTest, GetShapeDebugString) {
  IntArrayUniquePtr dims0 = BuildTfLiteArray({});
  EXPECT_EQ("[]", GetShapeDebugString(dims0.get()));

  IntArrayUniquePtr dims1 = BuildTfLiteArray({1});
  dims1->data[0] = 1;
  EXPECT_EQ("[1]", GetShapeDebugString(dims1.get()));

  IntArrayUniquePtr dims2 = BuildTfLiteArray({2, 3});
  dims2->data[0] = 2;
  dims2->data[1] = 3;
  EXPECT_EQ("[2,3]", GetShapeDebugString(dims2.get()));

  IntArrayUniquePtr dims3 = BuildTfLiteArray({4, 5, 6});
  dims3->data[0] = 4;
  dims3->data[1] = 5;
  dims3->data[2] = 6;
  EXPECT_EQ("[4,5,6]", GetShapeDebugString(dims3.get()));
}

class QuantizationParamsTest : public TestWithTfLiteContext {};

TEST_F(QuantizationParamsTest, PerChannelConvolution) {
  // Create input.
  TensorUniquePtr input = BuildTfLiteTensor();
  input->type = kTfLiteInt8;
  input->allocation_type = kTfLiteArenaRw;
  input->dims = TfLiteIntArrayCreate(1);
  input->dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {0.5, 5};
  input->params = input_quant;
  input->quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 0.5;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input->quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  TensorUniquePtr filter = BuildTfLiteTensor();
  filter->type = kTfLiteInt8;
  filter->allocation_type = kTfLiteArenaRw;
  filter->dims = TfLiteIntArrayCreate(4);
  filter->dims->data[0] = 3;
  filter->dims->data[1] = 4;
  filter->dims->data[2] = 5;
  filter->dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {0.25, 0};
  filter->params = filter_quant;
  filter->quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  filter_params->scale = TfLiteFloatArrayCreate(3);
  filter_params->scale->data[0] = 0.25;
  filter_params->scale->data[1] = 0.125;
  filter_params->scale->data[2] = 0.25;
  filter_params->zero_point = TfLiteIntArrayCreate(3);
  filter_params->zero_point->data[0] = 0;
  filter_params->zero_point->data[1] = 0;
  filter_params->zero_point->data[2] = 0;
  filter_params->quantized_dimension = 0;
  filter->quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias.
  TensorUniquePtr bias = BuildTfLiteTensor();
  bias->type = kTfLiteInt32;
  bias->allocation_type = kTfLiteArenaRw;
  bias->dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {0.125, 9};
  bias->params = bias_quant;
  bias->quantization.type = kTfLiteAffineQuantization;
  auto* bias_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  bias_params->scale = TfLiteFloatArrayCreate(3);
  bias_params->scale->data[0] = 0.125;
  bias_params->scale->data[1] = 0.0625;
  bias_params->scale->data[2] = 0.125;
  bias_params->zero_point = TfLiteIntArrayCreate(3);
  bias_params->zero_point->data[0] = 11;
  bias_params->zero_point->data[1] = 12;
  bias_params->zero_point->data[2] = 15;
  bias->quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output.
  TensorUniquePtr output = BuildTfLiteTensor();
  output->type = kTfLiteInt8;
  output->allocation_type = kTfLiteArenaRw;
  output->dims = nullptr;
  TfLiteQuantizationParams output_quant = {0.5, -128};
  output->params = output_quant;
  output->quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 0.5;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output->quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int32_t> per_channel_shift(3);

  // Call and verify results for per channel case.
  auto status = PopulateConvolutionQuantizationParams(
      &context_, input.get(), filter.get(), bias.get(), output.get(),
      kTfLiteActRelu, &multiplier, &shift, &output_activation_min,
      &output_activation_max, per_channel_multiplier.data(),
      per_channel_shift.data());
  EXPECT_EQ(kTfLiteOk, status);
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-1, -2, -1));
}

TEST_F(QuantizationParamsTest, CheckAndPopulateShift) {
  // Create input of type kTfLiteUInt8.

  TensorUniquePtr input = BuildTfLiteTensor();
  input->type = kTfLiteUInt8;
  input->allocation_type = kTfLiteArenaRw;
  input->dims = TfLiteIntArrayCreate(1);
  input->dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {0.5, 5};
  input->params = input_quant;
  input->quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 0.5;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input->quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter of type kTfLiteUInt8.
  TensorUniquePtr filter = BuildTfLiteTensor();
  filter->type = kTfLiteUInt8;
  filter->allocation_type = kTfLiteArenaRw;
  filter->dims = TfLiteIntArrayCreate(4);
  filter->dims->data[0] = 3;
  filter->dims->data[1] = 4;
  filter->dims->data[2] = 5;
  filter->dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {0.25, 0};
  filter->params = filter_quant;
  filter->quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  // Create scale of size one.
  filter_params->scale = TfLiteFloatArrayCreate(1);
  filter_params->scale->data[0] = 0.25;
  filter_params->zero_point = TfLiteIntArrayCreate(1);
  filter_params->zero_point->data[0] = 0;
  filter_params->quantized_dimension = 0;
  filter->quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias for kTfLiteUInt8.
  TensorUniquePtr bias = BuildTfLiteTensor();
  bias->type = kTfLiteUInt8;
  bias->allocation_type = kTfLiteArenaRw;
  bias->dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {0.125, 9};
  bias->params = bias_quant;
  bias->quantization.type = kTfLiteAffineQuantization;
  auto* bias_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  bias_params->scale = TfLiteFloatArrayCreate(3);
  bias_params->scale->data[0] = 0.125;
  bias_params->scale->data[1] = 0.0625;
  bias_params->scale->data[2] = 0.125;
  bias_params->zero_point = TfLiteIntArrayCreate(3);
  bias_params->zero_point->data[0] = 11;
  bias_params->zero_point->data[1] = 12;
  bias_params->zero_point->data[2] = 15;
  bias->quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output for kTfLiteUInt8.
  TensorUniquePtr output = BuildTfLiteTensor();
  output->type = kTfLiteUInt8;
  output->allocation_type = kTfLiteArenaRw;
  output->dims = nullptr;
  TfLiteQuantizationParams output_quant = {0.5, 128};
  output->params = output_quant;
  output->quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 0.5;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = 128;
  output->quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(kTfLiteOk,
            PopulateConvolutionQuantizationParams(
                &context_, input.get(), filter.get(), bias.get(), output.get(),
                kTfLiteActRelu, &multiplier, &shift, &output_activation_min,
                &output_activation_max, per_channel_multiplier.data(),
                per_channel_shift.data(), 3));
  // Since the filter scale has a size of one but the number of channels is
  // three, in our TC we expect three 1073741824 as output
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-1, -1, -1));
  EXPECT_EQ(shift, 1);
  EXPECT_EQ(multiplier, 1073741824);
}

#ifndef __APPLE__  // Some Apple toolchains don't support std::ldexp
TEST_F(QuantizationParamsTest, CheckAndPopulateZeroValue) {
  // Create input.
  auto input = BuildTfLiteTensor();
  input->type = kTfLiteInt8;
  input->allocation_type = kTfLiteArenaRw;
  input->dims = TfLiteIntArrayCreate(1);
  input->dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {1, 5};
  input->params = input_quant;
  input->quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 1;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input->quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  auto filter = BuildTfLiteTensor();
  filter->type = kTfLiteInt8;
  filter->allocation_type = kTfLiteArenaRw;
  filter->dims = TfLiteIntArrayCreate(4);
  filter->dims->data[0] = 3;
  filter->dims->data[1] = 4;
  filter->dims->data[2] = 5;
  filter->dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {4.6566129e-10, 0};
  filter->params = filter_quant;
  filter->quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  filter_params->scale = TfLiteFloatArrayCreate(3);
  filter_params->scale->data[0] = std::ldexp(1.0f, -31);
  filter_params->scale->data[1] = std::ldexp(1.0f, -32);
  filter_params->scale->data[2] = std::ldexp(1.0f, -33);
  filter_params->zero_point = TfLiteIntArrayCreate(3);
  filter_params->zero_point->data[0] = 0;
  filter_params->zero_point->data[1] = 0;
  filter_params->zero_point->data[2] = 0;
  filter_params->quantized_dimension = 0;
  filter->quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias.
  auto bias = BuildTfLiteTensor();
  bias->type = kTfLiteInt32;
  bias->allocation_type = kTfLiteArenaRw;
  bias->dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {4.6566129e-10, 9};
  bias->params = bias_quant;
  bias->quantization.type = kTfLiteAffineQuantization;
  auto* bias_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  bias_params->scale = TfLiteFloatArrayCreate(3);
  bias_params->scale->data[0] = std::ldexp(1.0f, -31);
  bias_params->scale->data[1] = std::ldexp(1.0f, -32);
  bias_params->scale->data[2] = std::ldexp(1.0f, -33);
  bias_params->zero_point = TfLiteIntArrayCreate(3);
  bias_params->zero_point->data[0] = 11;
  bias_params->zero_point->data[1] = 12;
  bias_params->zero_point->data[2] = 15;
  bias->quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output.
  auto output = BuildTfLiteTensor();
  output->type = kTfLiteInt8;
  output->allocation_type = kTfLiteArenaRw;
  output->dims = nullptr;
  TfLiteQuantizationParams output_quant = {1, -128};
  output->params = output_quant;
  output->quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output->quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(kTfLiteOk,
            PopulateConvolutionQuantizationParams(
                &context_, input.get(), filter.get(), bias.get(), output.get(),
                kTfLiteActRelu, &multiplier, &shift, &output_activation_min,
                &output_activation_max, per_channel_multiplier.data(),
                per_channel_shift.data(), 3));
  EXPECT_THAT(per_channel_multiplier, ElementsAre(1073741824, 1073741824, 0));
  EXPECT_THAT(per_channel_shift, ElementsAre(-30, -31, 0));
}
#endif

TEST_F(QuantizationParamsTest, CheckAndPopulateUint8) {
  // Create input.
  auto input = BuildTfLiteTensor();
  input->type = kTfLiteUInt8;
  input->allocation_type = kTfLiteArenaRw;
  input->dims = TfLiteIntArrayCreate(1);
  input->dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {1, 5};
  input->params = input_quant;
  input->quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 1;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input->quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  auto filter = BuildTfLiteTensor();
  filter->type = kTfLiteUInt8;
  filter->allocation_type = kTfLiteArenaRw;
  filter->dims = TfLiteIntArrayCreate(4);
  filter->dims->data[0] = 3;
  filter->dims->data[1] = 4;
  filter->dims->data[2] = 5;
  filter->dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {4.6566129e-10, 0};
  filter->params = filter_quant;
  filter->quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  filter_params->scale = TfLiteFloatArrayCreate(1);
  int32_t two_pow_neg_31 = 0x30000000;  // 2^-31 so shift = -30.
  filter_params->scale->data[0] = *reinterpret_cast<float*>(&two_pow_neg_31);
  filter_params->zero_point = TfLiteIntArrayCreate(1);
  filter_params->zero_point->data[0] = 0;
  filter_params->quantized_dimension = 0;
  filter->quantization.params = reinterpret_cast<void*>(filter_params);

  // Create bias.
  auto bias = BuildTfLiteTensor();
  bias->type = kTfLiteInt32;
  bias->allocation_type = kTfLiteArenaRw;
  bias->dims = TfLiteIntArrayCreate(4);
  TfLiteQuantizationParams bias_quant = {4.6566129e-10, 9};
  bias->params = bias_quant;
  bias->quantization.type = kTfLiteAffineQuantization;
  auto* bias_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  bias_params->scale = TfLiteFloatArrayCreate(1);
  bias_params->scale->data[0] = 4.6566129e-10;  // 2^-31
  bias_params->zero_point = TfLiteIntArrayCreate(1);
  bias_params->zero_point->data[0] = 11;
  bias->quantization.params = reinterpret_cast<void*>(bias_params);

  // Create output.
  auto output = BuildTfLiteTensor();
  output->type = kTfLiteUInt8;
  output->allocation_type = kTfLiteArenaRw;
  output->dims = nullptr;
  TfLiteQuantizationParams output_quant = {1, -128};
  output->params = output_quant;
  output->quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output->quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(kTfLiteOk,
            PopulateConvolutionQuantizationParams(
                &context_, input.get(), filter.get(), bias.get(), output.get(),
                kTfLiteActRelu, &multiplier, &shift, &output_activation_min,
                &output_activation_max, per_channel_multiplier.data(),
                per_channel_shift.data(), 3));
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-30, -30, -30));
}

TEST_F(QuantizationParamsTest, CheckAndPopulateWithoutBias) {
  // Create input.
  auto input = BuildTfLiteTensor();
  input->type = kTfLiteUInt8;
  input->allocation_type = kTfLiteArenaRw;
  input->dims = TfLiteIntArrayCreate(1);
  input->dims->data[0] = 2;
  TfLiteQuantizationParams input_quant = {1, 5};
  input->params = input_quant;
  input->quantization.type = kTfLiteAffineQuantization;
  auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  input_params->scale = TfLiteFloatArrayCreate(1);
  input_params->scale->data[0] = 1;
  input_params->zero_point = TfLiteIntArrayCreate(1);
  input_params->zero_point->data[0] = 5;
  input->quantization.params = reinterpret_cast<void*>(input_params);

  // Create filter.
  auto filter = BuildTfLiteTensor();
  filter->type = kTfLiteUInt8;
  filter->allocation_type = kTfLiteArenaRw;
  filter->dims = TfLiteIntArrayCreate(4);
  filter->dims->data[0] = 3;
  filter->dims->data[1] = 4;
  filter->dims->data[2] = 5;
  filter->dims->data[3] = 6;
  TfLiteQuantizationParams filter_quant = {4.6566129e-10, 0};
  filter->params = filter_quant;
  filter->quantization.type = kTfLiteAffineQuantization;
  auto* filter_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  filter_params->scale = TfLiteFloatArrayCreate(1);
  int32_t two_pow_neg_31 = 0x30000000;  // 2^-31 so shift = -30.
  filter_params->scale->data[0] = *reinterpret_cast<float*>(&two_pow_neg_31);
  filter_params->zero_point = TfLiteIntArrayCreate(1);
  filter_params->zero_point->data[0] = 0;
  filter_params->quantized_dimension = 0;
  filter->quantization.params = reinterpret_cast<void*>(filter_params);

  // Create output.
  auto output = BuildTfLiteTensor();
  output->type = kTfLiteUInt8;
  output->allocation_type = kTfLiteArenaRw;
  output->dims = nullptr;
  TfLiteQuantizationParams output_quant = {1, -128};
  output->params = output_quant;
  output->quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output->quantization.params = reinterpret_cast<void*>(output_params);

  // Create call parameters.
  int32_t multiplier;
  int shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  std::vector<int32_t> per_channel_multiplier(3);
  std::vector<int> per_channel_shift(3);

  // Call and verify results for per channel case.
  EXPECT_EQ(kTfLiteOk,
            PopulateConvolutionQuantizationParams(
                &context_, input.get(), filter.get(), nullptr, output.get(),
                kTfLiteActRelu, &multiplier, &shift, &output_activation_min,
                &output_activation_max, per_channel_multiplier.data(),
                per_channel_shift.data(), 3));
  EXPECT_THAT(per_channel_multiplier,
              ElementsAre(1073741824, 1073741824, 1073741824));
  EXPECT_THAT(per_channel_shift, ElementsAre(-30, -30, -30));
}

TEST_F(QuantizationParamsTest, ActivationRangeQuantizedOverflow) {
  // Create output.
  auto output = BuildTfLiteTensor();
  output->type = kTfLiteUInt8;
  output->allocation_type = kTfLiteArenaRw;
  output->dims = nullptr;
  TfLiteQuantizationParams output_quant = {1e-10, -128};
  output->params = output_quant;
  output->quantization.type = kTfLiteAffineQuantization;
  auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  output_params->scale = TfLiteFloatArrayCreate(1);
  output_params->scale->data[0] = 1;
  output_params->zero_point = TfLiteIntArrayCreate(1);
  output_params->zero_point->data[0] = -128;
  output->quantization.params = reinterpret_cast<void*>(output_params);

  // For bounded activation, a too small scale value may cause overflow.
  // Make sure overflow error is handled gracefully.
  int32_t act_min, act_max;
  ASSERT_EQ(kTfLiteOk,
            CalculateActivationRangeQuantized(
                &context_, kTfLiteActRelu, output.get(), &act_min, &act_max));
  ASSERT_NE(kTfLiteOk,
            CalculateActivationRangeQuantized(
                &context_, kTfLiteActRelu6, output.get(), &act_min, &act_max));
  EXPECT_TRUE(absl::StrContains(
      context_.error, "no_integer_overflow_from_quantization was not true"));
  ASSERT_NE(kTfLiteOk, CalculateActivationRangeQuantized(
                           &context_, kTfLiteActReluN1To1, output.get(),
                           &act_min, &act_max));
  EXPECT_TRUE(absl::StrContains(
      context_.error, "no_integer_overflow_from_quantization was not true"));
}

TEST_F(QuantizationParamsTest, IsMobilePlatform) {
  // Note: This isn't meant to be exhaustive, as that would require replicating
  // the method's implementation, but it is a basic smoke check.
#if defined(__ANDROID__)
  EXPECT_TRUE(IsMobilePlatform());
#elif defined(__linux__)
  EXPECT_FALSE(IsMobilePlatform());
#elif defined(_WIN32)
  EXPECT_FALSE(IsMobilePlatform());
#endif
}

TEST(HasUnspecifiedDimensions, ReturnsTrueIfADimIsMinusOne) {
  auto tensor = BuildTfLiteTensor(kTfLiteInt32, {1, 1, 3}, kTfLiteDynamic);
  tensor->dims_signature = ConvertVectorToTfLiteIntArray({1, -1, 3});
  EXPECT_TRUE(HasUnspecifiedDimension(tensor.get()));
}

TEST(HasUnspecifiedDimensions, ReturnsFalseIfAllPostiveDims) {
  auto tensor = BuildTfLiteTensor(kTfLiteInt32, {1, 1, 3}, kTfLiteDynamic);
  tensor->dims_signature = ConvertVectorToTfLiteIntArray({1, 1, 3});
  EXPECT_FALSE(HasUnspecifiedDimension(tensor.get()));
}

// Sets up a TFLite context and default values to initialize/resize test
// tensors.
class SetTensorAllocationTypeTest : public testing::Test {
 public:
  SetTensorAllocationTypeTest() {
    tensor_->type = kTfLiteInt32;
    tensor_->allocation_type = kTfLiteDynamic;
  }

 protected:
  Interpreter interpreter_;
  TfLiteContext& context_ = *interpreter_.primary_subgraph().context();
  IntArrayUniquePtr dims_ = BuildTfLiteArray({2, 3, 4});
  TensorUniquePtr tensor_ = BuildTfLiteTensor();
};

TEST_F(SetTensorAllocationTypeTest,
       SetUnallocatedDynamicTensorToDynamicIsANoop) {
  tensor_->allocation_type = kTfLiteDynamic;
  SetTensorToDynamic(tensor_.get());
  EXPECT_EQ(tensor_->data.data, nullptr);
  EXPECT_EQ(tensor_->allocation_type, kTfLiteDynamic);
}

TEST_F(SetTensorAllocationTypeTest, SetAllocatedDynamicTensorToDynamicIsANoop) {
  tensor_->allocation_type = kTfLiteDynamic;
  ASSERT_EQ(context_.ResizeTensor(&context_, tensor_.get(), dims_.release()),
            kTfLiteOk);
  const void* const original_data = tensor_->data.data;
  SetTensorToDynamic(tensor_.get());
  EXPECT_EQ(tensor_->data.data, original_data);
  EXPECT_EQ(tensor_->allocation_type, kTfLiteDynamic);
}

TEST_F(SetTensorAllocationTypeTest,
       SetAllocatedPersistentRoTensorToDynamicFreesExistingTensorData) {
  tensor_->allocation_type = kTfLitePersistentRo;
  ASSERT_EQ(context_.ResizeTensor(&context_, tensor_.get(), dims_.release()),
            kTfLiteOk);

  // Leak checker will raise an error if data is not freed.
  SetTensorToDynamic(tensor_.get());
  EXPECT_EQ(tensor_->data.data, nullptr);
  EXPECT_EQ(tensor_->allocation_type, kTfLiteDynamic);
}

TEST_F(SetTensorAllocationTypeTest,
       SetUnallocatedPersistentRoTensorToPersistentRoIsANoop) {
  tensor_->allocation_type = kTfLitePersistentRo;
  SetTensorToPersistentRo(tensor_.get());
  EXPECT_EQ(tensor_->data.data, nullptr);
  EXPECT_EQ(tensor_->allocation_type, kTfLitePersistentRo);
}

TEST_F(SetTensorAllocationTypeTest,
       SetAllocatedPersistentRoTensorToPersistentRoIsANoop) {
  tensor_->allocation_type = kTfLitePersistentRo;
  ASSERT_EQ(context_.ResizeTensor(&context_, tensor_.get(), dims_.release()),
            kTfLiteOk);
  const void* const original_data = tensor_->data.data;
  SetTensorToPersistentRo(tensor_.get());
  EXPECT_EQ(tensor_->data.data, original_data);
  EXPECT_EQ(tensor_->allocation_type, kTfLitePersistentRo);
}

TEST_F(SetTensorAllocationTypeTest,
       SetAllocatedDynamicTensorToPersistentRoFreesExistingTensorData) {
  tensor_->allocation_type = kTfLiteDynamic;
  ASSERT_EQ(context_.ResizeTensor(&context_, tensor_.get(), dims_.release()),
            kTfLiteOk);

  // Leak checker will raise an error if data is not freed.
  SetTensorToPersistentRo(tensor_.get());
  EXPECT_EQ(tensor_->data.data, nullptr);
  EXPECT_EQ(tensor_->allocation_type, kTfLitePersistentRo);
}

}  // namespace
}  // namespace tflite
