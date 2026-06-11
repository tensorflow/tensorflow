/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/xnnpack/odml_sdpa_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "flatbuffers/flexbuffers.h"
#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/string.h"  // from @flatbuffers
#include "flatbuffers/util.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/schema/schema_conversion_utils.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

std::vector<int32_t> ODMLSDPATester::OutputShape() const {
  std::vector<int32_t> output_shape = QueryShape();
  return output_shape;
}

namespace {

void ComputeReferenceSDPA(
    const float* query, const tflite::RuntimeShape& query_shape,
    const float* key, const tflite::RuntimeShape& key_shape, const float* value,
    const tflite::RuntimeShape& value_shape, const float* mask,
    const tflite::RuntimeShape& mask_shape, float scale, float logit_cap,
    float* output) {
  const int32_t B = query_shape.Dims(0);
  const int32_t T = query_shape.Dims(1);
  const int32_t N_q = query_shape.Dims(2);
  const int32_t H = query_shape.Dims(3);

  const int32_t S = key_shape.Dims(1);
  const int32_t N_kv = key_shape.Dims(2);

  const int32_t group_size = N_q / N_kv;

  for (int32_t b = 0; b < B; ++b) {
    for (int32_t hq = 0; hq < N_q; ++hq) {
      const int32_t hkv = hq / group_size;
      for (int32_t t = 0; t < T; ++t) {
        std::vector<float> logits(S);
        for (int32_t s = 0; s < S; ++s) {
          float sum = 0.0f;
          for (int32_t h = 0; h < H; ++h) {
            int q_idx = b * (T * N_q * H) + t * (N_q * H) + hq * H + h;
            int k_idx = b * (S * N_kv * H) + s * (N_kv * H) + hkv * H + h;
            sum += query[q_idx] * key[k_idx];
          }
          logits[s] = sum * scale;

          if (logit_cap > 0.0f) {
            logits[s] = std::tanh(logits[s] / logit_cap) * logit_cap;
          }

          if (mask != nullptr) {
            int mb = mask_shape.Dims(0) == 1 ? 0 : b;
            int mn = mask_shape.Dims(1) == 1 ? 0 : hq;
            int mt = mask_shape.Dims(2) == 1 ? 0 : t;
            int ms = mask_shape.Dims(3) == 1 ? 0 : s;

            int mask_idx = mb * (mask_shape.Dims(1) * mask_shape.Dims(2) *
                                 mask_shape.Dims(3)) +
                           mn * (mask_shape.Dims(2) * mask_shape.Dims(3)) +
                           mt * mask_shape.Dims(3) + ms;
            logits[s] += mask[mask_idx];
          }
        }

        float max_val = -std::numeric_limits<float>::infinity();
        for (float val : logits) {
          max_val = std::max(max_val, val);
        }
        float sum = 0.0f;
        for (int32_t s = 0; s < S; ++s) {
          logits[s] = std::exp(logits[s] - max_val);
          sum += logits[s];
        }
        for (int32_t s = 0; s < S; ++s) {
          logits[s] /= sum;
        }

        for (int32_t h = 0; h < H; ++h) {
          float sum_v = 0.0f;
          for (int32_t s = 0; s < S; ++s) {
            int v_idx = b * (S * N_kv * H) + s * (N_kv * H) + hkv * H + h;
            sum_v += logits[s] * value[v_idx];
          }
          int out_idx = b * (T * N_q * H) + t * (N_q * H) + hq * H + h;
          output[out_idx] = sum_v;
        }
      }
    }
  }
}

struct OpData {
  float scale;
  float logit_cap;
};

void* SDPAInit(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  op_data->scale = 0.0f;
  op_data->logit_cap = 0.0f;
  if (buffer != nullptr && length > 0) {
    auto flexbuffer_map =
        flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length)
            .AsMap();
    op_data->scale = flexbuffer_map["scale"].AsFloat();
    op_data->logit_cap = flexbuffer_map["logit_cap"].AsFloat();
  }
  return op_data;
}

void SDPAFree(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus SDPAPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, NumInputs(node) == 3 || NumInputs(node) == 4);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* query;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &query));
  const TfLiteTensor* key;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &key));
  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &value));

  TF_LITE_ENSURE_EQ(context, query->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, key->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, value->type, kTfLiteFloat32);

  if (op_data->scale == 0.0f) {
    int32_t head_dim = query->dims->data[query->dims->size - 1];
    op_data->scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  output->type = kTfLiteFloat32;

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(query->dims));
}

TfLiteStatus SDPAInvoke(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* query;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &query));
  const TfLiteTensor* key;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &key));
  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &value));

  const TfLiteTensor* mask = nullptr;
  if (NumInputs(node) > 3) {
    mask = GetInput(context, node, 3);
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

  ComputeReferenceSDPA(
      tflite::GetTensorData<float>(query), tflite::GetTensorShape(query),
      tflite::GetTensorData<float>(key), tflite::GetTensorShape(key),
      tflite::GetTensorData<float>(value), tflite::GetTensorShape(value),
      mask ? tflite::GetTensorData<float>(mask) : nullptr,
      mask ? tflite::GetTensorShape(mask) : tflite::RuntimeShape(),
      op_data->scale, op_data->logit_cap, tflite::GetTensorData<float>(output));

  return kTfLiteOk;
}

TfLiteRegistration* Register_SDPA_Reference() {
  static TfLiteRegistration r = {
      SDPAInit,
      SDPAFree,
      SDPAPrepare,
      SDPAInvoke,
  };
  return &r;
}

}  // namespace

void ODMLSDPATester::Test(TfLiteDelegate* delegate) const {
  // Test for SDPA XNNPACK delegate vs TfLite Reference SDPA op.
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<char> buffer = CreateTfLiteModel();
  const Model* model = GetModel(buffer.data());

  std::unique_ptr<Interpreter> delegate_interpreter;
  auto resolver =
      ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates();
  resolver.AddCustom("odml.scaled_dot_product_attention",
                     Register_SDPA_Reference());
  ASSERT_EQ(InterpreterBuilder(model, resolver)(&delegate_interpreter),
            kTfLiteOk);
  std::unique_ptr<Interpreter> default_interpreter;
  ASSERT_EQ(InterpreterBuilder(model, resolver)(&default_interpreter),
            kTfLiteOk);

  ASSERT_TRUE(delegate_interpreter);
  ASSERT_TRUE(default_interpreter);

  ASSERT_EQ(delegate_interpreter->inputs().size(), 4);
  ASSERT_EQ(default_interpreter->inputs().size(), 4);

  ASSERT_EQ(delegate_interpreter->outputs().size(), 1);
  ASSERT_EQ(default_interpreter->outputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  for (size_t i = 0; i < delegate_interpreter->inputs().size(); ++i) {
    const TfLiteTensor* delegate_input_tensor = delegate_interpreter->tensor(i);
    const size_t num_elts = NumElements(delegate_input_tensor);
    float* const delegate_input_data =
        delegate_interpreter->typed_input_tensor<float>(i);
    float* const default_input_data =
        default_interpreter->typed_input_tensor<float>(i);
    std::generate_n(delegate_input_data, num_elts, std::ref(input_rng));
    std::copy_n(delegate_input_data, num_elts, default_input_data);
  }

  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);

  float* delegate_output_data =
      delegate_interpreter->typed_output_tensor<float>(0);
  float* default_output_data =
      default_interpreter->typed_output_tensor<float>(0);
  const int32_t output_size =
      NumElements(delegate_interpreter->output_tensor(0));

  for (size_t i = 0; i < output_size; i++) {
    ASSERT_NEAR(default_output_data[i], delegate_output_data[i],
                std::numeric_limits<float>::epsilon() *
                    std::max(std::abs(default_output_data[i]) * 20.0f, 1.0f));
  }
}

std::vector<char> ODMLSDPATester::CreateTfLiteModel() const {
  if (!model_name_.empty() && model_name_ != kOdmlSdpaCustom) {
    const char kTestModelFolder[] =
        "tensorflow/lite/delegates/xnnpack/";
    const std::string test_model =
        kTestModelFolder + model_name_ + ".tflite.bin";
    std::string model_data;
    if (!flatbuffers::LoadFile(test_model.c_str(), /*binary=*/true,
                               &model_data)) {
      ADD_FAILURE() << "file not loaded: " << test_model;
    }
    return std::vector<char>(model_data.begin(), model_data.end());
  } else {
    flatbuffers::FlatBufferBuilder builder;
    flatbuffers::Offset<OperatorCode> operator_code = CreateOperatorCode(
        builder, BuiltinOperator_CUSTOM,
        builder.CreateString("odml.scaled_dot_product_attention"));

    const std::array<flatbuffers::Offset<Buffer>, 1> buffers{{
        CreateBuffer(builder, builder.CreateVector({})),
    }};

    const std::array<flatbuffers::Offset<Tensor>, 5> tensors{{
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(QueryShape().data(),
                                                   QueryShape().size()),
                     TensorType_FLOAT32),
        CreateTensor(
            builder,
            builder.CreateVector<int32_t>(KeyShape().data(), KeyShape().size()),
            TensorType_FLOAT32),
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(ValueShape().data(),
                                                   ValueShape().size()),
                     TensorType_FLOAT32),
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(MaskShape().data(),
                                                   MaskShape().size()),
                     TensorType_FLOAT32),
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(OutputShape().data(),
                                                   OutputShape().size()),
                     TensorType_FLOAT32),
    }};

    auto fbb = std::make_unique<flexbuffers::Builder>();
    float scale = 1 / sqrt(QueryShape().data()[QueryShape().size() - 1]);
    fbb->Map([&]() { fbb->Float("scale", scale); });
    fbb->Finish();

    const std::array<int32_t, 4> op_inputs{{0, 1, 2, 3}};
    const std::array<int32_t, 1> op_outputs{{4}};
    flatbuffers::Offset<Operator> op = CreateOperator(
        builder, /*opcode_index=*/0,
        builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
        builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
        tflite::BuiltinOptions_NONE, 0,
        builder.CreateVector<uint8_t>(
            reinterpret_cast<const uint8_t*>(fbb->GetBuffer().data()),
            fbb->GetSize()));

    const std::array<int32_t, 4> subgraph_inputs{{0, 1, 2, 3}};
    const std::array<int32_t, 1> subgraph_outputs{{4}};
    flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
        builder, builder.CreateVector(tensors.data(), tensors.size()),
        builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                      subgraph_inputs.size()),
        builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                      subgraph_outputs.size()),
        builder.CreateVector(&op, 1));

    flatbuffers::Offset<flatbuffers::String> description =
        builder.CreateString("ODML SDPA model");

    flatbuffers::Offset<Model> model_buffer = CreateModel(
        builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
        builder.CreateVector(&subgraph, 1), description,
        builder.CreateVector(buffers.data(), buffers.size()));

    builder.Finish(model_buffer);

    return std::vector<char>(builder.GetBufferPointer(),
                             builder.GetBufferPointer() + builder.GetSize());
  }
}

int32_t ODMLSDPATester::ComputeSize(const std::vector<int32_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

}  // namespace xnnpack
}  // namespace tflite
