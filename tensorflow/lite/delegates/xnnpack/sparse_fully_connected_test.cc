/* Copyright 2026 Google LLC.

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

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {
namespace {

// A float FULLY_CONNECTED whose 2x4 filter is stored in CSR form, with its
// operator code at `version`.
std::vector<char> SparseFullyConnectedModel(int version) {
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<float> filter_values = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<flatbuffers::Offset<Buffer>> buffers = {
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(filter_values.data()),
                       sizeof(float) * filter_values.size()))};

  const std::vector<int32_t> segments = {0, 4, 8};
  const std::vector<int32_t> indices = {0, 1, 2, 3, 0, 1, 2, 3};
  const std::vector<flatbuffers::Offset<DimensionMetadata>> dim_metadata = {
      CreateDimensionMetadata(builder, DimensionType_DENSE, /*dense_size=*/2),
      CreateDimensionMetadata(
          builder, DimensionType_SPARSE_CSR, /*dense_size=*/0,
          SparseIndexVector_Int32Vector,
          CreateInt32Vector(builder, builder.CreateVector(segments)).Union(),
          SparseIndexVector_Int32Vector,
          CreateInt32Vector(builder, builder.CreateVector(indices)).Union())};
  const std::vector<int32_t> traversal_order = {0, 1};
  const flatbuffers::Offset<SparsityParameters> sparsity =
      CreateSparsityParameters(builder, builder.CreateVector(traversal_order),
                               /*block_map=*/0,
                               builder.CreateVector(dim_metadata));

  const std::vector<int32_t> input_shape = {1, 4};
  const std::vector<int32_t> filter_shape = {2, 4};
  const std::vector<int32_t> output_shape = {1, 2};
  const std::vector<flatbuffers::Offset<Tensor>> tensors = {
      CreateTensor(builder, builder.CreateVector(input_shape),
                   TensorType_FLOAT32),
      CreateTensor(builder, builder.CreateVector(filter_shape),
                   TensorType_FLOAT32, /*buffer=*/1, /*name=*/0,
                   /*quantization=*/0, /*is_variable=*/false, sparsity),
      CreateTensor(builder, builder.CreateVector(output_shape),
                   TensorType_FLOAT32)};

  const std::vector<int32_t> op_inputs = {0, 1};
  const std::vector<int32_t> op_outputs = {2};
  const flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0, builder.CreateVector(op_inputs),
      builder.CreateVector(op_outputs), BuiltinOptions_FullyConnectedOptions,
      CreateFullyConnectedOptions(builder).Union());
  const flatbuffers::Offset<OperatorCode> operator_code = CreateOperatorCode(
      builder, BuiltinOperator_FULLY_CONNECTED, /*custom_code=*/0, version);

  const std::vector<int32_t> subgraph_inputs = {0};
  const std::vector<int32_t> subgraph_outputs = {2};
  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors),
      builder.CreateVector(subgraph_inputs),
      builder.CreateVector(subgraph_outputs), builder.CreateVector(&op, 1));
  builder.Finish(CreateModel(builder, TFLITE_SCHEMA_VERSION,
                             builder.CreateVector(&operator_code, 1),
                             builder.CreateVector(&subgraph, 1),
                             builder.CreateString("sparse fully connected"),
                             builder.CreateVector(buffers)));
  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

// XNNPACK has no sparse fully connected operator, so a node with sparse
// weights stays on the builtin kernel. Its version says nothing about that:
// all FULLY_CONNECTED nodes of a model share one version, the highest any of
// them needs, so a sparse node can carry any version from 8 up.
TEST(SparseFullyConnected, StaysOnTheBuiltinKernelAtEveryVersion) {
  for (int version : {8, 9, 10}) {
    SCOPED_TRACE(version);
    const std::vector<char> buffer = SparseFullyConnectedModel(version);
    std::unique_ptr<Interpreter> interpreter;
    ASSERT_EQ(InterpreterBuilder(
                  GetModel(buffer.data()),
                  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
                  &interpreter),
              kTfLiteOk);
    std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
        delegate(TfLiteXNNPackDelegateCreate(nullptr),
                 TfLiteXNNPackDelegateDelete);
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);

    ASSERT_EQ(interpreter->execution_plan().size(), 1);
    const TfLiteRegistration& registration =
        interpreter->node_and_registration(interpreter->execution_plan()[0])
            ->second;
    EXPECT_EQ(registration.builtin_code, BuiltinOperator_FULLY_CONNECTED);
  }
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite
