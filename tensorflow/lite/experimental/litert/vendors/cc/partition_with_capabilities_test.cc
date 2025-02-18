// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Utility types for mapping LiteRt IR to arbitrary backend specific
// types. Implementations of these types define mapping for ops and tensors
// that may be used in a stndalone fashion. They also may be composed
// to create lowerings of entire graphs with topology.

#include "tensorflow/lite/experimental/litert/vendors/cc/partition_with_capabilities.h"

#include <array>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_graph.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_conversion_impl.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_ir.h"

namespace litert {
namespace {

using ::litert::example::ExampleLegalizeAdd;
using ::litert::example::ExampleLegalizeMul;
using ::litert::example::ExampleOpAllocator;
using ::litert::example::ExampleOpType;
using ::litert::example::ExampleTensorAllocator;
using ::litert::example::ExampleTypes;
using ::litert::example::MakeTensorConverter;

bool ExampleCapability(const ExampleTypes::Op* op) {
  return op->op_code == ExampleOpType::ADD ||
         op->op_code == ExampleOpType::RELU;
}

TEST(PartitionWithCapabilitiesTest, EmptyGraph) {
  ExampleTypes::Legalizations legalizations;
  legalizations.push_back(ExampleLegalizeAdd::Make());

  LiteRtSubgraphT subgraph;
  Subgraph litert_subgraph(&subgraph);

  ExampleTensorAllocator tensor_alloc;
  ExampleOpAllocator op_alloc;

  auto ops = PartitionWithCapabilities<ExampleTypes>(
      legalizations, ExampleCapability, MakeTensorConverter, tensor_alloc,
      op_alloc, litert_subgraph);
  ASSERT_TRUE(ops);
  EXPECT_TRUE(ops->empty());
}

TEST(PartitionWithCapabilitiesTest, SingleSelectedOp) {
  static constexpr std::array kDims = {2, 2};

  ExampleTypes::Legalizations legalizations;
  legalizations.push_back(ExampleLegalizeAdd::Make());

  LiteRtSubgraphT subgraph;

  const auto type = MakeRankedTensorType(kLiteRtElementTypeFloat32, kDims);

  auto& input1 = subgraph.EmplaceTensor();
  input1.SetType(type);

  auto& input2 = subgraph.EmplaceTensor();
  input2.SetType(type);

  auto& output = subgraph.EmplaceTensor();
  output.SetType(type);

  auto& op = subgraph.EmplaceOp();
  op.SetOpCode(kLiteRtOpCodeTflAdd);

  internal::AttachInput(&input1, op);
  internal::AttachInput(&input2, op);
  internal::AttachOutput(&output, op);

  Subgraph litert_subgraph(&subgraph);

  ExampleTensorAllocator tensor_alloc;
  ExampleOpAllocator op_alloc;

  auto ops = PartitionWithCapabilities<ExampleTypes>(
      legalizations, ExampleCapability, MakeTensorConverter, tensor_alloc,
      op_alloc, litert_subgraph);

  ASSERT_TRUE(ops);
  EXPECT_EQ(ops->size(), 1);
}

TEST(PartitionWithCapabilitiesTest, MultiSelectedOp) {
  static constexpr std::array kDims = {2, 2};

  ExampleTypes::Legalizations legalizations;
  legalizations.push_back(ExampleLegalizeAdd::Make());

  LiteRtSubgraphT subgraph;

  const auto type = MakeRankedTensorType(kLiteRtElementTypeFloat32, kDims);

  auto& add1_input = subgraph.EmplaceTensor();
  add1_input.SetType(type);
  auto& add1_output = subgraph.EmplaceTensor();
  add1_output.SetType(type);
  auto& add1 = subgraph.EmplaceOp();
  add1.SetOpCode(kLiteRtOpCodeTflAdd);

  internal::AttachInput(&add1_input, add1);
  internal::AttachInput(&add1_input, add1);
  internal::AttachOutput(&add1_output, add1);

  auto& mul_output = subgraph.EmplaceTensor();
  mul_output.SetType(type);
  auto& mul = subgraph.EmplaceOp();
  mul.SetOpCode(kLiteRtOpCodeTflMul);

  internal::AttachInput(&add1_output, mul);
  internal::AttachOutput(&mul_output, mul);

  auto& add2_output = subgraph.EmplaceTensor();
  add2_output.SetType(type);
  auto& add2 = subgraph.EmplaceOp();
  add2.SetOpCode(kLiteRtOpCodeTflAdd);

  internal::AttachInput(&mul_output, add2);
  internal::AttachInput(&mul_output, add2);
  internal::AttachOutput(&add2_output, add2);

  Subgraph litert_subgraph(&subgraph);

  ExampleTensorAllocator tensor_alloc;
  ExampleOpAllocator op_alloc;

  auto ops = PartitionWithCapabilities<ExampleTypes>(
      legalizations, ExampleCapability, MakeTensorConverter, tensor_alloc,
      op_alloc, litert_subgraph);

  ASSERT_TRUE(ops);

  ASSERT_EQ(ops->size(), 2);
  EXPECT_EQ(ops->front(), &add1);
  EXPECT_EQ(ops->back(), &add2);
}

TEST(PartitionWithCapabilitiesTest, WithGeneralResult) {
  static constexpr std::array kDims = {2, 2};

  ExampleTypes::Legalizations legalizations;
  legalizations.push_back(ExampleLegalizeAdd::Make());

  LiteRtSubgraphT subgraph;

  const auto type = MakeRankedTensorType(kLiteRtElementTypeFloat32, kDims);

  auto& add1_input = subgraph.EmplaceTensor();
  add1_input.SetType(type);
  auto& add1_output = subgraph.EmplaceTensor();
  add1_output.SetType(type);
  auto& add1 = subgraph.EmplaceOp();
  add1.SetOpCode(kLiteRtOpCodeTflAdd);

  internal::AttachInput(&add1_input, add1);
  internal::AttachInput(&add1_input, add1);
  internal::AttachOutput(&add1_output, add1);

  tflite::AddOptionsT add_opts;
  add_opts.fused_activation_function = tflite::ActivationFunctionType_RELU;
  internal::TflOptions tfl_opts;
  tfl_opts.Set(std::move(add_opts));
  detail::SetTflOptions(add1, std::move(tfl_opts));

  Subgraph litert_subgraph(&subgraph);

  ExampleTensorAllocator tensor_alloc;
  ExampleOpAllocator op_alloc;

  auto ops = PartitionWithCapabilities<ExampleTypes>(
      legalizations, ExampleCapability, MakeTensorConverter, tensor_alloc,
      op_alloc, litert_subgraph);

  ASSERT_TRUE(ops);

  ASSERT_EQ(ops->size(), 1);
  EXPECT_EQ(ops->front(), &add1);
}

}  // namespace

}  // namespace litert
