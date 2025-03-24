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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_EXAMPLES_EXAMPLE_CONVERSION_IMPL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_EXAMPLES_EXAMPLE_CONVERSION_IMPL_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/log/absl_check.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/c/litert_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/conversion.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/ir_types.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_ir.h"

namespace litert::example {

// Conversion type implementations for the fictional "example" backend.

ExampleTypes::TensorConverter MakeTensorConverter(
    ExampleTypes::TensorAllocator alloc);

static constexpr absl::string_view kIntermediateTensorName =
    "intermediate_bin_output";

// Example legalization for simple binary ops.
template <ExampleOpType BackendOpType, LiteRtOpCode LiteRtOpType>
class ExampleBinOpLegalization : public Legalization<ExampleOp, ExampleTensor> {
 private:
  using Self = ExampleBinOpLegalization<BackendOpType, LiteRtOpType>;

 public:
  using Ptr = std::unique_ptr<Self>;

  static Ptr Make() { return std::make_unique<Self>(); }

  // Return the litert op code to match on.
  LiteRtOpCode OpToMatch() const override { return LiteRtOpType; }

  // Determines if the given litert op has a fused relu attribute.
  bool HasFusedRelu(const Op& litert_op) const {
    if constexpr (LiteRtOpType != kLiteRtOpCodeTflAdd) {
      return false;
    }
    uint32_t faf;
    if (LiteRtGetAddFusedActivationOption(litert_op.Get(), &faf) !=
        kLiteRtStatusOk) {
      return false;
    }
    return faf == 1;
  }

  // Transforms LiteRtAdd op into example op definition using the tensor
  // converter to map tensors within.
  ExampleTypes::ConversionResult LegalizeImpl(
      const Op& litert_op, const Tensors& inputs, const Tensors& outputs,
      ExampleTypes::TensorAllocator tensor_allocator,
      ExampleTypes::OpAllocator op_allocator) const override {
    ABSL_DCHECK_EQ(litert_op.Code(), LiteRtOpType);

    auto& bin_op = *op_allocator();
    bin_op.op_code = BackendOpType;

    if (inputs.size() != 2 || outputs.size() != 1) {
      return Error(kLiteRtStatusErrorInvalidArgument);
    }

    for (const auto* input : inputs) {
      bin_op.inputs.push_back(input->id);
      bin_op.input_names.push_back(input->name);
    }

    auto& output_tensor = *outputs.front();
    if (!HasFusedRelu(litert_op)) {
      bin_op.outputs.push_back(output_tensor.id);
      bin_op.output_names.push_back(output_tensor.name);
      return Expected<Result>(&bin_op);
    }

    auto* bin_output = tensor_allocator();
    bin_output->dims = output_tensor.dims;
    bin_output->type = output_tensor.type;
    bin_output->name = std::string(kIntermediateTensorName);
    bin_op.outputs.push_back(bin_output->id);
    bin_op.output_names.push_back(bin_output->name);

    auto& relu = *op_allocator();
    relu.op_code = ExampleOpType::RELU;
    relu.inputs.push_back(bin_output->id);
    relu.input_names.push_back(bin_output->name);
    relu.outputs.push_back(output_tensor.id);
    relu.output_names.push_back(output_tensor.name);

    ExampleTypes::GeneralConversionResult result;
    result.ops.push_back(&bin_op);
    result.ops.push_back(&relu);
    result.intermediate_tensors.push_back(bin_output);

    return ExampleTypes::ConversionResult(result);
  }
};

using ExampleLegalizeAdd =
    ExampleBinOpLegalization<ExampleOpType::ADD, kLiteRtOpCodeTflAdd>;
using ExampleLegalizeMul =
    ExampleBinOpLegalization<ExampleOpType::MUL, kLiteRtOpCodeTflMul>;

ExampleTypes::Legalizations MakeAllLegalizations();

}  // namespace litert::example

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_EXAMPLES_EXAMPLE_CONVERSION_IMPL_H_
