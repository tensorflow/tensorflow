/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/fused_eigen_output_kernels.h"

#include <string>

#include "absl/strings/str_join.h"

namespace tensorflow {

Status InitializeFusedComputation(
    OpKernelConstruction* context, const std::string& kernel_name,
    const std::vector<FusedComputationPattern>& patterns,
    FusedComputationType* fused_computation,
    FusedComputationArgs* fused_computation_args) {
  // 'fused_ops' and 'num_args' attributes are specified by the Grappler
  // Remapper optimizer (see grappler/optimizers/remapper.cc).

  std::vector<std::string> fused_ops;
  TF_RETURN_IF_ERROR(context->GetAttr("fused_ops", &fused_ops));
  if (fused_ops.empty()) {
    return errors::InvalidArgument("Fused ", kernel_name,
                                   " must have at least one fused op.");
  }

  int num_args;
  TF_RETURN_IF_ERROR(context->GetAttr("num_args", &num_args));
  int num_host_args;
  if (!context->GetAttr("num_host_args", &num_host_args).ok()) {
    num_host_args = 0;  // default value
  }
  int num_inputs = context->num_inputs();
  if (num_inputs != 2 + num_args + num_host_args) {
    return errors::InvalidArgument(
        "Fused ", kernel_name,
        " must have the number of inputs equal to 2 + num_args + num_host_args "
        "but in fact the number of inputs is ",
        num_inputs, " and num_args is ", num_args, " and num_host_args is ",
        num_host_args);
  }

  // TODO(ezhulenev): Add support for fusion element-wise op chains defined
  // at runtime, e.g. Relu+Sqrt+Tanh+etc.

  // Reset fused computation type.
  *fused_computation = FusedComputationType::kUndefined;

  // Match op fusion to one of the supported patterns.
  for (const auto& pattern : patterns) {
    if (fused_ops == pattern.fused_ops) {
      *fused_computation = pattern.fused_computation;
      break;
    }
  }
  if (*fused_computation == FusedComputationType::kUndefined) {
    return errors::Unimplemented("Fusion is not implemented: [",
                                 absl::StrJoin(fused_ops, ","), "]");
  }

  // Depending on a picked fusion type validate fusion-specific arguments.
  if (*fused_computation == FusedComputationType::kBiasAdd ||
      *fused_computation == FusedComputationType::kBiasAddWithRelu ||
      *fused_computation == FusedComputationType::kBiasAddWithRelu6 ||
      *fused_computation == FusedComputationType::kBiasAddWithTanh ||
      *fused_computation == FusedComputationType::kBiasAddWithSigmoid ||
      *fused_computation == FusedComputationType::kBiasAddWithElu ||
      *fused_computation == FusedComputationType::kBiasAddWithLeakyRelu ||
      *fused_computation == FusedComputationType::kBiasAddWithGeluApproximate ||
      *fused_computation == FusedComputationType::kBiasAddWithGeluExact) {
    if (num_args != 1 && !(num_args == 2 && num_host_args == 2)) {
      return errors::InvalidArgument(
          "Fused ", kernel_name,
          " with BiasAdd must have one extra argument: bias"
          " or 4 extra arguments: bias, side_input, conv_input_scale and "
          "side_input_scale");
    }
    constexpr int kConvInput = 0;
    constexpr int kFilter = 1;
    constexpr int kBias = 2;
    constexpr int kSideInput = 3;
    if (context->input_type(kConvInput) == DT_INT8) {
      if (num_inputs != 6) {
        return errors::InvalidArgument("Fused ", kernel_name,
                                       " for int8 must have 6 inputs and ",
                                       num_inputs, " is provided");
      }
      if (context->input_type(kFilter) != DT_INT8) {
        return errors::InvalidArgument("Fused ", kernel_name,
                                       " for int8 has filter type ",
                                       context->input_type(kFilter),
                                       " that does not match the input fype ",
                                       context->input_type(kConvInput));
      }
      if (context->input_type(kBias) != DT_FLOAT) {
        return errors::InvalidArgument(
            "Fused ", kernel_name, " for int8 has bias type ",
            context->input_type(kBias), " that must have the float type");
      }
      if (context->input_type(kSideInput) != DT_INT8) {
        return errors::InvalidArgument("Fused ", kernel_name,
                                       " for int8 has side_input type ",
                                       context->input_type(kSideInput),
                                       " that does not match the input fype ",
                                       context->input_type(kConvInput));
      }
    }
    if (*fused_computation == FusedComputationType::kBiasAddWithLeakyRelu) {
      TF_RETURN_IF_ERROR(context->GetAttr(
          "leakyrelu_alpha", &fused_computation_args->leakyrelu_alpha));
    }
  }

  if (*fused_computation == FusedComputationType::kFusedBatchNorm ||
      *fused_computation == FusedComputationType::kFusedBatchNormWithRelu ||
      *fused_computation == FusedComputationType::kFusedBatchNormWithRelu6 ||
      *fused_computation == FusedComputationType::kFusedBatchNormWithElu ||
      *fused_computation ==
          FusedComputationType::kFusedBatchNormWithLeakyRelu) {
    if (num_args != 4) {
      return errors::InvalidArgument(
          "Fused ", kernel_name,
          " with FusedBatchNorm must have four extra arguments: scale, offset, "
          "mean, variance.");
    }
    TF_RETURN_IF_ERROR(
        context->GetAttr("epsilon", &fused_computation_args->epsilon));
    if (*fused_computation ==
        FusedComputationType::kFusedBatchNormWithLeakyRelu) {
      TF_RETURN_IF_ERROR(context->GetAttr(
          "leakyrelu_alpha", &fused_computation_args->leakyrelu_alpha));
    }
  }

  return OkStatus();
}

}  // namespace tensorflow
