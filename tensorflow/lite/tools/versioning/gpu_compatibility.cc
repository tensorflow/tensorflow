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
#include "tensorflow/lite/tools/versioning/gpu_compatibility.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/tools/versioning/op_signature.h"
#include "tensorflow/lite/util.h"

namespace tflite {

namespace {

const std::string GetOpName(const OpSignature& op_sig) {
  if (op_sig.op == tflite::BuiltinOperator_CUSTOM) {
    return op_sig.custom_name;
  }
  return tflite::EnumNamesBuiltinOperator()[op_sig.op];
}

int NumElements(const std::vector<int32_t>& dims) {
  int count = 1;
  for (int i = 0; i < dims.size(); ++i) {
    count *= dims.at(i);
  }
  return count;
}

// Helper functions from
// tensorflow/lite/delegates/gpu/common/model_builder_helper.cc

#define RETURN_IF_ERROR(s) \
  {                        \
    auto c = (s);          \
    if (!c.ok()) return c; \
  }

template <typename ParamsT>
absl::Status RetrieveBuiltinData(const OpSignature& op_sig,
                                 const ParamsT** tf_options) {
  *tf_options = static_cast<const ParamsT*>(op_sig.builtin_data);
  if (!*tf_options) {
    return absl::InternalError("Unable to retrieve builtin_data.");
  }
  return absl::OkStatus();
}

template <typename ParamsT>
absl::Status RetrieveCustomInitialData(const OpSignature& op_sig,
                                       const ParamsT** tf_options) {
  *tf_options = static_cast<const ParamsT*>(op_sig.custom_initial_data);
  if (!*tf_options) {
    return absl::InternalError("Unable to retrieve custom_initial_data.");
  }
  return absl::OkStatus();
}

absl::Status IsActivationSupported(TfLiteFusedActivation fused_activation) {
  switch (fused_activation) {
    case kTfLiteActNone:
    case kTfLiteActRelu:
    case kTfLiteActReluN1To1:
    case kTfLiteActRelu6:
    case kTfLiteActTanh:
    case kTfLiteActSigmoid:
      return absl::OkStatus();
    case kTfLiteActSignBit:
      return absl::UnimplementedError(
          "TfLiteFusedActivation.kTfLiteActSignBit");

      // Do not add default; we want compilation error rather than run-time
      // error.
  }
}

// Returns the number of runtime inputs of the given OpSignature.
// runtime inputs are input tensors which are not constant or optional tensors.
int GetNumberOfRuntimeInputs(const OpSignature& op_sig) {
  int number_of_runtime_inputs = 0;
  for (auto& input : op_sig.inputs) {
    if (!input.is_const && input.type != kTfLiteNoType) {
      number_of_runtime_inputs++;
    }
  }
  return number_of_runtime_inputs;
}

// Checks if the given OpSignature has required number of inputs and outputs.
// - required_runtime_inputs: number of inputs which are not constants.
// - required_outputs: number of outputs
absl::Status CheckInputsOutputs(const OpSignature& op_sig,
                                const int required_runtime_inputs,
                                const int required_outputs) {
  const int runtime_inputs_from_model = GetNumberOfRuntimeInputs(op_sig);
  if (runtime_inputs_from_model != required_runtime_inputs) {
    return absl::InternalError(
        absl::StrCat("Expected ", required_runtime_inputs,
                     " runtime input tensor(s), but node has ",
                     runtime_inputs_from_model, " runtime input(s)."));
  }
  const int outputs_from_model = op_sig.outputs.size();
  if (outputs_from_model != required_outputs) {
    return absl::InternalError(absl::StrCat("Expected ", required_outputs,
                                            " output tensor(s), but node has ",
                                            outputs_from_model, " output(s)."));
  }
  return absl::OkStatus();
}

// Checks if the given OpSignature has required number of inputs and outputs.
// - required_runtime_inputs: number of inputs which are not constants.
// - required_const_inputs: number of inputs which are constants.
// - required_outputs: number of outputs
absl::Status CheckInputsConstsOutputs(const OpSignature& op_sig,
                                      int required_runtime_inputs,
                                      int required_const_inputs,
                                      int required_outputs) {
  int const_inputs_from_model = 0;
  for (auto& input : op_sig.inputs) {
    if (input.is_const) {
      ++const_inputs_from_model;
    }
  }
  if (const_inputs_from_model != required_const_inputs) {
    return absl::InternalError(
        absl::StrCat("Expected ", required_const_inputs,
                     " const input tensor(s), but node has ",
                     const_inputs_from_model, " const input(s)."));
  }
  return CheckInputsOutputs(op_sig, required_runtime_inputs, required_outputs);
}

absl::Status CheckTensorIsAvailable(const OpSignature& op_sig, int idx) {
  // If tensor id is in range, it's guaranteed that it'll be available.
  if (idx >= op_sig.inputs.size()) {
    return absl::OutOfRangeError(
        absl::StrCat("Requested index goes beyond array size: ", idx, " vs ",
                     op_sig.inputs.size()));
  }
  return absl::OkStatus();
}

// Checks if the given OpSignature has required number of inputs and outputs for
// convolution operators. The number of input should be either 2 runtime inputs
// or 1 runtime and 1 constant input. The number of output should be one.
absl::Status CheckConvoultionInputOutput(const OpSignature& op_sig) {
  const int runtime_inputs = GetNumberOfRuntimeInputs(op_sig);
  if (runtime_inputs > 2) {
    return absl::InternalError(
        absl::StrCat("Expected 1 or 2 input tensor(s), but node has ",
                     runtime_inputs, " runtime inputs."));
  }
  const int runtime_outputs = op_sig.outputs.size();
  if (runtime_outputs != 1) {
    return absl::InternalError(
        absl::StrCat("Expected 1 output tensor(s), but node has ",
                     runtime_outputs, " runtime outputs."));
  }
  if (runtime_inputs == 1) {
    RETURN_IF_ERROR(CheckTensorIsAvailable(op_sig, 1));
  }
  return absl::OkStatus();
}

absl::Status CheckStrides(int strides_h, int strides_w) {
  if (strides_h <= 0 || strides_w <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Incorrect stride values: stride_height = ", strides_h,
                     ", stride_width = ", strides_w));
  }
  return absl::OkStatus();
}

absl::Status CheckDilation(int dilation_h, int dilation_w) {
  if (dilation_h <= 0 || dilation_w <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Incorrect dilation values: dilation_height = ", dilation_h,
        ", dilation_width = ", dilation_w));
  }
  return absl::OkStatus();
}

absl::Status CheckStridesAndDilation(int strides_h, int strides_w,
                                     int dilation_h, int dilation_w) {
  RETURN_IF_ERROR(CheckStrides(strides_h, strides_w));
  RETURN_IF_ERROR(CheckDilation(dilation_h, dilation_w));
  return absl::OkStatus();
}

absl::Status CheckKernels(int kernel_h, int kernel_w) {
  if (kernel_h <= 0 || kernel_w <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Incorrect kernel values: kernel_height = ", kernel_h,
                     ", kernel_width = ", kernel_w));
  }
  return absl::OkStatus();
}

absl::Status CheckKernelsAndStrides(int kernel_h, int kernel_w, int strides_h,
                                    int strides_w) {
  RETURN_IF_ERROR(CheckKernels(kernel_h, kernel_w));
  RETURN_IF_ERROR(CheckStrides(strides_h, strides_w));
  return absl::OkStatus();
}

// Checks if the axes tensor at the given index is a integer32 constant tensor.
absl::Status CheckAxesAreInt32Const(const OpSignature& op_sig, int idx) {
  auto axes = op_sig.inputs.at(idx);
  if (!axes.is_const) {
    return absl::UnimplementedError(GetOpName(op_sig) +
                                    " is only supported with constant axes.");
  }
  if (axes.type != kTfLiteInt32) {
    return absl::UnimplementedError(absl::StrCat(
        GetOpName(op_sig) + " supports int32 tensor for axes. But node has ",
        TfLiteTypeGetName(axes.type)));
  }
  return absl::OkStatus();
}

absl::Status CheckPooling2DGpuDelegateCompatibility(const OpSignature& op_sig) {
  const TfLitePoolParams* tf_options;
  if (op_sig.custom_initial_data) {  // custom case with indices as a second
                                     // output
    RETURN_IF_ERROR(RetrieveCustomInitialData(op_sig, &tf_options));
    RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                       /*required_runtime_inputs=*/1,
                                       /*required_outputs=*/2));
  } else {  // common pooling with 1 output
    RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
    RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                       /*required_runtime_inputs=*/1,
                                       /*required_outputs=*/1));
  }
  RETURN_IF_ERROR(CheckKernelsAndStrides(
      tf_options->filter_height, tf_options->filter_width,
      tf_options->stride_height, tf_options->stride_width));
  return IsActivationSupported(tf_options->activation);
}

absl::Status CheckDepthwiseConvGpuDelegateCompatibility(
    const OpSignature& op_sig) {
  RETURN_IF_ERROR(CheckConvoultionInputOutput(op_sig));
  const TfLiteDepthwiseConvParams* tf_options;
  RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
  RETURN_IF_ERROR(CheckStridesAndDilation(
      tf_options->stride_height, tf_options->stride_width,
      tf_options->dilation_height_factor, tf_options->dilation_width_factor));
  RETURN_IF_ERROR(IsActivationSupported(tf_options->activation));

  const int depth_multiplier = tf_options->depth_multiplier;
  const auto* input = &op_sig.inputs[0];
  const auto* filter = &op_sig.inputs[1];
  const auto* bias = op_sig.inputs.size() > 2 ? &op_sig.inputs[2] : nullptr;
  const auto* output = &op_sig.outputs[0];
  if (input->dims.size() != 4) {
    return absl::InvalidArgumentError("input.dims.size != 4");
  }
  if (filter->dims.size() != 4) {
    return absl::InvalidArgumentError("filter.dims.size != 4");
  }
  if (output->dims.size() != 4) {
    return absl::InvalidArgumentError("output.dims.size != 4");
  }
  if (input->dims[0] != output->dims[0]) {
    return absl::InvalidArgumentError("input.b != output.b");
  }
  const int input_depth = input->dims[3];
  const int output_depth = output->dims[3];
  if (filter->dims[3] != output_depth) {
    return absl::InvalidArgumentError("filter.i != output.c");
  }
  if (output_depth != input_depth * depth_multiplier) {
    return absl::InvalidArgumentError("output.c != input.c * depth_multiplier");
  }
  if (bias && NumElements(bias->dims) != output_depth) {
    return absl::InvalidArgumentError("bias.size != output.c");
  }
  if (depth_multiplier != 1 && input_depth != 1) {
    return absl::UnimplementedError("depth_multiplier != 1 && input.c != 1");
  }
  return absl::OkStatus();
}

absl::Status CheckCumsumGpuDelegateCompatibility(const OpSignature& op_sig) {
  if (op_sig.inputs.size() != 2) {
    return absl::InvalidArgumentError("Expects 2 inputs and 1 output");
  }
  auto error = absl::InvalidArgumentError(
      "Input/output must be float type and indices must be constant int32 "
      "type");
  if ((op_sig.inputs.at(0).type != kTfLiteFloat16 &&
       op_sig.inputs.at(0).type != kTfLiteFloat32) ||
      (op_sig.outputs.at(0).type != op_sig.inputs.at(0).type) ||
      (op_sig.inputs.at(1).type != kTfLiteInt32 ||
       !op_sig.inputs.at(1).is_const)) {
    return error;
  }
  return absl::OkStatus();
}

absl::Status CheckOneHotGpuDelegateCompatibility(const OpSignature& op_sig) {
  if (op_sig.inputs.size() != 4 && op_sig.outputs.size() != 1) {
    return absl::InvalidArgumentError("Expects 4 inputs and 1 output");
  }
  // Supports int32 indices with float scalar on/off values.
  // Axis value must be -1 or last dimension.
  absl::Status error = absl::InvalidArgumentError(
      "Indices must be int32 type, on/off tensors must be constant, scalar, "
      "float type, axis must be -1 or last dim");
  if (op_sig.inputs[0].type != kTfLiteInt32) {
    return error;
  }
  auto* one_hot_options =
      reinterpret_cast<TfLiteOneHotParams*>(op_sig.builtin_data);
  const int num_dims = op_sig.inputs[0].dims.size();
  if (one_hot_options->axis != -1 &&
      one_hot_options->axis != op_sig.inputs[0].dims[num_dims - 1]) {
    return error;
  }
  // Can only have batch and channels as non-singleton.
  for (int i = 0; i < num_dims - 1; ++i) {
    if (num_dims > 3 && i == 0) {
      continue;
    }
    if (op_sig.inputs.at(0).dims[i] != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unspported non-singleton dim at ", i));
    }
  }
  // On and off value must be float, constant and scalar.
  if (op_sig.inputs.at(2).type != kTfLiteFloat32 ||
      op_sig.inputs.at(3).type != kTfLiteFloat32) {
    return error;
  }
  if (!op_sig.inputs.at(2).is_const || !op_sig.inputs.at(3).is_const ||
      op_sig.inputs.at(2).dims.size() > 1 ||
      op_sig.inputs.at(3).dims.size() > 1) {
    return error;
  }
  if ((!op_sig.inputs.at(2).dims.empty() && op_sig.inputs.at(2).dims[0] > 1) ||
      (!op_sig.inputs.at(3).dims.empty() && op_sig.inputs.at(3).dims[0] > 1)) {
    return error;
  }
  return absl::OkStatus();
}

absl::Status CheckSelectV2GpuDelegateCompatibility(const OpSignature& op_sig) {
  if (op_sig.inputs.size() != 3 || op_sig.outputs.size() != 1) {
    return absl::InvalidArgumentError("Expected 3 inputs and 1 output");
  }
  // Only supports float inputs with non-broadcastable or scalar if/else.
  absl::Status error = absl::InvalidArgumentError(
      "Cond must be float or bool type, if, else tensors must be "
      "either be same the shape as output or constant, scalar.");
  if (op_sig.inputs.at(0).type != kTfLiteBool &&
      op_sig.inputs.at(0).type != kTfLiteFloat16 &&
      op_sig.inputs.at(0).type != kTfLiteFloat32) {
    return error;
  }
  std::vector<int32_t> output_dims = op_sig.outputs[0].dims;
  if (!op_sig.inputs.at(1).dims.empty() &&
      (op_sig.inputs.at(1).dims != output_dims) &&
      (op_sig.inputs.at(1).dims.size() > 1 ||
       op_sig.inputs.at(1).dims[0] > 1)) {
    return error;
  }
  if (op_sig.inputs.at(1).is_const && op_sig.inputs.at(1).dims.size() == 2) {
    return absl::InvalidArgumentError(
        "2-D if tensor only supported if constant.");
  }
  if (!op_sig.inputs.at(2).dims.empty() &&
      (op_sig.inputs.at(2).dims != output_dims) &&
      (op_sig.inputs.at(2).dims.size() > 1 ||
       op_sig.inputs.at(2).dims[0] > 1)) {
    return error;
  }
  if (op_sig.inputs.at(2).is_const && op_sig.inputs.at(2).dims.size() == 2) {
    return absl::InvalidArgumentError(
        "2-D else tensor only supported if constant.");
  }
  return absl::OkStatus();
}

absl::Status CheckCustomOpsGpuDelegateCompatibility(const OpSignature& op_sig) {
  if (op_sig.custom_name == "Convolution2DTransposeBias") {
    RETURN_IF_ERROR(CheckTensorIsAvailable(op_sig, 1));
    const TfLiteTransposeConvParams* tf_options;
    RETURN_IF_ERROR(RetrieveCustomInitialData(op_sig, &tf_options));
    RETURN_IF_ERROR(
        CheckStrides(tf_options->stride_height, tf_options->stride_width));
    return absl::OkStatus();
  }
  if (op_sig.custom_name == "MaxPoolingWithArgmax2D") {
    return CheckPooling2DGpuDelegateCompatibility(op_sig);
  }
  if (op_sig.custom_name == "MaxUnpooling2D") {
    RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                       /*required_runtime_inputs=*/2,
                                       /*required_outputs=*/1));
    const TfLitePoolParams* tf_options;
    RETURN_IF_ERROR(RetrieveCustomInitialData(op_sig, &tf_options));
    RETURN_IF_ERROR(CheckKernelsAndStrides(
        tf_options->filter_height, tf_options->filter_width,
        tf_options->stride_height, tf_options->stride_width));
    return absl::OkStatus();
  }
  if (op_sig.custom_name == "Resampler") {
    return CheckInputsOutputs(op_sig,
                              /*required_runtime_inputs=*/2,
                              /*required_outputs=*/1);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Not supported custom op ", op_sig.custom_name));
}

bool CheckIsBroadcastable(const std::vector<int32_t>* longer_dims,
                          const std::vector<int32_t>* shorter_dims) {
  int idx_1 = longer_dims->size() - 1;
  int idx_2 = shorter_dims->size() - 1;
  int max_idx = std::max(idx_1, idx_2);
  int data_1 = 0;
  int data_2 = 0;
  for (int i = max_idx; i >= 0; --i) {
    data_1 = idx_1 < 0 ? 1 : longer_dims->at(idx_1);
    data_2 = idx_2 < 0 ? 1 : shorter_dims->at(idx_2);
    if (data_1 != data_2 && data_1 != 1 && data_2 != 1) {
      return false;
    }
    --idx_1;
    --idx_2;
  }
  return true;
}

absl::Status CheckAddMulBroadcastCompatibility(
    const OpSignatureTensorSpec& input0, const OpSignatureTensorSpec& input1,
    GpuCompatibilityFlags flags) {
  if (input0.dims.size() > 1 && input1.dims.size() > 1 &&
      input0.dims.size() != input1.dims.size()) {
    const std::vector<int32_t>*longer_dims, *shorter_dims;
    if (input0.dims.size() >= input1.dims.size()) {
      longer_dims = &input0.dims;
      shorter_dims = &input1.dims;
    } else {
      longer_dims = &input1.dims;
      shorter_dims = &input0.dims;
    }

    bool is_broadcastable = false;
    if (flags == GpuCompatibilityFlags::kEnhancedBroadcast) {
      is_broadcastable = CheckIsBroadcastable(longer_dims, shorter_dims);
    } else {
      if (longer_dims->size() == 4 && shorter_dims->size() == 3 &&
          longer_dims->at(0) == 1) {
        // Broadcasting 3D to 4D with batch 1 works.
        is_broadcastable = true;
      } else if (longer_dims->size() == 4 && shorter_dims->size() == 2 &&
                 longer_dims->at(0) == 1 && shorter_dims->at(0) == 1 &&
                 shorter_dims->at(1) == 1) {
        // Broadcasting 2D [1, 1] to 4D [1, x, y, z] works.
        is_broadcastable = true;
      } else if (longer_dims->size() == 4 && shorter_dims->size() == 2 &&
                 longer_dims->at(0) == shorter_dims->at(0) &&
                 longer_dims->at(3) == shorter_dims->at(1)) {
        // Broadcasting 2D [b, c] to 4D [b, x, y, c] works.
        is_broadcastable = true;
      }
    }

    if (!is_broadcastable) {
      return absl::UnimplementedError(
          absl::StrCat("Doesn't support broadcasting - input0: [",
                       absl::StrJoin(input0.dims, ","), "], input1: [",
                       absl::StrJoin(input1.dims, ","), "]"));
    }
  }
  return absl::OkStatus();
}

}  // namespace

// Logics here used to be in TFLiteOperationParser:IsSupported()
// of tensorflow/lite/delegates/gpu/common/model_builder.cc but they're all
// migrated into here.
absl::Status CheckGpuDelegateCompatibility(const OpSignature& op_sig,
                                           GpuCompatibilityFlags flags) {
  TfLiteBuiltinOperator opcode = static_cast<TfLiteBuiltinOperator>(op_sig.op);
  switch (opcode) {
    case kTfLiteBuiltinAdd: {
      if (op_sig.inputs.size() != 2) {
        return absl::UnimplementedError("ADD requires two input tensors.");
      }
      const auto& input0 = op_sig.inputs.at(0);
      const auto& input1 = op_sig.inputs.at(1);
      auto broadcastable =
          CheckAddMulBroadcastCompatibility(input0, input1, flags);
      if (!broadcastable.ok()) {
        return broadcastable;
      }
      const TfLiteAddParams* tf_options;
      return RetrieveBuiltinData(op_sig, &tf_options);
    }
    case kTfLiteBuiltinAddN: {
      return op_sig.inputs.size() == 2
                 ? absl::OkStatus()
                 : absl::UnimplementedError("ADD_N only supports 2 inputs.");
    }

    case kTfLiteBuiltinAveragePool2d:
      return CheckPooling2DGpuDelegateCompatibility(op_sig);

    case kTfLiteBuiltinBatchMatmul: {
      const int num_inputs = op_sig.inputs.size();
      const int num_outputs = op_sig.outputs.size();
      if (!(num_inputs == 2 && num_outputs == 1)) {
        return absl::InternalError(
            absl::StrCat("Expected 2 inputs and 1 output, got: ", num_inputs,
                         " inputs and ", num_outputs, " outputs"));
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinBitcast: {
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      std::vector<int32_t> input_dims = op_sig.inputs.at(0).dims;
      std::vector<int32_t> output_dims = op_sig.outputs.at(0).dims;
      size_t input_elem_size, output_elem_size;
      TfLiteStatus status = GetSizeOfType(
          /*context=*/nullptr, op_sig.inputs.at(0).type, &input_elem_size);
      if (status != kTfLiteOk) {
        return absl::InternalError("Could not parse input type");
      }
      status = GetSizeOfType(/*context=*/nullptr, op_sig.outputs.at(0).type,
                             &output_elem_size);
      if (status != kTfLiteOk) {
        return absl::InternalError("Could not parse output type");
      }
      if (input_elem_size == output_elem_size) {
        if (input_dims != output_dims) {
          return absl::InternalError(
              "If input and output types have the same element size, they must "
              "have the same shapes");
        }
      } else if (input_elem_size > output_elem_size) {
        if (input_dims.size() + 1 != output_dims.size()) {
          return absl::InternalError(
              "If input element size is greater than output element size, "
              "require that input rank is one greater than output rank");
        }
        for (int d = 0; d < input_dims.size(); ++d) {
          if (input_dims[d] != output_dims[d]) {
            return absl::InternalError("Shapes must match in all but last dim");
          }
        }
        if (output_dims[output_dims.size() - 1] * output_elem_size !=
            input_elem_size) {
          return absl::InternalError(
              "Last output dim must be equal to input element size divided by "
              "output element size");
        }
      } else {  // output_elem_size > input_elem_size
        if (input_dims.size() != output_dims.size() + 1) {
          return absl::InternalError(
              "If output element size is greater than input element size, "
              "require that output rank is on greater than input rank");
        }
        for (int d = 0; d < output_dims.size(); ++d) {
          if (input_dims[d] != output_dims[d]) {
            return absl::InternalError("Shapes must match in all but last dim");
          }
        }
        if (input_dims[input_dims.size() - 1] * input_elem_size !=
            output_elem_size) {
          return absl::InternalError(
              "Last input dim must be equal to output element size divided by "
              "input element size");
        }
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinCast:
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      if (op_sig.inputs.at(0).type == kTfLiteBool &&
          (op_sig.outputs.at(0).type == kTfLiteFloat16 ||
           op_sig.outputs.at(0).type == kTfLiteFloat32)) {
        return absl::OkStatus();
      } else if ((op_sig.inputs.at(0).type == kTfLiteFloat16 ||
                  op_sig.inputs.at(0).type == kTfLiteFloat32) &&
                 op_sig.outputs.at(0).type == kTfLiteBool) {
        return absl::OkStatus();
      } else if ((op_sig.inputs.at(0).type == kTfLiteFloat32 ||
                  op_sig.inputs.at(0).type == kTfLiteInt32) &&
                 (op_sig.outputs.at(0).type == kTfLiteFloat32 ||
                  op_sig.outputs.at(0).type == kTfLiteInt32)) {
        return absl::OkStatus();
      } else {
        return absl::UnimplementedError(absl::StrCat(
            "Not supported Cast case. Input type: ",
            TfLiteTypeGetName(op_sig.inputs.at(0).type), " and output type: ",
            TfLiteTypeGetName(op_sig.outputs.at(0).type)));
      }

    case kTfLiteBuiltinConcatenation: {
      const TfLiteConcatenationParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      return absl::OkStatus();
    }

    case kTfLiteBuiltinConv2d: {
      RETURN_IF_ERROR(CheckConvoultionInputOutput(op_sig));
      const TfLiteConvParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      RETURN_IF_ERROR(CheckStridesAndDilation(
          tf_options->stride_height, tf_options->stride_width,
          tf_options->dilation_height_factor,
          tf_options->dilation_width_factor));
      return IsActivationSupported(tf_options->activation);
    }

    case kTfLiteBuiltinCumsum:
      return CheckCumsumGpuDelegateCompatibility(op_sig);

    case kTfLiteBuiltinDensify:
      return CheckInputsOutputs(op_sig, /*required_runtime_inputs=*/0,
                                /*required_outputs=*/1);

    case kTfLiteBuiltinDepthwiseConv2d:
      return CheckDepthwiseConvGpuDelegateCompatibility(op_sig);

    case kTfLiteBuiltinDepthToSpace: {
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      const TfLiteDepthToSpaceParams* d2s_params;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &d2s_params));
      if (d2s_params->block_size == 1) {
        return absl::InvalidArgumentError(
            "DEPTH_TO_SPACE block_size = 1 is a no-op.");
      }
      if (d2s_params->block_size < 1) {
        return absl::InvalidArgumentError(
            "DEPTH_TO_SPACE block_size must be > 1.");
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinDequantize: {
      const int num_inputs = op_sig.inputs.size();
      const int num_outputs = op_sig.outputs.size();
      if (num_inputs != 1 || num_outputs != 1) {
        return absl::InternalError(absl::StrCat(
            "Expected 1 input & output each from Dequantize, got: %d, %d",
            num_inputs, num_outputs));
      }
      if (op_sig.inputs[0].type == kTfLiteInt16) {
        return absl::UnimplementedError("Unsupported dequantization type.");
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinEmbeddingLookup: {
      const int num_inputs = op_sig.inputs.size();
      const OpSignatureTensorSpec ids_spec = op_sig.inputs[0];
      const OpSignatureTensorSpec value_spec = op_sig.inputs[1];
      const OpSignatureTensorSpec output_spec = op_sig.outputs[0];
      if (num_inputs != 2) {
        return absl::InvalidArgumentError(
            absl::StrCat("Expected 2, but got ", num_inputs, " inputs."));
      }

      if (ids_spec.dims.size() != 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Expected 1D, but got ", ids_spec.dims.size(), "D input #0."));
      }

      if (value_spec.dims.size() < 2) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Expected > 1D, but got ", value_spec.dims.size(), "D input #1."));
      }

      if (op_sig.outputs.size() != 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Expected 1, but got ", op_sig.outputs.size(), " outputs."));
      }

      if (value_spec.dims.size() != output_spec.dims.size()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Expected ", value_spec.dims.size(), ", but got ",
                         output_spec.dims.size(), " for output."));
      }

      for (int i = 1; i < output_spec.dims.size(); ++i) {
        if (value_spec.dims[i] != output_spec.dims[i]) {
          return absl::InvalidArgumentError(
              absl::StrCat("Expected ", value_spec.dims[i], ", but got ",
                           output_spec.dims[i], " for output.dim[", i, "]."));
        }
      }

      if (value_spec.type != kTfLiteInt8 && value_spec.type != kTfLiteInt4 &&
          value_spec.type != kTfLiteFloat32) {
        return absl::InvalidArgumentError(
            absl::StrCat("Expected int8, int4, or float32, but got ",
                         TfLiteTypeGetName(value_spec.type), " for input #1."));
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinDynamicUpdateSlice: {
      if (op_sig.inputs.size() != 3) {
        return absl::UnimplementedError(
            "DynamicUpdateSlice requires 3 inputs.");
      }
      OpSignatureTensorSpec operand = op_sig.inputs[0];
      OpSignatureTensorSpec update_slice = op_sig.inputs[1];
      OpSignatureTensorSpec start_indices = op_sig.inputs[2];

      if (operand.type != update_slice.type) {
        return absl::InternalError(
            absl::StrCat("Array to update and updated slice must have the same "
                         "data type, but got: array to update: ",
                         operand.type, ", updated slice: ", update_slice.type));
      }

      if (start_indices.dims.size() != 1) {
        return absl::InternalError(absl::StrCat(
            "Start indices must be 1D, but got: ", start_indices.dims.size()));
      }

      if (start_indices.type != kTfLiteInt32) {
        return absl::InvalidArgumentError(
            "start_indices must be of type int32.");
      }

      if (update_slice.dims.size() != operand.dims.size()) {
        return absl::InternalError(absl::StrCat(
            "Operand and update must have the same number of "
            "dimensions, but got: operand: ",
            operand.dims.size(), ", update: ", update_slice.dims.size()));
      }

      return absl::OkStatus();
    }
    case kTfLiteBuiltinFullyConnected: {
      const TfLiteFullyConnectedParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      if (tf_options->weights_format !=
          kTfLiteFullyConnectedWeightsFormatDefault) {
        return absl::UnimplementedError(
            absl::StrCat("Unsupported FullyConnected weights format: ",
                         tf_options->weights_format));
      }
      if (GetNumberOfRuntimeInputs(op_sig) > 2) {
        return absl::UnimplementedError(
            "FullyConnected doesn't support more than 2 runtime inputs.");
      }
      if (op_sig.inputs[0].is_const) {
        return absl::UnimplementedError(
            "FullyConnected doesn't support constant input.");
      }
      if (tf_options->keep_num_dims == true) {
        const auto& input = op_sig.inputs.at(0);
        const auto& output = op_sig.outputs.at(0);
        if (input.dims.size() != output.dims.size()) {
          return absl::UnimplementedError(
              "Input and output dimensions different and FullyConnected "
              "doesn't "
              "support keep_num_dims.");
        }
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinGather:
      if (!CheckInputsConstsOutputs(op_sig, /*required_runtime_inputs=*/2,
                                    /*required_const_inputs=*/0,
                                    /*required_outputs=*/1)
               .ok() &&
          !CheckInputsConstsOutputs(op_sig, /*required_runtime_inputs=*/1,
                                    /*required_const_inputs=*/1,
                                    /*required_outputs=*/1)
               .ok()) {
        return absl::InvalidArgumentError(
            "Op can only handle 1 or 2 operand(s).");
      }
      if (op_sig.inputs[1].dims.size() != 1) {
        return absl::UnimplementedError("Only support 1D indices\n");
      }
      return op_sig.inputs.at(1).type == kTfLiteInt32
                 ? absl::OkStatus()
                 : absl::UnimplementedError("Only accept INT32 indices\n");

    case kTfLiteBuiltinHardSwish:
      return CheckInputsOutputs(op_sig, /*required_runtime_inputs=*/1,
                                /*required_outputs=*/1);

    case kTfLiteBuiltinLstm: {
      const TfLiteLSTMParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      switch (tf_options->kernel_type) {
        case kTfLiteLSTMFullKernel: {
          const int inputs = op_sig.inputs.size();
          if (inputs != 20 && inputs != 24) {
            return absl::InternalError(
                absl::StrCat("Expected 20 or 24 input tensors, but node has ",
                             inputs, " input(s)."));
          }
          const int runtime_outputs = op_sig.outputs.size();
          if (runtime_outputs != 1) {
            return absl::InternalError(
                absl::StrCat("Expected 1 output tensor, but node has ",
                             runtime_outputs, " output(s)."));
          }
          if (tf_options->activation != kTfLiteActSigmoid &&
              tf_options->activation != kTfLiteActTanh) {
            return absl::UnimplementedError(absl::StrCat(
                "Only sigmoid or tanh activation is supported, but node has ",
                tf_options->activation));
          }
          return absl::OkStatus();
        }
        case kTfLiteLSTMBasicKernel:
          RETURN_IF_ERROR(
              CheckInputsConstsOutputs(op_sig, /*required_runtime_inputs=*/3,
                                       /*required_const_inputs=*/2,
                                       /*required_outputs=*/4));
          if (tf_options->activation != kTfLiteActTanh) {
            return absl::UnimplementedError(
                absl::StrCat("Only TANH activation is supported. but node has ",
                             tf_options->activation));
          }
          if (tf_options->cell_clip != 0.0f) {
            return absl::UnimplementedError("cell_clip is not supported.");
          }
          if (tf_options->proj_clip != 0.0f) {
            return absl::UnimplementedError("proj_clip is not supported.");
          }
          return absl::OkStatus();
      }
    }

    case kTfLiteBuiltinMaxPool2d:
      return CheckPooling2DGpuDelegateCompatibility(op_sig);

    case kTfLiteBuiltinMean: {
      RETURN_IF_ERROR(CheckInputsConstsOutputs(op_sig,
                                               /*required_runtime_inputs=*/1,
                                               /*required_const_inputs=*/1,
                                               /*required_outputs=*/1));
      return CheckAxesAreInt32Const(op_sig, 1);
    }

    case kTfLiteBuiltinMul: {
      if (op_sig.inputs.size() != 2) {
        return absl::UnimplementedError("MUL requires two input tensors.");
      }
      const auto& input0 = op_sig.inputs.at(0);
      const auto& input1 = op_sig.inputs.at(1);
      if (input0.dims.size() != input1.dims.size()) {
        const auto& input0 = op_sig.inputs.at(0);
        const auto& input1 = op_sig.inputs.at(1);
        auto broadcastable =
            CheckAddMulBroadcastCompatibility(input0, input1, flags);
        if (!broadcastable.ok()) {
          return broadcastable;
        }
      }
      const TfLiteMulParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      return IsActivationSupported(tf_options->activation);
    }

    case kTfLiteBuiltinPack:
      return absl::OkStatus();

    case kTfLiteBuiltinOneHot:
      return CheckOneHotGpuDelegateCompatibility(op_sig);

    case kTfLiteBuiltinQuantize:
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      return absl::OkStatus();

    case kTfLiteBuiltinReluN1To1:
      return absl::OkStatus();

    case kTfLiteBuiltinPrelu:
      return absl::OkStatus();

    case kTfLiteBuiltinReshape:
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      return absl::OkStatus();
    case kTfLiteBuiltinSelect:
    case kTfLiteBuiltinSelectV2:
      return CheckSelectV2GpuDelegateCompatibility(op_sig);

    case kTfLiteBuiltinSlice: {
      if (op_sig.inputs.size() < 3) {
        return absl::UnimplementedError(
            absl::StrCat("SLICE requires 3 inputs, but node has ",
                         op_sig.inputs.size(), " inputs."));
      }
      const auto& input = op_sig.inputs.at(0);
      if (input.dims.size() != 3 && input.dims.size() != 4) {
        return absl::UnimplementedError(absl::StrCat(
            "SLICE supports for 3 or 4 dimensional tensors only, but node has ",
            input.dims.size(), " dimensional tensors."));
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinSoftmax: {
      const TfLiteSoftmaxParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      if (tf_options->beta != 1) {
        return absl::UnimplementedError("Softmax.beta != 1 is not supported.");
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinSpaceToDepth: {
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      const TfLiteSpaceToDepthParams* s2d_params;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &s2d_params));
      if (s2d_params->block_size == 1) {
        return absl::InvalidArgumentError(
            "SPACE_TO_DEPTH block_size = 1 is a no-op.");
      }
      if (s2d_params->block_size < 1) {
        return absl::InvalidArgumentError(
            "SPACE_TO_DEPTH block_size must be > 1.");
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinSplit:
      return absl::OkStatus();

    case kTfLiteBuiltinSplitV:
      return absl::OkStatus();

    case kTfLiteBuiltinStridedSlice: {
      const TfLiteStridedSliceParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      if (tf_options->ellipsis_mask) {
        return absl::UnimplementedError(
            "Slice does not support ellipsis_mask.");
      }
      if (tf_options->new_axis_mask) {
        return absl::UnimplementedError(
            "Slice does not support new_axis_mask.");
      }
      if (tf_options->shrink_axis_mask) {
        return absl::UnimplementedError(
            "Slice does not support shrink_axis_mask parameter. ");
      }

      if (op_sig.inputs.size() < 4) {
        return absl::UnimplementedError("STRIDED_SLICE requires 4 inputs.");
      }
      const auto& input = op_sig.inputs.at(0);
      if (input.dims.size() != 3 && input.dims.size() != 4) {
        return absl::UnimplementedError(
            "STRIDED_SLICE supports for 3 or 4 dimensional tensors only.");
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinTile:
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      return absl::OkStatus();

    case kTfLiteBuiltinTranspose:
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      return absl::OkStatus();

    case kTfLiteBuiltinTransposeConv: {
      RETURN_IF_ERROR(CheckConvoultionInputOutput(op_sig));
      const TfLiteTransposeConvParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      RETURN_IF_ERROR(
          CheckStrides(tf_options->stride_height, tf_options->stride_width));
      return absl::OkStatus();
    }

    case kTfLiteBuiltinResizeBilinear: {
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      const TfLiteResizeBilinearParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      if (tf_options->align_corners && tf_options->half_pixel_centers) {
        return absl::InternalError(
            "If half_pixel_centers is True, align_corners must be False.");
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinResizeNearestNeighbor: {
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      const TfLiteResizeNearestNeighborParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      return absl::OkStatus();
    }

    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinLeakyRelu:
      return absl::OkStatus();

    case kTfLiteBuiltinReduceAll:
    case kTfLiteBuiltinReduceAny:
    case kTfLiteBuiltinReduceMax:
    case kTfLiteBuiltinReduceMin:
    case kTfLiteBuiltinReduceProd:
    case kTfLiteBuiltinSum: {
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      return CheckAxesAreInt32Const(op_sig, 1);
    }

    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinPadv2:
    case kTfLiteBuiltinMirrorPad: {
      if (opcode == kTfLiteBuiltinMirrorPad) {
        const TfLiteMirrorPaddingParams* tf_options;
        RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
        if (tf_options->mode !=
            TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingReflect) {
          return absl::InvalidArgumentError(
              absl::StrCat("Only Reflective padding is supported for Mirror "
                           "Pad operation. But node has ",
                           tf_options->mode));
        }
      }
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_runtime_inputs=*/1,
                                         /*required_outputs=*/1));
      RETURN_IF_ERROR(CheckTensorIsAvailable(op_sig, 1));
      auto& pad_tensor = op_sig.inputs.at(1);
      if (pad_tensor.dims.size() != 2) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid paddings tensor dimension: expected 2 dim, got ",
            pad_tensor.dims.size(), " dim"));
      }
      bool supported = pad_tensor.dims[0] == 3 || pad_tensor.dims[0] == 4;
      if (!supported || pad_tensor.dims[1] != 2) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid paddings tensor shape: expected 4x2 or 3x2, got ",
            pad_tensor.dims[0], "x", pad_tensor.dims[1]));
      }
      return absl::OkStatus();
    }
    case kTfLiteBuiltinReverseV2: {
      RETURN_IF_ERROR(CheckInputsConstsOutputs(op_sig,
                                               /*required_runtime_inputs=*/1,
                                               /*required_const_inputs=*/1,
                                               /*required_outputs=*/1));
      return CheckAxesAreInt32Const(op_sig, 1);
    }

    // One argument elementwise operations
    case kTfLiteBuiltinAbs:
    case kTfLiteBuiltinCeil:
    case kTfLiteBuiltinCos:
    case kTfLiteBuiltinElu:
    case kTfLiteBuiltinExp:
    case kTfLiteBuiltinFloor:
    case kTfLiteBuiltinGelu:
    case kTfLiteBuiltinLog:
    case kTfLiteBuiltinLogicalNot:
    case kTfLiteBuiltinLogistic:  // Sigmoid
    case kTfLiteBuiltinNeg:
    case kTfLiteBuiltinRound:
    case kTfLiteBuiltinRsqrt:
    case kTfLiteBuiltinSign:
    case kTfLiteBuiltinSin:
    case kTfLiteBuiltinSqrt:
    case kTfLiteBuiltinSquare:
    case kTfLiteBuiltinTanh:
      return (CheckInputsConstsOutputs(op_sig, /*required_runtime_inputs=*/1,
                                       /*required_const_inputs=*/0,
                                       /*required_outputs=*/1));

    // Two arguments elementwise operations
    case kTfLiteBuiltinAtan2:
    case kTfLiteBuiltinDiv:
    case kTfLiteBuiltinEqual:
    case kTfLiteBuiltinFloorDiv:
    case kTfLiteBuiltinFloorMod:
    case kTfLiteBuiltinGreater:
    case kTfLiteBuiltinGreaterEqual:
    case kTfLiteBuiltinLogicalAnd:
    case kTfLiteBuiltinLogicalOr:
    case kTfLiteBuiltinLess:
    case kTfLiteBuiltinLessEqual:
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinMinimum:
    case kTfLiteBuiltinNotEqual:
    case kTfLiteBuiltinPow:
    case kTfLiteBuiltinRightShift:
    case kTfLiteBuiltinStablehloRemainder:
    case kTfLiteBuiltinStablehloShiftLeft:
    case kTfLiteBuiltinSquaredDifference:
    case kTfLiteBuiltinSub: {
      if (!CheckInputsConstsOutputs(op_sig, /*required_runtime_inputs=*/2,
                                    /*required_const_inputs=*/0,
                                    /*required_outputs=*/1)
               .ok() &&
          !CheckInputsConstsOutputs(op_sig, /*required_runtime_inputs=*/1,
                                    /*required_const_inputs=*/1,
                                    /*required_outputs=*/1)
               .ok()) {
        return absl::InvalidArgumentError(
            "Op can only handle 1 or 2 operand(s).");
      }
      TfLiteFusedActivation activation = kTfLiteActNone;
      if (opcode == kTfLiteBuiltinDiv) {
        const TfLiteDivParams* tf_options;
        auto status = RetrieveBuiltinData(op_sig, &tf_options);
        activation = status.ok() ? tf_options->activation : kTfLiteActNone;
      } else if (opcode == kTfLiteBuiltinSub) {
        const TfLiteSubParams* tf_options;
        auto status = RetrieveBuiltinData(op_sig, &tf_options);
        activation = status.ok() ? tf_options->activation : kTfLiteActNone;
      }
      return IsActivationSupported(activation);
    }

    // Stable HLO ops
    case kTfLiteBuiltinStablehloBroadcastInDim:
      if (!CheckInputsConstsOutputs(op_sig, /*required_runtime_inputs=*/1,
                                    /*required_const_inputs=*/1,
                                    /*required_outputs=*/1)
               .ok()) {
        return absl::InvalidArgumentError(
            "requires one runtime input, one const input, and one output");
      }
      if (op_sig.inputs[1].dims.size() != 1) {
        return absl::InvalidArgumentError("Only support 1D indices");
      }
      if (op_sig.inputs[1].type != kTfLiteInt32) {
        return absl::InvalidArgumentError("Only support int32 indices");
      }
      if (op_sig.inputs[0].dims.size() != op_sig.inputs[1].dims[0]) {
        return absl::InvalidArgumentError(
            "Require size(indices) = rank(operand)");
      }
      return absl::OkStatus();
    case kTfLiteBuiltinStablehloCbrt:
      if (op_sig.inputs[0].type != kTfLiteFloat16 &&
          op_sig.inputs[0].type != kTfLiteFloat32 &&
          op_sig.inputs[0].type != kTfLiteBFloat16) {
        return absl::InvalidArgumentError("Only support float inputs");
      }
      if (op_sig.inputs[0].type != op_sig.outputs[0].type) {
        return absl::InvalidArgumentError("Input and output types must match");
      }
      return CheckInputsConstsOutputs(op_sig, /*required_runtime_inputs=*/1,
                                      /*required_const_inputs=*/0,
                                      /*required_outputs=*/1);
    case kTfLiteBuiltinStablehloClamp:
      if ((op_sig.inputs.at(0).type != op_sig.inputs.at(1).type) ||
          (op_sig.inputs.at(1).type != op_sig.inputs.at(2).type)) {
        return absl::InvalidArgumentError(
            "Clamp tensors must all be the same type");
      }
      if ((op_sig.inputs.at(0).dims != op_sig.inputs.at(1).dims) &&
          (NumElements(op_sig.inputs.at(0).dims) != 1)) {
        return absl::InvalidArgumentError(
            "Min tensor must be the same shape as the input, or a scalar");
      }
      if ((op_sig.inputs.at(2).dims != op_sig.inputs.at(1).dims) &&
          (NumElements(op_sig.inputs.at(0).dims) != 1)) {
        return absl::InvalidArgumentError(
            "Max tensor must be the same shape as the input, or a scalar");
      }
      return CheckInputsConstsOutputs(op_sig, /*required_runtime_inputs=*/3,
                                      /*required_const_inputs=*/0,
                                      /*required_outputs=*/1);
    case kTfLiteBuiltinCustom:
      return CheckCustomOpsGpuDelegateCompatibility(op_sig);

    default:
      break;
  }

  return absl::InvalidArgumentError(absl::StrCat(
      "Not supported op ", tflite::EnumNamesBuiltinOperator()[op_sig.op]));
}  // NOLINT(readability/fn_size)

absl::Status CheckGpuDelegateCompatibility(const OperatorCode* op_code,
                                           const Operator* op,
                                           const SubGraph* subgraph,
                                           const Model* model) {
  OpSignature op_sig = GetOpSignature(op_code, op, subgraph, model);
  // Offline compatibility assumes enhanced broadcast is enabled.
  auto status = CheckGpuDelegateCompatibility(
      op_sig, GpuCompatibilityFlags::kEnhancedBroadcast);
  if (op_sig.builtin_data) {
    free(op_sig.builtin_data);
  }
  return status;
}

absl::Status CheckGpuDelegateCompatibility(
    const TfLiteContext* context, const TfLiteNode* node,
    const TfLiteRegistration* registration, GpuCompatibilityFlags flags) {
  return CheckGpuDelegateCompatibility(
      GetOpSignature(context, node, registration), flags);
}

}  // namespace tflite
