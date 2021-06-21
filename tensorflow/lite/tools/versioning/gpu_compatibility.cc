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

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/tools/versioning/op_signature.h"

namespace tflite {

namespace {

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

int GetNumberOfRuntimeInputs(const OpSignature& op_sig) {
  int number_of_runtime_inputs = 0;
  for (auto& input : op_sig.inputs) {
    if (!input.is_const) {
      number_of_runtime_inputs++;
    }
  }
  return number_of_runtime_inputs;
}

absl::Status CheckInputsOutputs(const OpSignature& op_sig,
                                const int required_nonconstant_inputs,
                                const int required_outputs) {
  const int runtime_inputs_from_model = GetNumberOfRuntimeInputs(op_sig);
  if (runtime_inputs_from_model != required_nonconstant_inputs) {
    return absl::InternalError(
        absl::StrCat("Expected ", required_nonconstant_inputs,
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

absl::Status CheckTensorIsAvailable(const OpSignature& op_sig, int idx) {
  // If tensor id is in range, it's guaranteed that it'll be available.
  if (idx >= op_sig.inputs.size()) {
    return absl::OutOfRangeError(
        absl::StrCat("Requested index goes beyond array size: ", idx, " vs ",
                     op_sig.inputs.size()));
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

}  // namespace

// TODO(b/189917229): Logics are copied from TFLiteOperationParser:IsSupported()
// in tensorflow/lite/delegates/gpu/common/model_builder.cc. Once this logic is
// stabilized, the original logic in model_builder.cc will be replaced by this.
absl::Status CheckGpuDelegateCompatibility(const OpSignature& op_sig) {
  switch (op_sig.op) {
    case kTfLiteBuiltinAbs:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinAdd:
      return CheckInputsOutputs(op_sig, /*required_nonconstant_inputs=*/2,
                                /*required_outputs=*/1);

    case kTfLiteBuiltinAveragePool2d:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinBatchMatmul:
      return CheckInputsOutputs(op_sig, /*required_nonconstant_inputs=*/2,
                                /*required_outputs=*/1);

    case kTfLiteBuiltinCast:
      return CheckInputsOutputs(op_sig, /*required_nonconstant_inputs=*/1,
                                /*required_outputs=*/1);

    case kTfLiteBuiltinConcatenation:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinConv2d: {
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
      const TfLiteConvParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      RETURN_IF_ERROR(CheckStridesAndDilation(
          tf_options->stride_height, tf_options->stride_width,
          tf_options->dilation_height_factor,
          tf_options->dilation_width_factor));
      return IsActivationSupported(tf_options->activation);
    }

    case kTfLiteBuiltinCos:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinDensify:
      return CheckInputsOutputs(op_sig, /*required_nonconstant_inputs=*/0,
                                /*required_outputs=*/1);

    case kTfLiteBuiltinDepthwiseConv2d: {
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
      const TfLiteDepthwiseConvParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(op_sig, &tf_options));
      RETURN_IF_ERROR(CheckStridesAndDilation(
          tf_options->stride_height, tf_options->stride_width,
          tf_options->dilation_height_factor,
          tf_options->dilation_width_factor));
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
        return absl::InvalidArgumentError(
            "output.c != input.c * depth_multiplier");
      }
      if (bias && bias->dims.size() != output_depth) {
        return absl::InvalidArgumentError("bias.size != output.c");
      }
      if (depth_multiplier != 1 && input_depth != 1) {
        return absl::UnimplementedError(
            "depth_multiplier != 1 && input.c != 1");
      }
      return absl::OkStatus();
    }

    case kTfLiteBuiltinDepthToSpace: {
      RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                         /*required_nonconstant_inputs=*/1,
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

    case kTfLiteBuiltinDiv:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinEqual:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinElu:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinExp:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinFloor:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinFloorDiv:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinFloorMod:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinFullyConnected:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinGreater:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinGreaterEqual:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinHardSwish:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinLess:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinLessEqual:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinLogistic:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinLog:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinLstm:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinMaximum:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinMaxPool2d:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinMean:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinMinimum:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinMirrorPad:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinMul:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinNeg:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinNotEqual:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinPack:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinPad:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinPow:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinReduceMax:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinReduceMin:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinReduceProd:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinQuantize:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinRelu:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinRelu6:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinReluN1To1:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinLeakyRelu:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinPrelu:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinReshape:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinResizeBilinear:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinResizeNearestNeighbor:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinRsqrt:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinSin:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinSlice:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinSoftmax:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinSpaceToDepth:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinSplit:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinSplitV:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinSqrt:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinSquare:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinSquaredDifference:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinStridedSlice:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinSub:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinSum:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinTanh:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinTile:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinTranspose:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinTransposeConv:
      // TODO(b/189917229): Implement logic.
      return absl::OkStatus();

    case kTfLiteBuiltinCustom: {
      if (op_sig.custom_name == "Convolution2DTransposeBias") {
        // TODO(b/189917229): Implement logic.
        return absl::OkStatus();
      }
      if (op_sig.custom_name == "MaxPoolingWithArgmax2D") {
        // TODO(b/189917229): Implement logic.
        return absl::OkStatus();
      }
      if (op_sig.custom_name == "MaxUnpooling2D") {
        // TODO(b/189917229): Implement logic.
        return absl::OkStatus();
      }
      if (op_sig.custom_name == "Resampler") {
        // TODO(b/189917229): Implement logic.
        return absl::OkStatus();
      }
      return absl::InvalidArgumentError(
          absl::StrCat("Not supported custom op ", op_sig.custom_name));
    }

    default:
      break;
  }

  return absl::InvalidArgumentError(absl::StrCat(
      "Not supported op ", tflite::EnumNamesBuiltinOperator()[op_sig.op]));
}

absl::Status CheckGpuDelegateCompatibility(const OperatorCode* op_code,
                                           const Operator* op,
                                           const SubGraph* subgraph,
                                           const Model* model) {
  OpSignature op_sig = GetOpSignature(op_code, op, subgraph, model);
  auto status = CheckGpuDelegateCompatibility(op_sig);
  if (op_sig.builtin_data) {
    free(op_sig.builtin_data);
  }
  return status;
}

}  // namespace tflite
