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
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

bool HardcodeMinMaxForIm2colArray(Model* model, Operator* op) {
  if (op->outputs.size() != 2) {
    return false;
  }
  auto& im2col_array = model->GetArray(op->outputs[1]);
  if (im2col_array.minmax) {
    return false;
  }
  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.minmax) {
    return false;
  }
  const auto& input_minmax = input_array.GetMinMax();
  CHECK(!im2col_array.minmax);
  auto& im2col_minmax = im2col_array.GetOrCreateMinMax();
  im2col_minmax.min = input_minmax.min;
  im2col_minmax.max = input_minmax.max;
  return true;
}

bool HardcodeMinMaxForL2Normalization(Model* model, Operator* op) {
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.minmax) {
    return false;
  }
  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.minmax) {
    return false;
  }
  const auto& input_minmax = input_array.GetMinMax();
  CHECK(!output_array.minmax);
  auto& output_minmax = output_array.GetOrCreateMinMax();
  output_minmax.min = input_minmax.min >= 0. ? 0. : -1.;
  output_minmax.max = input_minmax.max <= 0. ? 0. : 1.;
  return true;
}

bool HardcodeInputMinMaxFromOutput(Model* model, Operator* op) {
  auto& input = model->GetArray(op->inputs[0]);
  if (input.minmax) {
    const auto* minmax = input.minmax.get();
    if (minmax) {
      return false;
    }
  }
  auto& output = model->GetArray(op->outputs[0]);
  if (output.minmax) {
    const auto* minmax = model->GetArray(op->outputs[0]).minmax.get();
    if (minmax) {
      input.GetOrCreateMinMax() = *minmax;
      return true;
    }
  }
  return false;
}

bool HardcodeMinMaxForConcatenation(Model* model, Operator* op) {
  // Do not early return if the output already has min/max:
  // we may still need to adjust the inputs min/max.
  bool has_minmax = false;
  double overall_min = std::numeric_limits<double>::infinity();
  double overall_max = -std::numeric_limits<double>::infinity();
  for (const auto& input : op->inputs) {
    if (model->GetArray(input).minmax) {
      has_minmax = true;
      const auto* minmax = model->GetArray(input).minmax.get();
      if (minmax) {
        overall_min = std::min(overall_min, minmax->min);
        overall_max = std::max(overall_max, minmax->max);
      }
    }
  }
  auto& output = model->GetArray(op->outputs[0]);
  if (output.minmax) {
    has_minmax = true;
    const auto* minmax = model->GetArray(op->outputs[0]).minmax.get();
    if (minmax) {
      overall_min = std::min(overall_min, minmax->min);
      overall_max = std::max(overall_max, minmax->max);
    }
  }
  if (!has_minmax) {
    return false;
  }
  MinMax overall_minmax;
  overall_minmax.min = overall_min;
  overall_minmax.max = overall_max;
  bool changed = false;
  if (model->flags.change_concat_input_ranges()) {
    for (const auto& input : op->inputs) {
      auto& array = model->GetArray(input);
      if (!array.minmax) {
        changed = true;
      } else if (!(overall_minmax == array.GetMinMax())) {
        changed = true;
        LOG(WARNING)
            << "Tweaking the MinMax of array " << input << ", which is "
            << "an input to " << LogName(*op) << ", because we want all inputs "
            << "and outputs of a Concatenation operator to have the same "
            << "MinMax so that it can be implemented as a pure byte-copy, no "
               "arithmetic.";
      }
      array.GetOrCreateMinMax() = overall_minmax;
    }
  }
  if (!output.minmax) {
    changed = true;
  } else if (!(overall_minmax == output.GetMinMax())) {
    if (model->flags.change_concat_input_ranges()) {
      changed = true;
      LOG(WARNING)
          << "Tweaking the MinMax of the output array of " << LogName(*op)
          << ", because we want all inputs "
          << "and outputs of a Concatenation operator to have the same MinMax "
          << "so that it can be implemented as a pure byte-copy, no "
          << "arithmetic.";
    } else {
      return false;
    }
  }
  output.GetOrCreateMinMax() = overall_minmax;

  return changed;
}

bool HardcodeMinMaxForSplit(Model* model, Operator* op) {
  // Data is in second input.
  auto& input_array = model->GetArray(op->inputs[1]);
  if (!input_array.minmax) {
    return false;
  }
  bool changed = false;
  for (const auto& output : op->outputs) {
    auto& array = model->GetArray(output);
    if (!array.minmax || !(array.GetMinMax() == input_array.GetMinMax())) {
      changed = true;
      array.GetOrCreateMinMax() = *input_array.minmax;
    }
  }
  return changed;
}

// The output of average or max pooling is within the same range as its input.
bool HardcodeMinMaxForAverageOrMaxPool(Model* model, Operator* op) {
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.minmax) {
    return false;
  }
  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.minmax) {
    return false;
  }
  const auto& input_minmax = input_array.GetMinMax();
  CHECK(!output_array.minmax);
  auto& output_minmax = output_array.GetOrCreateMinMax();
  output_minmax.min = std::min(input_minmax.min, 0.);
  output_minmax.max = std::max(input_minmax.max, 0.);
  return true;
}

bool HardcodeMinMaxFromFirstInput(Model* model, Operator* op) {
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.minmax) {
    return false;
  }
  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.minmax) {
    return false;
  }
  const auto& input_minmax = input_array.GetMinMax();
  CHECK(!output_array.minmax);
  auto& output_minmax = output_array.GetOrCreateMinMax();
  output_minmax.min = input_minmax.min;
  output_minmax.max = input_minmax.max;
  return true;
}

bool HardcodeMinMaxForSelect(Model* model, Operator* op) {
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.minmax) {
    return false;
  }

  auto& input_array_1 = model->GetArray(op->inputs[1]);
  auto& input_array_2 = model->GetArray(op->inputs[2]);

  if (!input_array_1.minmax && !input_array_2.minmax) {
    return false;
  }

  // Propagate up if one input is quantized and the other is constant.
  if (!input_array_1.minmax &&
      IsConstantParameterArray(*model, op->inputs[1])) {
    auto& minmax_1 = input_array_1.GetOrCreateMinMax();
    const auto& minmax_2 = input_array_2.GetMinMax();
    minmax_1.min = minmax_2.min;
    minmax_1.max = minmax_2.max;
  }

  if (!input_array_2.minmax &&
      IsConstantParameterArray(*model, op->inputs[2])) {
    auto& minmax_2 = input_array_2.GetOrCreateMinMax();
    const auto& minmax_1 = input_array_1.GetMinMax();
    minmax_2.min = minmax_1.min;
    minmax_2.max = minmax_1.max;
  }

  if (!input_array_1.minmax || !input_array_2.minmax) {
    return false;
  }

  const auto& input_minmax_1 = input_array_1.GetMinMax();
  const auto& input_minmax_2 = input_array_2.GetMinMax();

  CHECK_EQ(input_minmax_1.min, input_minmax_2.min);
  CHECK_EQ(input_minmax_1.max, input_minmax_2.max);
  CHECK(!output_array.minmax);
  auto& output_minmax = output_array.GetOrCreateMinMax();
  output_minmax.min = input_minmax_1.min;
  output_minmax.max = input_minmax_1.max;
  return true;
}

bool HardcodeMinMaxForOutput(Model* model, Operator* op, double min,
                             double max) {
  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.minmax) {
    return false;
  }
  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.minmax) {
    return false;
  }
  CHECK(!output_array.minmax);
  auto& output_minmax = output_array.GetOrCreateMinMax();
  output_minmax.min = min;
  output_minmax.max = max;
  return true;
}

bool MinMaxApproximatelyEqual(const MinMax& minmax1, const MinMax& minmax2) {
  const double magnitude =
      std::min(minmax1.max - minmax1.min, minmax2.max - minmax2.min);
  const double tolerated = 1e-6 * magnitude;
  return std::abs(minmax1.min - minmax2.min) < tolerated &&
         std::abs(minmax1.max - minmax2.max) < tolerated;
}

// Propagates MinMax from any of the listed arrays, to all others.
// If multiple of these arrays have MinMax, then these are required
// to agree with each other.
bool PropagateMinMaxAmongArrays(Model* model,
                                const std::vector<string> array_names) {
  string reference_array_name;
  MinMax* reference_minmax = nullptr;
  for (const string& array_name : array_names) {
    if (model->GetArray(array_name).minmax) {
      reference_array_name = array_name;
      reference_minmax = model->GetArray(array_name).minmax.get();
      break;
    }
  }
  // No MinMax info is available to propagate.
  if (!reference_minmax) {
    return false;
  }
  bool changed = false;
  for (const string& array_name : array_names) {
    auto& array = model->GetArray(array_name);
    if (array.minmax) {
      CHECK(MinMaxApproximatelyEqual(*array.minmax, *reference_minmax))
          << "Both the following arrays have minmax, and they disagree: "
          << reference_array_name << " (" << reference_minmax->min << ","
          << reference_minmax->max << ") and " << array_name << " ("
          << array.minmax->min << "," << array.minmax->max
          << "). Expected that either only one of them would have minmax, or "
             "at "
             "least that they would agree.";
    } else {
      array.GetOrCreateMinMax() = *reference_minmax;
      changed = true;
    }
  }
  return changed;
}

bool HardcodeMinMaxForReshape(Model* model, Operator* op) {
  Array& input = model->GetArray(op->inputs[0]);
  Array& output = model->GetArray(op->outputs[0]);

  // If input and output both exist or do not exist, do nothing.
  if ((!input.minmax && !output.minmax) || (input.minmax && output.minmax)) {
    return false;
  }

  // Otherwise propagate info amongst the input and output array.
  return PropagateMinMaxAmongArrays(model, {op->inputs[0], op->outputs[0]});
}

bool HardcodeMinMaxForLstmCell(Model* model, Operator* op) {
  CHECK_EQ(op->inputs.size(), LstmCellOperator::NUM_INPUTS);
  CHECK_EQ(op->outputs.size(), LstmCellOperator::NUM_OUTPUTS);

  bool changed = false;
  changed |= PropagateMinMaxAmongArrays(
      model, {op->inputs[LstmCellOperator::PREV_STATE_INPUT],
              op->outputs[LstmCellOperator::STATE_OUTPUT]});

  auto& input_activations =
      model->GetArray(op->inputs[LstmCellOperator::DATA_INPUT]);
  if (!input_activations.minmax) {
    auto& minmax = input_activations.GetOrCreateMinMax();
    minmax.min = -1;
    minmax.max = 127. / 128.;
    changed = true;
  }

  auto& prev_output_activations =
      model->GetArray(op->inputs[LstmCellOperator::PREV_ACTIV_INPUT]);
  if (!prev_output_activations.minmax) {
    auto& minmax = prev_output_activations.GetOrCreateMinMax();
    minmax.min = -1;
    minmax.max = 127. / 128.;
    changed = true;
  }

  auto& output_concat_temp =
      model->GetArray(op->outputs[LstmCellOperator::CONCAT_TEMP]);
  if (!output_concat_temp.minmax) {
    auto& minmax = output_concat_temp.GetOrCreateMinMax();
    minmax.min = -1;
    minmax.max = 127. / 128.;
    changed = true;
  }

  auto& output_activations =
      model->GetArray(op->outputs[LstmCellOperator::ACTIV_OUTPUT]);
  if (!output_activations.minmax) {
    auto& minmax = output_activations.GetOrCreateMinMax();
    minmax.min = -1;
    minmax.max = 127. / 128.;
    changed = true;
  }

  // (This comment should morph into proper documentation for
  // quantization of LSTM models. It isn't just a local implementation detail,
  // the training code for LSTM models needs to be adjusted to that.)
  //
  // Finally, output_activations_temp holds the output of the fully-connected
  // node inside the LSTM cell. For it, we hardcode a minmax of [-8, 8].
  // The rationale for that is given in a lengthy comment on the LstmCell
  // quantized runtime implementation in reference_ops.h.
  auto& output_activations_temp =
      model->GetArray(op->outputs[LstmCellOperator::ACTIV_TEMP]);
  if (!output_activations_temp.minmax) {
    auto& minmax = output_activations_temp.GetOrCreateMinMax();
    minmax.min = -8;
    minmax.max = 8 * 32767. / 32768.;
    changed = true;
  }

  return changed;
}
}  // namespace

::tensorflow::Status HardcodeMinMax::Run(Model* model, std::size_t op_index,
                                         bool* modified) {
  *modified = false;
  auto it = model->operators.begin() + op_index;
  auto* op = it->get();
  bool changed = false;
  switch (op->type) {
    case OperatorType::kConv:
      changed = HardcodeMinMaxForIm2colArray(model, op);
      break;

    case OperatorType::kL2Normalization:
      changed = HardcodeMinMaxForL2Normalization(model, op);
      break;

    case OperatorType::kRelu:
      // For any normalization other than batch norm, the quantizations ranges
      // before and after relu are expected to be known. Having a quantization
      // op before relu would reduce the number of bits of precision for the
      // activation in half. So we deduce the range before relu from that after
      // the relu. This would eliminate the need for two fake quantization nodes
      // and would not reduce the bits of precision available for activation.
      changed = HardcodeInputMinMaxFromOutput(model, op);
      break;

    case OperatorType::kConcatenation:
      changed = HardcodeMinMaxForConcatenation(model, op);
      break;

    case OperatorType::kSplit:
      changed = HardcodeMinMaxForSplit(model, op);
      break;

    case OperatorType::kAveragePool:
    case OperatorType::kMaxPool:
      changed = HardcodeMinMaxForAverageOrMaxPool(model, op);
      break;

    case OperatorType::kResizeBilinear:
    case OperatorType::kResizeNearestNeighbor:
    case OperatorType::kSlice:
    case OperatorType::kStridedSlice:
    case OperatorType::kSqueeze:
    case OperatorType::kExpandDims:
    case OperatorType::kPad:
    case OperatorType::kGather:
    case OperatorType::kTranspose:
    case OperatorType::kMean:
    case OperatorType::kReduceMax:
    case OperatorType::kReduceMin:
      changed = HardcodeMinMaxFromFirstInput(model, op);
      break;
    case OperatorType::kSum:
      // reduce_sum is expected to change the output range. Hence
      // a fake_quant op is necessary in the output to minimize error. However
      // in special circumstances like when computing expected value using
      // reduce_sum the input range and the output range matches. Hence the
      // below code would act as a fallback. If a fake_quant node is observed in
      // the output that takes precedence over the hard coding logic below.
      changed = HardcodeMinMaxFromFirstInput(model, op);
      if (changed) {
        LOG(WARNING) << "Using the input range for output in reduce_sum op."
                     << "This could have an impact on your model accuracy.";
      }
      break;
    case OperatorType::kSelect:
      changed = HardcodeMinMaxForSelect(model, op);
      break;
    case OperatorType::kLogistic:
      // We hardcode quantization_params to: zero_point=0, scale=1/256.
      // This choice of minmax is the one that is equivalent to that.
      changed = HardcodeMinMaxForOutput(model, op, 0, 255. / 256.);
      break;

    case OperatorType::kSoftmax:
      // We hardcode quantization_params to: zero_point=0, scale=1/256.
      // This choice of minmax is the one that is equivalent to that.
      changed = HardcodeMinMaxForOutput(model, op, 0, 255. / 256.);
      break;

    case OperatorType::kTanh:
      // We hardcode quantization_params to: zero_point=127, scale=1/128.
      // This choice of minmax is the one that is equivalent to that.
      changed = HardcodeMinMaxForOutput(model, op, -127. / 128., 1.0);
      break;

    case OperatorType::kLstmCell:
      changed = HardcodeMinMaxForLstmCell(model, op);
      break;

    case OperatorType::kReshape:
      changed = HardcodeMinMaxForReshape(model, op);
      break;

    default:
      break;
  }
  if (changed) {
    AddMessageF("Hardcoded min-max through %s", LogName(*op));
  }
  *modified = changed;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
