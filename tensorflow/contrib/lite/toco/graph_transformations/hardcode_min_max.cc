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

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
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
  for (const auto& output : op->outputs) {
    if (model->GetArray(output).minmax) {
      LOG(WARNING) << "Skipping min-max setting for " << LogName(*op)
                   << " because output " << output << " already has min-max.";
      return false;
    }
  }
  // Data is in second input.
  auto& input_array = model->GetArray(op->inputs[1]);
  if (!input_array.minmax) {
    return false;
  } else {
    for (const auto& output : op->outputs) {
      auto& array = model->GetArray(output);
      array.GetOrCreateMinMax() = *input_array.minmax;
    }
    return true;
  }
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
  const auto& input_array_1 = model->GetArray(op->inputs[1]);
  if (!input_array_1.minmax) {
    return false;
  }
  const auto& input_array_2 = model->GetArray(op->inputs[2]);
  if (!input_array_2.minmax) {
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
      CHECK(*array.minmax == *reference_minmax)
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

bool HardcodeMinMax::Run(Model* model, std::size_t op_index) {
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
    case OperatorType::kSlice:
    case OperatorType::kStridedSlice:
    case OperatorType::kSqueeze:
    case OperatorType::kReshape:
    case OperatorType::kPad:
    case OperatorType::kGather:
    case OperatorType::kTranspose:
    case OperatorType::kMean:
      changed = HardcodeMinMaxFromFirstInput(model, op);
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

    default:
      break;
  }
  if (changed) {
    AddMessageF("Hardcoded min-max through %s", LogName(*op));
  }
  return changed;
}

}  // namespace toco
