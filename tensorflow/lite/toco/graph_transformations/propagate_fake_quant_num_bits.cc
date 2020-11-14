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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/quantization_util.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

bool ChangeArrayDataType(GraphTransformation* transformation, Array* array,
                         ArrayDataType new_data_type,
                         const MinMax* new_minmax) {
  // Ensure the array ends up in the new type (if it hasn't yet been quantized).
  bool data_type_changed = array->final_data_type != new_data_type;
  array->final_data_type = new_data_type;

  if (array->minmax && array->quantization_params && data_type_changed) {
    // The array is already quantized and has min/max info.
    // As we are changing the data type we need to fix up the existing min/max
    // to the new data type range.

    double old_quantized_min, old_quantized_max;
    CHECK(GetQuantizedDataTypeNumericalRange(
        array->data_type, &old_quantized_min, &old_quantized_max))
        << "Existing data type is not quantized: "
        << ArrayDataTypeName(array->data_type);
    double new_quantized_min, new_quantized_max;
    CHECK(GetQuantizedDataTypeNumericalRange(new_data_type, &new_quantized_min,
                                             &new_quantized_max))
        << "New data type is not quantized: "
        << ArrayDataTypeName(new_data_type);

    // Compute new minmax values.
    double min = (old_quantized_min - array->quantization_params->zero_point) *
                 array->quantization_params->scale;
    double max =
        (old_quantized_max + 1 - array->quantization_params->zero_point) *
        array->quantization_params->scale;
    max = max - 1.0 / (new_quantized_max + 1);

    auto& array_minmax = array->GetOrCreateMinMax();
    transformation->AddMessageF(
        "Rescaling min/max from %g,%g (%s) to %g,%g (%s)", array_minmax.min,
        array_minmax.max, ArrayDataTypeName(array->data_type), min, max,
        ArrayDataTypeName(new_data_type));
    array_minmax.min = min;
    array_minmax.max = max;
    ChooseQuantizationParamsForArrayAndQuantizedDataType(
        *array, new_data_type, array->quantization_params.get());
    // Directly change the type as the array was already quantized.
    array->data_type = new_data_type;
    return true;
  }

  // Array has not yet been quantized so we can just set the final data type
  // and assign the new min/max value (if provided).
  if (!array->quantization_params && !array->minmax && new_minmax) {
    transformation->AddMessageF("Forcing new minmax to %g,%g (%s)",
                                new_minmax->min, new_minmax->max,
                                ArrayDataTypeName(new_data_type));
    auto& array_minmax = array->GetOrCreateMinMax();
    array_minmax.min = new_minmax->min;
    array_minmax.max = new_minmax->max;
    return true;
  }

  return data_type_changed;
}

// Returns true if the op blocks our backward recursive data type propagation.
bool DoesOpBlockBackwardPropagation(const Operator& op) {
  switch (op.type) {
    case OperatorType::kConcatenation:
    case OperatorType::kConcat:
    case OperatorType::kConcatV2:
      // Concat shouldn't block propagation, but we do expect that all inputs
      // have the same range.
      return false;
    case OperatorType::kDequantize:
      // Dequantize ops are inserted between the value we care about and the
      // FakeQuant so make sure we move across them.
    case OperatorType::kGather:
      // Gathers need their parameters changed to the appropriate data type.
    case OperatorType::kReshape:
    case OperatorType::kTranspose:
    case OperatorType::kSelect:
    case OperatorType::kTile:
      // Reshapes and transposes don't change values.
    case OperatorType::kRelu:
    case OperatorType::kRelu1:
    case OperatorType::kRelu6:
      // Relus only clamp the output. If min/max of parent is unknown, just
      // prop the range backward. This only happens for cases where activations
      // are not fused to avoid a default being set on the RELU input and
      // propagating forward to the RELU output.
      return false;
    default:
      return true;
  }
}

// Returns true if the input of an op blocks our backward recursive data type
// propagation.
bool DoesOpInputBlockBackwardPropagation(const Operator& op, int input_index) {
  switch (op.type) {
    case OperatorType::kSelect:
      return input_index == 0;
    case OperatorType::kGather:
      // Ignore gather indices.
      return input_index != 0;
      break;
    case OperatorType::kReshape:
    case OperatorType::kTranspose:
      // Ignore reshape/transpose shapes/dimensions.
      return input_index != 0;
    case OperatorType::kTile:
      // Ignore tile multiples.
      return input_index != 0;
    default:
      return false;
  }
}

// Propagates the data type up into the input arrays if they are model inputs
// that may need their type changed. May act recursively if the inputs are
// produced by ops that we can move over (such as Dequantize).
bool RecursivelyBackwardPropagateDataType(GraphTransformation* transformation,
                                          Model* model, Operator* op,
                                          ArrayDataType new_data_type,
                                          const MinMax& new_minmax) {
  bool did_change = false;
  for (size_t input_index = 0; input_index < op->inputs.size(); ++input_index) {
    const auto& input = op->inputs[input_index];
    auto& input_array = model->GetArray(input);

    // Prevent moving into constant param args that we don't want to modify.
    if (DoesOpInputBlockBackwardPropagation(*op, input_index)) {
      continue;
    }

    bool array_did_change = ChangeArrayDataType(transformation, &input_array,
                                                new_data_type, &new_minmax);
    if (array_did_change) {
      transformation->AddMessageF(
          "Adjusting input final data type of array %s from %s to %s", input,
          ArrayDataTypeName(input_array.final_data_type),
          ArrayDataTypeName(new_data_type));
    }
    did_change |= array_did_change;

    // Walk up into all ops producing the inputs to this op.
    for (auto& producing_op : model->operators) {
      if (!DoesOpBlockBackwardPropagation(*producing_op)) {
        for (const auto& output : producing_op->outputs) {
          if (input == output) {
            did_change |= RecursivelyBackwardPropagateDataType(
                transformation, model, producing_op.get(), new_data_type,
                new_minmax);
          }
        }
      }
    }
  }
  return did_change;
}

// Returns true if the op blocks our forward recursive data type propagation.
bool DoesOpBlockForwardPropagation(const Operator& op) {
  switch (op.type) {
    case OperatorType::kFakeQuant:
      // Always stop at another FakeQuant, as it will likely have different
      // parameters.
      return true;
    default:
      return false;
  }
}

// Recurses down the graph setting the data type of all arrays until an operator
// that blocks propagation (like another FakeQuant) or a final_data_type is
// already specified.
bool RecursivelyForwardPropagateDataType(GraphTransformation* transformation,
                                         Model* model, Operator* op,
                                         ArrayDataType new_data_type) {
  bool did_change = false;
  for (const auto& output : op->outputs) {
    auto& output_array = model->GetArray(output);
    if (output_array.final_data_type == new_data_type) {
      // Final data type is already - skip.
      continue;
    }

    if (output_array.final_data_type == ArrayDataType::kNone ||
        output_array.final_data_type != new_data_type) {
      transformation->AddMessageF(
          "Adjusting output final data type of array %s from %s to %s", output,
          ArrayDataTypeName(output_array.final_data_type),
          ArrayDataTypeName(new_data_type));
      did_change |= ChangeArrayDataType(transformation, &output_array,
                                        new_data_type, nullptr);

      // Walk down into all ops consuming the output of this op.
      for (auto& consuming_op : model->operators) {
        if (!DoesOpBlockForwardPropagation(*consuming_op)) {
          for (const auto& input : consuming_op->inputs) {
            if (input == output) {
              did_change |= RecursivelyForwardPropagateDataType(
                  transformation, model, consuming_op.get(), new_data_type);
            }
          }
        }
      }
    }
  }
  return did_change;
}

}  // namespace

// Propagates the num_bits on a FakeQuant operator into the final data types
// of inputs and outputs. For example, if FakeQuant.num_bits==16 then we know
// the output must be int16 and assume all inputs up until the preceding op are
// also 16.
//
// This can be thought of as a bidirectional flood-fill of the num_bits implied
// final_data_type that terminates at other FakeQuant ops (and a few others as
// determined by DoesOpBlockBackwardPropagation/DoesOpBlockForwardPropagation).
// Once all FakeQuant ops have been visited the arrays should all have
// appropriate final_data_types if the source graph was annotated with the
// proper FakeQuant ops.
//
// Annotating a graph requires following a few hard rules:
// - every input MUST have a FakeQuant immediately following it
// - every output MUST have a FakeQuant immediately preceding it
// - important arithmetic ops (such as FullyConnected) SHOULD have a FakeQuant
//   immediately following it
// - all trained weights (RHS of FullyConnected ops, params on Gather ops, etc)
//   MUST have FakeQuants between them and the consuming op
// Additional FakeQuants may be used if desired, especially in areas that may
// suffer from large precision changes - such as between a Softmax and a
// FullyConnected. Only by validating accuracy differences between float
// inference with the FakeQuant ops simulating quantization and the actually
// quantized graph can you be sure the appropriate FakeQuant ops are present.
//
// You can tell if you're missing some FakeQuants by looking for warnings from
// quantize.cc about minmax ranges being determined by the contents of constant
// arrays. This will almost never produce functional models during inference.
//
// As this op may change the data types and ranges of input and output arrays
// downstream tools must also be sure to parse the output model flags to get the
// post-Transform values that may have changed due to this transformation.
//
// This isn't a GraphTransformation in the traditional respect as it affects ops
// outside of the one under transformation. This is primarily so that we can
// utilize the graph traversal and repeated pass system underlying the
// transformation system to exhaustively find all FakeQuant ops. It also gets us
// nice logging and integration with the graphviz video dumping mode.
// In general you should not copy this style of transformation and stick to
// local-only changes as seen in the other transformations.
::tensorflow::Status PropagateFakeQuantNumBits::Run(Model* model,
                                                    std::size_t op_index,
                                                    bool* modified) {
  *modified = false;
  auto it = model->operators.begin() + op_index;
  auto* op = it->get();
  if (op->type != OperatorType::kFakeQuant) {
    return ::tensorflow::Status::OK();
  }
  auto* fakequant_op = static_cast<FakeQuantOperator*>(op);

  ArrayDataType quantized_data_type = ArrayDataType::kNone;
  if (!InferQuantizedDataTypeFromFakeQuant(*fakequant_op,
                                           &quantized_data_type)) {
    AddMessageF("FakeQuant op %s num_bits=%d is out of range, ignoring",
                LogName(*op), fakequant_op->num_bits);
    return ::tensorflow::Status::OK();
  }
  const auto& final_minmax = *fakequant_op->minmax;

  AddMessageF(
      "Beginning propagation of fake quant %s num_bits=%d min=%g max=%g to %s",
      LogName(*op), fakequant_op->num_bits, final_minmax.min, final_minmax.max,
      ArrayDataTypeName(quantized_data_type));

  bool did_change = false;

  // Propagate the FakeQuant information backward up the graph.
  // This will possibly adjust input arrays or constant types (like Gather).
  did_change |= RecursivelyBackwardPropagateDataType(
      this, model, op, quantized_data_type, final_minmax);

  // Propagate the FakeQuant information forward down the graph.
  // This will possibly adjust output arrays.
  did_change |=
      RecursivelyForwardPropagateDataType(this, model, op, quantized_data_type);

  *modified = did_change;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
