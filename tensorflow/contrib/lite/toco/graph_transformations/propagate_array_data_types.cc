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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

ArrayDataType CommonDataTypeOfAllInputs(const Model& model,
                                        const Operator& op) {
  CHECK_GT(op.inputs.size(), 0);
  const ArrayDataType data_type = model.GetArray(op.inputs[0]).data_type;
  for (const auto& input : op.inputs) {
    const auto& array = model.GetArray(input);
    CHECK(array.data_type == data_type)
        << " Unexpected: this operator has inputs with different data types.";
  }
  return data_type;
}

void SetDataTypeForAllOutputs(Model* model, Operator* op,
                              ArrayDataType data_type) {
  for (const auto& output : op->outputs) {
    model->arrays[output]->data_type = data_type;
  }
}
}  // namespace

bool PropagateArrayDataTypes::Run(Model* model, std::size_t op_index) {
  auto it = model->operators.begin() + op_index;
  auto* op = it->get();

  // If the data type of some input is unknown, we need to yield.
  for (const auto& input : op->inputs) {
    if (model->arrays[input]->data_type == ArrayDataType::kNone) {
      return false;
    }
  }
  // Record data types of output before processing, so we can see at the
  // end if we changed anything, and return the correct boolean value.
  std::unordered_map<string, ArrayDataType> old_output_data_types;
  for (const auto& output : op->outputs) {
    old_output_data_types[output] = model->arrays[output]->data_type;
  }
  // Do the actual output data types propagation.
  if (op->type == OperatorType::kDequantize ||
      op->type == OperatorType::kResizeBilinear) {
    // These operators unconditionally produce float outputs
    SetDataTypeForAllOutputs(model, op, ArrayDataType::kFloat);
  } else if (op->type == OperatorType::kTensorFlowLess ||
             op->type == OperatorType::kTensorFlowLessEqual ||
             op->type == OperatorType::kTensorFlowGreater ||
             op->type == OperatorType::kTensorFlowGreaterEqual) {
    // These operators unconditionally produce bool outputs
    SetDataTypeForAllOutputs(model, op, ArrayDataType::kBool);
  } else if (op->type == OperatorType::kTensorFlowShape) {
    // These operators are assumed to produce int32 outputs.
    SetDataTypeForAllOutputs(model, op, ArrayDataType::kInt32);
  } else if (op->type == OperatorType::kAveragePool ||
             op->type == OperatorType::kMaxPool ||
             op->type == OperatorType::kL2Pool ||
             op->type == OperatorType::kConv ||
             op->type == OperatorType::kDepthwiseConv ||
             op->type == OperatorType::kFullyConnected ||
             op->type == OperatorType::kTensorFlowMax ||
             op->type == OperatorType::kTensorFlowMin ||
             op->type == OperatorType::kPad ||
             op->type == OperatorType::kStridedSlice ||
             op->type == OperatorType::kTensorFlowReshape ||
             op->type == OperatorType::kSlice ||
             op->type == OperatorType::kSqueeze ||
             op->type == OperatorType::kTensorFlowSum ||
             op->type == OperatorType::kTensorFlowSwitch ||
             op->type == OperatorType::kTensorFlowTile ||
             op->type == OperatorType::kTensorFlowAll ||
             op->type == OperatorType::kReorderAxes ||
             op->type == OperatorType::kTensorFlowConcatV2 ||
             op->type == OperatorType::kFloor ||
             op->type == OperatorType::kGather ||
             op->type == OperatorType::kSpaceToBatchND ||
             op->type == OperatorType::kBatchToSpaceND ||
             op->type == OperatorType::kMean) {
    // These operators produce outputs with the same type as their 1st input
    CHECK_GT(op->inputs.size(), 0);
    const ArrayDataType data_type = model->arrays[op->inputs[0]]->data_type;
    SetDataTypeForAllOutputs(model, op, data_type);
  } else if (op->type == OperatorType::kTensorFlowSplit ||
             op->type == OperatorType::kTensorFlowConcat) {
    // These operators produce an output with the same type as their 2nd input
    CHECK_GT(op->inputs.size(), 1);
    const ArrayDataType data_type = model->arrays[op->inputs[1]]->data_type;
    SetDataTypeForAllOutputs(model, op, data_type);
  } else if (op->type == OperatorType::kCast) {
    // Data type of the Cast op is specified.
    CHECK_EQ(op->outputs.size(), 1);
    auto* cast_op = static_cast<CastOperator*>(op);
    model->arrays[op->outputs[0]]->data_type = cast_op->dst_data_type;
  } else if (op->type == OperatorType::kTensorFlowUnsupported) {
    auto* unsupported_op = static_cast<TensorFlowUnsupportedOperator*>(op);
    if (unsupported_op->output_data_types.size() != op->outputs.size()) {
      return false;
    }
    for (int i = 0; i < unsupported_op->output_data_types.size(); ++i) {
      auto output = op->outputs[i];
      auto data_type = unsupported_op->output_data_types[i];
      model->arrays[output]->data_type = data_type;
    }
  } else {
    // These operators produce an output with the same type as any of their
    // inputs, which must always have the same type.
    const ArrayDataType data_type = CommonDataTypeOfAllInputs(*model, *op);
    SetDataTypeForAllOutputs(model, op, data_type);
  }
  // Return true if any output data type changed, false if none changed.
  for (const auto& output : op->outputs) {
    if (old_output_data_types[output] != model->arrays[output]->data_type) {
      return true;
    }
  }
  return false;
}

}  // namespace toco
