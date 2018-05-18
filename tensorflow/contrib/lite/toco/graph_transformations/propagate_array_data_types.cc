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
void SetDataTypeForAllOutputs(Model* model, Operator* op,
                              ArrayDataType data_type) {
  for (const auto& output : op->outputs) {
    model->GetArray(output).data_type = data_type;
  }
}
}  // namespace

bool PropagateArrayDataTypes::Run(Model* model, std::size_t op_index) {
  auto it = model->operators.begin() + op_index;
  auto* op = it->get();

  // If the data type of some input is unknown, we need to yield.
  for (const auto& input : op->inputs) {
    if (!model->IsOptionalArray(input) &&
        model->GetArray(input).data_type == ArrayDataType::kNone) {
      return false;
    }
  }
  // Record data types of output before processing, so we can see at the
  // end if we changed anything, and return the correct boolean value.
  std::unordered_map<string, ArrayDataType> old_output_data_types;
  for (const auto& output : op->outputs) {
    old_output_data_types[output] = model->GetArray(output).data_type;
  }
  // Do the actual output data types propagation.
  switch (op->type) {
    case OperatorType::kDequantize:
    case OperatorType::kResizeBilinear:
      // These operators unconditionally produce float outputs
      SetDataTypeForAllOutputs(model, op, ArrayDataType::kFloat);
      break;
    case OperatorType::kTensorFlowLess:
    case OperatorType::kTensorFlowLessEqual:
    case OperatorType::kTensorFlowGreater:
    case OperatorType::kTensorFlowGreaterEqual:
      // These operators unconditionally produce bool outputs
      SetDataTypeForAllOutputs(model, op, ArrayDataType::kBool);
      break;
    case OperatorType::kRank:
    case OperatorType::kTensorFlowShape:
      // These operators only produce int32 outputs.
      SetDataTypeForAllOutputs(model, op, ArrayDataType::kInt32);
      break;
    case OperatorType::kTensorFlowSplit:
    case OperatorType::kTensorFlowConcat:
    case OperatorType::kFill: {
      // These operators produce an output with the same type as their 2nd input
      CHECK_GE(op->inputs.size(), 2);
      const ArrayDataType data_type = model->GetArray(op->inputs[1]).data_type;
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    case OperatorType::kTransposeConv: {
      // These operators produce an output with the same type as their 3rd input
      CHECK_GE(op->inputs.size(), 3);
      const ArrayDataType data_type = model->GetArray(op->inputs[2]).data_type;
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    case OperatorType::kCast: {
      // Data type of the Cast op is specified.
      CHECK_EQ(op->outputs.size(), 1);
      auto* cast_op = static_cast<CastOperator*>(op);
      model->GetArray(op->outputs[0]).data_type = cast_op->dst_data_type;
      break;
    }
    case OperatorType::kArgMax: {
      // Data type of the ArgMax op is specified.
      CHECK_EQ(op->outputs.size(), 1);
      auto* argmax_op = static_cast<ArgMaxOperator*>(op);
      model->GetArray(op->outputs[0]).data_type = argmax_op->output_data_type;
      break;
    }
    case OperatorType::kRange: {
      auto* range_op = static_cast<RangeOperator*>(op);
      // Output type of the Range op can be set via an attribute
      ArrayDataType data_type;
      if (range_op->dtype != ArrayDataType::kNone) {
        // Use the type if specified
        data_type = range_op->dtype;
      } else {
        // Otherwise use the first input
        CHECK_GE(op->inputs.size(), 1);
        data_type = model->GetArray(op->inputs[0]).data_type;
      }
      CHECK_EQ(op->outputs.size(), 1);
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    case OperatorType::kRandomUniform: {
      auto* rand_op = static_cast<RandomUniformOperator*>(op);
      // The output type of RandomUniform is specified with an attribute
      if (rand_op->dtype == ArrayDataType::kNone) {
        return false;
      }
      CHECK_EQ(op->outputs.size(), 1);
      SetDataTypeForAllOutputs(model, op, rand_op->dtype);
      break;
    }
    case OperatorType::kTopK_V2: {
      // topk(values: T, k: int32) -> values: T, indices: int32
      CHECK_EQ(op->inputs.size(), 2);
      CHECK_EQ(op->outputs.size(), 2);
      CHECK(model->GetArray(op->inputs[1]).data_type == ArrayDataType::kInt32);
      model->GetArray(op->outputs[0]).data_type = model->GetArray(op->inputs[0]).data_type;
      model->GetArray(op->outputs[1]).data_type = ArrayDataType ::kInt32;
      break;
    }
    case OperatorType::kTensorFlowUnsupported: {
      auto* unsupported_op = static_cast<TensorFlowUnsupportedOperator*>(op);
      // Some output tensors from the op could be eliminated by optimization.
      // This can make unsupported_op->output_data_types have more elements than
      // op->outputs.
      if (unsupported_op->output_data_types.size() < op->outputs.size()) {
        return false;
      }
      for (int i = 0; i < op->outputs.size(); ++i) {
        auto output = op->outputs[i];
        auto data_type = unsupported_op->output_data_types[i];
        model->GetArray(output).data_type = data_type;
      }
      break;
    }
    case OperatorType::kExpandDims: {
      // Yield on ExpandDim until it is converted to Reshape
      return false;
    }
    case OperatorType::kSelect: {
      // Select produces outputs with the same type as their 2nd input
      CHECK_EQ(op->inputs.size(), 3);
      const ArrayDataType data_type_x =
          model->GetArray(op->inputs[1]).data_type;
      const ArrayDataType data_type_y =
          model->GetArray(op->inputs[2]).data_type;
      CHECK(data_type_x == data_type_y);
      SetDataTypeForAllOutputs(model, op, data_type_x);
      break;
    }
    default: {
      // These operators produce outputs with the same type as their 1st input
      CHECK_GT(op->inputs.size(), 0);
      const ArrayDataType data_type = model->GetArray(op->inputs[0]).data_type;
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
  }

  // Return true if any output data type changed, false if none changed.
  for (const auto& output : op->outputs) {
    if (old_output_data_types[output] != model->GetArray(output).data_type) {
      return true;
    }
  }
  return false;
}

}  // namespace toco
