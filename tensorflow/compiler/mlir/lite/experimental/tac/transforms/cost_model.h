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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_COST_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_COST_MODEL_H_

#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"

namespace mlir {
namespace TFL {
namespace tac {

// TODO(renjieliu): We need to come up with a better strategy to do cost
// estimatation. Maybe build a big lookup table for all the ops.

// TODO(renjieliu): We need to consider what's the default value if we cannot
// analyze the cost.

// ================== Interface  ========================

// Get the estimated cost for the op under the given hardware spec senario.
float GetCostForOp(Operation* op, const std::string& hardware);

// Get the estimated cost for the whole function under the given hardware.
float GetCostForFunc(func::FuncOp* func, const std::string& hardware);

// Get the transfer cost given from & to hardware info.
// We will only calculate for the "necessary" tensor transferred.
// from_graph & to_graph are used to compute the "necessary" tensors.
//     from_graph
//    /    \   \
//  out1   out2  out3
//           \   /
//           to_graph
// So only out2 & out3 are counted.
float GetTransferCost(const std::string& from_hardware_str,
                      const std::string& to_hardware_str,
                      func::CallOp from_graph, func::CallOp to_graph);

// Get the cross quantization/dequantization boundary cost.
float GetQuantDequantCost(InferenceType from_inference_type,
                          InferenceType to_inference_type,
                          func::CallOp from_graph, func::CallOp to_graph);

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_COST_MODEL_H_
