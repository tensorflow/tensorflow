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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_SUBGRAPH_QUANTIZER_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_SUBGRAPH_QUANTIZER_H_

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {
namespace internal {

// Quantizes a given subgraph, the subgraph needs to min/max information
// present.
//
// Assumes that some ops like Conv and Depthwise conv are quantized by
// per channel symmetric quantization.
class SubgraphQuantizer {
 public:
  SubgraphQuantizer(ModelT* model, SubGraphT* subgraph,
                    ErrorReporter* error_reporter)
      : model_(model), subgraph_(subgraph), error_reporter_(error_reporter) {}

  // Quantize operator at the given index.
  TfLiteStatus QuantizeOperator(int op_idx);

 private:
  // Quantizes ops with bias tensors.
  TfLiteStatus QuantizeOpWithBias(BuiltinOperator op_code, OperatorT* op);

  // Average and Max pool need special treatement. The scales are propagated
  // from inputs to outputs.
  TfLiteStatus PropagateMinMaxForAvgAndMaxPool(BuiltinOperator op_code,
                                               OperatorT* op);

  // Asymmetric quantizes inputs and outputs of an Op that has single input and
  // single output. E.g. Squeeze.
  TfLiteStatus AsymmetricQuantizeSingleInputOutputOp(BuiltinOperator op_code,
                                                     OperatorT* op);

  TfLiteStatus AsymmetricQuantizeTensor(BuiltinOperator op_code,
                                        int32_t tensor_idx);

  // Returns true if |tensor_idx| is one of the inputs in the subgraph.
  bool IsSubgraphInput(int32_t tensor_idx) const;

  ModelT* model_;
  SubGraphT* subgraph_;
  ErrorReporter* error_reporter_;
};
}  // namespace internal
}  // namespace optimize
}  // namespace tflite
#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_SUBGRAPH_QUANTIZER_H_
