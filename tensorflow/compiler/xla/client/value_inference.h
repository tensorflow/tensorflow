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
#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_VALUE_INFERENCE_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_VALUE_INFERENCE_H_

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
class ValueInference {
 public:
  // ValueInference analyzes values in XlaOp answers following questions:
  // - What's the upper-bound of each value in a tensor.
  // - What's the lower-bound of each value in a tensor.
  // - What's the constant value of each tensor.
  // - Whether or not each value in a tensor is dynamic.
  explicit ValueInference(XlaBuilder* builder) : builder_(builder) {}
  StatusOr<LiteralSlice> AnalyzeUpperBound(XlaOp op) {
    return Unimplemented("Analyzing upper-bound is not implemented yet.");
  }
  StatusOr<LiteralSlice> AnalyzeLowerBound(XlaOp op) {
    return Unimplemented("Analyzing lower-bound is not implemented yet.");
  }
  StatusOr<LiteralSlice> AnalyzeIsDynamic(XlaOp op) {
    return AnalyzeIsDynamic(op.handle());
  }
  StatusOr<LiteralSlice> AnalyzeConstant(XlaOp op) {
    return AnalyzeConstant(op.handle());
  }

 private:
  StatusOr<LiteralSlice> AnalyzeIsDynamic(int64 handle);
  StatusOr<LiteralSlice> AnalyzeConstant(int64 handle);

  StatusOr<Literal> AnalyzeIsDynamicLiteral(int64 handle);
  StatusOr<Literal> AnalyzeConstantLiteral(int64 handle);

  XlaBuilder* builder_;
  // Cache to avoid re-evaluating. Mapping of xla handle to evaluated
  // literals.
  absl::flat_hash_map<int64, Literal> upper_bound_;
  absl::flat_hash_map<int64, Literal> lower_bound_;
  absl::flat_hash_map<int64, Literal> is_dynamic_;
  absl::flat_hash_map<int64, Literal> constant_;
  HloEvaluator evaluator_;
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_VALUE_INFERENCE_H_
