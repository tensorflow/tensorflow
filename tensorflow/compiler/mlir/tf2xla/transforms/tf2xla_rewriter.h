/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_TF2XLA_REWRITER_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_TF2XLA_REWRITER_H_

#include <memory>
#include <string>

#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/mlir_hlo_builder.h"
#include "tensorflow/core/common_runtime/device_mgr.h"

namespace mlir {
namespace mhlo {

class Tf2XlaRewriter {
 public:
  static mlir::LogicalResult RewriteOp(mlir::Operation* op,
                                       mlir::PatternRewriter& rewriter,
                                       const std::string& device_type,
                                       bool is_module_pass);

 private:
  Tf2XlaRewriter(mlir::Operation* op, mlir::PatternRewriter& rewriter,
                 const std::string& device_type, bool is_module_pass);

  ~Tf2XlaRewriter();

  // Prepares OpKernelContext params common to all the ops.
  // Emits an error on failure.
  mlir::LogicalResult PrepareParams();

  // Tries to legalize the specified TensorFlow op, if supported.
  //
  // Emits an error and returns failure if an error is encountered during
  // conversion. Note that success return value doesn't mean successful
  // legalization.
  mlir::LogicalResult LegalizeOp();

  // Converts the given operand to expression of kind kConstant or kXlaOp.
  // Emits a remark and returns expression of kind kInvalid on failure.
  tensorflow::XlaExpression GetExprForOperand(mlir::Value operand,
                                              mlir::Operation* op);

  mlir::Operation* op_;
  std::string device_type_;

  mlir::PatternRewriter& rewriter_;
  ::xla::MlirHloBuilder hlo_builder_;
  tensorflow::OpOrArgLocNameMapper name_mapper_;

  tensorflow::XlaContext* context_;  // Ref-counted.

  std::unique_ptr<tensorflow::StaticDeviceMgr> device_mgr_;
  tensorflow::Device* device_;  // Owned by device_mgr_;
  std::unique_ptr<tensorflow::ScopedStepContainer> step_container_;
  std::unique_ptr<tensorflow::FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<tensorflow::ProcessFunctionLibraryRuntime> pflr_;
  tensorflow::OpKernelContext::Params params_;
};

}  // namespace mhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_TF2XLA_REWRITER_H_
