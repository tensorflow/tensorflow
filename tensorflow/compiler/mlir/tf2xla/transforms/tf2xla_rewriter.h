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
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/mlir_hlo_builder.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace mlir {
namespace mhlo {

class Tf2XlaRewriterTestPeer;

class Tf2XlaRewriter {
 public:
  static mlir::LogicalResult RewriteOp(mlir::Operation* op,
                                       mlir::PatternRewriter& rewriter,
                                       const std::string& device_type,
                                       bool is_module_pass,
                                       bool use_tf2xla_hlo_importer);

 private:
  friend class Tf2XlaRewriterTestPeer;

  Tf2XlaRewriter(mlir::Operation* op, mlir::PatternRewriter& rewriter,
                 const std::string& device_type, bool is_module_pass,
                 bool use_tf2xla_hlo_importer);

  ~Tf2XlaRewriter();

  // Compiles the given Operation with XlaBuilder and imports the generated HLO
  // via the HLO -> MHLO importer.
  tsl::StatusOr<mlir::func::FuncOp> CompileWithHloImporter();

  // Import the given XlaComputation into the parent module. Returns the given
  // generated function.
  tsl::StatusOr<mlir::func::FuncOp> ImportXlaComputation(
      xla::XlaComputation& computation);

  // Given the XlaComputation, return a new ModuleOp with MHLO that contains
  // the translated XlaComputation HLO.
  tsl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
  CreateModuleFromXlaComputation(xla::XlaComputation& computation);

  // Prepares OpKernelContext params common to all the ops.
  // Emits an error on failure.
  mlir::LogicalResult PrepareParams();

  // Given the required_consts, it will fill the 3 output vectors with
  // their respective data.
  // Expressions: Output XLA expressions as required by the compiled kernel.
  // Tensors: Vector of tensors that back the TensorValue inputs
  // Inputs: Vector of inputs that are backed by tensors.
  mlir::LogicalResult PrepareKernelInputs(
      const llvm::SmallDenseSet<int>& required_consts,
      std::vector<tensorflow::XlaExpression>& expressions,
      std::vector<tensorflow::Tensor>& tensors,
      std::vector<tensorflow::TensorValue>& inputs);

  mlir::LogicalResult VerifyOpResults(tensorflow::OpKernelContext& op_context);
  mlir::LogicalResult GetKernelOutputs(tensorflow::OpKernelContext& op_context,
                                       mlir::func::FuncOp translated_function,
                                       llvm::SmallVector<Value>& outputs);

  // When using the Hlo Importer, legalize the op into a call to the imported
  // MHLO function.
  mlir::LogicalResult InsertCallToTranslatedFunction(
      func::FuncOp translated_function, llvm::SmallVector<Value>& outputs);

  // Tries to legalize the specified TensorFlow op, if supported.
  //
  // Emits an error and returns failure if an error is encountered during
  // conversion. Note that success return value doesn't mean successful
  // legalization.
  mlir::LogicalResult LegalizeOp();

  // Converts the given operand to expression of kind kConstant or kXlaOp.
  // Emits a remark and returns expression of kind kInvalid on failure.
  tensorflow::XlaExpression GetExprForOperand(mlir::Value operand,
                                              mlir::Operation* op,
                                              int64_t operand_index);

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

  bool use_tf2xla_hlo_importer_;
  xla::XlaBuilder xla_builder_;
};

}  // namespace mhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_TF2XLA_REWRITER_H_
