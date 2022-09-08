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
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_common.h"
#include "tfrt/basic_kernels/opdefs/tfrt_base.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/attributes.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/types.h"  // from @tf_runtime

namespace tfrt {
namespace fallback_sync {

FallbackSyncDialect::FallbackSyncDialect(MLIRContext *context)
    : Dialect(/*name=*/"tfrt_fallback_sync", context,
              TypeID::get<FallbackSyncDialect>()) {
  context->getOrLoadDialect<tfrt::fallback::FallbackDialect>();
  context->getOrLoadDialect<compiler::TFRTDialect>();
  context->getOrLoadDialect<corert::CoreRTDialect>();

  allowUnknownTypes();

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.cpp.inc"
      >();
}

static Type GetTensorType(Builder *builder) {
  return tfrt::t::TensorType::get(builder->getContext());
}

LogicalResult SyncExecuteOp::verify() {
  return fallback_common::VerifyExecuteOpCommon(*this);
}

ParseResult SyncExecuteOp::parse(OpAsmParser &parser, OperationState &result) {
  fallback_common::ParseExecuteOpOptions parse_options;
  parse_options.has_chain = false;
  parse_options.has_key = false;
  parse_options.has_device = false;
  parse_options.has_func_attr = false;
  parse_options.has_cost = false;

  auto &builder = parser.getBuilder();
  return fallback_common::ParseExecuteOpCommon(
      parser, builder, result, GetTensorType(&builder), parse_options);
}

void SyncExecuteOp::print(OpAsmPrinter &p) {
  p << " " << (*this)->getAttr("op_name") << '(' << operands() << ')';

  fallback_common::PrintExecuteOpCommon(p, *this);
  if (!results().empty()) p << " : " << results().size();
}

void SyncExecuteOp::getOpAttrs(
    SmallVectorImpl<std::pair<StringRef, Attribute>> *op_attrs) {
  fallback_common::GetExecuteOpAttrsCommon(
      this->getContext(), this->op_attrs().getValue(), op_attrs);
}

}  // namespace fallback_sync
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.cpp.inc"
