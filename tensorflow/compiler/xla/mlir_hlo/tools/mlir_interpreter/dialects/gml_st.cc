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

#include "gml_st/IR/gml_st_ops.h"
#include "tools/mlir_interpreter/dialects/util.h"
#include "tools/mlir_interpreter/framework/interpreter.h"
#include "tools/mlir_interpreter/framework/interpreter_value.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

llvm::SmallVector<InterpreterValue> fusion(InterpreterState& state,
                                           gml_st::FusionOp op,
                                           ArrayRef<InterpreterValue> inputs,
                                           ArrayRef<InterpreterValue> inits) {
  llvm::SmallVector<InterpreterValue> args;
  llvm::append_range(args, inputs);
  llvm::append_range(args, inits);
  return interpret(state, op.getRegion(), args);
}

REGISTER_MLIR_INTERPRETER_OP(fusion);
REGISTER_MLIR_INTERPRETER_OP("gml_st.yield", noOpTerminator);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
