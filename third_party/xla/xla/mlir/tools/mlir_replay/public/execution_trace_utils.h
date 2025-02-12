/* Copyright 2022 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_MLIR_TOOLS_MLIR_REPLAY_PUBLIC_EXECUTION_TRACE_UTILS_H_
#define XLA_MLIR_TOOLS_MLIR_REPLAY_PUBLIC_EXECUTION_TRACE_UTILS_H_

#include "absl/status/statusor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/literal.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_replay/public/execution_trace.pb.h"
#include "xla/xla_data.pb.h"

namespace mlir {
namespace interpreter {

// Interpreter listener that builds a trace of all executed ops and regions.
class ExecutionTraceListener : public InterpreterListener {
 public:
  explicit ExecutionTraceListener(ExecutionTrace* trace) : trace_(trace) {}

  void BeforeOp(ArrayRef<InterpreterValue> args, Operation* op) override;
  void AfterOp(ArrayRef<InterpreterValue> results) override;
  void EnterRegion(ArrayRef<InterpreterValue> bbargs, Region& region) override;
  void LeaveRegion(ArrayRef<InterpreterValue> yielded) override;

 private:
  ExecutionTrace* trace_;
  SmallVector<RegionTrace*> regions_;
};

// Returns an attribute with the given contents and type.
llvm::SmallVector<mlir::Attribute> ValueToAttribute(
    const InterpreterValue& value, mlir::Type type);

// Deserializes the given literal.
absl::StatusOr<InterpreterValue> LiteralToValue(
    const xla::LiteralProto& literal);
// Deserializes the given literal and then casts it to the given type.
absl::StatusOr<InterpreterValue> LiteralToValue(
    const xla::LiteralProto& literal, mlir::Type type);

// Deserializes the given literal.
absl::StatusOr<InterpreterValue> LiteralToValue(const xla::Literal& literal);

// Serializes the given interpreter value.
TracedValue ValueToTracedValue(const InterpreterValue& value);

// Deserializes the given traced value.
absl::StatusOr<InterpreterValue> TracedValueToValue(
    const TracedValue& traced_value);

// Returns all executions of the given op in the given trace.
llvm::SmallVector<const InstructionTrace*> FindOpExecutionsInTrace(
    const ExecutionTrace& trace, mlir::Operation* op);

}  // namespace interpreter
}  // namespace mlir

#endif  // XLA_MLIR_TOOLS_MLIR_REPLAY_PUBLIC_EXECUTION_TRACE_UTILS_H_
