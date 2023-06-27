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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_IR_FUNCTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_IR_FUNCTION_H_

#include "absl/types/span.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace cpu {

// IrFunction creates and encapsulates an llvm::Function, exposing methods to
// emitters for function and function argument access.
// The llvm::Function is created with the standard function signature
// used in the XLA CPU backend (see ir_function.cc for argument details).
// In addition IrFunction saves the callers IR insert point during construction,
// and restores it after destruction.
//
// Example usage:
//
//    // Create and initialize new IrFunction.
//    std::unique_ptr<IrFunction> compute_function(new IrFunction(...));
//    // Emit IR for function body using IrFunction helper methods.
//    ...
//    // Store reference to llvm::Function for future invocation.
//    ir_functions.push_back(compute_function.function());
//    // Delete IrFunction (finalizes IR function and restores caller insertion
//    // point).
//    compute_function.reset();
//

class IrFunction {
 public:
  IrFunction(const std::string& function_name,
             llvm::Function::LinkageTypes linkage,
             const HloModuleConfig& module_config, llvm::Module* llvm_module,
             llvm::IRBuilder<>* b, int64_t num_dynamic_loop_bounds);
  ~IrFunction();

  // Emit IR to read and return the set of IR values representing the dynamic
  // loop bounds argument of this function. These bounds delimit the subset
  // of the output that will be written by the computation's root instruction at
  // runtime. This is used for parallel computations, where a single computation
  // is partitioned into N calls to a function with parallel loop bounds, and
  // then called N times in parallel with loop bounds limiting each call to
  // producing 1/N of the output.
  //
  // Each element in returned vector is a pair of ir values representing the
  // loop bounds for a specific dimension, where the first element of the pair
  // is the dimension start index, and the second element of the pair is the
  // dimension limit.
  //
  // EX: [dimension_i_index_start_ir_value, // dimension_i_index_limit_ir_value]
  DynamicLoopBounds GetDynamicLoopBounds();

  // Returns the encapculated llvm::Function.
  llvm::Function* function() { return function_; }

  // Get the llvm::Value* that represents this functions "retval" argument.
  llvm::Argument* result_arg() { return result_arg_; }

  // Get the xla::ExecutableRunOptions that represents this functions
  // "run_options" argument.
  llvm::Value* exec_run_options_arg() { return exec_run_options_arg_; }

  // Get the llvm::Value* that represents this functions parameters argument.
  llvm::Value* parameters_arg() { return parameters_arg_; }

  // Get the llvm::Value* that represents this functions "buffer_table"
  // argument.
  llvm::Value* buffer_table_arg() { return buffer_table_arg_; }

  // Get the llvm::Value* that represents this functions "prof_counters"
  // argument.
  llvm::Value* profile_counters_arg() { return profile_counters_arg_; }

  // Get the llvm::BasicBlock* that contains this function's "ret" instruction.
  llvm::BasicBlock* return_block() { return return_block_; }

  // Get the llvm::Value* that represents this function's "status" argument.
  llvm::Value* status_arg() { return status_arg_; }

 private:
  // Initialize an llvm::Function with standard signature based on arguments.
  void Initialize(const std::string& function_name,
                  llvm::Function::LinkageTypes linkage,
                  const HloModuleConfig& module_config);

  // Emit ir to read and return the ir value for the dynamic loop bound at
  // 'offset' from the "dynamic_loop_bounds" argument of this function.
  llvm::Value* GetDynamicLoopBound(int64_t offset);

  llvm::IRBuilder<>* b_;
  llvm::Module* llvm_module_;
  llvm::IRBuilder<>::InsertPointGuard caller_insert_point_guard_;

  int64_t num_dynamic_loop_bounds_ = 0;
  // Encapsulated llvm::Function.
  llvm::Function* function_;
  // Function argument IR values.
  llvm::Argument* result_arg_;
  llvm::Value* exec_run_options_arg_;
  llvm::Value* parameters_arg_;
  llvm::Value* buffer_table_arg_;
  llvm::Value* dynamic_loop_bounds_arg_ = nullptr;
  llvm::Value* profile_counters_arg_;
  llvm::Value* status_arg_;
  // Basic block containing return.
  llvm::BasicBlock* return_block_;
};

// Returns arguments in `arguments` encoded as a single buffer, suitable for a
// function call.
llvm::Value* EncodeArrayFunctionArguments(
    absl::Span<llvm::Value* const> arguments, absl::string_view name,
    llvm::IRBuilder<>* b);

// Returns an array of compute function call argument ir values.
std::vector<llvm::Value*> GetArrayFunctionCallArguments(
    absl::Span<llvm::Value* const> parameter_addresses, llvm::IRBuilder<>* b,
    absl::string_view name, llvm::Value* return_value_buffer,
    llvm::Value* exec_run_options_arg, llvm::Value* buffer_table_arg,
    llvm::Value* status_arg, llvm::Value* profile_counters_arg);

// Emits a call to a runtime fork/join function which dispatches parallel
// calls to 'parallel_function' (and joins threads before returning).
Status EmitCallToParallelForkJoin(
    const std::vector<llvm::Value*>& arguments, const Shape& shape,
    absl::Span<const int64_t> dimension_partition_counts, llvm::IRBuilder<>* b,
    llvm::Function* parallel_function, absl::string_view name);

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_IR_FUNCTION_H_
