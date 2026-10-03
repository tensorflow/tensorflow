/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/ir_function.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Value.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/status_macros.h"

namespace xla {
namespace cpu {

static std::vector<llvm::Type*> GetComputeFunctionParams(
    llvm::Module* llvm_module) {
  llvm::Type* ptr_type =
      llvm::PointerType::getUnqual(llvm_module->getContext());
  return std::vector<llvm::Type*>(6, ptr_type);
}

IrFunction::IrFunction(const std::string& function_name,
                       llvm::Function::LinkageTypes linkage,
                       const HloModuleConfig& module_config,
                       llvm::Module* llvm_module, llvm::IRBuilderBase* b)
    : b_(b), llvm_module_(llvm_module), caller_insert_point_guard_(*b) {
  Initialize(function_name, linkage, module_config);
}

IrFunction::IrFunction(llvm::IRBuilderBase* b, llvm::Module* llvm_module,
                       llvm::Function* function,
                       llvm::BasicBlock* return_block)
    : b_(b),
      llvm_module_(llvm_module),
      caller_insert_point_guard_(*b),
      function_(function),
      result_arg_(nullptr),
      exec_run_options_arg_(nullptr),
      parameters_arg_(nullptr),
      buffer_table_arg_(nullptr),
      profile_counters_arg_(nullptr),
      status_arg_(nullptr),
      return_block_(return_block) {};

IrFunction::~IrFunction() {
  // Branch to function return.
  b_->CreateBr(return_block_);
}

void IrFunction::Initialize(const std::string& function_name,
                            llvm::Function::LinkageTypes linkage,
                            const HloModuleConfig& module_config) {
  // The function signature is:
  //   void function(i8* retval, i8* run_options, i8** params, i8**
  //   buffer_table,
  //                 i64* dynamic_loop_bounds, i64* prof_counters)
  //
  // For thread local functions:
  //   retval: points to the returned value.
  //   params: address of an array with pointers to parameters.
  //   buffer_table: is null
  //
  // For global functions:
  //   retval: is null
  //   params: is null
  //   buffer_table: address of an array with pointers to temporary buffers and
  //     entry computation parameters (but not to constant buffers).
  //
  // Therefore, the generated function's signature (FunctionType) is
  // statically determined - parameter unpacking is done in code generated
  // into the function, rather than by a prologue dictated by the platform ABI.
  //
  //                      /--------------\
  //   retval ----------> | return value |
  //                      \--------------/
  //
  //                      /-------------------------------\
  //   run_options -----> | xla::ExecutableRunOptions |
  //                      \-------------------------------/
  //
  //                     /---------------------------------------------\
  //   params -------->  |  param 0  |  param 1  | ..... |  param N-1  |
  //                     |   addr    |   addr    |       |   addr      |
  //                     \---------------------------------------------/
  //                          |           |                   |
  //                          |           |                   |
  //                          V           V                   V
  //                     /---------\  /---------\         /-----------\
  //                     | param 0 |  | param 1 |         | param N-1 |
  //                     \---------/  \---------/         \-----------/
  //
  //                     /---------------------------------------------\
  //   buffer_table--->  |  buff  0  |  guff  1  | ..... |  buff  N-1  |
  //                     |   addr    |   addr    |       |   addr      |
  //                     \---------------------------------------------/
  //                          |           |                   |
  //                          |           |                   |
  //                          V           V                   V
  //                     /---------\  /---------\         /-----------\
  //                     | temp  0 |  | temp  1 |         | temp  N-1 |
  //                     \---------/  \---------/         \-----------/
  //
  //                        /--------------------------------------------\
  // dynamic loop bounds -> | outer_dim0_start | outer_dim0_limit | .....|
  //  (elided for aot)      \--------------------------------------------/
  //
  //                     /---------------------------------------------\
  //   prof counters ->  | counter 0 | counter 1 | ..... | counter N-1 |
  //                     \---------------------------------------------/

  // Even though the type of params and buffer_table is void** in the host's
  // view, in LLVM IR this is represented by i8*, similarly to void*. It's up
  // to the code to use GEPs to unravel the indirection layers.
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(llvm_module_->getContext()),
      /*Params=*/
      GetComputeFunctionParams(llvm_module_),
      /*isVarArg=*/false);

  // Functions with local linkage get an inlining bonus.  Because we know
  // a-priori that embedded functions (non-entry functions) will not have its
  // name resolved, give it local linkage.
  function_ = llvm_ir::CreateCpuFunction(function_type, linkage, module_config,
                                         function_name, llvm_module_);

  // Set meaningful names for the function's arguments: useful for debugging.
  llvm::Function::arg_iterator arg_iter = function_->arg_begin();
  arg_iter->setName("retval");
  result_arg_ = &*arg_iter;
  (++arg_iter)->setName("run_options");
  exec_run_options_arg_ = &*arg_iter;
  (++arg_iter)->setName("params");
  parameters_arg_ = &*arg_iter;
  (++arg_iter)->setName("buffer_table");
  buffer_table_arg_ = &*arg_iter;
  (++arg_iter)->setName("status");
  status_arg_ = &*arg_iter;
  (++arg_iter)->setName("prof_counters");
  profile_counters_arg_ = &*arg_iter;

  // We know a-priori that the function arguments are guaranteed to point to
  // disjoint objects.
  llvm::Argument* retval = result_arg();
  for (llvm::Argument& argument : function_->args()) {
    // However, the return buffer aliases the temporaries and thus cannot be
    // marked noalias.
    if (&argument == retval) {
      continue;
    }
    function_->addParamAttr(argument.getArgNo(), llvm::Attribute::NoAlias);
  }

  return_block_ =
      llvm::BasicBlock::Create(/*Context=*/llvm_module_->getContext(),
                               /*Name=*/"return", /*Parent=*/function_);

  b_->SetInsertPoint(return_block_);
  b_->CreateRetVoid();

  b_->SetInsertPoint(llvm::BasicBlock::Create(
      /*Context=*/llvm_module_->getContext(),
      /*Name=*/"entry",
      /*Parent=*/function_,
      /*InsertBefore=*/return_block_));
}

llvm::Value* EncodeArrayFunctionArguments(
    absl::Span<llvm::Value* const> arguments, absl::string_view name,
    llvm::IRBuilderBase* b) {
  llvm::Value* arguments_buffer;
  if (arguments.empty()) {
    arguments_buffer = llvm::Constant::getNullValue(b->getPtrTy());
  } else {
    arguments_buffer = llvm_ir::EmitAllocaAtFunctionEntryWithCount(
        b->getPtrTy(), b->getInt32(arguments.size()),
        absl::StrCat(name, "_parameter_addresses"), b);

    for (size_t i = 0; i < arguments.size(); i++) {
      llvm::Value* slot_in_param_addresses =
          b->CreateInBoundsGEP(b->getPtrTy(), arguments_buffer, b->getInt64(i));
      b->CreateStore(arguments[i], slot_in_param_addresses);
    }
  }
  return arguments_buffer;
}

// Emits code to allocate an array of parameter address pointers, and store
// each address from 'parameter_addresses'.
// Returns an array of compute function call arguments (including parameter
// address buffer).
std::vector<llvm::Value*> GetArrayFunctionCallArguments(
    absl::Span<llvm::Value* const> parameter_addresses, llvm::IRBuilderBase* b,
    absl::string_view name, llvm::Value* return_value_buffer,
    llvm::Value* exec_run_options_arg, llvm::Value* buffer_table_arg,
    llvm::Value* status_arg, llvm::Value* profile_counters_arg) {
  llvm::Value* parameter_addresses_buffer =
      EncodeArrayFunctionArguments(parameter_addresses, name, b);

  return std::vector<llvm::Value*>{
      return_value_buffer, exec_run_options_arg, parameter_addresses_buffer,
      buffer_table_arg,    status_arg,           profile_counters_arg};
}

}  // namespace cpu
}  // namespace xla
