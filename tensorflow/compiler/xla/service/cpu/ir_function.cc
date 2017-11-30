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

#include <iterator>

#include "tensorflow/compiler/xla/service/cpu/ir_function.h"

#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

namespace {
using llvm_ir::AsStringRef;
}  // namespace

namespace cpu {

IrFunction::IrFunction(const string& function_name,
                       llvm::Function::LinkageTypes linkage,
                       const bool optimize_for_size_requested,
                       const bool enable_fast_math, llvm::Module* llvm_module,
                       llvm::IRBuilder<>* ir_builder,
                       int64 num_dynamic_loop_bounds)
    : ir_builder_(ir_builder),
      llvm_module_(llvm_module),
      caller_insert_point_guard_(*ir_builder),
      num_dynamic_loop_bounds_(num_dynamic_loop_bounds) {
  Initialize(function_name, linkage, optimize_for_size_requested,
             enable_fast_math);
}

IrFunction::~IrFunction() {
  // Emit function return value.
  ir_builder_->CreateRetVoid();
}

void IrFunction::Initialize(const string& function_name,
                            llvm::Function::LinkageTypes linkage,
                            const bool optimize_for_size_requested,
                            const bool enable_fast_math) {
  // The function signature is:
  //   void function(i8* retval, i8* run_options, i8** params, i8** temps,
  //                 i64* dynamic_loop_bounds, i64* prof_counters)
  //
  // retval: points to the returned value.
  // params: address of an array with pointers to parameters.
  // temps: address of an array with pointers to temporary buffers.
  //
  // Therefore, the generated function's signature (FunctionType) is statically
  // determined - parameter unpacking is done in code generated into the
  // function, rather than by a prologue dictated by the platform ABI.
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
  //   temps --------->  |  temp  0  |  temp  1  | ..... |  temp  N-1  |
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

  // Even though the type of params and temps is void** in the host's view, in
  // LLVM IR this is represented by i8*, similarly to void*. It's up to the code
  // to use GEPs to unravel the indirection layers.
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(llvm_module_->getContext()),
      /*Params=*/GetComputeFunctionParams(),
      /*isVarArg=*/false);

  // Functions with local linkage get an inlining bonus.  Because we know
  // a-priori that embedded functions (non-entry functions) will not have its
  // name resolved, give it local linkage.
  function_ = llvm::Function::Create(/*Ty=*/function_type,
                                     /*Linkage=*/linkage,
                                     /*N=*/AsStringRef(function_name),
                                     /*M=*/llvm_module_);
  function_->setCallingConv(llvm::CallingConv::C);

  // Set meaningful names for the function's arguments: useful for debugging.
  llvm::Function::arg_iterator arg_iter = function_->arg_begin();
  arg_iter->setName("retval");
  result_arg_ = &*arg_iter;
  (++arg_iter)->setName("run_options");
  exec_run_options_arg_ = &*arg_iter;
  (++arg_iter)->setName("params");
  parameters_arg_ = &*arg_iter;
  (++arg_iter)->setName("temps");
  temp_buffers_arg_ = &*arg_iter;
  if (num_dynamic_loop_bounds_ > 0) {
    (++arg_iter)->setName("dynamic_loop_bounds");
    dynamic_loop_bounds_arg_ = &*arg_iter;
  }
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
    function_->addAttribute(argument.getArgNo() + 1, llvm::Attribute::NoAlias);
  }

  // Add the optize attribute to the function if optimizing for size. This
  // controls internal behavior of some optimization passes (e.g. loop
  // unrolling).
  if (optimize_for_size_requested) {
    function_->addFnAttr(llvm::Attribute::OptimizeForSize);
  }

  if (enable_fast_math) {
    function_->addFnAttr("unsafe-fp-math", "true");
    function_->addFnAttr("no-infs-fp-math", "true");
    function_->addFnAttr("no-nans-fp-math", "true");
    function_->addFnAttr("no-signed-zeros-fp-math", "true");
  }

  ir_builder_->SetInsertPoint(llvm::BasicBlock::Create(
      /*Context=*/llvm_module_->getContext(),
      /*Name=*/"entry",
      /*Parent=*/function_));
}

std::vector<llvm::Type*> IrFunction::GetComputeFunctionParams() {
  llvm::Type* i8_ptr_type =
      llvm::Type::getInt8PtrTy(llvm_module_->getContext());
  llvm::Type* i8_ptr_ptr_type = i8_ptr_type->getPointerTo();
  llvm::Type* i64_ptr_type =
      llvm::Type::getInt64PtrTy(llvm_module_->getContext());
  std::vector<llvm::Type*> compute_function_params(
      {i8_ptr_type, i8_ptr_type, i8_ptr_ptr_type, i8_ptr_ptr_type});
  if (num_dynamic_loop_bounds_ > 0) {
    compute_function_params.push_back(i64_ptr_type);
  }
  compute_function_params.push_back(i64_ptr_type);
  return compute_function_params;
}

llvm::Value* IrFunction::GetDynamicLoopBound(const int64 offset) {
  CHECK_GT(num_dynamic_loop_bounds_, 0);
  CHECK_LT(offset, num_dynamic_loop_bounds_ * 2);
  string name = tensorflow::strings::StrCat("dynamic_loop_bound_", offset);
  return ir_builder_->CreateLoad(
      ir_builder_->CreateGEP(CHECK_NOTNULL(dynamic_loop_bounds_arg_),
                             ir_builder_->getInt64(offset), AsStringRef(name)));
}

}  // namespace cpu
}  // namespace xla
