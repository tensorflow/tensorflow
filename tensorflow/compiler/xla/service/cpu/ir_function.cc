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

#include "tensorflow/compiler/xla/service/cpu/ir_function.h"

#include <iterator>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/shape_partition.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {
namespace cpu {

static std::vector<llvm::Type*> GetComputeFunctionParams(
    llvm::Module* llvm_module, const int64_t num_dynamic_loop_bounds) {
  llvm::Type* i8_ptr_type = llvm::Type::getInt8PtrTy(llvm_module->getContext());
  llvm::Type* i8_ptr_ptr_type = i8_ptr_type->getPointerTo();
  llvm::Type* i64_ptr_type =
      llvm::Type::getInt64PtrTy(llvm_module->getContext());
  std::vector<llvm::Type*> compute_function_params(
      {i8_ptr_type, i8_ptr_type, i8_ptr_ptr_type, i8_ptr_ptr_type,
       i8_ptr_type});
  if (num_dynamic_loop_bounds > 0) {
    compute_function_params.push_back(i64_ptr_type);
  }
  compute_function_params.push_back(i64_ptr_type);
  return compute_function_params;
}

IrFunction::IrFunction(const std::string& function_name,
                       llvm::Function::LinkageTypes linkage,
                       const HloModuleConfig& module_config,
                       llvm::Module* llvm_module, llvm::IRBuilder<>* b,
                       int64_t num_dynamic_loop_bounds)
    : b_(b),
      llvm_module_(llvm_module),
      caller_insert_point_guard_(*b),
      num_dynamic_loop_bounds_(num_dynamic_loop_bounds) {
  Initialize(function_name, linkage, module_config);
}

IrFunction::~IrFunction() {
  // Branch to function return.
  b_->CreateBr(return_block_);
}

DynamicLoopBounds IrFunction::GetDynamicLoopBounds() {
  DynamicLoopBounds dynamic_loop_bounds(num_dynamic_loop_bounds_);
  for (int i = 0; i < num_dynamic_loop_bounds_; ++i) {
    dynamic_loop_bounds[i].first = GetDynamicLoopBound(i * 2 + 0);
    dynamic_loop_bounds[i].second = GetDynamicLoopBound(i * 2 + 1);
  }
  return dynamic_loop_bounds;
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
  // view, in LLVM IR this is represented by i8*, similarly to void*. It's up to
  // the code to use GEPs to unravel the indirection layers.
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(llvm_module_->getContext()),
      /*Params=*/
      GetComputeFunctionParams(llvm_module_, num_dynamic_loop_bounds_),
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

llvm::Value* IrFunction::GetDynamicLoopBound(const int64_t offset) {
  CHECK_GT(num_dynamic_loop_bounds_, 0);
  CHECK_LT(offset, num_dynamic_loop_bounds_ * 2);
  llvm::Type* int64_ty = b_->getInt64Ty();
  auto gep = b_->CreateGEP(int64_ty, CHECK_NOTNULL(dynamic_loop_bounds_arg_),
                           b_->getInt64(offset));
  return b_->CreateLoad(int64_ty, gep,
                        "dynamic_loop_bound_" + llvm::Twine(offset));
}

llvm::Value* EncodeArrayFunctionArguments(
    absl::Span<llvm::Value* const> arguments, absl::string_view name,
    llvm::IRBuilder<>* b) {
  llvm::Value* arguments_buffer;
  llvm::Type* int8ptr_ty = b->getInt8PtrTy();
  if (arguments.empty()) {
    arguments_buffer = llvm::Constant::getNullValue(int8ptr_ty->getPointerTo());
  } else {
    arguments_buffer = llvm_ir::EmitAllocaAtFunctionEntryWithCount(
        int8ptr_ty, b->getInt32(arguments.size()),
        absl::StrCat(name, "_parameter_addresses"), b);

    for (size_t i = 0; i < arguments.size(); i++) {
      llvm::Value* parameter_as_i8ptr = b->CreateBitCast(
          arguments[i], b->getInt8PtrTy(),
          absl::StrCat(name, "_parameter_", i, "_address_as_i8ptr"));
      llvm::Value* slot_in_param_addresses =
          b->CreateInBoundsGEP(int8ptr_ty, arguments_buffer, b->getInt64(i));
      b->CreateStore(parameter_as_i8ptr, slot_in_param_addresses);
    }
  }
  return arguments_buffer;
}

// Emits code to allocate an array of parameter address pointers, and store
// each address from 'parameter_addresses'.
// Returns an array of compute function call arguments (including parameter
// address buffer).
std::vector<llvm::Value*> GetArrayFunctionCallArguments(
    absl::Span<llvm::Value* const> parameter_addresses, llvm::IRBuilder<>* b,
    absl::string_view name, llvm::Value* return_value_buffer,
    llvm::Value* exec_run_options_arg, llvm::Value* buffer_table_arg,
    llvm::Value* status_arg, llvm::Value* profile_counters_arg) {
  llvm::Value* parameter_addresses_buffer =
      EncodeArrayFunctionArguments(parameter_addresses, name, b);

  const auto to_int8_ptr = [=](llvm::Value* ptr) {
    return b->CreatePointerCast(ptr, b->getInt8PtrTy());
  };
  return std::vector<llvm::Value*>{to_int8_ptr(return_value_buffer),
                                   to_int8_ptr(exec_run_options_arg),
                                   parameter_addresses_buffer,
                                   buffer_table_arg,
                                   status_arg,
                                   profile_counters_arg};
}

// Emits a call to a runtime fork/join function which dispatches parallel
// calls to 'parallel_function' (and joins threads before returning).
Status EmitCallToParallelForkJoin(
    const std::vector<llvm::Value*>& arguments, const Shape& shape,
    absl::Span<const int64_t> dimension_partition_counts, llvm::IRBuilder<>* b,
    llvm::Function* parallel_function, absl::string_view name) {
  llvm::Module* module = b->GetInsertBlock()->getModule();

  // Build ParallelForkJoin function type.
  std::vector<llvm::Type*> compute_function_params =
      GetComputeFunctionParams(module, /*num_dynamic_loop_bounds=*/0);
  // Number of parallel compute functions.
  compute_function_params.push_back(b->getInt32Ty());
  // Array of partitions. There is an array element for each
  // partition x partition_dim x 2 (for dimension start and limit).
  compute_function_params.push_back(
      llvm::Type::getInt64PtrTy(module->getContext()));
  // Number of partitioned most-major dimensions in 'shape'.
  compute_function_params.push_back(b->getInt32Ty());
  // Function pointer for compute function to be dispatched in parallel.
  compute_function_params.push_back(
      llvm::Type::getInt8PtrTy(module->getContext()));

  llvm::FunctionType* fork_join_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(module->getContext()),
      /*Params=*/compute_function_params,
      /*isVarArg=*/false);

  llvm::Function* fork_join_func = llvm::dyn_cast<llvm::Function>(
      module
          ->getOrInsertFunction(runtime::kParallelForkJoinSymbolName,
                                fork_join_type)
          .getCallee());
  fork_join_func->setCallingConv(llvm::CallingConv::C);
  fork_join_func->setDoesNotThrow();

  // Add common compute function arguments.
  std::vector<llvm::Value*> fork_join_arguments(arguments);

  // Create ShapePartitionIterator to generate all partitions of 'shape'.
  ShapePartitionIterator partition_iterator(shape, dimension_partition_counts);
  const int64_t num_partitions = partition_iterator.GetTotalPartitionCount();
  // Add argument specifying the number of parallel partitions.
  fork_join_arguments.push_back(b->getInt32(num_partitions));

  // The number of partitioned most-major dimensions in 'shape'.
  const int32_t num_partitioned_dims = dimension_partition_counts.size();
  // A dimension partition consists of two elements: [start_index, limit_index).
  const int32_t dim_partition_size = 2;
  // Calculate array partition stride.
  const int32_t array_partition_stride =
      num_partitioned_dims * dim_partition_size;
  // Calculate the total number of elements in the partition array.
  const int32_t partition_array_size =
      dim_partition_size * num_partitioned_dims * num_partitions;

  // Store dimension partition values as llvm constants in 'partitions'.
  // See comments in runtime_fork_join.cc for array layout description.
  std::vector<llvm::Constant*> partitions(partition_array_size);
  for (int32_t i = 0; i < num_partitions; ++i) {
    std::vector<std::pair<int64_t, int64_t>> dim_partitions =
        partition_iterator.GetPartition(i);
    CHECK_EQ(num_partitioned_dims, dim_partitions.size());
    const int32_t partition_index = i * array_partition_stride;
    for (int32_t j = 0; j < num_partitioned_dims; ++j) {
      const std::pair<int64_t, int64_t>& dim_partition = dim_partitions[j];
      const int32_t index = partition_index + j * dim_partition_size;
      // Store partition [dim_start, dim_limit) intervals for each dimension.
      partitions[index] = b->getInt64(dim_partition.first);
      partitions[index + 1] =
          b->getInt64(dim_partition.first + dim_partition.second);
    }
  }

  // Create global variable out of dimension partitions in 'partitions'.
  llvm::ArrayType* partitions_array_type =
      llvm::ArrayType::get(b->getInt64Ty(), partition_array_size);
  llvm::Constant* partitions_array =
      llvm::ConstantArray::get(partitions_array_type, partitions);
  llvm::GlobalVariable* global_partitions_array = new llvm::GlobalVariable(
      /*M=*/*module,
      /*Ty=*/partitions_array_type,
      /*isConstant=*/true,
      /*Linkage=*/llvm::GlobalValue::PrivateLinkage,
      /*Initializer=*/partitions_array,
      /*Name=*/
      absl::StrCat(name, "_parallel_dimension_partitions"));

  // Add argument specifying parallel dimension partitions.
  fork_join_arguments.push_back(
      b->CreateBitCast(global_partitions_array,
                       llvm::Type::getInt64PtrTy(module->getContext())));
  // Add argument specifying the number of partitioned most-major dimensions.
  fork_join_arguments.push_back(b->getInt32(num_partitioned_dims));
  // Add argument for parallel compute function pointer.
  fork_join_arguments.push_back(
      b->CreateBitCast(parallel_function, b->getInt8PtrTy()));
  // Emit call to parallel fork/join.
  b->CreateCall(fork_join_func, fork_join_arguments);

  return OkStatus();
}

}  // namespace cpu
}  // namespace xla
