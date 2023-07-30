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

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include "tensorflow/compiler/xla/service/cpu/onednn_memory_util.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <vector>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_lightweight_check.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

namespace xla {
namespace cpu {

// Put structure definition together with dependant code
// to simplify consistency maintenance.
struct MemrefInfoPOD {
  int64_t dtype;
  int64_t rank;
  int64_t dims[kOneDnnMaxNDims];
  int64_t strides[kOneDnnMaxNDims];
  void* data;
};

StackAlloca GetAllocaAndEmitMemrefInfo(llvm::IRBuilder<>& builder,
                                       const llvm_ir::IrArray& ir_array) {
  const Shape& shape = ir_array.GetShape();
  int64_t rank = shape.rank();
  absl::Span<const int64_t> dims = shape.dimensions();

  std::vector<int64_t> strides(rank);
  int64_t stride = 1;
  for (int i : shape.layout().minor_to_major()) {
    strides.at(i) = stride;
    stride *= dims.at(i);
  }

  // Type of struct
  llvm::Type* i64_type = builder.getInt64Ty();
  llvm::Type* ptr_type = builder.getPtrTy();
  llvm::ArrayType* i64_array_type =
      llvm::ArrayType::get(builder.getInt64Ty(), kOneDnnMaxNDims);
  llvm::StructType* memref_info_type = llvm::StructType::get(
      builder.getContext(),
      {i64_type, i64_type, i64_array_type, i64_array_type, ptr_type});

  // Prepare array dims and strides.
  llvm::Value* dims_val = llvm::UndefValue::get(i64_array_type);
  llvm::Value* strides_val = llvm::UndefValue::get(i64_array_type);
  for (unsigned i = 0; i < rank; ++i) {
    llvm::Value* dim_val = builder.getInt64(dims[i]);
    llvm::Value* stride_val = builder.getInt64(strides[i]);
    dims_val = builder.CreateInsertValue(dims_val, dim_val, i);
    strides_val = builder.CreateInsertValue(strides_val, stride_val, i);
  }

  // Prepare values for struct MemrefInfo.
  llvm::Value* dtype_val = builder.getInt64(shape.element_type());
  llvm::Value* rank_val = builder.getInt64(rank);
  llvm::Value* data_ptr = ir_array.GetBasePointer();
  llvm::Value* memref_info_val = llvm::UndefValue::get(memref_info_type);
  memref_info_val = builder.CreateInsertValue(memref_info_val, dtype_val, 0);
  memref_info_val = builder.CreateInsertValue(memref_info_val, rank_val, 1);
  memref_info_val = builder.CreateInsertValue(memref_info_val, dims_val, 2);
  memref_info_val = builder.CreateInsertValue(memref_info_val, strides_val, 3);
  memref_info_val = builder.CreateInsertValue(memref_info_val, data_ptr, 4);

  // Allocate MemrefInfo on the stack
  llvm::Value* memref_info_ptr = llvm_ir::EmitAllocaAtFunctionEntry(
      memref_info_type, "memref.info", &builder);
  llvm::Value* memref_life_start =
      builder.CreateLifetimeStart(memref_info_ptr, builder.getInt64(-1));
  llvm::Value* memref_store =
      builder.CreateStore(memref_info_val, memref_info_ptr);

  return {&builder, memref_info_ptr};
}

MemrefInfo::MemrefInfo(void* pod_data)
    : pod_(reinterpret_cast<MemrefInfoPOD*>(pod_data)) {
  // TODO(intel-tf): verify pod_
}

dnnl::memory::dims MemrefInfo::GetOneDnnDims() const {
  return dnnl::memory::dims(pod_->dims, pod_->dims + pod_->rank);
}

dnnl::memory::dims MemrefInfo::GetOneDnnStrides() const {
  return dnnl::memory::dims(pod_->strides, pod_->strides + pod_->rank);
}

dnnl::memory::data_type MemrefInfo::GetOneDnnDataType() const {
  return ToOneDnnDataType(static_cast<PrimitiveType>(pod_->dtype));
}

dnnl::memory::desc MemrefInfo::GetOneDnnMemDesc() const {
  auto dims = GetOneDnnDims();
  auto dtype = GetOneDnnDataType();
  auto strides = GetOneDnnStrides();
  return dnnl::memory::desc{dims, dtype, strides};
}

void* MemrefInfo::Data() { return pod_->data; }

void MemrefInfo::Print() {
  std::cout << "Data type: " << pod_->dtype << "\t";
  std::cout << "Rank: " << pod_->rank << "\t";
  std::cout << "Dims: [ ";
  for (int i = 0; i < pod_->rank; ++i) {
    std::cout << pod_->dims[i] << " ";
  }
  std::cout << "]\t";

  std::cout << "Strides: [ ";
  for (int i = 0; i < pod_->rank; ++i) {
    std::cout << pod_->strides[i] << " ";
  }
  std::cout << "]\n";
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
