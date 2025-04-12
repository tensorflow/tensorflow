/* Copyright 2023 The OpenXLA Authors.

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

#if defined(INTEL_MKL)

#include "xla/service/cpu/onednn_memory_util.h"

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
#include "xla/service/cpu/runtime_lightweight_check.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla {
namespace cpu {

// Put structure definition together with dependant code
// to simplify consistency maintenance.
struct MemrefInfoPOD {
  int64_t dtype;
  int64_t rank;
  void* data;
  int64_t unused;  // This unused value pads the struct to align with a 64-byte
                   // cacheline
  int64_t dims[kOneDnnMaxNDims];
  int64_t strides[kOneDnnMaxNDims];
};

MemrefInfoHandler CreateMemrefFromShape(const Shape& shape, void* const buf) {
  MemrefInfoHandler result(new MemrefInfoPOD);
  result->dtype = shape.element_type();
  result->rank = shape.dimensions_size();
  auto dimensions = shape.dimensions();
  std::copy(dimensions.begin(), dimensions.end(),
            absl::MakeSpan(result->dims).begin());

  int64_t stride = 1;
  for (int i : shape.layout().minor_to_major()) {
    result->strides[i] = stride;
    stride *= dimensions.at(i);
  }
  result->data = buf;
  return result;
}

MemrefInfoHandler CreateMemrefInfoFromLiteral(const Literal* literal) {
  const auto& shape = literal->shape();
  void* const buf = const_cast<void*>(literal->untyped_data());
  return CreateMemrefFromShape(shape, buf);
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> GetDimsStrides(
    const Shape& shape) {
  // oneDNN handles scalar as a vector of size 1.
  const bool is_scalar = shape.dimensions_size() == 0;
  int64_t rank = is_scalar ? 1 : shape.dimensions_size();
  std::vector<int64_t> strides(rank);
  std::vector<int64_t> scalar_shape(1, 1);
  absl::Span<const int64_t> dimensions =
      is_scalar ? scalar_shape : shape.dimensions();
  std::vector<int64_t> dims(dimensions.begin(), dimensions.end());
  if (is_scalar) {
    strides[0] = 1;
  } else {
    int64_t stride = 1;
    for (int i : shape.layout().minor_to_major()) {
      strides.at(i) = stride;
      stride *= dims.at(i);
    }
  }
  return std::make_pair(dims, strides);
}

StackAlloca GetAllocaAndEmitMemrefInfo(llvm::IRBuilderBase& builder,
                                       const llvm_ir::IrArray& ir_array) {
  const Shape& shape = ir_array.GetShape();
  // oneDNN handles scalar as a vector of size 1.
  int64_t rank = shape.dimensions_size() == 0 ? 1 : shape.dimensions_size();
  auto [dims, strides] = GetDimsStrides(shape);

  // Type of struct
  llvm::Type* i64_type = builder.getInt64Ty();
  llvm::Type* ptr_type = builder.getPtrTy();
  llvm::ArrayType* i64_array_type =
      llvm::ArrayType::get(builder.getInt64Ty(), kOneDnnMaxNDims);
  llvm::StructType* memref_info_type = llvm::StructType::get(
      builder.getContext(),
      {i64_type, i64_type, ptr_type, i64_type, i64_array_type, i64_array_type});

  // Prepare array dims and strides.
  llvm::Value* dims_val = llvm::UndefValue::get(i64_array_type);
  llvm::Value* strides_val = llvm::UndefValue::get(i64_array_type);
  for (unsigned i = 0; i < rank; ++i) {
    llvm::Value* dim_val = builder.getInt64(dims[i]);
    llvm::Value* stride_val = builder.getInt64(strides[i]);
    dims_val = builder.CreateInsertValue(dims_val, dim_val, i);
    strides_val = builder.CreateInsertValue(strides_val, stride_val, i);
  }

  // Prepare values for struct MemrefInfo with padding to align to system
  // cacheline
  llvm::Value* dtype_val = builder.getInt64(shape.element_type());
  llvm::Value* rank_val = builder.getInt64(rank);
  llvm::Value* pad_val = builder.getInt64(0xff);
  llvm::Value* data_ptr = ir_array.GetBasePointer();
  llvm::Value* memref_info_val = llvm::UndefValue::get(memref_info_type);
  memref_info_val = builder.CreateInsertValue(memref_info_val, dtype_val, 0);
  memref_info_val = builder.CreateInsertValue(memref_info_val, rank_val, 1);
  memref_info_val = builder.CreateInsertValue(memref_info_val, data_ptr, 2);
  memref_info_val = builder.CreateInsertValue(memref_info_val, pad_val, 3);
  memref_info_val = builder.CreateInsertValue(memref_info_val, dims_val, 4);
  memref_info_val = builder.CreateInsertValue(memref_info_val, strides_val, 5);

  // Allocate MemrefInfo on the stack
  llvm::Value* memref_info_ptr = llvm_ir::EmitAllocaAtFunctionEntry(
      memref_info_type, "memref.info", &builder);
  builder.CreateLifetimeStart(memref_info_ptr, builder.getInt64(-1));
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
  auto dtype = GetOneDnnDataType();
  auto dims = GetOneDnnDims();
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

int64_t MemrefInfo::GetChannels() const { return pod_->dims[pod_->rank - 1]; }

int64_t MemrefInfo::GetRank() const { return pod_->rank; }

absl::StatusOr<dnnl::memory::desc> TransposeLastTwoDims(
    const dnnl::memory::desc& md) {
  int64_t ndims = md.get_ndims();
  if (ndims < 2) {
    return absl::InvalidArgumentError("Requires at least 2D shape.");
  }
  std::vector<int> permutation(ndims);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[ndims - 1], permutation[ndims - 2]);
  return md.permute_axes(permutation);
}

dnnl::memory::desc ShapeToMemDesc(const Shape& shape) {
  auto [dims, strides] = GetDimsStrides(shape);
  if (dims.empty()) {
    return dnnl::memory::desc{};
  }
  auto dt = ToOneDnnDataType(static_cast<PrimitiveType>(shape.element_type()));
  return dnnl::memory::desc(dims, dt, strides);
}

Shape MemDescToXlaShapeFlattened(const dnnl::memory::desc& md) {
  if (md.is_zero()) {
    LOG(FATAL) << "Memory descriptor is zero.";
  }
  auto dtype = md.get_data_type();
  auto element_size = dnnl::memory::data_type_size(dtype);
  int64_t bytes_num = md.get_size();
  int64_t elements_num = static_cast<int64_t>(bytes_num / element_size);
  return ShapeUtil::MakeShape(ToXlaPrimitiveType(dtype), {elements_num});
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL
