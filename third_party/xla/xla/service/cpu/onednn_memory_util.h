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

#ifndef XLA_SERVICE_CPU_ONEDNN_MEMORY_UTIL_H_
#define XLA_SERVICE_CPU_ONEDNN_MEMORY_UTIL_H_
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include <memory>

#include "dnnl.hpp"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "xla/literal.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

static const int kOneDnnMaxNDims = DNNL_MAX_NDIMS;

struct StackAlloca {
  llvm::IRBuilder<>* builder;
  llvm::Value* value;
  void EmitLifetimeEnd() {
    builder->CreateLifetimeEnd(value, builder->getInt64(-1));
  }
};

// Declare as opaque to put structure definition together with dependant code.
struct MemrefInfoPOD;
using MemrefInfoHandler = std::shared_ptr<MemrefInfoPOD>;

MemrefInfoHandler CreateMemrefInfoFromLiteral(const Literal* literal);

StackAlloca GetAllocaAndEmitMemrefInfo(llvm::IRBuilder<>& builder,
                                       const llvm_ir::IrArray& ir_array);

inline dnnl::memory::data_type ToOneDnnDataType(PrimitiveType ptype) {
  using dt = dnnl::memory::data_type;
  switch (ptype) {
    case S32:
      return dt::s32;
    case U8:
      return dt::u8;
    case S8:
      return dt::s8;
    case F16:
      return dt::f16;
    case BF16:
      return dt::bf16;
    case F32:
      return dt::f32;
    case F64:
      return dt::f64;

    // TODO(intel-tf): properly handle not supported types:
    // S16, S64, U16, U32, U64, C64, C128, F8E5M2, F8E4M3FN, S4, U4,
    // F8E4M3B11FNUZ
    default:
      return dt::undef;
  }
}

inline PrimitiveType ToXlaPrimitiveType(dnnl::memory::data_type dtype) {
  using dt = dnnl::memory::data_type;
  switch (dtype) {
    case dt::s32:
      return PrimitiveType::S32;
    case dt::u8:
      return PrimitiveType::U8;
    case dt::s8:
      return PrimitiveType::S8;
    case dt::f16:
      return PrimitiveType::F16;
    case dt::bf16:
      return PrimitiveType::BF16;
    case dt::f32:
      return PrimitiveType::F32;
    case dt::f64:
      return PrimitiveType::F64;
    // TODO(intel-tf): properly handle not supported type:
    default:
      return PRIMITIVE_TYPE_INVALID;
  }
}

class MemrefInfo {
 public:
  MemrefInfo(void* data);

  dnnl::memory::dims GetOneDnnDims() const;
  dnnl::memory::dims GetOneDnnStrides() const;
  dnnl::memory::data_type GetOneDnnDataType() const;
  dnnl::memory::desc GetOneDnnMemDesc() const;
  void* Data();

  void Print();

  int64_t GetChannels() const;
  int64_t GetRank() const;

 private:
  MemrefInfoPOD* pod_;
};

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
#endif  // XLA_SERVICE_CPU_ONEDNN_MEMORY_UTIL_H_
