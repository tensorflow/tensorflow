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

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_common_types.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "xla/literal.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

static const int kOneDnnMaxNDims = DNNL_MAX_NDIMS;

struct StackAlloca {
  llvm::IRBuilderBase* builder;
  llvm::Value* value;
  void EmitLifetimeEnd() { builder->CreateLifetimeEnd(value); }
};

// Declare as opaque to put structure definition together with dependant code.
struct MemrefInfoPOD;
using MemrefInfoHandler = std::shared_ptr<MemrefInfoPOD>;

MemrefInfoHandler CreateMemrefInfoFromLiteral(const Literal* literal);

MemrefInfoHandler CreateMemrefFromShape(const Shape& shape, const void* buf);

StackAlloca GetAllocaAndEmitMemrefInfo(llvm::IRBuilderBase& builder,
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
    // F8E4M3B11FNUZ, F8E4M3, F8E3M4, F4E2M1FN, F8E8M0FNU
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
  explicit MemrefInfo(void* pod_data);

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

absl::StatusOr<dnnl::memory::desc> TransposeLastTwoDims(
    const dnnl::memory::desc& md);
#define TRANSPOSE_LAST_TWO_DIMS_IF(pred, mem_desc)        \
  if (pred) {                                             \
    auto trans_mem_desc = TransposeLastTwoDims(mem_desc); \
    CHECK(trans_mem_desc.ok());                           \
    mem_desc = *trans_mem_desc;                           \
  }

dnnl::memory::desc ShapeToMemDesc(const Shape& shape);

Shape MemDescToXlaShapeFlattened(const dnnl::memory::desc& md);

// Base resources: common arg/result memrefs.
struct OneDnnBaseResources {
  std::vector<MemrefInfoHandler> arg_memrefs;
  std::vector<MemrefInfoHandler> result_memrefs;
  OneDnnBaseResources() = default;
  virtual ~OneDnnBaseResources() = default;
};

// oneDNN primitive resources.
struct OneDnnPrimResources : public OneDnnBaseResources {
  dnnl::primitive primitive;
  dnnl::memory src_mem;
  dnnl::memory wei_mem;
  dnnl::memory dst_mem;
  dnnl::memory scratch_mem;
  dnnl::memory scale_mem;
  dnnl::memory shift_mem;
  std::vector<std::pair<int, dnnl::memory>> postop_args;

  OneDnnPrimResources()
      : primitive(),
        src_mem(),
        wei_mem(),
        dst_mem(),
        scratch_mem(),
        scale_mem(),
        shift_mem(),
        postop_args() {}
};

// TODO(intel-tf): Add a child struct of OneDnnBaseResources for oneDNN graph.

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_ONEDNN_MEMORY_UTIL_H_
