/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_TPU_C_API_CONVERSIONS_H_
#define XLA_STREAM_EXECUTOR_TPU_C_API_CONVERSIONS_H_

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/xla_data.pb.h"

// APIs for converting between internal and external versions of
// XLA/StreamExecutor data structures.
namespace ApiConverter {

absl::Span<const float> MakeSpan(const FloatList& src_list);
void CreateVector(absl::Span<const float> src, FloatList* dst);
void Destroy(FloatList* float_list);

absl::Span<const int64_t> MakeSpan(const Int64List& src_list);
void CreateVector(absl::Span<const int64_t> src, Int64List* dst);

absl::Span<const int> MakeSpan(const IntList& src_list);
void CreateVector(absl::Span<const int> src, IntList* dst);

absl::Span<const bool> MakeSpan(const BoolList& src_list);
void CreateVector(absl::Span<const bool> src, BoolList* dst);

void CreateVector(absl::Span<const xla::DimLevelType> src, IntList* dst);

// se::DeviceMemoryBase
SE_DeviceMemoryBase ToC(const stream_executor::DeviceMemoryBase& base);
void ToC(const stream_executor::DeviceMemoryBase& base,
         SE_DeviceMemoryBase* se_base);
stream_executor::DeviceMemoryBase FromC(const SE_DeviceMemoryBase& se_base);
void Destroy(SE_DeviceMemoryBase*);

// xla::Tile
xla::Tile FromC(const XLA_Tile* c_tile);
void ToC(const xla::Tile& xla_tile, XLA_Tile* c_tile);
void Destroy(XLA_Tile* c_tile);

// xla::Layout
xla::Layout FromC(const XLA_Layout* c_layout);
void ToC(const xla::Layout& xla_layout, XLA_Layout* c_layout);
void Destroy(XLA_Layout* c_layout);

// xla::Shape
xla::Shape FromC(const XLA_Shape* c_shape);
void ToC(const xla::Shape& xla_shape, XLA_Shape* c_shape);
void Destroy(XLA_Shape* c_shape);

// xla::ShapeIndex
XLA_ShapeIndex ToC(const xla::ShapeIndex& xla_shape);
xla::ShapeIndex FromC(XLA_ShapeIndex* c_shape);
void Destroy(XLA_ShapeIndex*);

// Literal
void ToC(const xla::LiteralSlice& literal, XLA_Literal* c_literal);
xla::MutableBorrowingLiteral FromC(XLA_Literal* c_literal);
void Destroy(XLA_Literal* c_literal);

// ShapedBuffer
void ToC(const xla::ShapedBuffer& buffer, XLA_ShapedBuffer* c_device_buffer);
xla::ShapedBuffer FromC(XLA_ShapedBuffer* c_buffer);
void Destroy(XLA_ShapedBuffer* c_buffer);

// se::DeviceMemoryBase
SE_DeviceMemoryBase ToC(const stream_executor::DeviceMemoryBase& base);
stream_executor::DeviceMemoryBase FromC(const SE_DeviceMemoryBase& se_base);
void Destroy(SE_DeviceMemoryBase*);

// Literal
void ToC(const xla::LiteralSlice& literal, XLA_Literal* c_literal);
xla::MutableBorrowingLiteral FromC(XLA_Literal* c_literal);
void Destroy(XLA_Literal* c_literal);

// ShapedBuffer
void ToC(const xla::ShapedBuffer& buffer, XLA_ShapedBuffer* c_device_buffer);
xla::ShapedBuffer FromC(XLA_ShapedBuffer* c_buffer);
void Destroy(XLA_ShapedBuffer* c_buffer);

// TpuEmbeddingEngineParametersData
struct TpuEmbeddingEngineParametersData {
  // Backing vector for struct
  std::array<std::vector<FloatListRef*>, 8> vectors;
  TpuEmbeddingEngineParameters c_params;
};

std::unique_ptr<TpuEmbeddingEngineParametersData> Create(int num_tables);

xla::MaybeOwningDeviceMemory FromC(
    SE_MaybeOwningDeviceMemory* se_mem,
    stream_executor::DeviceMemoryAllocator* allocator);

// DeviceMemoryAllocator
SE_DeviceMemoryAllocator ToC(stream_executor::DeviceMemoryAllocator* allocator);
stream_executor::DeviceMemoryAllocator* FromC(
    const SE_DeviceMemoryAllocator& c_allocator);

// OwningDeviceMemory
SE_MaybeOwningDeviceMemory ToC(stream_executor::OwningDeviceMemory* mem);
// mem.HasOwnership() may be true if the buffer is aliased and shouldn't be
// released. 'aliased' should be true in this case. 'aliased' has no effect if
// 'mem' is unowned.
SE_MaybeOwningDeviceMemory ToC(xla::MaybeOwningDeviceMemory& mem, bool aliased);

// HloModule
XLA_HloModule ToC(const xla::HloModule& module);
absl::StatusOr<std::unique_ptr<xla::HloModule>> FromC(
    const XLA_HloModule& c_module);
void Destroy(XLA_HloModule* c_module);

// HloModuleConfig
XLA_HloModuleConfig ToC(const xla::HloModuleConfig& config);
xla::HloModuleConfig FromC(const XLA_HloModuleConfig& c_config);
void Destroy(XLA_HloModuleConfig* c_config);

// Helper for managing stack based C -> C++ conversions.
template <class CType>
struct StackHelper {
  explicit StackHelper() {}

  template <class CppType>
  explicit StackHelper(const CppType& t) {
    ::ApiConverter::ToC(t, &value);
  }
  ~StackHelper() { ::ApiConverter::Destroy(&value); }

  template <class CppType>
  CppType AsCpp() const {
    return ::ApiConverter::FromC(&value);
  }

  mutable CType value;
};

}  // namespace ApiConverter

#endif  // XLA_STREAM_EXECUTOR_TPU_C_API_CONVERSIONS_H_
