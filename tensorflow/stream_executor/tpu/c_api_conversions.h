/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_C_API_CONVERSIONS_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_C_API_CONVERSIONS_H_

#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"

// APIs for converting between internal and external versions of
// XLA/StreamExecutor data structures.
namespace ApiConverter {

absl::Span<const float> MakeSpan(const FloatList& src_list);
void CreateVector(const absl::Span<const float> src, FloatList* dst);
void Destroy(FloatList* float_list);

// se::DeviceMemoryBase
SE_DeviceMemoryBase ToC(const stream_executor::DeviceMemoryBase& base);
void ToC(const stream_executor::DeviceMemoryBase& base,
         SE_DeviceMemoryBase* se_base);
stream_executor::DeviceMemoryBase FromC(const SE_DeviceMemoryBase& se_base);
void Destroy(SE_DeviceMemoryBase*);

// xla::Shape
xla::Shape FromC(const XLA_Shape* c_shape);
void ToC(const xla::Shape& xla_shape, XLA_Shape* c_shape);
void Destroy(XLA_Shape* c_shape);

// xla::Layout
xla::Layout FromC(const XLA_Layout* c_layout);
void ToC(const xla::Layout& xla_layout, XLA_Layout* c_layout);
void Destroy(XLA_Layout* c_layout);

// xla::Tile
xla::Tile FromC(const XLA_Tile* c_tile);
void ToC(const xla::Tile& xla_tile, XLA_Tile* c_tile);
void Destroy(XLA_Tile* c_tile);

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

// OwningDeviceMemory
SE_MaybeOwningDeviceMemory ToC(stream_executor::OwningDeviceMemory* mem);
// mem.HasOwnership() may be true if the buffer is aliased and shouldn't be
// released. 'aliased' should be true in this case. 'aliased' has no effect if
// 'mem' is unowned.
SE_MaybeOwningDeviceMemory ToC(xla::MaybeOwningDeviceMemory& mem, bool aliased);

// HloModule
XLA_HloModule ToC(const xla::HloModule& module);
xla::StatusOr<std::unique_ptr<xla::HloModule>> FromC(
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

#endif
