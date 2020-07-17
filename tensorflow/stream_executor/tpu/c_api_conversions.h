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
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"
#include "tensorflow/stream_executor/tpu/tpu_stream.h"

class TpuConversions {
 public:
  static stream_executor::DeviceMemoryBase
  SE_DeviceMemoryBaseToDeviceMemoryBase(SE_DeviceMemoryBase se_base) {
    stream_executor::DeviceMemoryBase base(se_base.opaque, se_base.size);
    base.SetPayload(se_base.payload);
    return base;
  }

  static SE_DeviceMemoryBase DeviceMemoryBaseToSE_DeviceMemoryBase(
      const stream_executor::DeviceMemoryBase& base) {
    SE_DeviceMemoryBase se_base;
    se_base.opaque = const_cast<void*>(base.opaque());
    se_base.payload = base.payload();
    se_base.size = base.size();
    return se_base;
  }

  static xla::Shape CShapeToXlaShape(XLA_Shape* shape) {
    xla::ShapeProto p;
    p.ParseFromArray(shape->bytes, shape->size);
    return xla::Shape(p);
  }

  static void XlaShapeToCShape(const xla::Shape& xla_shape,
                               XLA_Shape* c_shape) {
    xla::ShapeProto p = xla_shape.ToProto();
    std::string p_str = p.SerializeAsString();
    c_shape->bytes = new char[p_str.size()];
    c_shape->size = p_str.size();
    memcpy(c_shape->bytes, p_str.data(), p_str.size());
  }

  static XLA_ShapeIndex XlaShapeIndexToCShapeIndex(
      const xla::ShapeIndex& xla_shape) {
    XLA_ShapeIndex c_shape;
    CHECK_LT(xla_shape.size(), 8);
    c_shape.count = xla_shape.size();
    for (int i = 0; i < xla_shape.size(); ++i) {
      c_shape.indices[i] = xla_shape[i];
    }
    return c_shape;
  }

  static xla::ShapeIndex CShapeIndexToXlaShapeIndex(XLA_ShapeIndex* c_shape) {
    return xla::ShapeIndex(&c_shape->indices[0],
                           &c_shape->indices[c_shape->count]);
  }

  static void XLAShapedBufferToCShapedBuffer(
      const xla::ShapedBuffer& buffer, XLA_ShapedBuffer* c_device_buffer) {
    XlaShapeToCShape(buffer.on_host_shape(), &c_device_buffer->on_host_shape);
    XlaShapeToCShape(buffer.on_device_shape(),
                     &c_device_buffer->on_device_shape);
    c_device_buffer->device_ordinal = buffer.device_ordinal();
    absl::InlinedVector<SE_DeviceMemoryBase, 2> bases;
    for (auto& pair : buffer.buffers()) {
      bases.push_back(DeviceMemoryBaseToSE_DeviceMemoryBase(pair.second));
    }
    c_device_buffer->count = bases.size();
    c_device_buffer->bases = new SE_DeviceMemoryBase[bases.size()];
    for (int i = 0; i < bases.size(); ++i) {
      c_device_buffer->bases[i] = bases[i];
    }
  }

  static void XLALiteralToCLiteral(const xla::LiteralSlice& literal,
                                   XLA_Literal* c_literal) {
    XlaShapeToCShape(literal.shape(), &c_literal->shape);
    auto shapes = xla::ShapeUtil::GetLeafShapes(literal.shape());
    c_literal->buffers = new char*[shapes.size()];
    c_literal->sizes = new size_t[shapes.size()];
    c_literal->count = shapes.size();
    for (int i = 0; i < shapes.size(); ++i) {
      c_literal->buffers[i] = reinterpret_cast<char*>(
          const_cast<void*>(literal.untyped_data(shapes[i].index)));
      c_literal->sizes[i] = literal.size_bytes(shapes[i].index);
    }
  }

  static xla::MutableBorrowingLiteral CLiteralToXLALiteral(
      XLA_Literal* c_literal) {
    xla::Shape shape = CShapeToXlaShape(&c_literal->shape);
    return xla::MutableBorrowingLiteral(
        absl::MakeSpan(c_literal->buffers, c_literal->count), shape);
  }

  static void CShapeCleanup(XLA_Shape* c_shape) { delete[] c_shape->bytes; }

  static void CLiteralCleanup(XLA_Literal* c_literal) {
    delete[] c_literal->buffers;
    delete[] c_literal->sizes;
    CShapeCleanup(&c_literal->shape);
  }

  static void CShapedBufferCleanup(XLA_ShapedBuffer* c_buffer) {
    CShapeCleanup(&c_buffer->on_device_shape);
    CShapeCleanup(&c_buffer->on_host_shape);
    delete[] c_buffer->bases;
  }

  static SE_DeviceMemoryAllocator AllocatorToSE_Allocator(
      stream_executor::DeviceMemoryAllocator* allocator) {
    SE_DeviceMemoryAllocator se_allocator;
    if (allocator == nullptr) {
      se_allocator.ctx = nullptr;
      se_allocator.platform = nullptr;
      se_allocator.allocate = nullptr;
      se_allocator.deallocate = nullptr;
      return se_allocator;
    }
    se_allocator.platform =
        static_cast<const tensorflow::TpuPlatform*>(allocator->platform())
            ->se_platform();
    se_allocator.ctx = allocator;
    se_allocator.allocate = [](void* ctx, int device_ordinal, uint64_t size,
                               bool retry_on_failure, int64_t memory_space,
                               SE_ScopedDeviceMemory* memory,
                               SE_Status* se_status) {
      auto allocation =
          reinterpret_cast<stream_executor::DeviceMemoryAllocator*>(ctx)
              ->Allocate(device_ordinal, size, retry_on_failure, memory_space);
      if (!allocation.ok()) {
        auto status = allocation.status();
        TpuStatus_Set(se_status, status.code(), status.error_message().data(),
                      status.error_message().size());
      } else {
        auto& scoped_memory = allocation.ValueOrDie();
        memory->wrapped =
            DeviceMemoryBaseToSE_DeviceMemoryBase(scoped_memory.Release());
        memory->device_ordinal = scoped_memory.device_ordinal();
      }
    };

    se_allocator.deallocate = [](void* ctx, SE_DeviceMemoryBase* base,
                                 int device_ordinal, SE_Status* se_status) {
      auto status =
          reinterpret_cast<stream_executor::DeviceMemoryAllocator*>(ctx)
              ->Deallocate(device_ordinal,
                           SE_DeviceMemoryBaseToDeviceMemoryBase(*base));
      if (!status.ok()) {
        TpuStatus_Set(se_status, status.code(), status.error_message().data(),
                      status.error_message().size());
      }
    };
    return se_allocator;
  }

  static SE_ExecutableRunOptions ExecutableRunOptionsToSE_ExecutableRunOptions(
      const xla::ServiceExecutableRunOptions& options) {
    SE_ExecutableRunOptions se_options;
    se_options.allocator =
        AllocatorToSE_Allocator(options.run_options().allocator());
    se_options.device_ordinal = options.run_options().device_ordinal();
    se_options.stream =
        static_cast<TpuStream*>(options.stream()->implementation())
            ->se_stream();
    return se_options;
  }

  static SE_MaybeOwningDeviceMemory SEOwningDeviceMemoryToC(
      stream_executor::OwningDeviceMemory* mem) {
    SE_MaybeOwningDeviceMemory se_mem;
    se_mem.device_ordinal = mem->device_ordinal();
    se_mem.memory = DeviceMemoryBaseToSE_DeviceMemoryBase(mem->Release());
    se_mem.allocator = AllocatorToSE_Allocator(mem->allocator());
    se_mem.owned = true;
    return se_mem;
  }

  static SE_MaybeOwningDeviceMemory SEMaybeOwningDeviceMemoryToC(
      xla::MaybeOwningDeviceMemory& mem) {
    SE_MaybeOwningDeviceMemory se_mem;
    se_mem.owned = mem.HasOwnership();
    se_mem.memory =
        DeviceMemoryBaseToSE_DeviceMemoryBase(mem.AsDeviceMemoryBase());
    if (mem.HasOwnership()) {
      auto owned = mem.Release().value();
      se_mem.device_ordinal = owned.device_ordinal();
      se_mem.allocator =
          TpuConversions::AllocatorToSE_Allocator(owned.allocator());
    } else {
      se_mem.allocator = AllocatorToSE_Allocator(nullptr);
      se_mem.device_ordinal = -1;
    }
    return se_mem;
  }

  static xla::MaybeOwningDeviceMemory COwningDeviceMemToSEOwningDeviceMem(
      SE_MaybeOwningDeviceMemory* se_mem,
      stream_executor::DeviceMemoryAllocator* allocator) {
    if (se_mem->owned) {
      return xla::MaybeOwningDeviceMemory(stream_executor::OwningDeviceMemory(
          SE_DeviceMemoryBaseToDeviceMemoryBase(se_mem->memory),
          se_mem->device_ordinal, allocator));
    } else {
      return xla::MaybeOwningDeviceMemory(
          SE_DeviceMemoryBaseToDeviceMemoryBase(se_mem->memory));
    }
  }
};

#endif  // THIRD_PARTY_TENSORFLOW_STREAM_EXECUTOR_TPU_C_API_CONVERSIONS_H_
