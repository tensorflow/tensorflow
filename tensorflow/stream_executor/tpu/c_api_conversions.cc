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

#include "tensorflow/stream_executor/tpu/c_api_conversions.h"

#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/tpu/c_api_defn.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"

namespace ApiConverter {

xla::ShapedBuffer FromC(XLA_ShapedBuffer* c_buffer) {
  xla::Shape xla_on_host_shape = ApiConverter::FromC(&c_buffer->on_host_shape);
  xla::Shape xla_on_device_shape =
      ApiConverter::FromC(&c_buffer->on_device_shape);

  xla::ShapeTree<stream_executor::DeviceMemoryBase> xla_shape_tree(
      xla_on_device_shape);
  size_t i = 0;
  for (auto& pair : xla_shape_tree) {
    pair.second = ApiConverter::FromC(c_buffer->bases[i]);
    i++;
  }

  xla::ShapedBuffer xla_shaped_buffer(
      xla_on_host_shape, xla_on_device_shape,
      tensorflow::tpu::TpuPlatformInterface::GetRegisteredPlatform(),
      c_buffer->device_ordinal);
  xla_shaped_buffer.set_buffers(xla_shape_tree);
  return xla_shaped_buffer;
}

SE_MaybeOwningDeviceMemory ToC(xla::MaybeOwningDeviceMemory& mem,
                               bool aliased) {
  SE_MaybeOwningDeviceMemory se_mem;
  se_mem.owned = mem.HasOwnership();
  se_mem.memory = ApiConverter::ToC(mem.AsDeviceMemoryBase());
  if (mem.HasOwnership()) {
    const stream_executor::OwningDeviceMemory* owned =
        mem.AsOwningDeviceMemory();
    se_mem.device_ordinal = owned->device_ordinal();
    se_mem.allocator = ApiConverter::ToC(owned->allocator());
    if (!aliased) {
      // Underlying buffer is owned by se_mem now.
      mem.Release()->Release();
    }
  } else {
    se_mem.allocator =
        ToC(static_cast<stream_executor::DeviceMemoryAllocator*>(nullptr));
    se_mem.device_ordinal = -1;
  }
  return se_mem;
}

xla::MaybeOwningDeviceMemory FromC(
    SE_MaybeOwningDeviceMemory* se_mem,
    stream_executor::DeviceMemoryAllocator* allocator) {
  if (se_mem->owned) {
    return xla::MaybeOwningDeviceMemory(
        stream_executor::OwningDeviceMemory(ApiConverter::FromC(se_mem->memory),
                                            se_mem->device_ordinal, allocator));
  } else {
    return xla::MaybeOwningDeviceMemory(ApiConverter::FromC(se_mem->memory));
  }
}

SE_DeviceMemoryAllocator ToC(
    stream_executor::DeviceMemoryAllocator* allocator) {
  SE_DeviceMemoryAllocator se_allocator;
  if (allocator == nullptr) {
    se_allocator.ctx = nullptr;
    se_allocator.platform = nullptr;
    se_allocator.allocate = nullptr;
    se_allocator.deallocate = nullptr;
    return se_allocator;
  }
  // N.B. Platform is assumed to be the registered backend platform.
  se_allocator.platform = nullptr;
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
      tensorflow::tpu::ExecutorApiFn()->TpuStatus_SetFn(
          se_status, status.code(), status.error_message().data(),
          status.error_message().size());
    } else {
      auto& scoped_memory = allocation.ValueOrDie();
      memory->wrapped = ApiConverter::ToC(scoped_memory.Release());
      memory->device_ordinal = scoped_memory.device_ordinal();
    }
  };

  se_allocator.deallocate = [](void* ctx, SE_DeviceMemoryBase* base,
                               int device_ordinal, SE_Status* se_status) {
    auto status = reinterpret_cast<stream_executor::DeviceMemoryAllocator*>(ctx)
                      ->Deallocate(device_ordinal, ApiConverter::FromC(*base));
    if (!status.ok()) {
      tensorflow::tpu::ExecutorApiFn()->TpuStatus_SetFn(
          se_status, status.code(), status.error_message().data(),
          status.error_message().size());
    }
  };
  return se_allocator;
}

SE_MaybeOwningDeviceMemory ToC(stream_executor::OwningDeviceMemory* mem) {
  SE_MaybeOwningDeviceMemory se_mem;
  se_mem.device_ordinal = mem->device_ordinal();
  se_mem.memory = ApiConverter::ToC(mem->Release());
  se_mem.allocator = ApiConverter::ToC(mem->allocator());
  se_mem.owned = true;
  return se_mem;
}

void ToC(const stream_executor::DeviceMemoryBase& base,
         SE_DeviceMemoryBase* se_base) {
  se_base->opaque = const_cast<void*>(base.opaque());
  se_base->payload = base.payload();
  se_base->size = base.size();
}

SE_DeviceMemoryBase ToC(const stream_executor::DeviceMemoryBase& base) {
  SE_DeviceMemoryBase se_base;
  ToC(base, &se_base);
  return se_base;
}

stream_executor::DeviceMemoryBase FromC(const SE_DeviceMemoryBase& se_base) {
  stream_executor::DeviceMemoryBase base(se_base.opaque, se_base.size);
  base.SetPayload(se_base.payload);
  return base;
}

xla::Shape FromC(const XLA_Shape* shape) {
  xla::ShapeProto p;
  p.ParseFromArray(shape->bytes, shape->size);
  return xla::Shape(p);
}

void ToC(const xla::Shape& xla_shape, XLA_Shape* c_shape) {
  xla::ShapeProto p = xla_shape.ToProto();
  std::string p_str = p.SerializeAsString();
  c_shape->bytes = new char[p_str.size()];
  c_shape->size = p_str.size();
  memcpy(c_shape->bytes, p_str.data(), p_str.size());
}

XLA_ShapeIndex ToC(const xla::ShapeIndex& xla_shape) {
  XLA_ShapeIndex c_shape;
  CHECK_LT(xla_shape.size(), 8);
  c_shape.count = xla_shape.size();
  for (int i = 0; i < xla_shape.size(); ++i) {
    c_shape.indices[i] = xla_shape[i];
  }
  return c_shape;
}

xla::ShapeIndex FromC(XLA_ShapeIndex* c_shape) {
  return xla::ShapeIndex(&c_shape->indices[0],
                         &c_shape->indices[c_shape->count]);
}

void ToC(const xla::LiteralSlice& literal, XLA_Literal* c_literal) {
  ApiConverter::ToC(literal.shape(), &c_literal->shape);
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

xla::MutableBorrowingLiteral FromC(XLA_Literal* c_literal) {
  xla::Shape shape = ApiConverter::FromC(&c_literal->shape);
  return xla::MutableBorrowingLiteral(
      absl::MakeSpan(c_literal->buffers, c_literal->count), shape);
}

void ToC(const xla::ShapedBuffer& buffer, XLA_ShapedBuffer* c_device_buffer) {
  ApiConverter::ToC(buffer.on_host_shape(), &c_device_buffer->on_host_shape);
  ApiConverter::ToC(buffer.on_device_shape(),
                    &c_device_buffer->on_device_shape);
  c_device_buffer->device_ordinal = buffer.device_ordinal();
  absl::InlinedVector<SE_DeviceMemoryBase, 2> bases;
  for (auto& pair : buffer.buffers()) {
    bases.push_back(ApiConverter::ToC(pair.second));
  }
  c_device_buffer->count = bases.size();
  c_device_buffer->bases = new SE_DeviceMemoryBase[bases.size()];
  for (int i = 0; i < bases.size(); ++i) {
    c_device_buffer->bases[i] = bases[i];
  }
}

void Free(XLA_Shape* shape) { delete[] shape->bytes; }
void Free(XLA_ShapeIndex* shape_index) { delete[] shape_index; }
void Free(SE_DeviceMemoryBase*) {}

void Free(XLA_Literal* c_literal) {
  delete[] c_literal->buffers;
  delete[] c_literal->sizes;
  ApiConverter::Free(&c_literal->shape);
}

void Free(XLA_ShapedBuffer* c_buffer) {
  ApiConverter::Free(&c_buffer->on_device_shape);
  ApiConverter::Free(&c_buffer->on_host_shape);
  delete[] c_buffer->bases;
}

XLA_HloModule ToC(const xla::HloModule& module) {
  XLA_HloModule c_module;
  c_module.proto = stream_executor::tpu::SerializeProto(module.ToProto());
  c_module.module_config = ApiConverter::ToC(module.config());
  return c_module;
}

xla::StatusOr<std::unique_ptr<xla::HloModule>> FromC(
    const XLA_HloModule& c_module) {
  xla::HloModuleProto module_proto =
      stream_executor::tpu::DeserializeProto<xla::HloModuleProto>(
          c_module.proto);
  return xla::HloModule::CreateFromProto(
      module_proto, ApiConverter::FromC(c_module.module_config));
}

void Free(XLA_HloModule* c_module) {
  stream_executor::tpu::SerializedProto_Free(c_module->proto);
  Free(&c_module->module_config);
}

static xla::HloModuleConfig ConfigWithLayout(
    const XLA_HloModuleConfig& se_config) {
  xla::ShapeLayout result_layout(
      FromC(&se_config.entry_computation_layout.result_layout));
  xla::ComputationLayout layout(result_layout);
  for (int i = 0; i < se_config.entry_computation_layout.parameter_count; ++i) {
    layout.add_parameter_layout(xla::ShapeLayout(
        FromC(&se_config.entry_computation_layout.parameter_layouts[i])));
  }
  return xla::HloModuleConfig(layout);
}

XLA_HloModuleConfig ToC(const xla::HloModuleConfig& config) {
  XLA_HloModuleConfig hlo_config;

  hlo_config.seed = config.seed();
  hlo_config.launch_id = config.launch_id();
  hlo_config.replica_count = config.replica_count();
  hlo_config.num_partitions = config.num_partitions();
  hlo_config.use_spmd_partitioning = config.use_spmd_partitioning();
  hlo_config.has_static_device_assignment =
      config.has_static_device_assignment();
  hlo_config.has_entry_computation_layout =
      config.has_entry_computation_layout();

  if (config.has_static_device_assignment()) {
    xla::DeviceAssignmentProto dev_proto;
    config.static_device_assignment().Serialize(&dev_proto).IgnoreError();
    hlo_config.static_device_assignment =
        stream_executor::tpu::SerializeProto(dev_proto);
  }
  if (config.has_entry_computation_layout()) {
    const auto& layout = config.entry_computation_layout();
    ApiConverter::ToC(layout.result_layout().shape(),
                      &hlo_config.entry_computation_layout.result_layout);
    hlo_config.entry_computation_layout.parameter_layouts =
        new XLA_Shape[layout.parameter_count()];
    for (int i = 0; i < layout.parameter_count(); ++i) {
      ApiConverter::ToC(
          layout.parameter_layout(i).shape(),
          &hlo_config.entry_computation_layout.parameter_layouts[i]);
    }
    hlo_config.entry_computation_layout.parameter_count =
        layout.parameter_count();
  }
  return hlo_config;
}

xla::HloModuleConfig FromC(const XLA_HloModuleConfig& c_config) {
  xla::HloModuleConfig config = c_config.has_entry_computation_layout
                                    ? ConfigWithLayout(c_config)
                                    : xla::HloModuleConfig();
  config.set_launch_id(c_config.launch_id);
  config.set_seed(c_config.seed);
  config.set_replica_count(c_config.replica_count);
  config.set_num_partitions(c_config.num_partitions);
  config.set_use_spmd_partitioning(c_config.use_spmd_partitioning);
  if (c_config.has_static_device_assignment) {
    auto device_assignment = xla::DeviceAssignment::Deserialize(
        stream_executor::tpu::DeserializeProto<xla::DeviceAssignmentProto>(
            c_config.static_device_assignment));
    config.set_static_device_assignment(
        *(device_assignment.ConsumeValueOrDie()));
  }
  return config;
}

void Free(XLA_HloModuleConfig* c_config) {
  for (auto i = 0; i < c_config->entry_computation_layout.parameter_count;
       ++i) {
    ApiConverter::Free(
        &c_config->entry_computation_layout.parameter_layouts[i]);
  }
  delete[] c_config->entry_computation_layout.parameter_layouts;
  ApiConverter::Free(&c_config->entry_computation_layout.result_layout);
  if (c_config->has_static_device_assignment) {
    stream_executor::tpu::SerializedProto_Free(
        c_config->static_device_assignment);
  }
}

}  // namespace ApiConverter
