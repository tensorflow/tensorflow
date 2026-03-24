/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/pjrt/pjrt_client.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/utils.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {

PjRtBuffer::ExternalReference::~ExternalReference() = default;

absl::StatusOr<std::uintptr_t> PjRtClient::UnsafeBufferPointer(
    PjRtBuffer* buffer) {
  if (buffer->on_device_shape().IsTuple()) {
    return Unimplemented(
        "unsafe_buffer_pointer is not implemented for tuple buffers.");
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold,
      buffer->AcquireExternalReference());
  const void* ptr = external_reference_hold->OpaqueDeviceMemoryDataPointer();
  return absl::bit_cast<std::uintptr_t>(ptr);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> PjRtClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtBuffer* donated_dst, const Layout* device_layout) {
  return BufferFromHostBuffer(data, type, dims, byte_strides,
                              host_buffer_semantics,
                              std::move(on_done_with_host_buffer),
                              donated_dst->memory_space(), device_layout);
}

Future<> PjRtBuffer::CopyRawToHostFuture(Future<void*> dst, int64_t offset,
                                         int64_t transfer_size) {
  return Future<>(absl::UnimplementedError(
      "PjRtBuffer::CopyRawToHostFuture is not implemented"));
}

std::string CompiledMemoryStats::DebugString() const {
  return absl::Substitute(
      "CompiledMemoryStats("
      "generated_code_size_in_bytes=$0, "
      "argument_size_in_bytes=$1, "
      "output_size_in_bytes=$2, "
      "alias_size_in_bytes=$3, "
      "temp_size_in_bytes=$4, "
      "host_generated_code_size_in_bytes=$5, "
      "host_argument_size_in_bytes=$6, "
      "host_output_size_in_bytes=$7, "
      "host_alias_size_in_bytes=$8, "
      "host_temp_size_in_bytes=$9)",
      generated_code_size_in_bytes, argument_size_in_bytes,
      output_size_in_bytes, alias_size_in_bytes, temp_size_in_bytes,
      host_generated_code_size_in_bytes, host_argument_size_in_bytes,
      host_output_size_in_bytes, host_alias_size_in_bytes,
      host_temp_size_in_bytes);
}

// Defining the first virtual non-pure method, which is usually the virtual
// destructor, makes it a key function. This reduces the program size and takes
// fewer linker resources.
PjRtHostMemoryForDeviceManager::~PjRtHostMemoryForDeviceManager() = default;

CopyToDeviceStream::~CopyToDeviceStream() = default;

absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
PjRtLoadedExecutable::GetCostAnalysis() const {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloCostAnalysis> hlo_cost_analysis,
                      client()->GetHloCostAnalysis());
  return PjRtExecutableUtil::RunHloCostAnalysis(*GetExecutable(),
                                                hlo_cost_analysis.get());
}

PjRtExecutable* PjRtLoadedExecutable::GetExecutable() const {
  return executable_forwarder_.get();
}

absl::StatusOr<Shape> PjRtBuffer::HostShape() {
  Shape device_shape;
  if (!IsTuple()) {
    absl::Span<const int64_t> literal_dims;
    std::optional<std::vector<int64_t>> logical_dims_storage;
    if (has_dynamic_dimensions()) {
      TF_ASSIGN_OR_RETURN(std::vector<int64_t> logical_dims,
                          logical_dimensions());
      logical_dims_storage.emplace(std::move(logical_dims));
      literal_dims = *logical_dims_storage;
    } else {
      literal_dims = dimensions();
    }
    if (element_type() == TOKEN) {
      device_shape = ShapeUtil::MakeTokenShape();
    } else {
      device_shape = ShapeUtil::MakeShape(element_type(), literal_dims);
      // TODO(b/327524065): use PjRtLayout directly instead of xla::Layout
      *device_shape.mutable_layout() = layout()->xla_layout();
    }
  } else {
    // TODO(skyewm): does anything need to create tuple literals? The PJRT C
    // API doesn't support tuples or {logical_}on_device_shape(), so we prefer
    // to use the above non-tuple code path where possible.
    device_shape = on_device_shape();
    if (device_shape.is_dynamic()) {
      TF_ASSIGN_OR_RETURN(device_shape, logical_on_device_shape());
    }
  }
  return ShapeUtil::DeviceShapeToHostShape(device_shape);
}

xla::Future<std::shared_ptr<Literal>> PjRtBuffer::ToLiteral() {
  absl::StatusOr<Shape> host_shape = HostShape();
  if (!host_shape.ok()) {
    return xla::Future<std::shared_ptr<Literal>>(host_shape.status());
  }
  auto [promise, future] = xla::MakePromise<std::shared_ptr<Literal>>();
  auto shared_literal = std::make_shared<Literal>();
  Literal* literal = shared_literal.get();
  LazyToLiteral([literal, host_shape = *std::move(
                              host_shape)]() -> Future<MutableLiteralBase*> {
    auto literal_or = Literal::Make(host_shape);
    if (!literal_or.ok()) {
      return Future<MutableLiteralBase*>(literal_or.status());
    }
    *literal = *std::move(literal_or);
    return Future<MutableLiteralBase*>(literal);
  })
      .OnReady(
          [promise = std::move(promise),
           shared_literal = std::move(shared_literal)](absl::Status s) mutable {
            if (!s.ok()) {
              std::move(promise).Set(s);
            } else {
              std::move(promise).Set(std::move(shared_literal));
            }
          });
  return future;
}

}  // namespace xla
