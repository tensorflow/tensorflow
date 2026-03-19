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

#include "xla/pjrt/cpu/abstract_cpu_buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/pjrt/cpu/raw_buffer.h"
#include "xla/pjrt/cpu/tracked_cpu_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/primitive_util.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/cpu_xfeed.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace {

constexpr size_t kSmallDataTransferByteSize = 102400;  // 100 KiB

// Unpacks and copies the packed data at `input` into the literal at the given
// ShapeIndex.
void UnpackIntNToLiteral(PrimitiveType input_element_type,
                         const CpuDeviceMemory& input,
                         MutableLiteralBase* literal,
                         const ShapeIndex& shape_index) {
  absl::Span<const char> input_span{
      static_cast<const char*>(input.untyped_data()), input.size_bytes()};
  size_t output_size = static_cast<size_t>(ShapeUtil::ByteSizeOf(
      ShapeUtil::GetSubshape(literal->shape(), shape_index)));
  absl::Span<char> output_span{
      static_cast<char*>(literal->untyped_data(shape_index)), output_size};
  primitive_util::UnpackIntN(input_element_type, input_span, output_span);
}

}  //  namespace

void PackOrCopy(PrimitiveType element_type, const LiteralSlice& literal,
                void* data, int64_t size) {
  if (primitive_util::IsSubByteNonPredType(element_type)) {
    const int bit_width = primitive_util::BitWidth(element_type);
    absl::Span<const char> src_data_span(
        static_cast<const char*>(literal.untyped_data()), literal.size_bytes());
    absl::Span<char> dst_data_span(static_cast<char*>(data), size);
    PackIntN(bit_width, src_data_span, dst_data_span);
  } else {
    CHECK_EQ(literal.size_bytes(), size);
    std::memcpy(data, literal.untyped_data(), size);
  }
}

/*static*/ bool AbstractCpuBuffer::BufferFromHostBufferSupportsZeroCopy(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides, const Shape& shape) {
  if (byte_strides && !HasMajorToMinorLayout(type, dims, *byte_strides)) {
    return false;
  }
  // Packed arrays are unpacked on host and packed on device.
  if (primitive_util::IsSubByteNonPredType(type)) {
    return false;
  }

  // If the input buffer has a default layout and is sufficiently aligned, we
  // can simply point to the input array's data without any further copies. At
  // the time of writing we require a 16-byte alignment because XLA may generate
  // code which requires it.
  if ((absl::bit_cast<std::uintptr_t>(data) & (cpu::MinAlign() - 1)) != 0) {
    return false;
  }
  return true;
}

}  // namespace xla
