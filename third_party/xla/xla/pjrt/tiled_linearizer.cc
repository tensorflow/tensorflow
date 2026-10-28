/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/tiled_linearizer.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/tiled_linearizer_highway.h"
#include "xla/pjrt/transpose.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tpu/noncopyable_buffer.h"
#include "xla/tsl/platform/byte_order.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla {
namespace {

using Buffer = tensorflow::tpu::NoncopyableBuffer;

bool HasLinearLayout(const Layout& layout) {
  return !layout.tiles().empty() &&
         layout.tiles()[0].dimensions() == absl::Span<const int64_t>({1});
}

template <int N>
struct ComponentTraits {};
template <>
struct ComponentTraits<64> {
  static constexpr PrimitiveType kFloatingPointComponent = F32;
  static constexpr PrimitiveType kIntegralComponent = U32;
};
template <>
struct ComponentTraits<128> {
  static constexpr PrimitiveType kFloatingPointComponent = F64;
  static constexpr PrimitiveType kIntegralComponent = U64;
};

template <int N>
constexpr PrimitiveType ComponentType(const PrimitiveType type) {
  if (!primitive_util::IsArrayType(type) ||
      primitive_util::BitWidth(type) != N) {
    return type;
  }
  if (primitive_util::IsComplexType(type)) {
    return primitive_util::ComplexComponentType(type);
  }
  if (primitive_util::IsFloatingPointType(type)) {
    return ComponentTraits<N>::kFloatingPointComponent;
  }
  return ComponentTraits<N>::kIntegralComponent;
}

template <int N>
Shape ComponentShape(const Shape& shape) {
  if (shape.IsTuple()) {
    std::vector<Shape> component_shapes;
    component_shapes.reserve(shape.tuple_shapes().size());
    for (const Shape& tuple_shape : shape.tuple_shapes()) {
      component_shapes.push_back(ComponentShape<N>(tuple_shape));
    }
    return ShapeUtil::MakeTupleShape(component_shapes);
  }
  if (!ShapeUtil::ElementHasBitWidth(shape, N)) {
    return shape;
  }
  return ShapeUtil::ChangeElementType(shape,
                                      ComponentType<N>(shape.element_type()));
}

constexpr PrimitiveType X64ComponentType(const PrimitiveType x64_type) {
  return ComponentType<64>(x64_type);
}

Shape X64ComponentShape(const Shape& shape) {
  return ComponentShape<64>(shape);
}

Shape X128ComponentShape(const Shape& shape) {
  return ComponentShape<128>(shape);
}

// The value here is bit_cast<uint32_t>(NAN), to more easily catch errors
// related to ops depending on pad values that they should not depend on. To be
// BF16 compatible, the value is still NaN when unpacked into two BF16 values.
constexpr uint32_t kPadValue = 0x7fc07fc0;

// TensorCore lane count as constexpr used for template specialization.
template <int64_t kTcLaneCount>
using TcLaneCountConstant = std::integral_constant<int64_t, kTcLaneCount>;

template <typename F>
absl::Status TcLaneCountSwitch(F&& f, int64_t lane_count) {
  switch (lane_count) {
    case 128:
      std::invoke(f, TcLaneCountConstant<128>());
      return absl::OkStatus();
    default:
      return absl::UnimplementedError(
          absl::StrCat("unhandled lane_count ", lane_count));
  }
}

// TensorCore lane count as constexpr used for template specialization.
template <int64_t kElementPackingFactor>
using ElementPackingFactorConstant =
    std::integral_constant<int64_t, kElementPackingFactor>;

template <typename F>
auto ElementPackingFactorSwitch(F&& f, const int64_t packing_factor) {
  switch (packing_factor) {
    case 1:
      return std::invoke(f, ElementPackingFactorConstant<1>());
    case 2:
      return std::invoke(f, ElementPackingFactorConstant<2>());
    case 4:
      return std::invoke(f, ElementPackingFactorConstant<4>());
    case 8:
      return std::invoke(f, ElementPackingFactorConstant<8>());
    case 16:
      return std::invoke(f, ElementPackingFactorConstant<16>());
    case 32:
      return std::invoke(f, ElementPackingFactorConstant<32>());
    default:
      LOG(FATAL) << "unhandled packing_factor" << packing_factor;
  }
}

template <typename R, typename F>
R ElementPackingFactorSwitch(F&& f, const int64_t packing_factor) {
  return ElementPackingFactorSwitch(std::forward<F>(f), packing_factor);
}

std::string FormatLinearizedLiteral(const Shape& literal_shape,
                                    absl::Span<const uint32_t> values) {
  return absl::StrJoin(
      values, ", ", [&literal_shape](std::string* out, uint64_t value) {
        auto component_type = X64ComponentType(literal_shape.element_type());
        primitive_util::PrimitiveTypeSwitch<void>(
            [&](auto constant_type) {
              if constexpr (primitive_util::IsArrayType(constant_type) &&
                            primitive_util::BitWidth(constant_type) <= 32) {
                using NativeT = primitive_util::NativeTypeOf<constant_type>;
                using BitsT = UnsignedIntegerTypeForSizeType<sizeof(NativeT)>;
                constexpr auto kBitWidth =
                    primitive_util::BitWidth(constant_type);
                constexpr int kSubElements = 32 / kBitWidth;
                for (int i = 0; i < kSubElements; ++i) {
                  if (i > 0) {
                    absl::StrAppendFormat(out, ", ");
                  }
                  auto bits = static_cast<BitsT>(value);
                  if constexpr (primitive_util::IsFloatingPointType(
                                    constant_type)) {
                    auto native = Eigen::numext::bit_cast<NativeT>(bits);
                    absl::StrAppendFormat(out, "%g",
                                          static_cast<float>(native));
                  } else if constexpr (primitive_util::IsSignedIntegralType(
                                           constant_type)) {
                    auto native = static_cast<NativeT>(value);
                    absl::StrAppendFormat(out, "%d",
                                          static_cast<int32_t>(native));
                  } else {
                    absl::StrAppendFormat(out, "%#x", bits);
                  }
                  value >>= kBitWidth;
                }
              }
            },
            component_type);
      });
}

template <int64_t kPackingFactor>
void RepackFromSparseCoreLayout(absl::Span<const uint32_t> input,
                                absl::Span<uint8_t> output) {
  const size_t output_bytes = output.size();
  if constexpr (kPackingFactor <= 4) {
    std::memcpy(output.data(), input.data(), output_bytes);
  } else {
    constexpr int kBitsPerElement = 32 / kPackingFactor;
    const size_t num_elements = output_bytes;
    std::memset(output.data(), 0, output_bytes);
    const uint8_t* input_bytes = reinterpret_cast<const uint8_t*>(input.data());
    for (size_t i = 0; i < num_elements; ++i) {
      size_t byte_idx = (i * kBitsPerElement) / CHAR_BIT;
      size_t bit_offset = (i * kBitsPerElement) % CHAR_BIT;
      uint8_t mask = (1 << kBitsPerElement) - 1;
      output[i] = (input_bytes[byte_idx] >> bit_offset) & mask;
    }
  }
}

absl::Span<uint8_t> GetLiteralAsMutableSpan(MutableLiteralBase* literal) {
  return absl::MakeSpan(
      tsl::safe_reinterpret_cast<uint8_t*>(literal->untyped_data()),
      literal->size_bytes());
}

void CreateGarbageFilledLiteral(const Shape& shape,
                                MutableLiteralBase* literal) {
  // Use Compatible instead of Equal because host-offloaded tensors may have
  // layout/tiling differences (such as linear tiled layouts T(1)L(1024))
  // between the device shape and the host literal shape, but their logical
  // shapes (dimensions and element type) are compatible.
  DCHECK(ShapeUtil::Compatible(shape, literal->shape()))
      << "shape: " << shape.ToString(true)
      << ", literal->shape(): " << literal->shape().ToString(true);
  primitive_util::ArrayTypeSwitch(
      [&](auto ptype) {
        using NativeT = primitive_util::NativeTypeOf<ptype>;
        if constexpr (ptype == PRED) {
          literal->PopulateWithValue<bool>(false);
          return;
        }
        if constexpr (primitive_util::IsSignedIntegralType(ptype)) {
          constexpr int64_t kGarbage{0x5a5a5a5a5a5a5a5a};
          literal->PopulateWithValue<NativeT>(static_cast<NativeT>(kGarbage));
          return;
        }
        if constexpr (primitive_util::IsUnsignedIntegralType(ptype)) {
          constexpr uint64_t kGarbage{0xcdcdcdcdcdcdcdcd};
          literal->PopulateWithValue<NativeT>(static_cast<NativeT>(kGarbage));
          return;
        }
        if constexpr (primitive_util::IsFloatingPointType(ptype)) {
          const NativeT kGarbage =
              static_cast<NativeT>(std::numeric_limits<NativeT>::quiet_NaN());
          literal->PopulateWithValue<NativeT>(kGarbage);
          return;
        }
        if constexpr (primitive_util::IsComplexType(ptype)) {
          using ComponentT = typename NativeT::value_type;
          constexpr ComponentT kGarbage =
              std::numeric_limits<ComponentT>::quiet_NaN();
          literal->PopulateWithValue<NativeT>(NativeT{kGarbage, kGarbage});
          return;
        }
        LOG(FATAL) << "unhandled primitive type: "
                   << PrimitiveType_Name(ptype());
      },
      shape.element_type());
}
}  // namespace

namespace {

// Reorders a 4/8/16-bit packed input to the hardware expected layout: For
// each sublane, split the inputs into four/two parts, then zip four/two
// elements from the same lane together.
//
// `size` represents the number of 32-bit elements that need to be repacked.
//
// Precondition: Input size is a multiple of chunk lanes. If source buffer and
// target buffer aliases, then source_buffer == target_buffer must hold.
//
// Example for BF16: "|" is the pack operation for two 16 bits number.
//
// Input:  a1  |a2   a3  |a4    ....     a127|a128 (64 elements, 256 bytes)
//         b1  |b2   b3  |b4    ....     b127|b128 (64 elements, 256 bytes)
//         c1  |c2   c3  |c4    ....     c127|c128 (64 elements, 256 bytes)
//         d1  |d2   d3  |d4    ....     d127|d128 (64 elements, 256 bytes)
//
// Output: b1  |a1   b2  |a2    ....     b128|a128 (128 elements, 512 bytes)
//         d1  |c1   d2  |c2    ....     d128|c128 (128 elements, 512 bytes)
//
template <int64_t kPackingFactor, int64_t kLaneCount>
void RepackToHardwareLayout(absl::Span<const uint8_t> source_buffer,
                            uint32_t* target_buffer, bool set_padding_to_nan) {
  static_assert(kLaneCount % kPackingFactor == 0,
                "ChunkLanes not a multiple of kPackingFactor");
  constexpr int kTpuToCpuExpansionFactor =
      CeilOfRatio<size_t>(kPackingFactor, sizeof(uint32_t));
  constexpr int kExpandedLaneCount = kLaneCount * kTpuToCpuExpansionFactor;
  uint32_t staging_buffer[kExpandedLaneCount];
  constexpr int kBytesReadPerSublaneWritten = sizeof(staging_buffer);
  const size_t input_bytes = source_buffer.size();
  const size_t full_reads = input_bytes / kBytesReadPerSublaneWritten;
  const size_t partial_bytes_remaining =
      input_bytes % kBytesReadPerSublaneWritten;
  bool is_alias = source_buffer.data() ==
                  tsl::safe_reinterpret_cast<uint8_t*>(target_buffer);

  // Don't leave uninitialized memory if we are in MSAN mode.
#ifdef MEMORY_SANITIZER
  set_padding_to_nan = true;
#endif

  if constexpr (kPackingFactor == 1) {
    if (!is_alias) {
      std::memcpy(target_buffer, source_buffer.data(), input_bytes);
    }
  } else {
    for (size_t i = 0; i < full_reads; ++i) {
      // If inputs are pointer aliases, copy source to staging buffer so we can
      // modify source/target in place. Compiler actually optimizes the memcpy
      // away by prefetching a sublane into vector registers.
      // NOTE: Another way to handle pointer aliases is
      //
      // ```
      // staging = is_alias ? staging_buffer : target_sublane
      // RepackToHardwareLayoutImpl(source_sublane, staging)
      // if (is_alias) memcpy(target_sublane, staging_buffer)
      // ```
      //
      // However, compiler does not optimize the memcpy away in this case.
      if (is_alias) {
        __builtin_memcpy_inline(staging_buffer,
                                &source_buffer[kBytesReadPerSublaneWritten * i],
                                sizeof(staging_buffer));
      }
      const uint32_t* staging =
          is_alias ? staging_buffer
                   : tsl::safe_reinterpret_cast<const uint32_t*>(
                         &source_buffer[kBytesReadPerSublaneWritten * i]);

      RepackToHardwareLayoutSublaneHighway<kPackingFactor, kLaneCount>(
          staging, &target_buffer[kLaneCount * i]);
    }
    if (partial_bytes_remaining > 0) {
      uint8_t* first_uninitialized_byte = std::copy_n(
          tsl::safe_reinterpret_cast<const uint8_t*>(
              &source_buffer[kBytesReadPerSublaneWritten * full_reads]),
          partial_bytes_remaining,
          tsl::safe_reinterpret_cast<uint8_t*>(staging_buffer));
      // The caller may want to make sure uninitialized memory does not make it
      // to the linearized literal. This is important for executable
      // reproducibility as literals get baked into the TPU program data
      // segment.
      if (set_padding_to_nan) {
        // Fill the rest of the staging buffer with an all-ones bit-pattern.
        // This bit-pattern is nice because it is a NaN regardless of the width
        // of the exponent field and as such works with F32, F16, BF16, etc.
        std::fill_n(first_uninitialized_byte,
                    kBytesReadPerSublaneWritten - partial_bytes_remaining, -1);
      } else {
        std::fill_n(first_uninitialized_byte,
                    kBytesReadPerSublaneWritten - partial_bytes_remaining, 0);
      }
      RepackToHardwareLayoutSublaneHighway<kPackingFactor, kLaneCount>(
          staging_buffer, &target_buffer[kLaneCount * full_reads]);
    }
  }
}

// Reorders a 4/8/16-bit packed input to the host expected layout. This is the
// reverse transform of RepackToHardwareLayout
//
// Precondition: Input size is a multiple of chunk lanes.
//
// Example of BF16: "|" is the pack operation for two 16 bits number.
//
// Input:  b1  |a1   b2  |a2    ....     b128|a128 (128 elements, 512 bytes)
//         d1  |c1   d2  |c2    ....     d128|c128 (128 elements, 512 bytes)
//
// Output: a1  |a2   a3  |a4    ....     a127|a128 (64 elements, 256 bytes)
//         b1  |b2   b3  |b4    ....     b127|b128 (64 elements, 256 bytes)
//         c1  |c2   c3  |c4    ....     c127|c128 (64 elements, 256 bytes)
//         d1  |d2   d3  |d4    ....     d127|d128 (64 elements, 256 bytes)
//
template <int64_t kPackingFactor, int64_t kLaneCount>
void RepackToHostLayout(absl::Span<const uint32_t> input,
                        absl::Span<uint8_t> output) {
  constexpr int kTpuToCpuExpansionFactor =
      CeilOfRatio<size_t>(kPackingFactor, sizeof(uint32_t));
  constexpr int kExpandedLaneCount = kLaneCount * kTpuToCpuExpansionFactor;
  // We repack a sublane at a time, more than enough to occupy the SIMD lanes on
  // the host. While our input is a multiple of kLaneCount, our output may not
  // be because linearized data is padded to a multiple of the number of lanes.
  // Each lane delinearizes to `kTpuToCpuExpansionFactor` uint32_t values. If
  // our output is smaller, we delinearize into a staging buffer and copy the
  // needed values out.
  // N.B. Expansion occurs when we are delinearizing types which are smaller
  // than a byte because those types take up less space on the TPU (typically
  // the number of bits in the type) than on the CPU (typeically 8 bits).
  uint32_t staging_buffer[kExpandedLaneCount];
  constexpr int kBytesReadPerSublane = kLaneCount * sizeof(uint32_t);
  constexpr int kBytesWrittenPerSublane =
      kBytesReadPerSublane * kTpuToCpuExpansionFactor;
  const size_t output_bytes = output.size();
  const size_t full_writes = output_bytes / kBytesWrittenPerSublane;
  const size_t partial_bytes_remaining = output_bytes % kBytesWrittenPerSublane;
  if constexpr (kPackingFactor > 1) {
    for (size_t i = 0; i < full_writes; ++i) {
      RepackToHostLayoutSublaneHighway<kPackingFactor, kLaneCount>(
          &input[kLaneCount * i], tsl::safe_reinterpret_cast<uint32_t*>(
                                      &output[kBytesWrittenPerSublane * i]));
    }
    if (partial_bytes_remaining > 0) {
      RepackToHostLayoutSublaneHighway<kPackingFactor, kLaneCount>(
          &input[kLaneCount * full_writes], staging_buffer);
      std::memcpy(&output[kBytesWrittenPerSublane * full_writes],
                  staging_buffer, partial_bytes_remaining);
    }
  } else {
    std::memcpy(output.data(), input.data(), output.size());
  }
}

struct LinearizerCache {
  LinearizerCache() : cache(1024) {}
  absl::Mutex mu;
  TransposePlanCache cache ABSL_GUARDED_BY(mu);
};

LinearizerCache* linearizer_cache = new LinearizerCache;

absl::InlinedVector<int64_t, 2> GetResolvedTopLevelTiling(
    const Shape& device_shape) {
  absl::InlinedVector<int64_t, 2> tiling_vec;
  if (device_shape.layout().tiles().empty()) {
    return tiling_vec;
  }
  const auto tile_dims = device_shape.layout().tiles(0).dimensions();
  for (int i = 0; i < tile_dims.size(); ++i) {
    int64_t dim = tile_dims[i];
    if (dim == Tile::kMatchDimension) {
      int64_t minor = tile_dims.size() - 1 - i;
      if (minor >= device_shape.layout().minor_to_major().size()) {
        continue;
      }
      int64_t logical_dim = device_shape.layout().minor_to_major(minor);
      int64_t dim_size = device_shape.dimensions(logical_dim);
      int64_t packing =
          (device_shape.layout().tiles().size() > 1 &&
           minor < device_shape.layout().tiles(1).dimensions().size())
              ? device_shape.layout().tiles(1).dimension(
                    device_shape.layout().tiles(1).dimensions().size() - 1 -
                    minor)
              : 1;
      dim = RoundUpTo(std::max<int64_t>(dim_size, 1),
                      std::max<int64_t>(packing, 1));
    }
    tiling_vec.push_back(dim);
  }
  return tiling_vec;
}

}  // namespace

/* static */ absl::Status TiledLinearizer::LinearizeUsingTransposeInto(
    absl::Span<uint8_t> dest, const void* data, const Shape& host_shape,
    absl::Span<int64_t const> byte_strides, const Layout& device_layout,
    bool set_padding_to_nan, absl::Span<const uint32_t> dynamic_sizes,
    const TiledLinearizer::ParallelismOptions& parallelism_options) {
  if (host_shape.element_type() == C128) {
    TF_RET_CHECK(dest.size() % 2 == 0);
    Shape f64_shape = ShapeUtil::ChangeElementType(host_shape, F64);
    size_t half_size = dest.size() / 2;
    ABSL_RETURN_IF_ERROR(LinearizeUsingTransposeInto(
        dest.subspan(0, half_size), data, f64_shape, byte_strides,
        device_layout, set_padding_to_nan, dynamic_sizes, parallelism_options));
    return LinearizeUsingTransposeInto(
        dest.subspan(half_size),
        static_cast<const char*>(data) + sizeof(double), f64_shape,
        byte_strides, device_layout, set_padding_to_nan, dynamic_sizes,
        parallelism_options);
  }

  Shape device_shape = host_shape;
  *(device_shape.mutable_layout()) = device_layout;

  int64_t byte_width = primitive_util::ByteWidth(host_shape.element_type());
  int64_t prefix_size =
      dynamic_sizes.empty()
          ? 0
          : device_layout.dynamic_shape_metadata_prefix_bytes();
  TF_RET_CHECK(prefix_size % sizeof(uint32_t) == 0);
  int64_t prefix_u32_size = prefix_size / sizeof(uint32_t);
  TF_RET_CHECK(dest.size() >= prefix_size);
  int64_t size_in_bytes = dest.size() - prefix_size;
  TF_RET_CHECK(size_in_bytes % sizeof(uint32_t) == 0);
  TF_RET_CHECK(reinterpret_cast<uintptr_t>(dest.data()) % alignof(uint32_t) ==
               0);

  int ndim = host_shape.dimensions().size();

  absl::InlinedVector<int64_t, 4> input_strides(ndim);
  absl::c_copy(byte_strides, input_strides.begin());

  absl::InlinedVector<int64_t, 4> permutation;
  permutation.reserve(ndim + 1);

  absl::Span<int64_t const> dims;
  absl::InlinedVector<int64_t, 1> tmp_dims;
  int64_t transpose_byte_width = byte_width;
  TransposePlan::Transformation transformation =
      TransposePlan::Transformation::kNone;
  if (byte_width <= 4) {
    dims = host_shape.dimensions();
    // For scalars, add a size-1 dimension because the tiling will have 1
    // dimension.
    if (ndim == 0) {
      tmp_dims = {1};
      input_strides = {byte_width};
      permutation = {0};
      ++ndim;
      dims = tmp_dims;
    }
  } else {
    DCHECK_EQ(byte_width, 8);

    // For 8-byte types, change the element size to 4 bytes, add a trailing
    // input dimension [..., 2], and transpose it to an output major dimension.
    static_assert(tsl::port::kLittleEndian);
    TF_RET_CHECK(byte_width == 8) << byte_width;
    // Add a size-1 dimension for scalars, since it becomes a tiled dimension.
    if (ndim == 0) {
      tmp_dims = {1, 2};
      input_strides = {8, 4};
      permutation = {1, 0};
      ++ndim;
    } else {
      tmp_dims.resize(ndim + 1);
      absl::c_copy(host_shape.dimensions(), tmp_dims.begin());
      tmp_dims.back() = 2;
      permutation.push_back(ndim);
      input_strides.push_back(4);
    }
    transpose_byte_width = 4;
    dims = tmp_dims;
    if (host_shape.element_type() == F64) {
      transformation = TransposePlan::Transformation::kF64ToEf57;
    }
  }

  absl::Span<int64_t const> minor_to_major = device_layout.minor_to_major();
  for (auto it = minor_to_major.rbegin(); it != minor_to_major.rend(); ++it) {
    permutation.push_back(*it);
  }

  absl::InlinedVector<int64_t, 2> output_tiling_vec =
      GetResolvedTopLevelTiling(device_shape);

  const auto tiles = device_shape.layout().tiles();
  const int bits_per_pack = device_layout.element_size_in_bits() != 0
                                ? device_layout.element_size_in_bits()
                                : std::min<int64_t>(byte_width * 8, 32);
  const int packing_factor = 32 / bits_per_pack;
  const bool has_subtile_packing =
      packing_factor > 1 &&
      (tiles.size() > 1 ||
       (!tiles.empty() && tiles[0].dimensions().size() > 1));

  TransposePlan::Options options;
  options.elem_size_in_bytes = transpose_byte_width;
  options.dims = dims;
  options.permutation = permutation;
  options.input_striding = TransposePlan::Striding{input_strides};
  options.output_tiling = TransposePlan::Tiling{output_tiling_vec};
  options.transformation = transformation;
  if (!has_subtile_packing && bits_per_pack < 8) {
    options.dest_bits_per_element = bits_per_pack;
  }
  options.num_threads = parallelism_options.max_parallelism;
  std::shared_ptr<TransposePlan> plan;
  {
    absl::MutexLock lock(linearizer_cache->mu);
    ABSL_ASSIGN_OR_RETURN(plan, linearizer_cache->cache.GetOrCreate(options));
  }

  size_t elements_in_the_packed_data = size_in_bytes / sizeof(uint32_t);
  // The `data` from host memory has 1 int4 in 1 byte, then we need double size
  // for `scratch` to support the transpose Execute below. It is then shrunk
  // when repacking into `dest`.
  if (has_subtile_packing && bits_per_pack < 8) {
    elements_in_the_packed_data *= 8 / bits_per_pack;
  }

  const size_t written_bytes =
      options.dest_bits_per_element.has_value()
          ? CeilOfRatio<size_t>(static_cast<size_t>(Product(dims)) *
                                    *options.dest_bits_per_element,
                                CHAR_BIT)
          : static_cast<size_t>(Product(dims)) * transpose_byte_width;
  TF_RET_CHECK(elements_in_the_packed_data * sizeof(uint32_t) >= written_bytes);
  const bool has_padding =
      written_bytes < elements_in_the_packed_data * sizeof(uint32_t);
  const uint32_t pad_u32 = set_padding_to_nan ? kPadValue : uint32_t{0};

  std::optional<Buffer> scratch;
  absl::Span<uint8_t> dst;
  if (has_subtile_packing && bits_per_pack < 8) {
    if (prefix_u32_size > 0) {
      std::fill_n(tsl::safe_reinterpret_cast<uint32_t*>(dest.data()),
                  prefix_u32_size, pad_u32);
    }
    scratch.emplace(
        elements_in_the_packed_data,
        /*value=*/has_padding ? std::make_optional(pad_u32) : std::nullopt,
        &tensorflow::tpu::DefaultAllocator);
    dst = scratch->mutable_data<uint8_t>();
  } else {
    auto dest_u32 =
        absl::MakeSpan(tsl::safe_reinterpret_cast<uint32_t*>(dest.data()),
                       elements_in_the_packed_data + prefix_u32_size);
    if (has_padding) {
      absl::c_fill(dest_u32, pad_u32);
    } else if (prefix_u32_size > 0) {
      std::fill_n(dest_u32.data(), prefix_u32_size, pad_u32);
    }
    dst = dest.subspan(prefix_size);
  }
  if (plan->Parallelism() <= 1) {
    plan->Execute(data, dst.data());
  } else {
    parallelism_options.run_plan(*plan, data, dst.data());
  }
  // TODO(phawkins): it would be preferable to sink the repacking into the
  // transpose computation to improve cache locality.

  // TODO(b/188059010, b/194422623): Ideally we want to check LaneCount ==
  // kTcLaneCount here, but chip variants without TC have zero LaneCount, and
  // it's not clear what that means.
  // TODO (b/274171729): For SparseCore currently the data is being packed
  // linearly.
  if (has_subtile_packing) {
    const auto lane_tile_dims =
        (tiles.size() > 1 ? tiles[tiles.size() - 2] : tiles[0]).dimensions();
    TF_RET_CHECK(!lane_tile_dims.empty());
    ABSL_RETURN_IF_ERROR(TcLaneCountSwitch(
        [&](auto kTcLaneCount) {
          ElementPackingFactorSwitch<void>(
              [&](auto kPackingFactor) {
                if constexpr (kPackingFactor > 1) {
                  RepackToHardwareLayout<kPackingFactor, kTcLaneCount>(
                      dst,
                      tsl::safe_reinterpret_cast<uint32_t*>(
                          dest.subspan(prefix_size).data()),
                      set_padding_to_nan);
                }
              },
              packing_factor);
        },
        lane_tile_dims.back()));
  }

  if (!dynamic_sizes.empty()) {
    absl::c_copy(dynamic_sizes,
                 tsl::safe_reinterpret_cast<uint32_t*>(dest.data()));
  }

  return absl::OkStatus();
}

/* static */ absl::Status TiledLinearizer::X64FromTuple(
    const LiteralSlice& low, const LiteralSlice& high,
    MutableBorrowingLiteral literal) {
  tsl::profiler::TraceMe trace_me(
      [&] {
        return absl::StrCat("X64FromTuple ",
                            ShapeUtil::HumanStringWithLayout(literal.shape()));
      },
      /*level=*/2);
  TF_RET_CHECK(ShapeUtil::ElementHasBitWidth(literal.shape(), 64));
  Shape component_shape = X64ComponentShape(literal.shape());
  TF_RET_CHECK(ShapeUtil::Compatible(low.shape(), component_shape));
  TF_RET_CHECK(ShapeUtil::Compatible(high.shape(), component_shape));
  switch (literal.shape().element_type()) {
    case C64: {
      LiteralSlice real = low;
      LiteralSlice imag = high;
      ABSL_RETURN_IF_ERROR(
          literal.Populate<complex64>([&](absl::Span<const int64_t> indices) {
            return complex64(real.Get<float>(indices),
                             imag.Get<float>(indices));
          }));
      break;
    }
    case S64:
      ABSL_RETURN_IF_ERROR(
          literal.Populate<int64_t>([&](absl::Span<const int64_t> indices) {
            return static_cast<int64_t>(
                low.Get<uint32_t>(indices) +
                (static_cast<uint64_t>(high.Get<uint32_t>(indices)) << 32));
          }));
      break;
    case U64:
      ABSL_RETURN_IF_ERROR(
          literal.Populate<uint64_t>([&](absl::Span<const int64_t> indices) {
            return low.Get<uint32_t>(indices) +
                   (static_cast<uint64_t>(high.Get<uint32_t>(indices)) << 32);
          }));
      break;
    case F64:
      ABSL_RETURN_IF_ERROR(
          literal.Populate<double>([&](absl::Span<const int64_t> indices) {
            auto most_significant = low.Get<float>(indices);
            auto least_significant = high.Get<float>(indices);
            return double{least_significant} + double{most_significant};
          }));
      break;
    default:
      return Unimplemented("Unsupported complex type: %s",
                           PrimitiveType_Name(literal.shape().element_type()));
  }
  return absl::OkStatus();
}

/* static */ absl::Status TiledLinearizer::X128FromTuple(
    const LiteralSlice& low, const LiteralSlice& high,
    MutableBorrowingLiteral literal) {
  tsl::profiler::TraceMe trace_me(
      [&] {
        return absl::StrCat("X128FromTuple ",
                            ShapeUtil::HumanStringWithLayout(literal.shape()));
      },
      /*level=*/2);
  TF_RET_CHECK(ShapeUtil::ElementHasBitWidth(literal.shape(), 128));
  Shape component_shape = X128ComponentShape(literal.shape());
  TF_RET_CHECK(component_shape.element_type() == F64);
  TF_RET_CHECK(ShapeUtil::Compatible(low.shape(), component_shape));
  TF_RET_CHECK(ShapeUtil::Compatible(high.shape(), component_shape));
  switch (literal.shape().element_type()) {
    case C128: {
      LiteralSlice real = low;
      LiteralSlice imag = high;
      ABSL_RETURN_IF_ERROR(
          literal.Populate<complex128>([&](absl::Span<const int64_t> indices) {
            return complex128(real.Get<double>(indices),
                              imag.Get<double>(indices));
          }));
      break;
    }
    default:
      return Unimplemented("Unsupported type: %s",
                           PrimitiveType_Name(literal.shape().element_type()));
  }
  return absl::OkStatus();
}

namespace {

absl::Status DelinearizeComponents(
    int bitwidth, const Shape& literal_shape, absl::Span<const uint8_t> data,
    const Layout& device_layout, MutableLiteralBase* literal,
    const TiledLinearizer::ParallelismOptions& parallelism_options) {
  Shape device_shape = literal_shape;
  (*device_shape.mutable_layout()) = device_layout;
  Shape component_subshape;
  Shape component_subshape_device;
  if (bitwidth == 64) {
    component_subshape = X64ComponentShape(literal_shape);
    component_subshape_device = X64ComponentShape(device_shape);
  } else if (bitwidth == 128) {
    component_subshape = X128ComponentShape(literal_shape);
    component_subshape_device = X128ComponentShape(device_shape);
  } else {
    return absl::InternalError(
        "Unsupported bitwidth for tuple delinearization.");
  }

  int64_t component_size_bytes =
      ShapeUtil::ArraySize(component_subshape_device);
  Literal low_tmp(component_subshape);
  Literal high_tmp(component_subshape);
  ABSL_RETURN_IF_ERROR(TiledLinearizer::Delinearize(
      low_tmp.shape(), data.subspan(0, component_size_bytes), device_layout,
      &low_tmp, parallelism_options));

  ABSL_RETURN_IF_ERROR(TiledLinearizer::Delinearize(
      high_tmp.shape(),
      data.subspan(component_size_bytes, component_size_bytes), device_layout,
      &high_tmp, parallelism_options));

  if (bitwidth == 64) {
    return TiledLinearizer::X64FromTuple(low_tmp, high_tmp, literal);
  }
  return TiledLinearizer::X128FromTuple(low_tmp, high_tmp, literal);
}

template <int kPackingFactor>
ABSL_ATTRIBUTE_FUNC_ALIGN(64)
absl::Status
    DelinearizeR1(const Shape& literal_shape, absl::Span<const uint32_t> values,
                  const Layout& device_layout, MutableLiteralBase* literal) {
  CHECK(!HasLinearLayout(device_layout));
  DCHECK(literal_shape.IsArray());
  DCHECK_EQ(literal_shape.dimensions().size(), 1);
#ifndef NDEBUG
  CreateGarbageFilledLiteral(literal_shape, literal);
#endif
  auto output = GetLiteralAsMutableSpan(literal);
  const auto tiles = device_layout.tiles();
  if (tiles.size() <= 1) {
    // Transfer in host layout or 1D SparseCore layout. No subtile pack/unpack.
    RepackFromSparseCoreLayout<kPackingFactor>(values, output);
    return absl::OkStatus();
  }
  const auto lane_tile_dims = tiles[tiles.size() - 2].dimensions();
  TF_RET_CHECK(!lane_tile_dims.empty());
  return TcLaneCountSwitch(
      [&](auto kLaneCount) {
        RepackToHostLayout<kPackingFactor, kLaneCount>(values, output);
      },
      lane_tile_dims.back());
}

absl::Status DelinearizeUsingTranspose(
    const Shape& literal_shape, absl::Span<const uint8_t> data,
    const Layout& device_layout, MutableLiteralBase* literal,
    const TiledLinearizer::ParallelismOptions& parallelism_options) {
  tsl::profiler::TraceMe traceme("DelinearizeUsingTranspose");
  DCHECK_OK(LayoutUtil::ValidateLayoutForShape(device_layout, literal_shape));
  TF_RET_CHECK(!device_layout.tiles().empty());

#ifndef NDEBUG
  CreateGarbageFilledLiteral(literal_shape, literal);
#endif

  // The fast path does not parallelize. Above a certain size it's worth using
  // the transpose plan to get parallelism.
  static const int64_t kFastPathMaxBytes = int64_t{1} << 26;
  int64_t size_bytes = literal->size_bytes();

  const int64_t element_size_in_bits =
      device_layout.element_size_in_bits() != 0
          ? device_layout.element_size_in_bits()
          : ShapeUtil::ByteSizeOfPrimitiveType(literal_shape.element_type()) *
                CHAR_BIT;
  int packing_factor = 32 / element_size_in_bits;

  if (packing_factor <= 4 && HasLinearLayout(device_layout) &&
      Layout::Equal().MinorToMajorOnly()(device_layout,
                                         literal_shape.layout()) &&
      size_bytes <= kFastPathMaxBytes) {
    tsl::profiler::TraceMe trace_me("the memcpy");

    std::memcpy(literal->untyped_data(), data.data(), size_bytes);
    return absl::OkStatus();
  }

  auto values = absl::MakeConstSpan(
      tsl::safe_reinterpret_cast<const uint32_t*>(data.data()),
      data.size() / sizeof(uint32_t));

  // Fast path for R1 tensors.
  if (literal_shape.dimensions().size() == 1 &&
      !HasLinearLayout(device_layout) && size_bytes <= kFastPathMaxBytes) {
    return ElementPackingFactorSwitch<absl::Status>(
        [&](auto kPackingFactor) {
          return DelinearizeR1<kPackingFactor>(literal_shape, values,
                                               device_layout, literal);
        },
        packing_factor);
  }

  Shape device_shape = literal_shape;
  *(device_shape.mutable_layout()) = device_layout;

  absl::InlinedVector<int64_t, 4> input_dims(literal_shape.dimensions().size());
  absl::InlinedVector<int64_t, 4> input_logical_dim_to_physical_dim(
      input_dims.size());
  auto dim_it = input_dims.rbegin();
  int physical_dim = input_dims.size();
  for (int logical_dim : device_layout.minor_to_major()) {
    --physical_dim;
    input_logical_dim_to_physical_dim[logical_dim] = physical_dim;
    *dim_it = literal_shape.dimensions(logical_dim);
    ++dim_it;
  }

  absl::InlinedVector<int64_t, 2> input_tiling_vec =
      literal_shape.dimensions().size() > 1
          ? GetResolvedTopLevelTiling(device_shape)
          : absl::InlinedVector<int64_t, 2>();

  // minor_to_major maps from physical dim to logical dim.
  // permutation maps from output physical dims to input physical dims.
  absl::InlinedVector<int64_t, 4> permutation(
      literal_shape.dimensions().size());
  auto it = permutation.rbegin();
  for (int dim : literal_shape.layout().minor_to_major()) {
    *it = input_logical_dim_to_physical_dim[dim];
    ++it;
  }

  int byte_width = primitive_util::ByteWidth(literal_shape.element_type());

  TransposePlan::Options options;
  options.elem_size_in_bytes = byte_width;
  options.dims = input_dims;
  options.permutation = permutation;
  options.input_tiling = TransposePlan::Tiling{input_tiling_vec};
  options.output_tiling = TransposePlan::Tiling{};
  options.transformation = TransposePlan::Transformation::kNone;
  options.num_threads = parallelism_options.max_parallelism;
  std::shared_ptr<TransposePlan> plan;
  {
    absl::MutexLock lock(linearizer_cache->mu);
    ABSL_ASSIGN_OR_RETURN(plan, linearizer_cache->cache.GetOrCreate(options));
  }

  // TODO(phawkins): sink unpacking code into transpose plan.
  Buffer unpacked_values;
  const auto tiles = device_shape.layout().tiles();
  const bool has_subtile_packing =
      packing_factor > 1 &&
      (tiles.size() > 1 ||
       (!tiles.empty() && tiles[0].dimensions().size() > 1));
  // TODO (b/274171729): For SparseCore currently the data is being packed
  // linearly.
  if (has_subtile_packing) {
    const auto lane_tile_dims =
        (tiles.size() > 1 ? tiles[tiles.size() - 2] : tiles[0]).dimensions();
    TF_RET_CHECK(!lane_tile_dims.empty());
    ABSL_RETURN_IF_ERROR(TcLaneCountSwitch(
        [&](auto kTcLaneCount) {
          ElementPackingFactorSwitch<void>(
              [&](auto kPackingFactor) {
                if constexpr (kPackingFactor > 1) {
                  constexpr int kTpuToCpuExpansionFactor =
                      CeilOfRatio<size_t>(kPackingFactor, sizeof(uint32_t));
                  unpacked_values =
                      Buffer(values.size() * kTpuToCpuExpansionFactor *
                             sizeof(uint32_t));
                  RepackToHostLayout<kPackingFactor, kTcLaneCount>(
                      values, unpacked_values.mutable_data<uint8_t>());
                  values = unpacked_values.const_data<uint32_t>();
                }
              },
              packing_factor);
        },
        lane_tile_dims.back()));
  } else if (const int bits_per_pack = 32 / packing_factor; bits_per_pack < 8) {
    ElementPackingFactorSwitch<void>(
        [&](auto kPackingFactor) {
          if constexpr (kPackingFactor > 4) {
            constexpr int kTpuToCpuExpansionFactor =
                CeilOfRatio<size_t>(kPackingFactor, sizeof(uint32_t));
            unpacked_values = Buffer(values.size() * kTpuToCpuExpansionFactor *
                                     sizeof(uint32_t));
            RepackFromSparseCoreLayout<kPackingFactor>(
                values, unpacked_values.mutable_data<uint8_t>());
            values = unpacked_values.const_data<uint32_t>();
          }
        },
        packing_factor);
  }

  if (plan->Parallelism() <= 1) {
    plan->Execute(values.data(), literal->untyped_data());
  } else {
    parallelism_options.run_plan(*plan, values.data(), literal->untyped_data());
  }
  return absl::OkStatus();
}

}  // namespace

/* static */ ABSL_ATTRIBUTE_FUNC_ALIGN(64) absl::Status
    TiledLinearizer::Delinearize(
        const Shape& literal_shape, absl::Span<const uint8_t> data,
        const Layout& device_layout, MutableLiteralBase* literal,
        const TiledLinearizer::ParallelismOptions& parallelism_options) {
  tsl::profiler::TraceMe trace_me(
      [&] {
        return tsl::profiler::TraceMeEncode(
            "Delinearize", {{"shape", literal_shape.ToString()},
                            {"layout", device_layout.ToString()}});
      },
      /*level=*/2);
  if (literal_shape.IsToken()) {
    return absl::OkStatus();
  }
  if (!literal_shape.IsArray()) {
    return Unimplemented("Unsupported shape for Delinearize.");
  }
  TF_RET_CHECK(reinterpret_cast<uintptr_t>(data.data()) % alignof(uint32_t) ==
               0);
  VLOG(2) << "delinearized: " << ShapeUtil::HumanString(literal_shape) << " @ "
          << LayoutUtil::HumanString(device_layout) << " {"
          << FormatLinearizedLiteral(
                 literal_shape,
                 absl::MakeConstSpan(
                     tsl::safe_reinterpret_cast<const uint32_t*>(data.data()),
                     data.size() / sizeof(uint32_t)))
          << "}";
  // Zero-element arrays carry no tiles in their device layout and have nothing
  // to delinearize.
  if (ShapeUtil::IsZeroElementArray(literal_shape)) {
    return absl::OkStatus();
  }
  const int bitwidth = primitive_util::BitWidth(literal_shape.element_type());
  if (bitwidth == 64 || bitwidth == 128) {
    return DelinearizeComponents(bitwidth, literal_shape, data, device_layout,
                                 literal, parallelism_options);
  }
  return DelinearizeUsingTranspose(literal_shape, data, device_layout, literal,
                                   parallelism_options);
}

}  // namespace xla
