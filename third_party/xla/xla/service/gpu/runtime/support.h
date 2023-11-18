/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_SUPPORT_H_
#define XLA_SERVICE_GPU_RUNTIME_SUPPORT_H_

#include <string>
#include <string_view>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "xla/runtime/custom_call.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"

namespace xla {
namespace gpu {

template <auto T>
using FunctionWrapper = xla::runtime::CustomCall::FunctionWrapper<T>;

struct DotDimensionNumbers {
  absl::Span<const int64_t> lhs_batch;
  absl::Span<const int64_t> lhs_contract;
  absl::Span<const int64_t> rhs_batch;
  absl::Span<const int64_t> rhs_contract;
};

// Disable expensive CustomCall checks in optimized build.
inline constexpr runtime::CustomCall::RuntimeChecks checks =  // NOLINT
#if defined(NDEBUG)
    runtime::CustomCall::RuntimeChecks::kLess;
#else
    runtime::CustomCall::RuntimeChecks::kDefault;
#endif

template <typename T>
absl::StatusOr<T> ToAbsl(StatusOr<T> status_or) {
  if (!status_or.ok()) return status_or.status();
  return std::move(status_or).value();
}

inline se::DeviceMemoryBase GetDeviceAddress(
    const runtime::FlatMemrefView& memref) {
  return se::DeviceMemoryBase(memref.data, memref.size_in_bytes);
}

inline se::DeviceMemoryBase GetDeviceAddress(
    const runtime::MemrefView& memref) {
  uint64_t size = primitive_util::ByteWidth(memref.dtype);
  for (auto dim : memref.sizes) size *= dim;
  return se::DeviceMemoryBase(memref.data, size);
}

inline se::DeviceMemoryBase GetDeviceAddress(
    const runtime::StridedMemrefView& memref) {
  uint64_t size = primitive_util::ByteWidth(memref.dtype);
  for (auto dim : memref.sizes) size *= dim;
  if (primitive_util::Is4BitType(memref.dtype)) {
    size = (size + 1) / 2;
  }
  return se::DeviceMemoryBase(memref.data, size);
}

inline Shape ToShape(const runtime::StridedMemrefView& memref) {
  // Recover `minor_to_major` dimensions permutation from strides.
  auto indexed_strides_range =
      llvm::map_range(llvm::enumerate(memref.strides), [](auto pair) {
        return std::pair<int64_t, size_t>{pair.value(), pair.index()};
      });

  auto indexed_strides = llvm::to_vector(indexed_strides_range);
  llvm::stable_sort(indexed_strides);

  llvm::SmallVector<int64_t> minor_to_major;
  minor_to_major.reserve(indexed_strides.size());
  for (auto& pair : indexed_strides) minor_to_major.push_back(pair.second);

  return ShapeUtil::MakeShapeWithDenseLayout(memref.dtype, memref.sizes,
                                             minor_to_major);
}

inline StatusOr<GemmConfig> GetGemmConfig(
    const runtime::StridedMemrefView& lhs,
    const runtime::StridedMemrefView& rhs,
    const runtime::StridedMemrefView& out, int64_t algorithm, double alpha_real,
    double alpha_imag, double beta, absl::Span<const int64_t> lhs_batch,
    absl::Span<const int64_t> lhs_contract, absl::Span<const int64_t> rhs_batch,
    absl::Span<const int64_t> rhs_contract, int64_t compute_precision,
    const std::optional<runtime::StridedMemrefView> c = std::nullopt,
    const std::optional<runtime::StridedMemrefView>& bias = std::nullopt,
    bool grad_x = false, bool grad_y = false) {
  Shape c_shape = ToShape(c.value_or(out));
  Shape bias_shape;
  Shape* bias_shape_ptr = nullptr;
  if (bias) {
    bias_shape = ToShape(*bias);
    bias_shape_ptr = &bias_shape;
  }
  return GemmConfig::For(ToShape(lhs), lhs_batch, lhs_contract, ToShape(rhs),
                         rhs_batch, rhs_contract, c_shape, bias_shape_ptr,
                         ToShape(out), alpha_real, alpha_imag, beta, algorithm,
                         compute_precision, grad_x, grad_y);
}

// adds Dot Dimension Attribute encodings for calls to Gemm and cuBLASLt
inline void PopulateDotDimsAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding) {
  using DotDimsAttr = mlir::mhlo::DotDimensionNumbersAttr;
  encoding.Add<
      xla::runtime::AggregateAttrEncoding<DotDimsAttr, DotDimensionNumbers>>(
      encoding,
      xla::runtime::AggregateAttrDef<DotDimsAttr>()
          .Add("lhs_batch", &DotDimsAttr::getLhsBatchingDimensions)
          .Add("lhs_contract", &DotDimsAttr::getLhsContractingDimensions)
          .Add("rhs_batch", &DotDimsAttr::getRhsBatchingDimensions)
          .Add("rhs_contract", &DotDimsAttr::getRhsContractingDimensions));
}

// Appends to `diagnostic_engine` a handler that appends all emitted errors to
// the `diagnostic` string. If `append_annotation_stack` is true, it will append
// current profiler annotation stack to the diagnostic message (annotation used
// in Xprof).
void AppendDiagnosticToString(runtime::DiagnosticEngine& diagnostic_engine,
                              std::string* diagnostic,
                              bool append_annotation_stack = false);

// Sets the current tracing scope that will be added to all emitted diagnostics.
void SetCurrentTracingScope(std::string_view scope);
void ResetCurrentTracingScope();

}  // namespace gpu
}  // namespace xla

namespace xla {
namespace runtime {

// using llvm::ArrayRef;

XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(
    xla::gpu::DotDimensionNumbers,
    AggregateMember<absl::Span<const int64_t>>("lhs_batch"),
    AggregateMember<absl::Span<const int64_t>>("lhs_contract"),
    AggregateMember<absl::Span<const int64_t>>("rhs_batch"),
    AggregateMember<absl::Span<const int64_t>>("rhs_contract"));

}  // namespace runtime
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_SUPPORT_H_
