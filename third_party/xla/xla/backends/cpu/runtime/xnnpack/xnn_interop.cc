/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/xnnpack/xnn_interop.h"

#include "experimental.h"  // xnnpack
#include "xnnpack.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/primitive_util.h"
#include "xla/util.h"

namespace xla::cpu {

absl::Status InitializeXnnPack() {
  static xnn_status status = xnn_initialize(/*allocator=*/nullptr);
  if (status != xnn_status_success) {
    return Internal("XNNPACK initialization failed");
  }
  return absl::OkStatus();
}

absl::StatusOr<XnnSubgraph> CreateXnnSubgraph(
    absl::FunctionRef<xnn_status(xnn_subgraph_t*)> builder) {
  xnn_subgraph_t subgraph = nullptr;
  XNN_RETURN_IF_ERROR(builder(&subgraph));
  return XnnSubgraph(subgraph);
}

absl::StatusOr<XnnRuntime> CreateXnnRuntime(
    absl::FunctionRef<xnn_status(xnn_runtime_t*)> builder) {
  xnn_runtime_t runtime = nullptr;
  XNN_RETURN_IF_ERROR(builder(&runtime));
  return XnnRuntime(runtime);
}

absl::StatusOr<XnnThreadpool> CreateXnnThreadpool(
    absl::FunctionRef<xnn_status(xnn_threadpool_t*)> builder) {
  xnn_threadpool_t threadpool = nullptr;
  XNN_RETURN_IF_ERROR(builder(&threadpool));
  return XnnThreadpool(threadpool);
}

absl::StatusOr<xnn_datatype> XnnDatatype(const PrimitiveType& type) {
  switch (type) {
    case BF16:
      return xnn_datatype_bf16;
    case F16:
      return xnn_datatype_fp16;
    case F32:
      return xnn_datatype_fp32;
    default:
      return InvalidArgument("Unsupported XNNPACK data type: %s",
                             primitive_util::LowercasePrimitiveTypeName(type));
  }
}

}  // namespace xla::cpu
