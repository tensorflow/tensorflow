/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"

#include "ynnpack/include/ynnpack.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "xla/primitive_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

absl::StatusOr<YnnSubgraph> CreateYnnSubgraph(
    absl::FunctionRef<ynn_status(ynn_subgraph_t*)> builder) {
  ynn_subgraph_t subgraph = nullptr;
  YNN_RETURN_IF_ERROR(builder(&subgraph));
  return YnnSubgraph(subgraph);
}

absl::StatusOr<YnnRuntime> CreateYnnRuntime(
    absl::FunctionRef<ynn_status(ynn_runtime_t*)> builder) {
  ynn_runtime_t runtime = nullptr;
  YNN_RETURN_IF_ERROR(builder(&runtime));
  return YnnRuntime(runtime);
}

absl::StatusOr<YnnThreadpool> CreateYnnThreadpool(
    absl::FunctionRef<ynn_status(ynn_threadpool_t*)> builder) {
  ynn_threadpool_t threadpool = nullptr;
  YNN_RETURN_IF_ERROR(builder(&threadpool));
  return YnnThreadpool(threadpool);
}

absl::StatusOr<ynn_type> YnnType(const PrimitiveType& type) {
  switch (type) {
    case S4:
      return ynn_type_int4;
    case U4:
      return ynn_type_uint4;
    case S8:
      return ynn_type_int8;
    case U8:
      return ynn_type_uint8;
    case BF16:
      return ynn_type_bf16;
    case F16:
      return ynn_type_fp16;
    case F32:
      return ynn_type_fp32;
    case S32:
      return ynn_type_int32;
    default:
      return InvalidArgument("Unsupported YNNPACK type: %s",
                             primitive_util::LowercasePrimitiveTypeName(type));
  }
}

}  // namespace xla::cpu
