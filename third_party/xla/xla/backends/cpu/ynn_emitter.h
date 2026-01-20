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

#ifndef XLA_BACKENDS_CPU_YNN_EMITTER_H_
#define XLA_BACKENDS_CPU_YNN_EMITTER_H_

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/stream_executor/device_address.h"

namespace xla::cpu {

absl::StatusOr<absl::AnyInvocable<absl::StatusOr<YnnSubgraph>(
    absl::Span<const se::DeviceAddressBase> arguments_buffers)>>
EmitYnnFusionBuilder(const HloComputation* computation);

absl::StatusOr<absl::AnyInvocable<absl::StatusOr<YnnSubgraph>(
    absl::Span<const se::DeviceAddressBase> arguments_buffers)>>
EmitYnnDotBuilder(const HloDotInstruction* dot, bool capture_rhs);

absl::StatusOr<absl::AnyInvocable<absl::StatusOr<YnnSubgraph>(
    absl::Span<const se::DeviceAddressBase> arguments_buffers)>>
EmitYnnConvolutionBuilder(const HloConvolutionInstruction* conv);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_YNN_EMITTER_H_
