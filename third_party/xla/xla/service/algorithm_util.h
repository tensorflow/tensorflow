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

#ifndef XLA_SERVICE_ALGORITHM_UTIL_H_
#define XLA_SERVICE_ALGORITHM_UTIL_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla {

// We try to keep most algorithm-specific queries in this file, so that we only
// have to update one file when we add a new one.
// We can also add some platform-specific queries as long as we don't need to
// depend on specific targets, such as the "gpu" folder.
namespace algorithm_util {

// Get the ComputationType corresponding to an algorithm. See the
// ComputationType definition for more info.
absl::StatusOr<stream_executor::blas::ComputationType> GetBlasComputationType(
    PrecisionConfig::Algorithm algorithm);

// Get the accumulator type of an algorithm.
absl::StatusOr<PrimitiveType> GetDotAccumulatorType(
    PrecisionConfig::Algorithm algorithm);

// Are the AType & BType TF32?
bool HasTf32InputType(PrecisionConfig::Algorithm algorithm);

// Checks if the algorithm uses fast accumulation as in
// CUBLASLT_MATMUL_DESC_FAST_ACCUM.
bool HasFastAccum(PrecisionConfig::Algorithm algorithm);

// Checks if we support the given algorithm using cuBLAS or cuBLASLt.
//
// It's clear that those libraries could support more, but we only list the ones
// which we explicitly test for now.
//
// We may want to also check storage types, but for now those are checked in
// IsSupportedDotAlgorithmOnGpu.
bool IsSupportedByCublasOrCublasLt(
    PrecisionConfig::Algorithm algorithm,
    stream_executor::GpuComputeCapability gpu_compute_capability,
    const HloDotInstruction* dot = nullptr, int64_t rhs_contracting_index = -1);

// Checks if we support the given algorithm using cuDNN.
bool IsSupportedByCudnn(PrecisionConfig::Algorithm algorithm);

// Checks if we support the given algorithm using the elemental IR emitter.
bool IsSupportedByElementalIrEmitter(PrecisionConfig::Algorithm algorithm);

// Is the given algorithm supported on GPU with the given compute capability and
// input/output storage types.
bool IsSupportedDotAlgorithmOnGpu(
    PrecisionConfig::Algorithm algorithm,
    stream_executor::GpuComputeCapability gpu_compute_capability,
    PrimitiveType input_storage_type, PrimitiveType output_storage_type);

}  // namespace algorithm_util
}  // namespace xla

#endif  // XLA_SERVICE_ALGORITHM_UTIL_H_
