/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_HLO_ALGORITHM_DENYLIST_H_
#define XLA_SERVICE_GPU_HLO_ALGORITHM_DENYLIST_H_

#include <string>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/dnn.h"

namespace xla {
namespace gpu {

// Key is the tuple of canonicalized hlo, compute capability major/minor,
// cudnn version major/minor/patch, blas version.
using DenyListMapType = absl::flat_hash_map<
    std::tuple<std::string, int, int, int, int, int, std::string>,
    std::vector<stream_executor::dnn::AlgorithmDesc>>;

// Get the list of convolution algorithms which are disabled for the given
// 'instr' when using compute capability 'cc', cudnn version 'cudnn_version' and
// blas version 'blas_version'. In addition to the hardcoded denylist used in
// this function, extra entries for the denylist can be added via a file pointed
// to by the --xla_gpu_algorithm_denylist_path flag.
std::vector<stream_executor::dnn::AlgorithmDesc> GetDisabledConvAlgorithms(
    ComputeCapability cc, CudnnVersion cudnn_version,
    absl::string_view blas_version, const HloCustomCallInstruction& instr);

// Get the list of convolution algorithms which are present in the denylist for
// the given 'instr' when using compute capability 'cc', cudnn version
// 'cudnn_version' and blas version 'blas_version'.
std::vector<stream_executor::dnn::AlgorithmDesc> GetDisabledConvAlgorithms(
    const DenyListMapType& denylist, ComputeCapability cc,
    CudnnVersion cudnn_version, absl::string_view blas_version,
    const HloCustomCallInstruction& instr);

// Parses the text format denylist proto into the given map.
absl::Status ParseTextFormatDenyList(DenyListMapType& list,
                                     absl::string_view denylist_text);

// Creates a text format denylist proto entry for the given convolution
// instruction. Its output can be added to either the default denylist or to a
// user denylist file.
absl::StatusOr<std::string> GenerateDenyListEntry(
    const HloCustomCallInstruction& instr,
    const stream_executor::dnn::AlgorithmDesc& algo,
    const ComputeCapability& cc, const CudnnVersion& cudnn_version,
    absl::string_view blas_version);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_HLO_ALGORITHM_DENYLIST_H_
