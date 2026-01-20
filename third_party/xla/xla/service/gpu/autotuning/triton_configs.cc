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

#include "xla/service/gpu/autotuning/triton_configs.h"

#include <initializer_list>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "xla/autotuning.pb.h"
#include "xla/service/gpu/matmul_utils.h"

namespace xla {
namespace gpu {
namespace {

// TODO(b/467265599): Replace string constants with cc_embed_data when
// https://github.com/bazelbuild/rules_cc/issues/41 is fixed.

constexpr absl::string_view kBlackwellTritonConfigs = R"(
config { block_m: 128 block_n: 128 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 1 num_stages: 1 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 8 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 16 split_k: 512 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 32 split_k: 16 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 1 num_stages: 5 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 64 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 1 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 2 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 4 num_stages: 3 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 64 block_k: 64 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 64 block_k: 64 split_k: 16 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 64 block_k: 64 split_k: 8 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 64 split_k: 8 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 32 block_k: 64 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 256 block_n: 128 block_k: 64 split_k: 1 num_stages: 3 num_warps: 8 num_ctas: 1 }
config { block_m: 256 block_n: 16 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 256 block_n: 32 block_k: 32 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 16 block_k: 32 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 32 block_n: 16 block_k: 512 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 16 block_k: 64 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 32 block_n: 16 block_k: 64 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 16 split_k: 1 num_stages: 1 num_warps: 16 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 16 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 64 split_k: 1 num_stages: 2 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 128 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 32 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 64 split_k: 64 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 128 split_k: 8 num_stages: 1 num_warps: 8 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 16 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 1 num_stages: 1 num_warps: 8 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 16 block_k: 32 split_k: 16 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 1 num_stages: 4 num_warps: 8 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 64 block_k: 64 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 64 block_k: 64 split_k: 16 num_stages: 4 num_warps: 8 num_ctas: 1 is_tma_allowed: true }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 16 block_n: 16 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 16 block_n: 32 block_k: 64 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 256 block_n: 128 block_k: 64 split_k: 1 num_stages: 3 num_warps: 8 num_ctas: 1 is_tma_allowed: true }
config { block_m: 256 block_n: 16 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 256 block_n: 32 block_k: 32 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 32 block_n: 16 block_k: 32 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 32 block_n: 16 block_k: 64 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 32 block_n: 16 block_k: 64 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 128 block_k: 16 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 128 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 32 block_k: 128 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 32 block_k: 32 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 64 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 64 block_k: 16 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
)";

constexpr absl::string_view kDefaultCudaTritonConfigs = R"(
config { block_m: 32 block_n: 32 block_k: 256 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 32 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 64 block_k: 64 split_k: 4 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 4 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 256 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 128 block_k: 32 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 64 block_k: 128 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 128 block_k: 32 split_k: 8 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 512 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 16 block_k: 512 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 64 split_k: 1 num_stages: 2 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 32 split_k: 1 num_stages: 3 num_warps: 8 num_ctas: 1 }
config { block_m: 256 block_n: 128 block_k: 32 split_k: 1 num_stages: 3 num_warps: 8 num_ctas: 1 }
config { block_m: 256 block_n: 64 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 256 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 64 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 256 block_n: 128 block_k: 128 split_k: 1 num_stages: 3 num_warps: 8 num_ctas: 1 }
config { block_m: 256 block_n: 64 block_k: 128 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 256 block_k: 128 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 128 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 64 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 32 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 128 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 256 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 2 num_stages: 1 num_warps: 8 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 64 split_k: 1 num_stages: 2 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 64 block_k: 256 split_k: 8 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 256 block_n: 256 block_k: 128 split_k: 1 num_stages: 3 num_warps: 8 num_ctas: 1 }
)";

constexpr absl::string_view kDefaultRocmTritonConfigs = R"(
config { block_m: 32 block_n: 32 block_k: 256 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 32 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 64 block_k: 64 split_k: 4 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 4 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 256 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 128 block_k: 32 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
)";

constexpr absl::string_view kAmpereTritonConfigs = R"(
config { block_m: 16 block_n: 16 block_k: 64 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 128 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 16 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 256 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 32 block_n: 32 block_k: 128 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 256 block_k: 32 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 32 block_n: 256 block_k: 32 split_k: 16 num_stages: 3 num_warps: 8 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 32 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 32 split_k: 16 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 64 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 64 split_k: 4 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 64 split_k: 16 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 128 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 128 split_k: 16 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 64 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 128 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 128 split_k: 128 num_stages: 2 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 64 split_k: 4 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 128 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 256 split_k: 16 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 16 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 64 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 128 split_k: 8 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 256 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 32 split_k: 8 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 16 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 32 block_k: 32 split_k: 8 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 32 split_k: 8 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 32 split_k: 1 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 1 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 64 block_n: 8 block_k: 128 split_k: 2 num_stages: 3 num_warps: 4 num_ctas: 1 }
)";

constexpr absl::string_view kHopperTritonConfigs = R"(
config { block_m: 16 block_n: 8 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 8 block_k: 256 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 32 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 64 split_k: 128 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 16 block_n: 16 block_k: 128 split_k: 8 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 32 block_n: 8 block_k: 64 split_k: 1 num_stages: 5 num_warps: 2 num_ctas: 1 }
config { block_m: 32 block_n: 16 block_k: 64 split_k: 1 num_stages: 4 num_warps: 2 num_ctas: 1 }
config { block_m: 32 block_n: 16 block_k: 64 split_k: 32 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 32 block_n: 16 block_k: 256 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 8 block_k: 16 split_k: 16 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 8 block_k: 128 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 8 block_k: 128 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 8 block_k: 128 split_k: 16 num_stages: 2 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 32 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 32 split_k: 1 num_stages: 5 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 64 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 16 block_k: 64 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 16 block_k: 128 split_k: 1 num_stages: 5 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 16 block_k: 256 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 32 block_k: 64 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 32 block_k: 64 split_k: 64 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 128 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 32 block_k: 128 split_k: 16 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 32 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 64 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 64 split_k: 1 num_stages: 5 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 64 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 64 split_k: 64 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 64 block_k: 128 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 64 block_n: 128 block_k: 16 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 128 block_k: 64 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 64 block_n: 128 block_k: 128 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 16 block_k: 64 split_k: 8 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 16 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 128 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 32 split_k: 2 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 32 split_k: 4 num_stages: 3 num_warps: 8 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 1 num_stages: 3 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 128 block_k: 64 split_k: 64 num_stages: 3 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 256 block_k: 32 split_k: 1 num_stages: 5 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 32 split_k: 32 num_stages: 4 num_warps: 16 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 1 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 1 num_stages: 4 num_warps: 8 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 256 block_k: 64 split_k: 8 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 32 block_k: 64 split_k: 8 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 64 block_k: 16 split_k: 1 num_stages: 1 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 64 block_k: 16 split_k: 1 num_stages: 3 num_warps: 2 num_ctas: 1 }
config { block_m: 128 block_n: 64 block_k: 32 split_k: 1 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 64 block_k: 32 split_k: 128 num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 64 block_k: 32 split_k: 16 num_stages: 3 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 64 block_k: 32 split_k: 8 num_stages: 3 num_warps: 4 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 64 block_k: 64 split_k: 2 num_stages: 5 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 8 block_k: 128 split_k: 1 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 128 block_n: 8 block_k: 256 split_k: 32 num_stages: 4 num_warps: 8 num_ctas: 1 }
config { block_m: 128 block_n: 8 block_k: 32 split_k: 1 num_stages: 2 num_warps: 2 num_ctas: 1 is_tma_allowed: true }
config { block_m: 128 block_n: 8 block_k: 64 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 }
config { block_m: 256 block_n: 128 block_k: 32 split_k: 1 num_stages: 5 num_warps: 8 num_ctas: 1 is_tma_allowed: true }
config { block_m: 256 block_n: 16 block_k: 16 split_k: 1 num_stages: 1 num_warps: 2 num_ctas: 1 }
config { block_m: 256 block_n: 8 block_k: 32 split_k: 16 num_stages: 3 num_warps: 4 num_ctas: 1 }
)";

absl::flat_hash_map<TritonConfigsPlatform, std::vector<TritonGemmConfig>>
LoadTritonConfigs() {
  absl::flat_hash_map<TritonConfigsPlatform, std::vector<TritonGemmConfig>>
      result;

  auto parse_config =
      [](absl::string_view config_str) -> std::vector<TritonGemmConfig> {
    TritonGemmConfigsProto proto;
    CHECK(tsl::protobuf::TextFormat::ParseFromString(config_str, &proto))
        << config_str;
    std::vector<TritonGemmConfig> configs;
    absl::c_transform(proto.config(), std::back_inserter(configs),
                      [](const AutotuneResult::TritonGemmKey& config_proto) {
                        absl::StatusOr<TritonGemmConfig> config =
                            TritonGemmConfig::FromProto(config_proto);
                        CHECK_OK(config);
                        return *config;
                      });
    return configs;
  };

  const std::initializer_list<
      std::pair<TritonConfigsPlatform, absl::string_view>>
      kConfigsMap = {
          {TritonConfigsPlatform::kAmpere, kAmpereTritonConfigs},
          {TritonConfigsPlatform::kBlackwell, kBlackwellTritonConfigs},
          {TritonConfigsPlatform::kDefaultCuda, kDefaultCudaTritonConfigs},
          {TritonConfigsPlatform::kDefaultRocm, kDefaultRocmTritonConfigs},
          {TritonConfigsPlatform::kHopper, kHopperTritonConfigs},
      };
  for (const auto& [platform, config_str] : kConfigsMap) {
    result[platform] = parse_config(config_str);
  }

  return result;
}

}  // namespace

const std::vector<TritonGemmConfig>& GetTritonConfigsForPlatform(
    TritonConfigsPlatform platform) {
  static const absl::NoDestructor<
      absl::flat_hash_map<TritonConfigsPlatform, std::vector<TritonGemmConfig>>>
      kConfigs(LoadTritonConfigs());
  return kConfigs->at(platform);
}

}  // namespace gpu
}  // namespace xla
