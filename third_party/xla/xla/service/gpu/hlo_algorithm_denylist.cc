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

#include "xla/service/gpu/hlo_algorithm_denylist.h"

#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/backend_config.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/service/gpu/autotuning/gpu_autotuning.pb.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

constexpr char kDefaultDenylist[] = R"pb(
  entries {
    hlo: "(f32[512,512,7,7]{3,2,1,0}, u8[0]{0}) custom-call(f32[512,512,7,7]{3,2,1,0}, f32[512,512,3,3]{3,2,1,0}, f32[512]{0}), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_oi01->bf01, custom_call_target=\"__cudnn$convBiasActivationForward\""
    backend_config {
      operation_queue_id: 0
      wait_on_operation_queues: []
      cudnn_conv_backend_config: {
        activation_mode: kNone
        conv_result_scale: 1
        side_input_scale: 0
        leakyrelu_alpha: 0
      },
      force_earliest_schedule: false
      device_type: DEVICE_TYPE_DEVICE
    }
    cc { major: 7 }
    cudnn_version { major: 9 }
    algos { id: 14 }
  }
  entries {
    hlo: "(f32[512,512,7,7]{3,2,1,0}, u8[0]{0}) custom-call(f32[512,512,7,7]{3,2,1,0}, f32[512,512,3,3]{3,2,1,0}, f32[512]{0}), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_oi01->bf01, custom_call_target=\"__cudnn$convBiasActivationForward\""
    backend_config {
      operation_queue_id: 0
      wait_on_operation_queues: []
      cudnn_conv_backend_config: {
        activation_mode: kNone
        conv_result_scale: 1
        side_input_scale: 0
        leakyrelu_alpha: 0
      },
      force_earliest_schedule: false
    }
    cc { major: 7 }
    cudnn_version { major: 9 minor: 1 patch: 1 }
    algos { id: 14 }
  }
  entries {
    hlo: "(f32[27,256,32,32]{3,2,1,0}, u8[0]{0}) custom-call(f32[27,256,32,32]{3,2,1,0}, f32[256,256,3,3]{3,2,1,0}, f32[256]{0}, f32[27,256,32,32]{3,2,1,0}), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_oi01->bf01, custom_call_target=\"__cudnn$convBiasActivationForward\""
    backend_config {
      operation_queue_id: 0
      wait_on_operation_queues: []
      cudnn_conv_backend_config: {
        activation_mode: kNone
        conv_result_scale: 1
        side_input_scale: 1,
        leakyrelu_alpha: 0
      },
      force_earliest_schedule: false
    }
    cc { major: 7 }
    cudnn_version { major: 9 }
    algos { id: 14 }
  }
  entries {
    hlo: "(f32[27,256,32,32]{3,2,1,0}, u8[0]{0}) custom-call(f32[27,256,32,32]{3,2,1,0}, f32[256,256,3,3]{3,2,1,0}, f32[256]{0}, f32[27,256,32,32]{3,2,1,0}), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_oi01->bf01, custom_call_target=\"__cudnn$convBiasActivationForward\""
    backend_config {
      operation_queue_id: 0
      wait_on_operation_queues: []
      cudnn_conv_backend_config: {
        activation_mode: kNone
        conv_result_scale: 1
        side_input_scale: 1
        leakyrelu_alpha: 0
      },
      force_earliest_schedule: false
    }
    cc { major: 7 minor: 5 }
    cudnn_version { major: 9 }
    algos { id: 14 }
  }
  entries {
    hlo: "(f32[27,256,32,32]{3,2,1,0}, u8[0]{0}) custom-call(f32[27,256,32,32]{3,2,1,0}, f32[256,256,3,3]{3,2,1,0}, f32[256]{0}, f32[27,256,32,32]{3,2,1,0}), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_oi01->bf01, custom_call_target=\"__cudnn$convBiasActivationForward\""
    backend_config {
      operation_queue_id: 0
      wait_on_operation_queues: []
      cudnn_conv_backend_config: {
        activation_mode: kNone
        conv_result_scale: 1
        side_input_scale: 1
        leakyrelu_alpha: 0
      },
      force_earliest_schedule: false
    }
    cc { major: 7 }
    cudnn_version { major: 9 minor: 1 patch: 1 }
    algos { id: 14 }
  }
  entries {
    hlo: "(f32[27,256,32,32]{3,2,1,0}, u8[0]{0}) custom-call(f32[27,256,32,32]{3,2,1,0}, f32[256,256,3,3]{3,2,1,0}, f32[256]{0}, f32[27,256,32,32]{3,2,1,0}), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_oi01->bf01, custom_call_target=\"__cudnn$convBiasActivationForward\""
    backend_config {
      operation_queue_id: 0
      wait_on_operation_queues: []
      cudnn_conv_backend_config: {
        activation_mode: kNone
        conv_result_scale: 1
        side_input_scale: 1
        leakyrelu_alpha: 0
      },
      force_earliest_schedule: false
    }
    cc { major: 7 minor: 5 }
    cudnn_version { major: 9 minor: 1 patch: 1 }
    algos { id: 14 }
  }
  entries {
    hlo: "(f32[7,2500,3072]{2,1,0}, u8[0]{0}) custom-call(f32[7,2500,3072]{2,1,0}, f32[3072,513,512]{2,1,0}), window={size=513 pad=256_256}, dim_labels=b0f_o0i->b0f, feature_group_count=6, custom_call_target=\"__cudnn$convForward\""
    backend_config {
      force_earliest_schedule: false
      operation_queue_id: 0
      wait_on_operation_queues: []
      cudnn_conv_backend_config: {
        activation_mode: kNone
        conv_result_scale: 1
        leakyrelu_alpha: 0
        side_input_scale: 0
      }
    }
    cc { major: 9 }
    cudnn_version { major: 9 minor: 10 }
    algos { id: 0 }
  }
  entries {
    hlo: "(f32[7,2500,3072]{2,1,0}, u8[0]{0}) custom-call(f32[7,2500,3072]{2,1,0}, f32[3072,257,512]{2,1,0}), window={size=257 pad=128_128}, dim_labels=b0f_o0i->b0f, feature_group_count=6, custom_call_target=\"__cudnn$convForward\""
    cc { major: 9 }
    cudnn_version { major: 9 minor: 10 }
    algos { id: 0 }
    backend_config { cudnn_conv_backend_config { conv_result_scale: 1 } }
  }
  entries {
    hlo: "(f32[7,2500,3072]{2,1,0}, u8[0]{0}) custom-call(f32[7,2500,3072]{2,1,0}, f32[3072,129,512]{2,1,0}), window={size=129 pad=64_64}, dim_labels=b0f_o0i->b0f, feature_group_count=6, custom_call_target=\"__cudnn$convForward\""
    cc { major: 9 }
    cudnn_version { major: 9 minor: 10 }
    algos { id: 0 }
    backend_config { cudnn_conv_backend_config { conv_result_scale: 1 } }
  }
)pb";

static std::string HloStringWithGpuBackendConfig(const std::string& hlo,
                                                 GpuBackendConfig config) {
  BackendConfigWrapper backend_config(config);
  return absl::StrCat(hlo, ", backend_config=", backend_config.GetRawString());
}

absl::Status ParseTextFormatDenyList(DenyListMapType& list,
                                     absl::string_view denylist_text) {
  AlgorithmDenylist proto;
  if (!tsl::protobuf::TextFormat::ParseFromString(denylist_text, &proto)) {
    return absl::InvalidArgumentError("Failed to parse denylist text proto");
  }

  for (const auto& entry : proto.entries()) {
    for (const auto& algo : entry.algos()) {
      list[std::make_tuple(HloStringWithGpuBackendConfig(
                               entry.hlo(), entry.backend_config()),
                           entry.cc().major(), entry.cc().minor(),
                           entry.cudnn_version().major(),
                           entry.cudnn_version().minor(),
                           entry.cudnn_version().patch(), entry.blas_version())]
          .emplace_back(algo.id(), algo.tensor_ops(), std::nullopt);
    }
  }
  return absl::OkStatus();
}

std::vector<stream_executor::dnn::AlgorithmDesc> GetDisabledConvAlgorithms(
    ComputeCapability cc, CudnnVersion cudnn_version,
    absl::string_view blas_version, const HloCustomCallInstruction& instr) {
  static DenyListMapType* denylist = [] {
    auto* list = new DenyListMapType();
    std::string file_path =
        GetDebugOptionsFromFlags().xla_gpu_algorithm_denylist_path();
    if (!file_path.empty()) {
      std::string denylist_text;
      CHECK_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path,
                                     &denylist_text));
      CHECK_OK(ParseTextFormatDenyList(*list, denylist_text));
    }
    CHECK_OK(ParseTextFormatDenyList(*list, kDefaultDenylist));
    return list;
  }();

  return GetDisabledConvAlgorithms(*denylist, cc, cudnn_version, blas_version,
                                   instr);
}

std::vector<stream_executor::dnn::AlgorithmDesc> GetDisabledConvAlgorithms(
    const DenyListMapType& denylist, ComputeCapability cc,
    CudnnVersion cudnn_version, absl::string_view blas_version,
    const HloCustomCallInstruction& instr) {
  std::vector<stream_executor::dnn::AlgorithmDesc> algorithms;
  auto add_matching_disabled_algorithms_to_result = [&](const auto& key) {
    auto iter = denylist.find(key);
    if (iter != denylist.end()) {
      algorithms.insert(algorithms.end(), iter->second.begin(),
                        iter->second.end());
    }
  };

  std::string hlo = instr.ToString(
      ::xla::HloPrintOptions::Fingerprint().set_print_backend_config(true));

  // Exclude algorithms with explicit BLAS version set
  auto key = std::make_tuple(hlo, cc.major(), cc.minor(), cudnn_version.major(),
                             cudnn_version.minor(), cudnn_version.patch(),
                             std::string{blas_version});
  add_matching_disabled_algorithms_to_result(key);

  // Exclude algorithms with no BLAS version set
  std::get<6>(key) = std::string{};
  add_matching_disabled_algorithms_to_result(key);

  return algorithms;
}

absl::StatusOr<std::string> GenerateDenyListEntry(
    const HloCustomCallInstruction& instr,
    const stream_executor::dnn::AlgorithmDesc& algo,
    const ComputeCapability& cc, const CudnnVersion& cudnn_version,
    absl::string_view blas_version) {
  AlgorithmDenylist list;
  AlgorithmDenylistEntry* entry = list.add_entries();
  entry->set_hlo(instr.ToString(::xla::HloPrintOptions::Fingerprint()));
  TF_ASSIGN_OR_RETURN(*entry->mutable_backend_config(),
                      instr.backend_config<GpuBackendConfig>());

  *entry->mutable_cc() = cc;
  *entry->mutable_cudnn_version() = cudnn_version;
  entry->set_blas_version(blas_version);

  DenylistedAlgorithm* denylisted_algo = entry->add_algos();
  denylisted_algo->set_id(algo.algo_id());
  denylisted_algo->set_tensor_ops(algo.tensor_ops_enabled());

  std::string denylist_string;
  tsl::protobuf::TextFormat::PrintToString(list, &denylist_string);
  return denylist_string;
}

}  // namespace gpu
}  // namespace xla
