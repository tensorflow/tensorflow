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
#include "absl/strings/str_cat.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/backend_config.h"
#include "xla/service/gpu/autotuning/gpu_autotuning.pb.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/dnn.h"
#include "tsl/platform/env.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"

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
)pb";

std::vector<stream_executor::dnn::AlgorithmDesc> GetDisabledConvAlgorithms(
    ComputeCapability cc, CudnnVersion cudnn_version,
    const std::string& blas_version, const std::string& hlo) {
  // Key is the tuple of canonicalized hlo, compute capability major/minor,
  // cudnn version major/minor/patch, blas version.
  using MapType = absl::flat_hash_map<
      std::tuple<std::string, int, int, int, int, int, std::string>,
      std::vector<stream_executor::dnn::AlgorithmDesc>>;

  static MapType* denylist = [] {
    auto* list = new MapType();
    AlgorithmDenylist proto;
    auto process_denylist = [list](const AlgorithmDenylist& proto) {
      for (const auto& entry : proto.entries()) {
        for (const auto& algo : entry.algos()) {
          (*list)[std::make_tuple(HloStringWithGpuBackendConfig(
                                      entry.hlo(), entry.backend_config()),
                                  entry.cc().major(), entry.cc().minor(),
                                  entry.cudnn_version().major(),
                                  entry.cudnn_version().minor(),
                                  entry.cudnn_version().patch(),
                                  entry.blas_version())]
              .emplace_back(algo.id(), algo.tensor_ops(), std::nullopt);
        }
      }
    };

    std::string file_path =
        GetDebugOptionsFromFlags().xla_gpu_algorithm_denylist_path();
    if (!file_path.empty()) {
      TF_CHECK_OK(tsl::ReadTextProto(tsl::Env::Default(), file_path, &proto));
      process_denylist(proto);
    }
    CHECK(tsl::protobuf::TextFormat::ParseFromString(
        std::string(kDefaultDenylist), &proto));
    process_denylist(proto);
    return list;
  }();

  std::vector<stream_executor::dnn::AlgorithmDesc> algorithms;
  auto add_matching_disabled_algorithms_to_result = [&](const auto& key) {
    auto iter = denylist->find(key);
    if (iter != denylist->end()) {
      algorithms.insert(algorithms.end(), iter->second.begin(),
                        iter->second.end());
    }
  };

  // Exclude algorithms with explicit BLAS version set
  auto key = std::make_tuple(hlo, cc.major(), cc.minor(), cudnn_version.major(),
                             cudnn_version.minor(), cudnn_version.patch(),
                             blas_version);
  add_matching_disabled_algorithms_to_result(key);

  // Exclude algorithms with no BLAS version set
  std::get<6>(key) = std::string{};
  add_matching_disabled_algorithms_to_result(key);

  return algorithms;
}

std::string HloStringWithGpuBackendConfig(const std::string& hlo,
                                          GpuBackendConfig config) {
  BackendConfigWrapper backend_config(config);
  return absl::StrCat(hlo, ", backend_config=", backend_config.GetRawString());
}

}  // namespace gpu
}  // namespace xla
