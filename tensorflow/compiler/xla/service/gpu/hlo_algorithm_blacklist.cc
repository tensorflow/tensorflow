/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/hlo_algorithm_blacklist.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_autotuning.pb.h"

namespace xla {
namespace gpu {

constexpr absl::string_view kDefaultBlacklist = R"pb(
  entries {
    hlo: "(f16[256,112,112,64]{3,2,1,0}, u8[0]{0}) custom-call(f16[256,224,224,4]{3,2,1,0}, f16[7,7,4,64]{2,1,0,3}), window={size=7x7 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f, custom_call_target=\"__cudnn$convForward\", backend_config=\"{conv_result_scale:1}\""
    cc { major: 7 }
    cudnn_version { major: 7 minor: 6 patch: 2 }
    blas_version: "10201"
    algos { id: 1 tensor_ops: true }
  }
  entries {
    hlo: "(f16[7,7,4,64]{2,1,0,3}, u8[0]{0}) custom-call(f16[256,224,224,4]{3,2,1,0}, f16[256,112,112,64]{3,2,1,0}), window={size=7x7 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f, custom_call_target=\"__cudnn$convBackwardFilter\", backend_config=\"{conv_result_scale:1}\""
    cc { major: 7 }
    cudnn_version { major: 7 minor: 6 patch: 2 }
    blas_version: "10201"
    algos { id: 1 tensor_ops: true }
  })pb";

absl::Span<const stream_executor::dnn::AlgorithmDesc>
GetBlacklistedConvAlgorithms(tensorflow::ComputeCapability cc,
                             tensorflow::CudnnVersion cudnn_version,
                             absl::string_view blas_version,
                             absl::string_view hlo) {
  // Key is the tuple of canonicalized hlo, compute capability major/minor,
  // cudnn version major/minor/patch, blas version.
  using MapType = absl::flat_hash_map<
      std::tuple<std::string, int, int, int, int, int, std::string>,
      std::vector<stream_executor::dnn::AlgorithmDesc>>;

  static MapType* blacklist = [] {
    MapType* list = new MapType();
    AlgorithmBlacklist proto;
    std::string file_path =
        GetDebugOptionsFromFlags().xla_gpu_algorithm_blacklist_path();
    if (!file_path.empty()) {
      TF_CHECK_OK(tensorflow::ReadTextProto(tensorflow::Env::Default(),
                                            file_path, &proto));
    } else {
      CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
          std::string(kDefaultBlacklist), &proto));
    }
    for (const auto& entry : proto.entries()) {
      for (const auto& algo : entry.algos()) {
        (*list)[std::make_tuple(
                    std::string(entry.hlo()), entry.cc().major(),
                    entry.cc().minor(), entry.cudnn_version().major(),
                    entry.cudnn_version().minor(),
                    entry.cudnn_version().patch(), entry.blas_version())]
            .push_back({algo.id(), algo.tensor_ops()});
      }
    }
    return list;
  }();

  auto iter = blacklist->find(std::make_tuple(
      std::string(hlo), cc.major(), cc.minor(), cudnn_version.major(),
      cudnn_version.minor(), cudnn_version.patch(), std::string(blas_version)));
  if (iter != blacklist->end()) {
    return iter->second;
  }
  return {};
}

}  // namespace gpu
}  // namespace xla
