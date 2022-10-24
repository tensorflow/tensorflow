/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// For Google-internal use only.
#include "tensorflow/core/util/autotune_maps/autotune_serialize.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/stream_executor/dnn.h"
#include "tensorflow/compiler/xla/stream_executor/dnn.pb.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/util/activation_mode.h"
#include "tensorflow/core/util/autotune_maps/autotune_map.pb.h"
#include "tensorflow/core/util/autotune_maps/autotune_maps_utils.h"
#include "tensorflow/core/util/autotune_maps/conv_autotune_maps.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.pb.h"

namespace tensorflow {

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
namespace {
using stream_executor::dnn::AlgorithmConfig;
using stream_executor::dnn::AlgorithmConfigProto;
using stream_executor::dnn::AlgorithmDesc;
using stream_executor::dnn::AlgorithmProto;

template <typename Op>
ConvMapProto ConvMapToProto(
    const AutotuneMap<ConvParameters, AutotuneEntry<Op>> &autotune_map) {
  ConvMapProto proto;

  // Deterministically sort the entries in autotune maps
  // according to the serialized string of ConvParametersProto in order to
  // enable deterministic serialization. The actual order is meaningless.
  //
  // This step also filters out duplicate entries (only device_id's are
  // different) in the autotune maps. So that there is only one entry for a
  // convolution operation with a specific GPU device type.
  std::map<string, ConvMapProto::Entry> sorted_map;

  for (auto const &p : autotune_map.GetMap()) {
    const ConvParameters &params = p.first;
    const ConvParametersProto &params_proto = params.proto();
    VLOG(1) << "Reading: " << params.ToString();

    ConvMapProto::Entry kv;
    *kv.mutable_key() = params_proto;

    if (p.second.is_algorithm_config()) {
      *kv.mutable_value() = p.second.GetAlgorithmConfig().ToProto();
    } else {
      const auto &runners = p.second.GetOpRunners();
      *kv.mutable_value()->mutable_algorithm() =
          runners.primary->ToAlgorithmDesc().ToProto();
      if (runners.no_scratch_fallback) {
        *kv.mutable_value()->mutable_algorithm_no_scratch() =
            runners.no_scratch_fallback->ToAlgorithmDesc().ToProto();
      }
    }

    sorted_map.insert(std::make_pair(
        autotune_maps_utils::SerializeProtoDeterministic(params_proto), kv));
  }

  for (auto const &p : sorted_map) {
    ConvMapProto::Entry *kv = proto.add_kv_pairs();
    *kv = p.second;
  }
  return proto;
}

template <typename Op>
Status PopulateConvMap(
    const ConvMapProto &m,
    AutotuneMap<ConvParameters, AutotuneEntry<Op>> *autotune_map) {
  if (m.kv_pairs().size() == 0) {
    return OkStatus();
  }
  std::set<std::string> unmatched_device_ids;
  // Map device_id's to corresponding device_identifiers.
  std::vector<string> device_ids_map =
      autotune_maps_utils::GetDeviceIdToIdentifierMap();
  // Map device_identifiers to device_ids whose corresponding GPU devices have
  // the given device_identifier.
  std::unordered_map<string, std::vector<int>> device_identifiers_map;
  bool devices_matched = false;
  for (const ConvMapProto::Entry &kv : m.kv_pairs()) {
    const ConvParametersProto &params_proto = kv.key();
    // Abort loading process whenever there is an entry whose version number
    // doesn't match runtime version because the autotune results may be
    // incorrect.
    if (params_proto.version() != ConvParameters::kVersion) {
      VLOG(1) << "ConvParametersProto with the incompatible version:"
              << params_proto.DebugString();
      return errors::Aborted(
          "Aborted because the loaded autotune results for convolution "
          "operations have a version different "
          "from runtime's version. Expected version: ",
          ConvParameters::kVersion,
          ". Actual version: ", params_proto.version());
    }

    auto iter = device_identifiers_map.find(params_proto.device_identifier());
    std::vector<int> device_ids;
    if (iter == device_identifiers_map.end()) {
      for (int i = 0; i < device_ids_map.size(); i++) {
        if (device_ids_map[i] == params_proto.device_identifier()) {
          device_ids.push_back(i);
        }
      }
      device_identifiers_map.insert(
          std::make_pair(params_proto.device_identifier(), device_ids));
    } else {
      device_ids = iter->second;
    }

    if (device_ids.empty()) {
      unmatched_device_ids.insert(params_proto.device_identifier());
    } else {
      devices_matched = true;
    }

    const AlgorithmConfigProto &algorithm_config_proto = kv.value();
    const AlgorithmDesc primary(algorithm_config_proto.algorithm());
    const absl::optional<AlgorithmDesc> fallback =
        algorithm_config_proto.has_algorithm_no_scratch()
            ? absl::optional<AlgorithmDesc>(
                  AlgorithmDesc(algorithm_config_proto.algorithm_no_scratch()))
            : absl::nullopt;

    for (int device_id : device_ids) {
      AutotuneEntry<Op> entry;
#if TENSORFLOW_USE_ROCM
      // ROCm doesn't yet support the OpRunner-based API, so for the time being
      // we still need legacy AlgorithmDesc entries in the autotune map.
      // Long-term, this should be folded into the next case.
      entry = AutotuneEntry<Op>(AlgorithmConfig(algorithm_config_proto));
#else
      entry = AutotuneEntry<Op>(primary, fallback);
#endif

      autotune_map->Insert(ConvParameters(device_id, params_proto), entry);
    }
  }

  if (!unmatched_device_ids.empty()) {
    LOG(WARNING) << "Unmatched device id's from AoT autotuning data: "
                 << str_util::Join(unmatched_device_ids, ", ")
                 << "; existing devices: "
                 << str_util::Join(device_ids_map, ", ");
  }

  // When no matching devices are found, populating autotuning map will not
  // happen. Instead of silently reporting an OK status, report an error back.
  if (!devices_matched) {
    return errors::NotFound("No matching devices found for ",
                            str_util::Join(device_ids_map, ", "));
  }
  return OkStatus();
}

}  // namespace
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

Status SerializeAutotuneMaps(std::string *output) {
  AutotuneMapsProto proto;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  *proto.mutable_conv_map() = ConvMapToProto(*ConvAutotuneMap::GetInstance());
  *proto.mutable_fused_conv_map() =
      ConvMapToProto(*FusedConvAutotuneMap::GetInstance());
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  *output = autotune_maps_utils::SerializeProtoDeterministic(proto);
  return OkStatus();
}

Status LoadSerializedAutotuneMaps(absl::string_view s) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  AutotuneMapsProto proto;
  // The explicit string conversion here is a workaround for
  // resolving the issue that OSS proto library's ParseFromString only accepts
  // std::string.
  if (!proto.ParseFromString(string(s))) {
    return errors::InvalidArgument(
        "Failed to parse the autotune maps from string.");
  }
  TF_RETURN_IF_ERROR(
      PopulateConvMap(proto.conv_map(), ConvAutotuneMap::GetInstance()));
  TF_RETURN_IF_ERROR(PopulateConvMap(proto.fused_conv_map(),
                                     FusedConvAutotuneMap::GetInstance()));
  // TODO(b/189530096): Populate autotune maps for more ops.
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return OkStatus();
}

void ResetAutotuneMaps() {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  ConvAutotuneMap::GetInstance()->ClearMap();
  FusedConvAutotuneMap::GetInstance()->ClearMap();
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

}  // namespace tensorflow
