/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/autotune_serialize.h"

#include <string>

#include "absl/base/call_once.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/autotune_results.pb.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/gpu/triton_autotuner.h"
#include "tensorflow/tsl/platform/env.h"

namespace xla {
namespace {

// Bump this version whenever you change the structure of the results.
// LINT.IfChange(version)
constexpr int kVersion = 2;
// LINT.ThenChange()

bool IsTextProtoPath(absl::string_view file_path) {
  return absl::EndsWith(file_path, ".txt") ||
         absl::EndsWith(file_path, ".textproto");
}

}  // anonymous namespace

Status LoadAutotuneResults(absl::string_view data, bool as_textproto) {
  AutotuneResults results;
  // The cast here is necessary for MacOS builds.
  bool parse_success =
      as_textproto ? tsl::protobuf::TextFormat::ParseFromString(
                         std::string(data), &results)             // NOLINT
                   : results.ParseFromString(std::string(data));  // NOLINT
  if (!parse_success) {
    return tsl::errors::InvalidArgument(
        "Failed to parse autotune results string.");
  }
  if (results.version() != kVersion) {
    return tsl::errors::InvalidArgument(absl::StrFormat(
        "Version mismatch in autotune results. Expected %d but was %d",
        kVersion, results.version()));
  }

  TF_RETURN_IF_ERROR(gpu::GpuConvAlgorithmPicker::LoadAutotuneResults(results));
  TF_RETURN_IF_ERROR(gpu::GemmAlgorithmPicker::LoadAutotuneResults(results));
  TF_RETURN_IF_ERROR(gpu::TritonAutotuner::LoadAutotuneResults(results));
  return OkStatus();
}

StatusOr<std::string> SerializeAutotuneResults(bool as_textproto) {
  AutotuneResults results;
  results.set_version(kVersion);

  TF_RETURN_IF_ERROR(
      gpu::GpuConvAlgorithmPicker::WriteAutotuneResults(&results));
  TF_RETURN_IF_ERROR(gpu::GemmAlgorithmPicker::WriteAutotuneResults(&results));
  TF_RETURN_IF_ERROR(gpu::TritonAutotuner::WriteAutotuneResults(&results));

  if (as_textproto) {
    std::string textproto;
    if (tsl::protobuf::TextFormat::PrintToString(results, &textproto)) {
      return textproto;
    } else {
      return tsl::errors::Internal("Failed to serialize autotune results.");
    }
  }
  return results.SerializeAsString();
}

Status SerializeAutotuneResultsToFile(absl::string_view file_path) {
  TF_RET_CHECK(!file_path.empty());
  // Some APIs need a const std::string&.
  std::string file_path_str(file_path);

  std::string autotune_results_str;
  TF_ASSIGN_OR_RETURN(
      autotune_results_str,
      xla::SerializeAutotuneResults(IsTextProtoPath(file_path_str)));
  TF_RETURN_IF_ERROR(tsl::WriteStringToFile(tsl::Env::Default(), file_path_str,
                                            autotune_results_str));
  LOG(INFO) << "Autotune results serialized to file: " << file_path_str;

  return OkStatus();
}

Status LoadAutotuneResultsFromFile(absl::string_view file_path) {
  TF_RET_CHECK(!file_path.empty());
  // Some APIs need a const std::string&.
  std::string file_path_str(file_path);

  if (!tsl::Env::Default()->FileExists(file_path_str).ok()) {
    return FailedPrecondition("Autotune results file does not exist: %s",
                              file_path_str);
  }
  std::string autotune_results_str;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(), file_path_str,
                                           &autotune_results_str));

  TF_RETURN_IF_ERROR(LoadAutotuneResults(autotune_results_str,
                                         IsTextProtoPath(file_path_str)));

  LOG(INFO) << "Autotune results loaded from file: " << file_path_str;

  return OkStatus();
}

Status LoadAutotuneResultsFromFileOnce(absl::string_view file_path) {
  Status status = OkStatus();

  static absl::once_flag once_flag;
  absl::call_once(once_flag, [&file_path, &status] {
    status = LoadAutotuneResultsFromFile(file_path);
  });

  return status;
}

}  // namespace xla
