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

#include "tensorflow/compiler/xla/autotune_results.pb.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.h"

namespace xla {
namespace {

// Bump this version whenever you change the structure of the results.
// LINT.IfChange(version)
constexpr int kVersion = 1;
// LINT.ThenChange()

}  // anonymous namespace

Status LoadAutotuneResults(absl::string_view data) {
  AutotuneResults results;
  // The cast here is necessary for MacOS builds.
  if (!results.ParseFromString(std::string(data))) {  // NOLINT
    return tsl::errors::InvalidArgument(
        "Failed to parse autotune results string.");
  }
  if (results.version() != kVersion) {
    return tsl::errors::InvalidArgument(absl::StrFormat(
        "Version mismatch in autotune results.  Expected %d but was %d",
        kVersion, results.version()));
  }

  TF_RETURN_IF_ERROR(gpu::GpuConvAlgorithmPicker::LoadAutotuneResults(results));
  TF_RETURN_IF_ERROR(gpu::GemmAlgorithmPicker::LoadAutotuneResults(results));
  return OkStatus();
}

StatusOr<std::string> SerializeAutotuneResults() {
  AutotuneResults results;
  results.set_version(kVersion);

  TF_RETURN_IF_ERROR(
      gpu::GpuConvAlgorithmPicker::WriteAutotuneResults(&results));
  TF_RETURN_IF_ERROR(gpu::GemmAlgorithmPicker::WriteAutotuneResults(&results));

  return results.SerializeAsString();
}

}  // namespace xla
