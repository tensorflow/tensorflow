/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/service/data_service.h"

#include <string>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace data {

namespace {
constexpr const char kParallelEpochs[] = "parallel_epochs";
constexpr const char kDistributedEpoch[] = "distributed_epoch";
constexpr const char kAuto[] = "AUTO";
constexpr const char kAny[] = "ANY";
constexpr const char kLocal[] = "LOCAL";
}  // namespace

Status ParseProcessingMode(const std::string& s, ProcessingMode& mode) {
  if (s == kParallelEpochs) {
    mode = ProcessingMode::PARALLEL_EPOCHS;
  } else if (s == kDistributedEpoch) {
    mode = ProcessingMode::DISTRIBUTED_EPOCH;
  } else {
    return errors::InvalidArgument("Unrecognized processing mode: ", s);
  }
  return Status::OK();
}

std::string ProcessingModeToString(ProcessingMode mode) {
  switch (mode) {
    case ProcessingMode::PARALLEL_EPOCHS:
      return kParallelEpochs;
    case ProcessingMode::DISTRIBUTED_EPOCH:
      return kDistributedEpoch;
    default:
      DCHECK(false);
      return "Unknown";
  }
}

StatusOr<TargetWorkers> ParseTargetWorkers(absl::string_view s) {
  std::string str_upper = absl::AsciiStrToUpper(s);
  if (str_upper.empty() || str_upper == kAuto) {
    return TargetWorkers::AUTO;
  }
  if (str_upper == kAny) {
    return TargetWorkers::ANY;
  }
  if (str_upper == kLocal) {
    return TargetWorkers::LOCAL;
  }
  return errors::InvalidArgument("Unrecognized target workers: ", s);
}

std::string TargetWorkersToString(TargetWorkers target_workers) {
  switch (target_workers) {
    case TargetWorkers::AUTO:
      return kAuto;
    case TargetWorkers::ANY:
      return kAny;
    case TargetWorkers::LOCAL:
      return kLocal;
    default:
      DCHECK(false);
      return "UNKNOWN";
  }
}

}  // namespace data
}  // namespace tensorflow
