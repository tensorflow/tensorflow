// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/core/environment.h"

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert::internal {

Environment* Environment::the_instance_ = nullptr;

Expected<void> Environment::CreateWithOptions(
    absl::Span<const LiteRtEnvOption> options) {
  LITERT_LOG(LITERT_INFO, "Environment::CreateWithOptions the_instance_=%p",
             the_instance_);
  if (the_instance_) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "LiteRT environment cannot be created with options, it has "
                 "already been created");
  }
  LITERT_LOG(LITERT_INFO, "Creating LiteRT environment with options");
  the_instance_ = new Environment();
  for (auto& option : options) {
    the_instance_->options_[option.tag] = option.value;
  }
  return {};
}

void Environment::Destroy() {
  delete the_instance_;
  the_instance_ = nullptr;
}

Expected<Environment*> Environment::Instance() {
  if (!the_instance_) {
    LITERT_LOG(LITERT_INFO, "Creating LiteRT environment with no options");
    the_instance_ = new Environment();
  }
  return the_instance_;
}

}  // namespace litert::internal
