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
#include "tensorflow/core/tpu/kernels/tpu_execute_op_options.h"

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace internal {

namespace {
static TpuCancellationClosesChipsMode tpu_cancellation_closes_chips =
    TpuCancellationClosesChipsMode::kUnset;
}  // namespace

absl::Status SetTpuCancellationClosesChips(int val) {
  if (val < 0 || val > 2) {
    return errors::InvalidArgument(
        "SetTpuCancellationClosesChips: input must be 0 (kUnset), 1 (kEnabled) "
        "or 2 (kDisabled); got ",
        val);
  }
  tpu_cancellation_closes_chips =
      static_cast<TpuCancellationClosesChipsMode>(val);
  return absl::OkStatus();
}

bool TpuCancellationClosesChipsGetOrDefault(bool default_value) {
  if (tpu_cancellation_closes_chips == TpuCancellationClosesChipsMode::kUnset) {
    return default_value;
  }
  return tpu_cancellation_closes_chips ==
         TpuCancellationClosesChipsMode::kEnabled;
}

}  // namespace internal
}  // namespace tensorflow
