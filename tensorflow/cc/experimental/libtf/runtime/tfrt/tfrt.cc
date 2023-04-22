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
#include "tensorflow/cc/experimental/libtf/runtime/tfrt/tfrt.h"

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/cc/experimental/libtf/value.h"

namespace tf {
namespace libtf {
namespace runtime {
namespace tfrt {

runtime::Runtime Runtime() {
  TFE_Context* ctx;
  TFE_ContextOptions* ctx_options = TFE_NewContextOptions();
  TFE_ContextOptionsSetTfrt(ctx_options, true);
  TFE_ContextOptionsSetDevicePlacementPolicy(ctx_options,
                                             TFE_DEVICE_PLACEMENT_WARN);
  TF_Status* status = TF_NewStatus();
  ctx = TFE_NewContext(ctx_options, status);
  TF_DeleteStatus(status);
  TFE_DeleteContextOptions(ctx_options);
  return runtime::Runtime(tensorflow::unwrap(ctx));
}

}  // namespace tfrt
}  // namespace runtime
}  // namespace libtf
}  // namespace tf
