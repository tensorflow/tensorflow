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

#include "tensorflow/lite/delegates/telemetry.h"

#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace delegates {

// TODO(b/153131797): Add an IFTTT here once we have a profiler to interpret
// these events, so that the two components don't go out of sync.

TfLiteStatus ReportDelegateSettings(TfLiteContext* context,
                                    TfLiteDelegate* delegate,
                                    const TFLiteSettings& settings) {
  auto* profiler = reinterpret_cast<Profiler*>(context->profiler);
  const int64_t event_metadata1 = reinterpret_cast<int64_t>(delegate);
  const int64_t event_metadata2 = reinterpret_cast<int64_t>(&settings);
  TFLITE_ADD_RUNTIME_INSTRUMENTATION_EVENT(profiler, kDelegateSettingsTag,
                                           event_metadata1, event_metadata2);
  return kTfLiteOk;
}

TfLiteStatus ReportDelegateStatus(TfLiteContext* context,
                                  TfLiteDelegate* delegate,
                                  const DelegateStatus& status) {
  auto* profiler = reinterpret_cast<Profiler*>(context->profiler);
  TFLITE_ADD_RUNTIME_INSTRUMENTATION_EVENT(profiler, kDelegateStatusTag,
                                           status.full_status(),
                                           static_cast<int64_t>(kTfLiteOk));
  return kTfLiteOk;
}

}  // namespace delegates
}  // namespace tflite
