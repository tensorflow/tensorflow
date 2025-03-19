// Copyright 2025 Google LLC.
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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_DISPATCH_DELEGATE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_DISPATCH_DELEGATE_H_

#include <memory>

#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert {

using DispatchDelegateOptionsPtr =
    std::unique_ptr<LiteRtDispatchDelegateOptions,
                    void (*)(LiteRtDispatchDelegateOptions*)>;

using DispatchDelegatePtr = tflite::TfLiteOpaqueDelegateUniquePtr;

DispatchDelegateOptionsPtr CreateDispatchDelegateOptionsPtr(
    LiteRtEnvironmentOptions environment_options);

DispatchDelegatePtr CreateDispatchDelegatePtr(
    LiteRtEnvironmentOptions environment_options,
    DispatchDelegateOptionsPtr&& options);

using DispatchDelegateMetricsPtr =
    std::unique_ptr<LiteRtDispatchDelegateMetricsT,
                    void (*)(LiteRtDispatchDelegateMetricsT*)>;

Expected<void> StartDispatchDelegateMetricsCollection(
    DispatchDelegatePtr& delegate, int detail_level);

Expected<DispatchDelegateMetricsPtr> StopDispatchDelegateMetricsCollection(
    DispatchDelegatePtr& delegate);

Expected<int> DispatchDelegateGetNumMetrics(
    DispatchDelegateMetricsPtr& metrics);

Expected<LiteRtMetric> DispatchDelegateGetMetric(
    DispatchDelegateMetricsPtr& metrics, int metric_index);

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_DISPATCH_DELEGATE_H_
