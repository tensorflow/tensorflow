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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/benchmark_result_evaluator.h"

#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

namespace tflite {
namespace acceleration {

bool EmbeddedResultEvaluator::IsValidationSuccessEvent(
    const BenchmarkEvent& event) {
  return event.event_type() == BenchmarkEventType_END && event.result() &&
         event.result()->ok();
}

bool CustomResultEvaluator::IsValidationSuccessEvent(
    const BenchmarkEvent& event) {
  return event.event_type() == BenchmarkEventType_END;
}

}  // namespace acceleration
}  // namespace tflite
