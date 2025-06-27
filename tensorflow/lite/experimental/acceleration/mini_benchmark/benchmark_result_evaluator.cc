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

#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"

namespace tflite {
namespace acceleration {

EmbeddedResultEvaluator* EmbeddedResultEvaluator::GetInstance() {
  static EmbeddedResultEvaluator* const instance =
      new EmbeddedResultEvaluator();
  return instance;
}

bool EmbeddedResultEvaluator::HasPassedAccuracyCheck(
    const BenchmarkResult& result) {
  return result.ok();
}

}  // namespace acceleration
}  // namespace tflite
