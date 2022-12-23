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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_BENCHMARK_RESULT_EVALUATOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_BENCHMARK_RESULT_EVALUATOR_H_

#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

namespace tflite {
namespace acceleration {

// Evaluates the BenchmarkEvent output from validator.
class AbstractBenchmarkResultEvaluator {
 public:
  virtual ~AbstractBenchmarkResultEvaluator() = default;

  // Returns whether this event means the validation test has passed. It checks
  // that the test has finished successfully, and the test result passed
  // accuracy checks.
  virtual bool IsValidationSuccessEvent(const BenchmarkEvent& event) {
    return event.event_type() == BenchmarkEventType_END && event.result() &&
           HasPassedAccuracyCheck(*event.result());
  }

  // Returns whether this BenchmarkResult should pass the accuracy check.
  virtual bool HasPassedAccuracyCheck(const BenchmarkResult& result) = 0;
};

// Evaluator for embedded validation scenario.
class EmbeddedResultEvaluator : public AbstractBenchmarkResultEvaluator {
 public:
  ~EmbeddedResultEvaluator() override = default;

  static EmbeddedResultEvaluator* GetInstance();

  bool HasPassedAccuracyCheck(const BenchmarkResult& result) override;

 private:
  EmbeddedResultEvaluator() = default;
};

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_BENCHMARK_RESULT_EVALUATOR_H_
