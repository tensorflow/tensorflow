/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/contrib/lite/kernels/eigen_support.h"

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace eigen_support {

struct RefCountedEigenContext {
  int num_references = 0;
};

void IncrementUsageCounter(TfLiteContext* context) {
  auto* ptr = reinterpret_cast<RefCountedEigenContext*>(context->eigen_context);
  if (ptr == nullptr) {
    if (context->recommended_num_threads != -1) {
      Eigen::setNbThreads(context->recommended_num_threads);
    }
    ptr = new RefCountedEigenContext;
    ptr->num_references = 0;
    context->eigen_context = ptr;
  }
  ptr->num_references++;
}

void DecrementUsageCounter(TfLiteContext* context) {
  auto* ptr = reinterpret_cast<RefCountedEigenContext*>(context->eigen_context);
  if (ptr == nullptr) {
    TF_LITE_FATAL(
        "Call to DecrementUsageCounter() not preceded by "
        "IncrementUsageCounter()");
  }
  if (--ptr->num_references == 0) {
    delete ptr;
  }
}

}  // namespace eigen_support
}  // namespace tflite
