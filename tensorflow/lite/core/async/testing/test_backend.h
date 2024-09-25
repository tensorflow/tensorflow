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
#ifndef TENSORFLOW_LITE_CORE_ASYNC_TESTING_TEST_BACKEND_H_
#define TENSORFLOW_LITE_CORE_ASYNC_TESTING_TEST_BACKEND_H_

#include <limits>
#include <memory>

#include "tensorflow/lite/core/async/async_kernel_internal.h"
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace async {
namespace testing {

// A test backend that takes in arbitrary TfLiteAsyncKernel.
class TestBackend {
 public:
  explicit TestBackend(TfLiteAsyncKernel* kernel);

  TfLiteDelegate* get_delegate() { return &delegate_; }
  TfLiteAsyncKernel* get_kernel() { return kernel_; }

  // Maximum delegate partitions.
  int NumPartitions() const { return num_partitions_; }
  void SetNumPartitions(int num_partitions) {
    num_partitions_ = num_partitions;
  }

  // Minimal number of nodes the backend delegates.
  int MinPartitionedNodes() const { return min_partioned_nodes_; }
  void SetMinPartitionedNodes(int min_partioned_nodes) {
    min_partioned_nodes_ = min_partioned_nodes;
  }

 private:
  // Not owned.
  TfLiteAsyncKernel* kernel_ = nullptr;

  // Owned.
  TfLiteDelegate delegate_;

  int num_partitions_ = std::numeric_limits<int>::max();
  int min_partioned_nodes_ = 0;
};

}  // namespace testing
}  // namespace async
}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_ASYNC_TESTING_TEST_BACKEND_H_
