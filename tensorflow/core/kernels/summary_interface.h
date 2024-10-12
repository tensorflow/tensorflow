/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_SUMMARY_INTERFACE_H_
#define TENSORFLOW_CORE_KERNELS_SUMMARY_INTERFACE_H_

#include <memory>

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Event;
class GraphDef;

// Main interface for the summary writer resource.
class SummaryWriterInterface : public ResourceBase {
 public:
  virtual ~SummaryWriterInterface() override {}

  // Flushes all unwritten messages in the queue.
  virtual absl::Status Flush() = 0;

  // These are called in the OpKernel::Compute methods for the summary ops.
  virtual absl::Status WriteTensor(int64_t global_step, Tensor t,
                                   const string& tag,
                                   const string& serialized_metadata) = 0;

  virtual absl::Status WriteScalar(int64_t global_step, Tensor t,
                                   const string& tag) = 0;

  virtual absl::Status WriteHistogram(int64_t global_step, Tensor t,
                                      const string& tag) = 0;

  virtual absl::Status WriteImage(int64_t global_step, Tensor t,
                                  const string& tag, int max_images,
                                  Tensor bad_color) = 0;

  virtual absl::Status WriteAudio(int64_t global_step, Tensor t,
                                  const string& tag, int max_outputs_,
                                  float sample_rate) = 0;

  virtual absl::Status WriteGraph(int64_t global_step,
                                  std::unique_ptr<GraphDef> graph) = 0;

  virtual absl::Status WriteEvent(std::unique_ptr<Event> e) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SUMMARY_INTERFACE_H_
