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
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {

// Main interface for the summary writer resource.
class SummaryWriterInterface : public ResourceBase {
 public:
  virtual ~SummaryWriterInterface() override {}

  // Flushes all unwritten messages in the queue.
  virtual Status Flush() = 0;

  // These are called in the OpKernel::Compute methods for the summary ops.
  virtual Status WriteTensor(int64 global_step, Tensor t, const string& tag,
                             const string& serialized_metadata) = 0;

  virtual Status WriteScalar(int64 global_step, Tensor t,
                             const string& tag) = 0;

  virtual Status WriteHistogram(int64 global_step, Tensor t,
                                const string& tag) = 0;

  virtual Status WriteImage(int64 global_step, Tensor t, const string& tag,
                            int max_images, Tensor bad_color) = 0;

  virtual Status WriteAudio(int64 global_step, Tensor t, const string& tag,
                            int max_outputs_, float sample_rate) = 0;

  virtual Status WriteEvent(std::unique_ptr<Event> e) = 0;
};

// Creates a SummaryWriterInterface instance which writes to a file. It will
// enqueue up to max_queue summaries, and flush at least every flush_millis
// milliseconds. The summaries will be written to the directory specified by
// logdir and with the filename suffixed by filename_suffix. The caller owns a
// reference to result if the returned status is ok. The Env object must not
// be destroyed until after the returned writer.
Status CreateSummaryWriter(int max_queue, int flush_millis,
                           const string& logdir, const string& filename_suffix,
                           Env* env, SummaryWriterInterface** result);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SUMMARY_INTERFACE_H_
