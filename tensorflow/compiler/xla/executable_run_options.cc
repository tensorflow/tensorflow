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

#include "tensorflow/compiler/xla/executable_run_options.h"

namespace xla {

ExecutableRunOptions& ExecutableRunOptions::set_device_ordinal(
    int device_ordinal) {
  device_ordinal_ = device_ordinal;
  return *this;
}

int ExecutableRunOptions::device_ordinal() const { return device_ordinal_; }

ExecutableRunOptions& ExecutableRunOptions::set_allocator(
    DeviceMemoryAllocator* allocator) {
  allocator_ = allocator;
  return *this;
}

DeviceMemoryAllocator* ExecutableRunOptions::allocator() const {
  return allocator_;
}

ExecutableRunOptions& ExecutableRunOptions::set_stream(
    perftools::gputools::Stream* stream) {
  stream_ = stream;
  return *this;
}

perftools::gputools::Stream* ExecutableRunOptions::stream() const {
  return stream_;
}

ExecutableRunOptions& ExecutableRunOptions::set_inter_op_thread_pool(
    tensorflow::thread::ThreadPool* inter_op_thread_pool) {
  inter_op_thread_pool_ = inter_op_thread_pool;
  return *this;
}

tensorflow::thread::ThreadPool* ExecutableRunOptions::inter_op_thread_pool()
    const {
  return inter_op_thread_pool_;
}

ExecutableRunOptions& ExecutableRunOptions::set_intra_op_thread_pool(
    const Eigen::ThreadPoolDevice* intra_op_thread_pool) {
  intra_op_thread_pool_ = intra_op_thread_pool;
  return *this;
}

const Eigen::ThreadPoolDevice* ExecutableRunOptions::intra_op_thread_pool()
    const {
  return intra_op_thread_pool_;
}

ExecutableRunOptions& ExecutableRunOptions::set_execution_profile(
    ExecutionProfile* profile) {
  execution_profile_ = profile;
  return *this;
}

ExecutionProfile* ExecutableRunOptions::execution_profile() const {
  return execution_profile_;
}

}  // namespace xla
