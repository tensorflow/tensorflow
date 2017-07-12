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

#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"

#include <functional>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {
namespace cpu {
namespace runtime {

XfeedManager* GetXfeedManager() {
  static XfeedManager* manager = new XfeedManager;
  return manager;
}

}  // namespace runtime
}  // namespace cpu
}  // namespace xla

void* __xla_cpu_runtime_AcquireInfeedBufferForDequeue(
    xla::int32 buffer_length) {
  VLOG(2) << "AcquireInfeedBufferForDequeue";
  xla::cpu::runtime::XfeedManager* xfeed = xla::cpu::runtime::GetXfeedManager();
  // Wait until there's a buffer to dequeue.
  xla::cpu::runtime::XfeedBuffer* buffer =
      xfeed->infeed()->BlockingDequeueBuffer();
  CHECK_EQ(buffer->length(), buffer_length);
  return buffer->data();
}

void __xla_cpu_runtime_ReleaseInfeedBufferAfterDequeue(xla::int32 buffer_length,
                                                       void* buffer_ptr) {
  VLOG(2) << "ReleaseInfeedBufferAfterDequeue";
  xla::cpu::runtime::XfeedManager* xfeed = xla::cpu::runtime::GetXfeedManager();
  xfeed->infeed()->ReleaseCurrentBuffer(buffer_length, buffer_ptr);
}

void* __xla_cpu_runtime_AcquireOutfeedBufferForPopulation(
    xla::int32 buffer_length) {
  VLOG(2) << "AcquireOutfeedBufferForPopulation";
  xla::cpu::runtime::XfeedManager* xfeed = xla::cpu::runtime::GetXfeedManager();
  // Wait until there's a buffer to dequeue.
  xla::cpu::runtime::XfeedBuffer* buffer =
      xfeed->outfeed()->BlockingDequeueBuffer();
  CHECK_EQ(buffer->length(), buffer_length);
  return buffer->data();
}

void __xla_cpu_runtime_ReleaseOutfeedBufferAfterPopulation(
    xla::int32 buffer_length, void* buffer_ptr) {
  VLOG(2) << "ReleaseOutfeedBufferAfterPopulation";
  xla::cpu::runtime::XfeedManager* xfeed = xla::cpu::runtime::GetXfeedManager();
  xfeed->outfeed()->ReleaseCurrentBuffer(buffer_length, buffer_ptr);
}
