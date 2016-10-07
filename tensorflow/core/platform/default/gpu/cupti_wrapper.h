/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_PLATFORM_DEFAULT_CUPTI_WRAPPER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_PLATFORM_DEFAULT_CUPTI_WRAPPER_H_

#if GOOGLE_CUDA

#include <stddef.h>
#include <stdint.h>

#include "cuda/extras/CUPTI/include/cupti.h"

namespace perftools {
namespace gputools {
namespace profiler {

// Wraps the CUPTI API so that we can dynamically load the library.
class CuptiWrapper {
 public:
  CuptiWrapper() {}

  // CUPTI activity API
  CUptiResult ActivityDisable(CUpti_ActivityKind kind);

  CUptiResult ActivityEnable(CUpti_ActivityKind kind);

  CUptiResult ActivityFlushAll(uint32_t flag);

  CUptiResult ActivityGetNextRecord(uint8_t* buffer,
                                    size_t valid_buffer_size_bytes,
                                    CUpti_Activity** record);

  CUptiResult ActivityGetNumDroppedRecords(CUcontext context,
                                           uint32_t stream_id, size_t* dropped);

  CUptiResult ActivityRegisterCallbacks(
      CUpti_BuffersCallbackRequestFunc func_buffer_requested,
      CUpti_BuffersCallbackCompleteFunc func_buffer_completed);

  CUptiResult GetDeviceId(CUcontext context, uint32_t* deviceId);

  CUptiResult GetTimestamp(uint64_t* timestamp);

  // CUPTI callback API
  CUptiResult EnableCallback(uint32_t enable, CUpti_SubscriberHandle subscriber,
                             CUpti_CallbackDomain domain,
                             CUpti_CallbackId cbid);

  CUptiResult EnableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber,
                           CUpti_CallbackDomain domain);

  CUptiResult Subscribe(CUpti_SubscriberHandle* subscriber,
                        CUpti_CallbackFunc callback, void* userdata);

  CUptiResult Unsubscribe(CUpti_SubscriberHandle subscriber);
};

}  // namespace profiler
}  // namespace gputools
}  // namespace perftools

#endif  // GOOGLE_CUDA

#endif  // THIRD_PARTY_TENSORFLOW_CORE_PLATFORM_DEFAULT_CUPTI_WRAPPER_H_
