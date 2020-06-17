/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_STREAM_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_STREAM_H_

#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_stream_interface.h"

class TpuStream : public tensorflow::tpu::TpuStreamInterface {
 public:
  using Status = stream_executor::port::Status;

  explicit TpuStream(SE_Stream* stream) : stream_(stream) {}
  ~TpuStream() override { TpuStream_Free(stream_); }

  bool IsSameSharedMemoryLocation(
      tensorflow::tpu::TpuStreamInterface* other) override {
    return TpuStream_IsSameSharedMemoryLocation(
        stream_, static_cast<TpuStream*>(other)->stream_);
  }

  Status EnqueueOnTpuDeviceSendRecvLocal(
      stream_executor::DeviceMemoryBase send_buffer,
      stream_executor::DeviceMemoryBase recv_buffer) override {
    StatusHelper status;
    TpuStream_TpuEnqueueOnDeviceSendRecvLocal(
        stream_,
        TpuConversions::DeviceMemoryBaseToSE_DeviceMemoryBase(send_buffer),
        TpuConversions::DeviceMemoryBaseToSE_DeviceMemoryBase(recv_buffer),
        status.c_status);
    return status.status();
  }

 private:
  SE_Stream* stream_;
};

class TpuEvent : public ::stream_executor::internal::EventInterface {
 public:
  explicit TpuEvent(SE_Event* event) : event_(event) {}
  ~TpuEvent() override { TpuEvent_Free(event_); }

 private:
  SE_Event* event_;
};

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_STREAM_H_
