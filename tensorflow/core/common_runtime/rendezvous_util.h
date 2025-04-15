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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_RENDEZVOUS_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_RENDEZVOUS_UTIL_H_

#include <map>

#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

typedef std::map<string, Tensor> NamedTensors;
typedef std::function<void(const absl::Status&)> StatusCallback;

// Uses `rendezvous` to send tensors in `tensors_to_send`. `device_context`
// should be the DeviceContext associated with the source of the tensors.
// `alloc_attrs` contains information about how the `tensors_to_send` are
// allocated. `alloc_attrs` should either be {} or should match the length of
// `keys`.
absl::Status SendTensorsToRendezvous(
    RendezvousInterface* rendezvous, DeviceContext* device_context,
    const std::vector<AllocatorAttributes>& alloc_attrs,
    const std::vector<string>& keys, absl::Span<const Tensor> tensors_to_send);

// Uses `rendezvous` to obtain tensors. `device_context` should be the
// DeviceContext associated with the receiving device. `alloc_attrs` contains
// information as how to store the received tensors. Should be {} or match the
// length of `keys`.
void RecvOutputsFromRendezvousAsync(
    RendezvousInterface* rendezvous, DeviceContext* device_context,
    const std::vector<AllocatorAttributes>& alloc_attrs,
    const std::vector<string>& keys, std::vector<Tensor>* received_tensors,
    StatusCallback done);

absl::Status RecvOutputsFromRendezvous(RendezvousInterface* rendezvous,
                                       NamedTensors* out,
                                       const Rendezvous::Args& args);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_RENDEZVOUS_UTIL_H_
