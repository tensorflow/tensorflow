/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ERROR_PAYLOADS_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ERROR_PAYLOADS_H_

// This file lists the proto payloads that may be inserted  by the code within
// `tensorflow/core/distributed_runtime/` into Status instances.

namespace tensorflow {
// Proto: tensorflow::distributed_runtime::WorkerPossiblyRestarted
// Location: tensorflow/core/protobuf/distributed_runtime_payloads.proto
// Usage: Flags the Status to be a possible outcome of a worker restart.
constexpr char kWorkerPossiblyRestarted[] =
    "type.googleapis.com/"
    "tensorflow.distributed_runtime.WorkerPossiblyRestarted";

constexpr char kWorkerPreemption[] =
    "type.googleapis.com/tensorflow.distributed_runtime.WorkerPreemption";

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_ERROR_PAYLOADS_H_
