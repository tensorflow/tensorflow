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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_VERSIONED_COMPUTATION_HANDLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_VERSIONED_COMPUTATION_HANDLE_H_

#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// A data structure encapsulating a ComputationHandle and version value of that
// computation. This object is used to unambiguously refer to a particular
// computation in the service.
struct VersionedComputationHandle {
  // A version value unambiguously specifying the state of the computation at a
  // particular point in time as it is being built. This value is the
  // ComputationDataHandle of the current root instruction.
  using Version = int64;

  ComputationHandle handle;
  Version version;
  bool operator==(const VersionedComputationHandle& other) const {
    return (handle.handle() == other.handle.handle()) &&
           (version == other.version);
  }
  bool operator<(const VersionedComputationHandle& other) const {
    return ((handle.handle() < other.handle.handle()) ||
            ((handle.handle() == other.handle.handle()) &&
             (version < other.version)));
  }
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_VERSIONED_COMPUTATION_HANDLE_H_
