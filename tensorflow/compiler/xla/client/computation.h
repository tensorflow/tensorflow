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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_COMPUTATION_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_COMPUTATION_H_

#include <memory>

#include "tensorflow/compiler/xla/service/session.pb.h"
#include "tensorflow/compiler/xla/service_interface.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// Wraps a ComputationHandle protobuf with a lifetime. Computation is
// movable and not copyable to capture the same kind of unique
// ownership that std::unique_ptr represents.
//
// TODO(b/74197823): Deprecated. Use XlaComputation instead.
class Computation {
 public:
  // Creates a null Computation.
  Computation();

  // parent: stub for the service on which we will deallocate the computation
  //   when it is no longer needed.
  // handle: the computation handle protobuf from the service.
  Computation(ServiceInterface* parent, const ComputationHandle& handle);

  Computation(Computation&& computation);

  // Deallocates the computation.
  ~Computation();

  Computation& operator=(Computation&& computation);

  // Returns the underlying handle.
  const ComputationHandle& handle() const { return handle_; }

  // Sets handle to a null state and clears any owned computation.
  void Reset();

  // Requests that we snapshot the computation into a serializable protocol
  // buffer form.
  StatusOr<std::unique_ptr<SessionModule>> Snapshot() const;

  // Returns true if this object is a null Computation.
  bool IsNull() const { return parent_ == nullptr; }

  // Returns the "program shape" (parameter and return shapes) for this
  // computation.
  StatusOr<ProgramShape> GetProgramShape() const;

 private:
  void ResetWithoutFreeing();

  ComputationHandle handle_;  // Handle that is wrapped by this class.

  // Stub that the handle is deallocated on when this object's lifetime ends.
  ServiceInterface* parent_;

  TF_DISALLOW_COPY_AND_ASSIGN(Computation);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_COMPUTATION_H_
