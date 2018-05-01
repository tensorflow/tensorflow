/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_XLA_CLIENT_XLA_COMPUTATION_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_XLA_CLIENT_XLA_COMPUTATION_H_

#include <utility>

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// The computation graph that the user builds up with the XlaBuilder.
//
// TODO(b/74197823): Replace xla::Computation with this one.
class XlaComputation {
 public:
  XlaComputation() : unique_id_(-1) {}
  XlaComputation(const HloModuleProto& proto)
      : unique_id_(proto.id()), proto_(proto) {}

  ~XlaComputation() {}

  XlaComputation(const XlaComputation&) = delete;
  XlaComputation& operator=(const XlaComputation&) = delete;

  XlaComputation(XlaComputation&& from) = default;

  XlaComputation& operator=(XlaComputation&& from) = default;

  // Returns the "program shape" (parameter and return shapes) for this
  // computation.
  StatusOr<ProgramShape> GetProgramShape() const;

  const HloModuleProto& proto() const { return proto_; }

  // Requests that we snapshot the computation into a serializable protocol
  // buffer form.
  StatusOr<std::unique_ptr<HloSnapshot>> Snapshot() const;

  // Returns true if this object is a null Computation.
  bool IsNull() const { return unique_id_ == -1; }

 private:
  XlaComputation(const int64 unique_id) : unique_id_(unique_id) {}
  HloModuleProto* mutable_proto() { return &proto_; }
  friend class XlaBuilder;

  int64 unique_id_;
  HloModuleProto proto_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_XLA_CLIENT_XLA_COMPUTATION_H_
