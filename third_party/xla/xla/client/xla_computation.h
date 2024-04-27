/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_CLIENT_XLA_COMPUTATION_H_
#define XLA_CLIENT_XLA_COMPUTATION_H_

#include <memory>
#include <string>
#include <utility>

#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla {

// The computation graph that the user builds up with the XlaBuilder.
class XlaComputation {
 public:
  XlaComputation() : unique_id_(-1) {}
  XlaComputation(HloModuleProto proto)
      : unique_id_(proto.id()), proto_(std::move(proto)) {}

  ~XlaComputation() = default;

  XlaComputation(const XlaComputation&) = delete;
  XlaComputation& operator=(const XlaComputation&) = delete;

  XlaComputation(XlaComputation&& from) = default;

  XlaComputation& operator=(XlaComputation&& from) = default;

  // Returns the "program shape" (parameter and return shapes) for this
  // computation.
  absl::StatusOr<ProgramShape> GetProgramShape() const;

  const std::string& name() const { return proto().name(); }

  const HloModuleProto& proto() const { return proto_; }
  HloModuleProto* mutable_proto() { return &proto_; }

  // Requests that we snapshot the computation into a serializable protocol
  // buffer form.
  absl::StatusOr<std::unique_ptr<HloSnapshot>> Snapshot() const;

  // Returns true if this object is a null Computation.
  bool IsNull() const { return unique_id_ == -1; }

 private:
  XlaComputation(const int64_t unique_id) : unique_id_(unique_id) {}
  friend class XlaBuilder;

  int64_t unique_id_;
  HloModuleProto proto_;
};

}  // namespace xla

#endif  // XLA_CLIENT_XLA_COMPUTATION_H_
