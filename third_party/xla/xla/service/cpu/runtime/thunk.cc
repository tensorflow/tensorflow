/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/cpu/runtime/thunk.h"

#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla::cpu {

std::string_view Thunk::KindToString(Kind kind) {
  switch (kind) {
    case Kind::kCall:
      return "call";
    case Kind::kCopy:
      return "copy";
    case Kind::kKernel:
      return "kernel";
    case Kind::kWhile:
      return "while";
  }
}

// Encodes thunk info into the TraceMe compatible format.
std::string Thunk::TraceMeEncode() const {
  return tsl::profiler::TraceMeEncode(info_.op_name,
                                      {{"hlo_op", info_.op_name},
                                       {"hlo_module", info_.module_name},
                                       {"hlo_module_id", info_.module_id}});
}

std::ostream& operator<<(std::ostream& os, Thunk::Kind kind) {
  os << Thunk::KindToString(kind);
  return os;
}

ThunkSequence::ThunkSequence(std::unique_ptr<Thunk> thunk) {
  push_back(std::move(thunk));
}

void ThunkSequence::Append(ThunkSequence other) {
  reserve(size() + other.size());
  for (auto& thunk : other) {
    push_back(std::move(thunk));
  }
}

absl::Status ThunkSequence::Execute(const Thunk::ExecuteParams& params) {
  VLOG(2) << "Execute thunk sequence of size " << size();
  for (auto& thunk : *this) {
    TF_RETURN_IF_ERROR(thunk->Execute(params));
  }
  return absl::OkStatus();
}

}  // namespace xla::cpu
