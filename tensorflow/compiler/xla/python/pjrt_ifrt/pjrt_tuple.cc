/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_tuple.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/client.h"
#include "tfrt/concurrency/ref_count.h"  // from @tf_runtime

namespace xla {
namespace ifrt {

/*static*/ StatusOr<tsl::RCReference<PjRtTuple>> PjRtTuple::Create(
    PjRtCompatibleClient* client, absl::Span<tsl::RCReference<Value>> values) {
  return tsl::MakeRef<PjRtTuple>(client, values);
}

Future<Status> PjRtTuple::GetReadyFuture() const {
  std::vector<Future<Status>> futures;
  futures.reserve(values_.size());
  for (const auto& value : values_) {
    futures.push_back(value->GetReadyFuture());
  }
  return JoinFutures(absl::MakeSpan(futures));
}

Future<Status> PjRtTuple::Delete() {
  {
    absl::MutexLock lock(&mu_);
    if (!is_deleted_.HasBeenNotified()) {
      is_deleted_.Notify();
    }
  }
  std::vector<Future<Status>> futures;
  futures.reserve(values_.size());
  for (const auto& value : values_) {
    futures.push_back(value->Delete());
  }
  return JoinFutures(absl::MakeSpan(futures));
}

bool PjRtTuple::IsDeleted() const {
  if (is_deleted_.HasBeenNotified()) {
    return true;
  }
  for (const auto& value : values_) {
    if (value->IsDeleted()) {
      return true;
    }
  }
  return false;
}

std::string PjRtTuple::DebugString() const {
  return absl::StrFormat(
      "PjRtTuple(%s)",
      absl::StrJoin(values_, ",",
                    [](std::string* out, const tsl::RCReference<Value>& value) {
                      out->append(value->DebugString());
                    }));
}
int PjRtTuple::Arity() { return values_.size(); }

Status PjRtTuple::Unpack(absl::Span<tsl::RCReference<Value>> values_out) {
  if (values_out.size() != values_.size()) {
    return InvalidArgument(
        "Wrong number of output values for "
        "PjRtTuple::Unpack(); got %d expected %d.",
        values_out.size(), values_.size());
  }
  absl::c_copy(values_, values_out.begin());
  return OkStatus();
}

char PjRtTuple::ID = 0;

PjRtTuple::PjRtTuple(PjRtCompatibleClient* client,
                     absl::Span<tsl::RCReference<Value>> values)
    : client_(client), values_(values.begin(), values.end()) {}

}  // namespace ifrt
}  // namespace xla
