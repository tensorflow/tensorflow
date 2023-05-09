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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_TUPLE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_TUPLE_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/client.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_client.h"
#include "tfrt/concurrency/ref_count.h"  // from @tf_runtime

namespace xla {
namespace ifrt {

class PjRtTuple final : public llvm::RTTIExtends<PjRtTuple, Tuple> {
 public:
  static StatusOr<tsl::RCReference<PjRtTuple>> Create(
      PjRtCompatibleClient* client, absl::Span<tsl::RCReference<Value>> values);

  ~PjRtTuple() override = default;

  PjRtCompatibleClient* client() const override {
    DCHECK(this);
    return client_;
  }

  Future<Status> GetReadyFuture() const override;

  Future<Status> Delete() override;

  bool IsDeleted() const override;

  std::string DebugString() const override;

  int Arity() override;

  Status Unpack(absl::Span<tsl::RCReference<Value>> values) override;

  static char ID;  // NOLINT

 private:
  PjRtTuple(PjRtCompatibleClient* client,
            absl::Span<tsl::RCReference<Value>> values);

  template <typename T, typename... Args>
  friend tsl::RCReference<T> tsl::MakeRef(Args&&... args);

  PjRtCompatibleClient* client_;
  absl::InlinedVector<tsl::RCReference<Value>, 4> values_;

  absl::Mutex mu_;

  // Notifying requires holding mu_.
  absl::Notification is_deleted_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_TUPLE_H_
