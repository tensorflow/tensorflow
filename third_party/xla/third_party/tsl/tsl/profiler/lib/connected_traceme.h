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
#ifndef TENSORFLOW_TSL_PROFILER_LIB_CONNECTED_TRACEME_H_
#define TENSORFLOW_TSL_PROFILER_LIB_CONNECTED_TRACEME_H_

#include <optional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xla/tsl/platform/types.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace tsl {
namespace profiler {

/*
 * TraceMeProducer and TraceMeConsumer are used to correlate TraceMe events on
 * different threads. TraceMeProducer generates the context information to be
 * passed to TraceMeConsumer, which consists of the context id and optionally
 * the context type. They may be provided by the user. Then, the events of the
 * same context information can be correlated during the analysis.
 *
 * Example Usages:
 * (1) Using the user-provided context type and id. The user is responsible for
 *     providing the same context type and id to TraceMeProducer and
 *     TraceMeConsumer.
 * [Producer Thread]
 * // user_context_id is provided by the user.
 * TraceMeProducer producer(
 *     [&] { return TraceMeEncode("op_dispatch", {{"op_type", "matmul"}}); },
 *     ContextType::kTfExecutor, user_context_id);
 * [Consumer Thread]
 * // user_context_id is provided by the user.
 * TraceMeConsumer consumer(
 *     [&] { return "op_execute"; }, ContextType::kTfExecutor, user_context_id);
 *
 * (2) Using the user-provided context type and generic id. The user is
 *     responsible for passing the TraceMeProducer's context id to
 *     TraceMeConsumer as well as providing the same context type to
 *     TraceMeProducer and TraceMeConsumer.
 * [Producer Thread]
 * TraceMeProducer producer(
 *     [&] { return TraceMeEncode("op_dispatch", {{"op_type", "matmul"}}); },
 *     ContextType::kTfExecutor);
 * context_id = producer.GetContextId();
 * // Pass context_id to the consumer thread.
 * [Consumer Thread]
 * // context_id is passed from the producer thread.
 * TraceMeConsumer consumer(
 *     [&] { return "op_execute"; }, ContextType::kTfExecutor, context_id);
 *
 * (3) Using the generic context information. The user is responsible for
 *     passing the TraceMeProducer's context id to TraceMeConsumer.
 * [Producer Thread]
 * TraceMeProducer producer(
 *     [&] { return TraceMeEncode("op_dispatch", {{"op_type", "matmul"}}); });
 * context_id = producer.GetContextId();
 * // Pass context_id to the consumer thread.
 * [Consumer Thread]
 * // context_id is passed from the producer thread.
 * TraceMeConsumer consumer([&] { return "op_execute"; }, context_id);
 */
class TraceMeProducer : public TraceMe {
 public:
  template <typename NameT>
  explicit TraceMeProducer(NameT&& name,
                           ContextType context_type = ContextType::kGeneric,
                           std::optional<uint64> context_id = std::nullopt,
                           int level = 2)
      : TraceMe(std::forward<NameT>(name), level),
        context_id_(context_id.has_value() ? context_id.value()
                                           : TraceMe::NewActivityId()) {
    AppendMetadata([&] {
      return TraceMeEncode({{"_pt", context_type}, {"_p", context_id_}});
    });
  }

  uint64 GetContextId() const { return context_id_; }

 private:
  uint64 context_id_;
};

class TraceMeConsumer : public TraceMe {
 public:
  template <typename NameT>
  TraceMeConsumer(NameT&& name, ContextType context_type, uint64 context_id,
                  int level = 2)
      : TraceMe(std::forward<NameT>(name), level) {
    AppendMetadata([&] {
      return TraceMeEncode({{"_ct", context_type}, {"_c", context_id}});
    });
  }

  template <typename NameT>
  TraceMeConsumer(NameT&& name, uint64 context_id, int level = 2)
      : TraceMeConsumer(std::forward<NameT>(name), ContextType::kGeneric,
                        context_id, level) {}
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_LIB_CONNECTED_TRACEME_H_
