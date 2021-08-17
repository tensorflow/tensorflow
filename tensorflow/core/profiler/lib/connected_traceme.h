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
#ifndef TENSORFLOW_CORE_PROFILER_LIB_CONNECTED_TRACEME_H_
#define TENSORFLOW_CORE_PROFILER_LIB_CONNECTED_TRACEME_H_

#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"

namespace tensorflow {
namespace profiler {

enum class ContextType : int {
  kGeneric = 0,
  kLegacy,
  kTfExecutor,
  kTfrtExecutor,
  kSharedBatchScheduler,
  kPjRt,
  kAdaptiveSharedBatchScheduler,
  kTfrtTpuRuntime,
};

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
class TraceMeProducer {
 public:
  template <typename NameT>
  explicit TraceMeProducer(NameT name,
                           ContextType context_type = ContextType::kGeneric,
                           absl::optional<uint64> context_id = absl::nullopt,
                           int level = 2)
      : trace_me_(name, level) {
    trace_me_.AppendMetadata([&] {
      context_id_ =
          context_id.has_value() ? *context_id : TraceMe::NewActivityId();
      return TraceMeEncode({{"_pt", context_type}, {"_p", context_id_}});
    });
  }

  uint64 GetContextId() const { return context_id_; }

 private:
  TraceMe trace_me_;
  uint64 context_id_ = 0;
};

class TraceMeConsumer {
 public:
  template <typename NameT>
  TraceMeConsumer(NameT name, ContextType context_type, uint64 context_id,
                  int level = 2)
      : trace_me_(name, level) {
    trace_me_.AppendMetadata([&] {
      return TraceMeEncode({{"_ct", context_type}, {"_c", context_id}});
    });
  }

  template <typename NameT>
  TraceMeConsumer(NameT name, uint64 context_id, int level = 2)
      : TraceMeConsumer(name, ContextType::kGeneric, context_id, level) {}

 private:
  TraceMe trace_me_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_LIB_CONNECTED_TRACEME_H_
