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
#ifndef TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_PREEMPTION_PREEMPTION_NOTIFIER_H_
#define TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_PREEMPTION_PREEMPTION_NOTIFIER_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "tsl/platform/env.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/statusor.h"

namespace tsl {

// Static registration for preemption notifiers.
#define REGISTER_PREEMPTION_NOTIFIER(notifier_type_name, factory_fn)        \
  REGISTER_PREEMPTION_NOTIFIER_UNIQ_HELPER(__COUNTER__, notifier_type_name, \
                                           factory_fn)
#define REGISTER_PREEMPTION_NOTIFIER_UNIQ_HELPER(counter, notifier_type_name, \
                                                 factory_fn)                  \
  static bool static_preemption_notifier_##counter TF_ATTRIBUTE_UNUSED =      \
      []() {                                                                  \
        ::tsl::PreemptionNotifier::RegisterPreemptionNotifier(                \
            notifier_type_name, factory_fn);                                  \
        return true;                                                          \
      }()

// Base class for listening and propagating task preemption notices.
//
// This class provides common mechanism to block on waiting for preemption
// signals, or register callbacks that will be triggered upon preemption.
//
// Example:
//
//    // Monitors the SIGTERM preemption signal
//    notifier = PreemptionNotifier::CreatePreemptionNotifier("sigterm", env);
//
//    // Register callback that will be invoked once preempted
//    notifier->WillBePreemptedAtAsync(
//      [](StatusOr<absl::Time> status_or_time) {
//        if (status_or_time.ok()) {
//          LOG(INFO) << "Preempted at time: " << status_or_time.value();
//        } else {
//          LOG(ERROR) << "Received error: " << status_or_time.status();
//        }
//      });
//
//    // Block current thread until preemption
//    absl::Time preempt_time = notifier->WillBePreemptedAt().value();
//
// Users can extend this class to support custom preemption signals, by subclass
// `PreemptionNotifier` with a custom constructor, register its creator (factory
// function) with `REGISTER_PREEMPTION_NOTIFIER`. The custom constructor should
// set up the communication with the cluster scheduler, and invoke the
// `NotifyRegisteredListeners` method once a preemption signal is received.
// See `SigtermNotifier` as an example.

class PreemptionNotifier {
 public:
  typedef std::function<void(absl::StatusOr<absl::Time>)> PreemptTimeCallback;
  using PreemptionNotifierFactory =
      std::function<std::unique_ptr<PreemptionNotifier>(Env* env)>;

  explicit PreemptionNotifier(Env* env) : env_(env) {}
  virtual ~PreemptionNotifier() = default;

  static void RegisterPreemptionNotifier(const std::string& notifier_type_name,
                                         PreemptionNotifierFactory factory_fn) {
    GetPreemptionNotifierFactories()->emplace(notifier_type_name,
                                              std::move(factory_fn));
  }

  static std::unique_ptr<PreemptionNotifier> CreatePreemptionNotifier(
      const std::string& notifier_type, Env* env) {
    const auto* factories = GetPreemptionNotifierFactories();
    auto it = factories->find(notifier_type);
    if (it == factories->end()) {
      std::vector<std::string> registered_types;
      registered_types.reserve(factories->size());
      for (auto& kv : *factories) {
        registered_types.push_back(kv.first);
      }
      LOG(ERROR) << "No preemption notifier factory found for notifier type "
                 << notifier_type
                 << ". All registered preemption notifier types are: "
                 << absl::StrJoin(registered_types, ", ")
                 << ". Make sure the library is loaded to the program.";
      return nullptr;
    }
    return it->second(env);
  }

  // This is a blocking call that returns a death time when preemption /
  // termination will occur once the listener receives the preemption
  // notification. If no death time is specified, absl::Now() is returned.
  // Returns error::Cancelled if UnregisterListeners() is called.
  absl::StatusOr<absl::Time> WillBePreemptedAt();

  // Registers a callback that takes the death time as input once the listener
  // receives the preemption notification.
  // If no death time is specified, absl::Now() is specified as input.
  // Note: callback should be kept as simple and fast as possible (e.g. simply
  // retrieve result). It should not wait for work done by another callback, and
  // invoke ahy PreemptionNotifier method (e.g. Reset(), destructor).
  void WillBePreemptedAtAsync(PreemptTimeCallback callback);

 protected:
  Env* GetEnv() { return env_; }
  // Invokes all pending callbacks upon receipt of preemption notice with death
  // time or errors (e.g. cancellation during shutdown).
  void NotifyRegisteredListeners(absl::StatusOr<absl::Time> death_time);

 private:
  static std::unordered_map<std::string, PreemptionNotifierFactory>*
  GetPreemptionNotifierFactories() {
    static auto* preemption_notifier_factories =
        new std::unordered_map<std::string, PreemptionNotifierFactory>();
    return preemption_notifier_factories;
  }

  Env* env_;  // Not owned.
  mutex mu_;
  absl::Time death_time_ TF_GUARDED_BY(mu_) = absl::InfinitePast();
  std::vector<PreemptTimeCallback> callbacks_ TF_GUARDED_BY(mu_);
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_DISTRIBUTED_RUNTIME_PREEMPTION_PREEMPTION_NOTIFIER_H_
