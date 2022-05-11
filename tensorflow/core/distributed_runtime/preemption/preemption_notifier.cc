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
#include "tensorflow/core/distributed_runtime/preemption/preemption_notifier.h"

#include <atomic>
#include <csignal>
#include <functional>
#include <memory>

#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

namespace {
constexpr absl::Duration kListenInterval = absl::Seconds(1);
static std::atomic_bool sigterm_received(false);

class SigtermNotifier : public PreemptionNotifier {
 public:
  explicit SigtermNotifier(Env* env);
  ~SigtermNotifier() override = default;
  void Reset() override;

 private:
  void StartListenerThread();
  std::unique_ptr<Thread> preempt_listener_thread_;
};

SigtermNotifier::SigtermNotifier(Env* env) {
  env_ = env;
  StartListenerThread();
  std::signal(SIGTERM, [](int signal) { sigterm_received.store(true); });
}

void SigtermNotifier::StartListenerThread() {
  preempt_listener_thread_.reset(
      env_->StartThread({}, "PreemptionNotifier_Listen", [this]() {
        int64_t listen_interval_micros =
            absl::ToInt64Microseconds(kListenInterval);

        // Poll for SIGTERM receipt every kListenInterval.
        while (!sigterm_received.load()) {
          env_->SleepForMicroseconds(listen_interval_micros);
        }

        mutex_lock l(mu_);
        death_time_ = absl::Now();
        LOG(WARNING) << "SIGTERM caught at " << death_time_;
        // Notify registered listeners.
        NotifyRegisteredListeners();
      }));
}

void SigtermNotifier::Reset() {
  {
    mutex_lock l(mu_);
    death_time_ = absl::InfinitePast();
    callbacks_.clear();
  }
  sigterm_received.store(false);
  StartListenerThread();
}
}  // namespace

absl::Time PreemptionNotifier::WillBePreemptedAt() {
  absl::Notification n;
  absl::Time death_time;
  WillBePreemptedAtAsync([&n, &death_time](absl::Time time) {
    death_time = time;
    n.Notify();
  });
  n.WaitForNotification();
  return death_time;
}

void PreemptionNotifier::WillBePreemptedAtAsync(PreemptTimeCallback callback) {
  mutex_lock l(mu_);
  if (death_time_ == absl::InfinitePast()) {
    // Did not receive preemption notice yet.
    callbacks_.push_back(callback);
  } else {
    // Already received preemption notice, respond immediately.
    callback(death_time_);
  }
}

void PreemptionNotifier::NotifyRegisteredListeners() {
  for (const auto& callback : callbacks_) {
    env_->SchedClosure(
        [callback, death_time = death_time_]() { callback(death_time); });
  }
  callbacks_.clear();
}

std::unique_ptr<PreemptionNotifier> CreateSigtermNotifier(Env* env) {
  return std::make_unique<SigtermNotifier>(env);
}

}  // namespace tensorflow
