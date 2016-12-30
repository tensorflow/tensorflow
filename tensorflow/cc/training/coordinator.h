/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CC_TRAINING_COORDINATOR_H_
#define THIRD_PARTY_TENSORFLOW_CC_TRAINING_COORDINATOR_H_

#include <atomic>
#include <memory>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// The abstract interface for runners which must implement the Join function.
class RunnerInterface {
 public:
  virtual ~RunnerInterface() {}
  virtual Status Join() = 0;

  // Returns true iff the runner is running, i.e. if it is trying to populate
  // its queue.
  virtual bool IsRunning() const = 0;
};

// Coordinator class manages the termination of a collection of QueueRunners.
// Without a coordinator, QueueRunners have to be joined in a specific order;
// otherwise the QueueRunner::Join() could sometimes hang. The
// Coordinator::RequestStop() plays the key role which notifies all running
// threads under a coordinator to stop. This function could be called by any
// thread or any client.
// Usage, in the client:
//   Coordinator coord;
//   std::unique_ptr<QueueRunner> qr(&coord, ...);
//   qr.Start(session);
//   coord.RegisterRunner(std::move(qr));
//   // do some work
//   TF_CHECK_OK(coord.Join());
// In each thread of QueueRunner, the coordinator needs to be used as:
//   void Run() {
//     while (!coord->ShouldStop()) {
//       // do some work
//       if (error) {
//         coord->RequestStop();
//         coord->ReportStatus(error_status);
//       }
//     }
//   }
class Coordinator {
 public:
  Coordinator();

  // Constructor with a list of error codes which would not be taken as errors
  // in status reporting.
  Coordinator(const std::vector<error::Code>& clean_stop_errors);

  // In the destructor, RequestStop() and Join() would be called.
  ~Coordinator();

  // Registers a runner, i.e. a unit of running threads which is usually a
  // QueueRunner. It takes the ownership of runner to avoid lifecycle-related
  // problems. Note, the coordinator would not start these threads; they are
  // supposed to be in running state when they are registered here.
  Status RegisterRunner(std::unique_ptr<RunnerInterface> runner);

  // Returns true iff all the registered runners have been stopped.
  bool AllRunnersStopped();

  // Requests all running threads to stop.
  Status RequestStop();

  // Returns true if its RequestStop() has been called.
  bool ShouldStop();

  // Joins all threads, returns OK or the first reported and unexpected status.
  Status Join();

  // Reports status to the coordinator. This is usually called by threads.
  void ReportStatus(const Status& status);

  // Returns the latest status.
  Status GetStatus();

  // Returns immediately if the coordinator is stopped or blocks until
  // RequestStop() is called.
  void WaitForStop();

 private:
  std::unordered_set<int> clean_stop_errors_;
  condition_variable wait_for_stop_;

  mutex mu_;
  bool should_stop_ GUARDED_BY(mu_);

  mutex status_lock_;
  Status status_ GUARDED_BY(status_lock_);

  mutex runners_lock_;
  std::vector<std::unique_ptr<RunnerInterface>> runners_
      GUARDED_BY(runners_lock_);

  std::atomic<int> num_runners_to_cancel_;

  TF_DISALLOW_COPY_AND_ASSIGN(Coordinator);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CC_TRAINING_COORDINATOR_H_
