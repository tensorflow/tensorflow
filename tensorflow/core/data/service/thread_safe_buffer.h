/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_THREAD_SAFE_BUFFER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_THREAD_SAFE_BUFFER_H_

#include <deque>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace data {

// A thread-safe bounded buffer with cancellation support.
template <class T>
class ThreadSafeBuffer final {
 public:
  // Creates a buffer with the specified `buffer_size`.
  // REQUIRES: buffer_size > 0
  explicit ThreadSafeBuffer(size_t buffer_size);

  // Gets the next element. Blocks if the buffer is empty. Returns an error if
  // a non-OK status was pushed or the buffer has been cancelled.
  StatusOr<T> Pop();

  // Writes the next element. Blocks if the buffer is full. Returns an error if
  // the buffer has been cancelled.
  Status Push(StatusOr<T> value);

  // Cancels the buffer with `status` and notifies waiting threads. After
  // cancelling, all `Push` and `Pop` calls will return `status`.
  // REQUIRES: !status.ok()
  void Cancel(Status status);

 private:
  const size_t buffer_size_;

  mutex mu_;
  condition_variable ready_to_pop_;
  condition_variable ready_to_push_;
  std::deque<StatusOr<T>> results_ TF_GUARDED_BY(mu_);
  Status status_ TF_GUARDED_BY(mu_) = Status::OK();

  TF_DISALLOW_COPY_AND_ASSIGN(ThreadSafeBuffer);
};

template <class T>
ThreadSafeBuffer<T>::ThreadSafeBuffer(size_t buffer_size)
    : buffer_size_(buffer_size) {
  DCHECK_GT(buffer_size, 0)
      << "ThreadSafeBuffer must have a postive buffer size. Got " << buffer_size
      << ".";
}

template <class T>
StatusOr<T> ThreadSafeBuffer<T>::Pop() {
  mutex_lock l(mu_);
  while (status_.ok() && results_.empty()) {
    ready_to_pop_.wait(l);
  }
  if (!status_.ok()) {
    return status_;
  }
  StatusOr<T> result = std::move(results_.front());
  results_.pop_front();
  ready_to_push_.notify_one();
  return result;
}

template <class T>
Status ThreadSafeBuffer<T>::Push(StatusOr<T> value) {
  mutex_lock l(mu_);
  while (status_.ok() && results_.size() >= buffer_size_) {
    ready_to_push_.wait(l);
  }
  if (!status_.ok()) {
    return status_;
  }
  results_.push_back(std::move(value));
  ready_to_pop_.notify_one();
  return Status::OK();
}

template <class T>
void ThreadSafeBuffer<T>::Cancel(Status status) {
  DCHECK(!status.ok())
      << "Cancelling ThreadSafeBuffer requires a non-OK status. Got " << status;
  mutex_lock l(mu_);
  status_ = std::move(status);
  ready_to_push_.notify_all();
  ready_to_pop_.notify_all();
}

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_THREAD_SAFE_BUFFER_H_
