/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_JIT_PRODUCER_CONSUMER_QUEUE_H_
#define TENSORFLOW_COMPILER_JIT_PRODUCER_CONSUMER_QUEUE_H_

#include <deque>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

// A thread-safe, first-in-first-out queue.
template <typename T>
class ProducerConsumerQueue {
 public:
  ProducerConsumerQueue()
      : capacity_(std::numeric_limits<std::size_t>::max()) {}
  ~ProducerConsumerQueue() = default;

  // Wait until the queue is non-full, then append a copy of v.
  void Put(const T &v);

  // Wait until the queue is non-empty, then remove and return the head value.
  T Get();

  // If the queue is non-empty, remove the head value, placing it in *pv, and
  // return true; otherwise return false.
  bool TryGet(T *pv);

  // Set the capacity of the queue; the queue is full whenever count() >=
  // capacity().  The initial value is the maximum size_t.  Requires size > 0.
  void set_capacity(std::size_t size);

  // Return the capacity of the queue.
  std::size_t capacity() const;

  // Return the number of elements in the queue.
  std::size_t count() const;

  // Implementation details follow.  Clients should ignore.
 private:
  mutable tensorflow::mutex mu_;  // protects all fields below
  tensorflow::condition_variable non_empty_ GUARDED_BY(mu_);
  tensorflow::condition_variable non_full_ GUARDED_BY(mu_);
  std::size_t capacity_ GUARDED_BY(mu_);
  std::deque<T> queue_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(ProducerConsumerQueue);
};

// ------------------------------------------------------
// Implementation details follow.  Clients should ignore.

// Wait until the queue is non-full, then append a copy of v.
template <typename T>
void ProducerConsumerQueue<T>::Put(const T &v) {
  mutex_lock lock(mu_);
  while (queue_.size() >= capacity_) {
    non_full_.wait(lock);
  }
  queue_.push_back(v);
  non_empty_.notify_one();
}

// Wait until the queue is non-empty, then remove and return the head value.
template <typename T>
T ProducerConsumerQueue<T>::Get() {
  mutex_lock lock(mu_);
  while (queue_.empty()) {
    non_empty_.wait(lock);
  }
  non_full_.notify_one();
  T result_value = queue_.front();
  queue_.pop_front();
  return result_value;
}

// If the queue is non-empty, remove the head value, placing it in *pv, and
// return true; otherwise return false.
template <typename T>
bool ProducerConsumerQueue<T>::TryGet(T *pv) {
  mutex_lock lock(mu_);
  bool got_element = !queue_.empty();
  if (got_element) {
    non_full_.notify_one();
    *pv = queue_.front();
    queue_.pop_front();
  }
  return got_element;
}

// Set the capacity of the queue; the queue is full whenever count() >=
// capacity().  The initial value is the maximum size_t.  Requires size > 0.
template <typename T>
void ProducerConsumerQueue<T>::set_capacity(std::size_t size) {
  mutex_lock lock(mu_);
  CHECK_NE(size, 0);
  capacity_ = size;
  non_full_.notify_all();
}

// Return the capacity of the queue.
template <typename T>
std::size_t ProducerConsumerQueue<T>::capacity() const {
  mutex_lock lock(mu_);
  std::size_t max_elements = capacity_;
  return max_elements;
}

// Return the number of elements in the queue.
template <typename T>
std::size_t ProducerConsumerQueue<T>::count() const {
  mutex_lock lock(mu_);
  std::size_t num_elements = queue_.size();
  return num_elements;
}
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_PRODUCER_CONSUMER_QUEUE_H_
