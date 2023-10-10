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

#ifndef TENSORFLOW_CORE_KERNELS_RECORD_YIELDER_H_
#define TENSORFLOW_CORE_KERNELS_RECORD_YIELDER_H_

#include <atomic>
#include <random>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

// RecordYielder produces value records from a set of tfrecord files
// in a random order.
//
// It guarantees that:
//   1) all records in tfrecords are yielded within every epoch;
//   2) each record is yielded only once within every epoch;
//   3) the order in which records are yielded is highly randomized.
//   4) the peak memory usage is roughly avg record size *
//      (opts.bufsize + opts.parallelism * 16).
//
// Usage example:
//   RecordYielder::Options opts;
//   opts.file_pattern = "input-*";
//   opts.seed = 301;
//   opts.bufsize = 1000000;    // A randomized buffer with 1M records.
//   opts.parallelism = 8;      // Uses 8 tfrecord iterators to iterate
//                              // through all files.
//   RecordYielder yielder(opts);
//   string val;
//   while (true) {
//     yielder.YieldOne(&val);
//     // process val
//   }
//
// RecordYielder can be accessed by multiple threads concurrently.
class RecordYielder {
 public:
  struct Options {
    // Glob pattern for tfrecords.
    string file_pattern;

    // Random seed. It determines how data files are shuffled and how
    // records are shuffled.
    int64_t seed = 0;

    // Each epoch, all files are first shuffled according to the
    // random seed and the epoch number, and then all files are
    // left-shifted by file_shuffle_shift_ratio * num_files slots.  If
    // file_shuffle_shift_ratio is not within [0, 1), the
    // implementation clip it to [0, 1).
    float file_shuffle_shift_ratio = 0;

    // Randomization buffer keeps these many records.
    uint64 bufsize = 1;

    // Uses these many concurrent tfrecord iterators to iterate through
    // tfrecords.
    int32 parallelism = 1;

    string compression_type;
  };

  explicit RecordYielder(OpKernelConstruction* context,
                         const RecordYielder::Options& opts);
  ~RecordYielder();

  RecordYielder(const RecordYielder&) = delete;
  RecordYielder& operator=(const RecordYielder&) = delete;

  // Yields one 'value'.
  Status YieldOne(tstring* value);

  // Returns the current epoch number.
  int64_t current_epoch() const { return epoch_; }

 private:
  typedef RecordYielder ME;

  Options opts_;

  // Backgrounds threads. Owned.
  thread::ThreadPool* thread_;

  // Epoch number.
  std::atomic<int64_t> epoch_;

  mutex mu_;

  // Turned to true when this is deleted.
  bool stop_ TF_GUARDED_BY(mu_) = false;
  Status status_ TF_GUARDED_BY(mu_);

  // PRG used for randomization.
  std::mt19937_64 rnd_ TF_GUARDED_BY(mu_);

  // Randomization buffer.
  std::vector<string> buf_ TF_GUARDED_BY(mu_);

  // True iff we are draining an epoch.
  bool epoch_end_ = false;

  int64_t num_records_added_in_epoch_ = 0;
  int64_t num_records_yielded_in_epoch_ = 0;

  // Trigger when the main loop has exited.
  Notification main_loop_done_;

  // condition_variables.
  condition_variable buf_empty_;
  bool BufEmpty() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    return stop_ || buf_.empty();
  }

  condition_variable buf_not_full_;
  bool BufNotFull() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    return stop_ || buf_.size() < opts_.bufsize;
  }

  condition_variable buf_enough_;
  bool BufEnough() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    // NOTE: Unless we are finishing an epoch, we want to make sure
    // the buf_ contains enough randomized elements before yielding
    // any.
    return stop_ || !status_.ok() || (epoch_end_ && !buf_.empty()) ||
           (!epoch_end_ &&
            buf_.size() >= std::max<uint64>(1, opts_.bufsize / 2));
  }

  void MainLoop();
  struct Shard;
  void ShardLoop(Shard* shard);
  bool ShouldFinish(const Status& s);
  bool Add(std::vector<string>* values);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RECORD_YIELDER_H_
