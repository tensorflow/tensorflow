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

#include "tensorflow/core/kernels/record_yielder.h"

#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

RecordYielder::RecordYielder(OpKernelConstruction* context,
                             const RecordYielder::Options& opts)
    : opts_(opts),
      thread_(new thread::ThreadPool(context->env(), ThreadOptions(),
                                     "record_yielder", 1 + opts.parallelism,
                                     /* low_latency_hint */ false)),
      epoch_(0),
      rnd_(opts.seed) {
  thread_->Schedule([this]() { MainLoop(); });
}

RecordYielder::~RecordYielder() {
  {
    mutex_lock l(mu_);
    stop_ = true;
    buf_empty_.notify_all();
    buf_enough_.notify_all();
    buf_not_full_.notify_all();
  }
  main_loop_done_.WaitForNotification();
  delete thread_;
}

Status RecordYielder::YieldOne(string* value) {
  mutex_lock l(mu_);
  while (!BufEnough()) {
    buf_enough_.wait(l);
  }
  if (status_.ok()) {
    bool notify_no_longer_full = !BufNotFull();
    CHECK(!stop_ && !buf_.empty());
    *value = std::move(buf_.back());
    buf_.pop_back();
    ++num_records_yielded_in_epoch_;
    // Assumption is that an epoch always has something in the buffer
    // until it ends.  If the input pipeline was slower than the consumers
    // by a lot this might not be true.  Not sure how to handle.
    if (buf_.empty()) {
      buf_empty_.notify_all();
    }
    if (notify_no_longer_full) {
      buf_not_full_.notify_all();
    }
  }
  return status_;
}

struct RecordYielder::Shard {
  int index;                      // Shard index.
  std::vector<string> filenames;  // File names given to this shard.
  Notification done;              // Notified when this shard is done.
  Status status;                  // Shard status.
};

bool RecordYielder::ShouldFinish(const Status& s) {
  mutex_lock l(mu_);
  status_.Update(s);
  return stop_ || !status_.ok();
}

static Status MatchFiles(const string& patterns,
                         std::vector<string>* filenames) {
  for (const auto& file_pattern : str_util::Split(patterns, ',')) {
    std::vector<string> tmp_filenames;
    TF_RETURN_IF_ERROR(
        Env::Default()->GetMatchingPaths(file_pattern, &tmp_filenames));
    filenames->insert(filenames->end(),
                      std::make_move_iterator(tmp_filenames.begin()),
                      std::make_move_iterator(tmp_filenames.end()));
  }
  return Status::OK();
}

void RecordYielder::MainLoop() {
  while (true) {
    ++epoch_;
    num_records_yielded_in_epoch_ = 0;

    // Finds all files.
    std::vector<string> filenames;
    Status s = MatchFiles(opts_.file_pattern, &filenames);
    if (ShouldFinish(s)) break;

    if (filenames.empty()) {
      s = errors::NotFound("Found no files at ", opts_.file_pattern);
      if (ShouldFinish(s)) break;
    }

    // Shuffles these files according to the epoch # and random seed.
    std::mt19937_64 shuffle_rnd(
        Hash64(reinterpret_cast<char*>(&epoch_), sizeof(epoch_), opts_.seed));
    std::shuffle(filenames.begin(), filenames.end(), shuffle_rnd);

    // Left-shift the filename list.
    const std::vector<string>::size_type num = filenames.size();
    int64 shift;
    if (0 <= opts_.file_shuffle_shift_ratio &&
        opts_.file_shuffle_shift_ratio < 1) {
      shift = opts_.file_shuffle_shift_ratio * num;
      std::rotate(filenames.begin(), filenames.begin() + shift,
                  filenames.end());
    }

    // Shards files and use one thread to go through each shard.
    const int N = opts_.parallelism;
    std::vector<Shard> shards(N);
    for (int i = 0; i < N; ++i) {
      Shard* shard = &shards[i];
      shard->index = i;
      for (std::vector<string>::size_type j = i; j < filenames.size(); j += N) {
        shard->filenames.push_back(filenames[j]);
      }
      thread_->Schedule([this, shard]() { ShardLoop(shard); });
    }
    for (int i = 0; i < N; ++i) {
      shards[i].done.WaitForNotification();
      s.Update(shards[i].status);
    }
    if (ShouldFinish(s)) break;

    // Starts the next epoch once all buffered records are consumed.
    {
      mutex_lock l(mu_);
      epoch_end_ = true;
      while (!BufEmpty()) {
        buf_empty_.wait(l);
      }
      epoch_end_ = false;
    }
  }
  main_loop_done_.Notify();
}

bool RecordYielder::Add(std::vector<string>* values) {
  mutex_lock l(mu_);
  while (!BufNotFull()) {
    buf_not_full_.wait(l);
  }
  while (BufNotFull() && !values->empty()) {
    // Adds values->back(). Swaps its position with another random
    // element.
    auto index = rnd_() % (buf_.size() + 1);
    if (index == buf_.size()) {
      buf_.push_back(std::move(values->back()));
    } else {
      buf_.push_back(std::move(buf_[index]));
      buf_[index] = std::move(values->back());
    }
    values->pop_back();
  }
  if (BufEnough()) {
    buf_enough_.notify_all();
  }
  return stop_;
}

void RecordYielder::ShardLoop(Shard* shard) {
  std::vector<string> values;
  const int64 kRecords = 16;
  for (const string& filename : shard->filenames) {
    std::unique_ptr<RandomAccessFile> file;
    if (ShouldFinish(Status::OK())) break;
    Status s = Env::Default()->NewRandomAccessFile(filename, &file);
    if (!s.ok()) {
      shard->status = errors::InvalidArgument("Can't open ", filename);
      break;
    }
    io::RecordReader rdr(file.get());
    uint64 offset = 0;
    string record;
    while (true) {
      Status s = rdr.ReadRecord(&offset, &record);
      if (s.ok()) {
        values.emplace_back(std::move(record));
        if (values.size() >= kRecords && Add(&values)) {
          shard->status = errors::Aborted("stopped");
          break;
        }
      } else if (errors::IsOutOfRange(s)) {
        break;
      } else {
        shard->status = s;
        break;
      }
    }
  }
  // Adds the remaining values of this shard to buf_.
  while (!values.empty()) {
    Add(&values);
  }
  shard->done.Notify();
}

}  // namespace tensorflow
