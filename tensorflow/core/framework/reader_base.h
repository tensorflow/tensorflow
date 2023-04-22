/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_READER_BASE_H_
#define TENSORFLOW_CORE_FRAMEWORK_READER_BASE_H_

#include <memory>
#include <string>
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/reader_interface.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

class ReaderBaseState;

// Default implementation of ReaderInterface.
class ReaderBase : public ReaderInterface {
 public:
  // name: For use in error messages, should mention both the name of
  // the op and the node.
  explicit ReaderBase(const string& name);

  // Note that methods with names ending in "Locked" are called while
  // the ReaderBase's mutex is held.

  // Implement this function in descendants -----------------------------------

  // Produce the next key/value pair from the current work item.
  // This is called "Locked" since it is executed under a mutex
  // that serializes all Reader calls.
  // Usage:
  //  a) If a record was successfully produced, set *produced = true,
  //  and fill in *key and *value.
  //  b) If no more records will be produced for this work item, set
  //  *at_end = true.
  //  c) If a record was produced, but no more will be produced, you
  //     may either do both (a) and (b), or do (a) in this call and do (b) in
  //     the next call to ReadLocked().
  //  d) If there was an error producing (e.g. an error reading the file,
  //     data corruption), return a non-OK() status.  ReadLocked may be
  //     called again if the user reruns this part of the graph.
  virtual Status ReadLocked(tstring* key, tstring* value, bool* produced,
                            bool* at_end) = 0;

  // Descendants may optionally implement these -------------------------------

  // Produce up to num_records next key/value pairs from the current
  // work item, in the same manner of ReadLocked.
  virtual Status ReadUpToLocked(int64 num_records, std::vector<tstring>* keys,
                                std::vector<tstring>* values, int64* num_read,
                                bool* at_end);

  // Called when work starts / finishes.
  virtual Status OnWorkStartedLocked() { return Status::OK(); }
  virtual Status OnWorkFinishedLocked() { return Status::OK(); }

  // Called to reset the Reader to a newly constructed state.
  virtual Status ResetLocked();

  // Default implementation generates an Unimplemented error.
  // See the protected helper methods below.
  virtual Status SerializeStateLocked(tstring* state);
  virtual Status RestoreStateLocked(const tstring& state);

  // Accessors ----------------------------------------------------------------

  // Always true during a call to ReadLocked().
  bool work_in_progress() const { return work_finished_ < work_started_; }

  // Returns the name of the current work item (valid if
  // work_in_progress() returns true).  May change between calls to
  // ReadLocked().
  const tstring& current_work() const { return work_; }

  // What was passed to the constructor.
  const string& name() const { return name_; }

  // Produce the key name (from current_work and the actual key).
  tstring KeyName(const tstring& key) const;

 protected:
  // For descendants wishing to implement serialize & restore state.

  // Writes ReaderBase state to *state.
  void SaveBaseState(ReaderBaseState* state) const;

  // Restores ReaderBase state from state. Assumes state was filled
  // using SaveBaseState() above.
  Status RestoreBaseState(const ReaderBaseState& state);

 private:
  // For descendants that wish to obtain the next work item in a different way.
  // For implementing Read().  Dequeues the next work item from
  // *queue, and if successful returns "work" (a string). May block.
  virtual string GetNextWorkLocked(QueueInterface* queue,
                                   OpKernelContext* context) const;

  // Implementations of ReaderInterface methods.  These ensure thread-safety
  // and call the methods above to do the work.
  void Read(QueueInterface* queue, tstring* key, tstring* value,
            OpKernelContext* context) override;

  // Produces up to num_records.
  // In this implementation all the records come from the same work unit.
  int64 ReadUpTo(const int64 num_records, QueueInterface* queue,
                 std::vector<tstring>* keys, std::vector<tstring>* value,
                 OpKernelContext* context) override;

  Status Reset() override;
  int64 NumRecordsProduced() override;
  int64 NumWorkUnitsCompleted() override;
  Status SerializeState(tstring* state) override;
  Status RestoreState(const tstring& state) override;

  mutable mutex mu_;
  const string name_;
  int64 work_started_ = 0;
  int64 work_finished_ = 0;
  int64 num_records_produced_ = 0;
  tstring work_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_READER_BASE_H_
