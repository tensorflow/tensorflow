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
#ifndef TENSORFLOW_CORE_FRAMEWORK_MODEL_H_
#define TENSORFLOW_CORE_FRAMEWORK_MODEL_H_

#include <list>
#include <memory>
#include <string>
#include <thread>  // (b/114492873): move this include into core/platform
#include <utility>
#include <vector>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace data {
namespace model {

class Model;
class Node;

// Abstract representation of a TensorFlow input pipeline node. It collects
// information about inputs to this node, processing time spent executing the
// node logic, number of elements produced by the node, various other
// information (e.g. batch size or execution parallelism).
//
// Developers of tf.data transformations are not expected to interact with this
// class directly. Boiler plate code for creating the abstract representation of
// the input pipeline and collecting common information has been added to the
// implementation of `DatasetBase` and `DatasetBaseIterator` respectively.
//
// In addition, `DatasetBaseIterator` provides wrappers that can be used for
// transformation-specific information collection. The `SetMetadata` wrapper can
// be used to pass arbitrary metadata to the modeling framework, while the
// `StartWork` and `StopWork` wrappers should be used to correctly account for
// processing time of multi-threaded transformation that yield the CPU; such
// transformations should invoke `StartWork()` when a transformation thread
// starts executing (e.g. when created or woken up) and `StopWork()` when a
// transformation thread stops executing (e.g. when returning or waiting).
//
// TODO(jsimsa): Create an API to capture the abstract semantics of each
// tf.data transformation and replace switch-case blocks with inheritance.
class Node {
 public:
  Node(int64 id, std::shared_ptr<Node> output) : id_(id), output_(output) {}

  // Adds a constant parameter.
  void add_constant_param(const string& name, int64 value) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    constant_params_[name] = value;
  }

  // Records that the node produced an element.
  void add_element() LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    num_elements_++;
  }

  // Adds an input.
  void add_input(std::shared_ptr<Node> node) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    inputs_.push_back(node);
  }

  // Increments the aggregate processing time by the given delta.
  void add_processing_time(int64 delta) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    processing_time_ += delta;
  }

  // Adds a tunable parameter.
  void add_tunable_param(const string& name, int64 value, int64 min, int64 max,
                         std::function<void(int64)>&& set_fn)
      LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    tunable_params_[name] =
        std::make_shared<Tunable>(value, min, max, std::move(set_fn));
  }

  // Returns the unique node ID.
  int64 id() LOCKS_EXCLUDED(mu_) { return id_; }

  // Returns the node inputs.
  std::list<std::shared_ptr<Node>> inputs() LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return inputs_;
  }

  // Returns the node name.
  const string& name() LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return name_;
  }

  // Returns the number of elements produced by the node.
  int64 num_elements() LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return num_elements_;
  }

  // Returns the node output.
  std::shared_ptr<Node> output() LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return output_;
  }

  // Removes an input.
  void remove_input(std::shared_ptr<Node> input) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    inputs_.remove(input);
  }

  // Sets the node name.
  void set_name(const string& name) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    name_ = name;
    type_ = TypeFromName(name);
  }

  // Set the node output.
  void set_output(std::shared_ptr<Node> output) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    output_ = output;
  }

  // Records that a node thread has started work.
  void start_work() LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    work_start_[std::this_thread::get_id()] = Env::Default()->NowNanos();
  }

  // Records that a node thread has stopped work.
  void stop_work() LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    auto iter = work_start_.find(std::this_thread::get_id());
    CHECK(work_start_.end() != iter)
        << "Encountered a stop event that was not preceded by a start event.";
    processing_time_ += Env::Default()->NowNanos() - iter->second;
    work_start_.erase(iter);
  }

 private:
  // Represents a tunable parameter.
  struct Tunable {
    Tunable(int64 value, int64 min, int64 max,
            std::function<void(int64)> set_fn)
        : value(value), min(min), max(max), set_fn(std::move(set_fn)) {}

    int64 value;
    int64 min;
    int64 max;
    std::function<void(int64)> set_fn;
  };

  enum class Type {
    BATCH = 0,
    CACHE,
    CONCATENATE,
    FILTER,
    FLAT_MAP,
    INTERLEAVE,
    MAP,
    MAP_AND_BATCH,
    PADDED_BATCH,
    PARALLEL_INTERLEAVE,
    PARALLEL_INTERLEAVE_V2,
    PARALLEL_MAP,
    PREFETCH,
    REPEAT,
    SHUFFLE,
    SKIP,
    TAKE,
    ZIP,
    UNKNOWN,
  };

  // Collects tunable parameters in the subtree rooted in this node.
  void CollectTunables(std::vector<std::shared_ptr<Node::Tunable>>* tunables)
      LOCKS_EXCLUDED(mu_);

  // Gets a value of the given parameter (tunable or constant).
  int64 GetParameterValue(const string& name) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns the per-element processing time spent in this node.
  int64 NanosPerElement() LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    return NanosPerElementLocked();
  }

  int64 NanosPerElementLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (num_elements_ == 0) {
      return 0;
    }
    return (int64)((double)processing_time_ / (double)num_elements_);
  }

  // Returns the per-element output time for this node.
  int64 OutputTime(std::vector<int64>* input_times) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    return OutputTimeLocked(input_times);
  }

  int64 OutputTimeLocked(std::vector<int64>* input_times)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  int64 OutputTimeForInputs(std::vector<int64>* input_times)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    int64 sum = 0;
    for (auto input : inputs_) {
      sum += input->OutputTime(input_times);
    }
    return sum;
  }

  // Returns the per-element processing time spent in the subtree rooted in this
  // node.
  int64 ProcessingTime() LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    return ProcessingTimeLocked();
  }

  int64 ProcessingTimeLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns the per-element processing time spent in the inputs of this node.
  int64 ProcessingTimeForInputs() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    int64 sum = 0;
    for (auto input : inputs_) {
      sum += input->ProcessingTimeLocked();
    }
    return sum;
  }

  Type TypeFromName(const string& name) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (name_ == "Batch") {
      return Type::BATCH;
    }
    if (str_util::EndsWith(name_, "Cache")) {
      return Type::CACHE;
    }
    if (name_ == "Concatenate") {
      return Type::CONCATENATE;
    }
    if (name_ == "Filter") {
      return Type::FILTER;
    }
    if (name_ == "FlatMap") {
      return Type::FLAT_MAP;
    }
    if (name_ == "Interleave") {
      return Type::INTERLEAVE;
    }
    if (name_ == "Map") {
      return Type::MAP;
    }
    if (name_ == "MapAndBatch") {
      return Type::MAP_AND_BATCH;
    }
    if (name_ == "PaddedBatch") {
      return Type::PADDED_BATCH;
    }
    if (name_ == "ParallelInterleave") {
      return Type::PARALLEL_INTERLEAVE;
    }
    if (name_ == "ParallelInterleaveV2") {
      return Type::PARALLEL_INTERLEAVE_V2;
    }
    if (name_ == "ParallelMap") {
      return Type::PARALLEL_MAP;
    }
    if (name_ == "Prefetch") {
      return Type::PREFETCH;
    }
    if (str_util::EndsWith(name_, "Repeat")) {
      return Type::REPEAT;
    }
    if (name_ == "Shuffle") {
      return Type::SHUFFLE;
    }
    if (str_util::EndsWith(name_, "Skip")) {
      return Type::SKIP;
    }
    if (str_util::EndsWith(name_, "Take")) {
      return Type::TAKE;
    }
    if (name_ == "Zip") {
      return Type::ZIP;
    }
    return Type::UNKNOWN;
  }

  mutex mu_;
  const int64 id_;
  Type type_ GUARDED_BY(mu_);
  string name_ GUARDED_BY(mu_);
  int64 processing_time_ GUARDED_BY(mu_) = 0;
  int64 num_elements_ GUARDED_BY(mu_) = 0;
  std::map<std::thread::id, int64> work_start_ GUARDED_BY(mu_);
  std::map<string, int64> constant_params_ GUARDED_BY(mu_);
  // Tunables are shared with the model during optimization.
  std::map<string, std::shared_ptr<Tunable>> tunable_params_ GUARDED_BY(mu_);
  std::list<std::shared_ptr<Node>> inputs_ GUARDED_BY(mu_);
  std::shared_ptr<Node> output_ GUARDED_BY(mu_);

  friend class Model;
};

// Abstract representation of a TensorFlow input pipeline that can be used
// for collecting runtime information and optimizing performance. It collects
// runtime information about execution of the input pipeline that is used to
// create a performance model, which is in turn used to identify optimal values
// of tunable parameters.
//
// Developers of tf.data transformations are not expected to interact with this
// class directly. Boiler plate code for creating the abstract representation of
// the input pipeline and collecting runtime information has been added to the
// implementation of `DatasetBase` and `DatasetBaseIterator` respectively.
class Model {
 public:
  Model() = default;

  // Returns the model output node.
  std::shared_ptr<Node> output() LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return output_;
  }

  // Adds a node with the given name and given output (identified by name).
  std::shared_ptr<Node> AddNode(const string& name, const string& output_name)
      LOCKS_EXCLUDED(mu_);

  // Looks up the node using the given name.
  std::shared_ptr<Node> LookupNode(const string& name) LOCKS_EXCLUDED(mu_);

  // Runs optimization.
  void Optimize(int64 cpu_budget) LOCKS_EXCLUDED(mu_);

  // Removes the node identified by the given name.
  void RemoveNode(const string& prefix) LOCKS_EXCLUDED(mu_);

 private:
  std::vector<std::shared_ptr<Node::Tunable>> CollectTunables()
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  int64 OutputTime() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  int64 ProcessingTime() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Used for coordination between different input pipeline threads.
  mutex mu_;
  // Used for preventing iterator deletion when optimization is in progress
  // because the optimization may try to update the values of tunable
  // parameters.
  mutex optimization_mu_ ACQUIRED_BEFORE(mu_);
  int64 id_counter_ GUARDED_BY(mu_) = 1;
  std::shared_ptr<Node> output_ GUARDED_BY(mu_);
  std::map<string, std::shared_ptr<Node>> lookup_table_ GUARDED_BY(mu_);
};

}  // namespace model
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_MODEL_H_
