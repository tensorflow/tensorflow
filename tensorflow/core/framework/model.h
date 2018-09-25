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
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace data {
namespace model {

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

  // Adds a constant parameter for the given node.
  void AddConstantParameter(const string& node_name,
                            const string& parameter_name, int64 value)
      LOCKS_EXCLUDED(mu_);

  // Adds a node with the given name and given output (identified by name).
  void AddNode(const string& name, const string& output_name)
      LOCKS_EXCLUDED(mu_);

  // Increments the processing time for the given node..
  void AddProcessingTime(const string& name, int64 delta) LOCKS_EXCLUDED(mu_);

  // Adds a tunable parameter for the given node.
  void AddTunableParameter(const string& node_name,
                           const string& parameter_name,
                           std::atomic<int64>* value, int64 min, int64 max,
                           condition_variable* cond_var) LOCKS_EXCLUDED(mu_);

  // Runs optimization.
  void Optimize(int64 cpu_budget) LOCKS_EXCLUDED(mu_);

  // Records that a node has produced an element.
  void RecordElement(const string& name) LOCKS_EXCLUDED(mu_);

  // Records that the given node has started work. If `stop_output` is set, it
  // also records that the output of the given node has stopped work.
  void RecordStart(const string& name, bool stop_output) LOCKS_EXCLUDED(mu_);

  // Records that the given node has stopped work. If `stop_output` is set, it
  // also records that the output of the given node has started work.
  void RecordStop(const string& name, bool start_output) LOCKS_EXCLUDED(mu_);

  // Removes the given node.
  void RemoveNode(const string& name) LOCKS_EXCLUDED(mu_);

 private:
  // Abstract representation of a TensorFlow input pipeline node. It collects
  // information about inputs to this node, processing time spent executing the
  // node logic, number of elements produced by the node, various other
  // information (e.g. batch size or execution parallelism).
  //
  // Developers of tf.data transformations are not expected to interact with
  // this class directly. Boiler plate code for creating the abstract
  // representation of the input pipeline and collecting common information has
  // been added to the implementation of `DatasetBase` and `DatasetBaseIterator`
  // respectively.
  //
  // In addition, `DatasetBaseIterator` provides wrappers that can be used for
  // transformation-specific information collection. The `SetMetadata` wrapper
  // can be used to pass arbitrary metadata to the modeling framework, while the
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
    // Represents a tunable parameter.
    struct Tunable {
      Tunable(std::atomic<int64>* value, int64 min, int64 max,
              condition_variable* cond_var)
          : value(*value),
            min(min),
            max(max),
            value_ptr(value),
            cond_var(cond_var) {}

      // Identifies the model value of the parameter. This can be different from
      // the actual value (e.g. during optimization search).
      int64 value;

      // Identifies the minimum value of the parameter.
      int64 min;

      // Identifies the maximum value of the parameter.
      int64 max;

      // Points to the actual value of the parameter. Not owned.
      std::atomic<int64>* value_ptr;

      // If non-null, this condition variable is notified when the model updates
      // the actual value of the parameter (via `value_ptr`). Not owned.
      condition_variable* cond_var;
    };

    Node(int64 id, const string& name, std::shared_ptr<Node> output)
        : id_(id), name_(name), type_(TypeFromName(name)), output_(output) {}

    // Adds a constant parameter.
    void add_constant_param(const string& name, int64 value)
        LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      constant_params_[name] = value;
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
    void add_tunable_param(const string& name, std::atomic<int64>* value,
                           int64 min, int64 max, condition_variable* cond_var)
        LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      tunable_params_[name] =
          std::make_shared<Tunable>(value, min, max, cond_var);
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

    // Records that the node produced an element.
    void record_element() LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      num_elements_++;
    }

    // Records that a node thread has started executing.
    void record_start() LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      work_start_[std::this_thread::get_id()] = Env::Default()->NowNanos();
    }

    // Records that a node thread has stopped executing.
    void record_stop() LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      std::thread::id tid = std::this_thread::get_id();
      auto start_time = gtl::FindOrNull(work_start_, tid);
      DCHECK(start_time)
          << "Encountered a stop event that was not preceded by a start event.";
      if (start_time) {
        processing_time_ += Env::Default()->NowNanos() - *start_time;
        work_start_.erase(tid);
      }
    }

    // Removes an input.
    void remove_input(std::shared_ptr<Node> input) LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      inputs_.remove(input);
    }

    // Set the node output.
    void set_output(std::shared_ptr<Node> output) LOCKS_EXCLUDED(mu_) {
      mutex_lock l(mu_);
      output_ = output;
    }

    // Collects tunable parameters in the subtree rooted in this node.
    void CollectTunables(std::vector<std::shared_ptr<Tunable>>* tunables)
        LOCKS_EXCLUDED(mu_);

    // Returns the per-element output time for this node.
    int64 OutputTime(std::vector<int64>* input_times) LOCKS_EXCLUDED(mu_) {
      tf_shared_lock l(mu_);
      return OutputTimeLocked(input_times);
    }

    // Returns the per-element processing time spent in the subtree rooted in
    // this node.
    int64 ProcessingTime() LOCKS_EXCLUDED(mu_) {
      tf_shared_lock l(mu_);
      return ProcessingTimeLocked();
    }

   private:
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

    // Gets a value of the given parameter (tunable or constant).
    int64 GetParameterValue(const string& name) SHARED_LOCKS_REQUIRED(mu_);

    // Returns the per-element processing time spent in this node.
    int64 NanosPerElement() LOCKS_EXCLUDED(mu_) {
      tf_shared_lock l(mu_);
      return NanosPerElementLocked();
    }

    int64 NanosPerElementLocked() SHARED_LOCKS_REQUIRED(mu_) {
      if (num_elements_ == 0) {
        return 0;
      }
      return (int64)((double)processing_time_ / (double)num_elements_);
    }

    int64 OutputTimeLocked(std::vector<int64>* input_times)
        SHARED_LOCKS_REQUIRED(mu_);

    int64 OutputTimeForInputs(std::vector<int64>* input_times)
        SHARED_LOCKS_REQUIRED(mu_) {
      int64 sum = 0;
      for (auto input : inputs_) {
        sum += input->OutputTime(input_times);
      }
      return sum;
    }

    int64 ProcessingTimeLocked() SHARED_LOCKS_REQUIRED(mu_);

    // Returns the per-element processing time spent in the inputs of this node.
    int64 ProcessingTimeForInputs() SHARED_LOCKS_REQUIRED(mu_) {
      int64 sum = 0;
      for (auto input : inputs_) {
        sum += input->ProcessingTime();
      }
      return sum;
    }

    Type TypeFromName(const string& name) SHARED_LOCKS_REQUIRED(mu_) {
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
    const string name_;
    const Type type_;
    int64 processing_time_ GUARDED_BY(mu_) = 0;
    int64 num_elements_ GUARDED_BY(mu_) = 0;
    std::map<std::thread::id, int64> work_start_ GUARDED_BY(mu_);
    std::map<string, int64> constant_params_ GUARDED_BY(mu_);
    // Tunables are shared with the model during optimization.
    std::map<string, std::shared_ptr<Tunable>> tunable_params_ GUARDED_BY(mu_);
    std::list<std::shared_ptr<Node>> inputs_ GUARDED_BY(mu_);
    std::shared_ptr<Node> output_ GUARDED_BY(mu_);
  };

  std::vector<std::shared_ptr<Node::Tunable>> CollectTunables()
      SHARED_LOCKS_REQUIRED(mu_);

  int64 OutputTime() SHARED_LOCKS_REQUIRED(mu_);

  int64 ProcessingTime() SHARED_LOCKS_REQUIRED(mu_);

  // Used for coordination between different input pipeline threads. Exclusive
  // access is required only when adding or removing nodes. Concurrent access to
  // existing nodes is protected by a node mutex.
  mutex mu_;
  int64 id_counter_ GUARDED_BY(mu_) = 1;
  std::shared_ptr<Node> output_ GUARDED_BY(mu_);
  std::map<string, std::shared_ptr<Node>> lookup_table_ GUARDED_BY(mu_);
};

}  // namespace model
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_MODEL_H_
