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
// TODO(b/114492873): Move this include into core/platform.
#include <thread>  // NOLINT
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

// Represents thread-safe state that can be shared between an input pipeline and
// the performance model.
struct SharedState {
 public:
  SharedState(int64 value, std::shared_ptr<mutex> mu,
              std::shared_ptr<condition_variable> cond_var)
      : value(value), mu(std::move(mu)), cond_var(std::move(cond_var)) {}

  int64 value;
  std::shared_ptr<mutex> mu;
  std::shared_ptr<condition_variable> cond_var;
  bool tunable = false;
};

// Represents a parameter.
struct Parameter {
  Parameter(const string& name, std::shared_ptr<SharedState> state, int64 min,
            int64 max)
      : name(name),
        value(state->value),
        min(min),
        max(max),
        state(std::move(state)) {}

  // Human-readable name of the parameter.
  string name;

  // Identifies the model value of the parameter. This can be different from
  // the actual value (e.g. during optimization search).
  int64 value;

  // Identifies the minimum value of the parameter.
  int64 min;

  // Identifies the maximum value of the parameter.
  int64 max;

  // Shared state of the parameter.
  std::shared_ptr<SharedState> state;
};

std::shared_ptr<Parameter> MakeParameter(const string& name,
                                         std::shared_ptr<SharedState> state,
                                         int64 min, int64 max);

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
class Node {
 public:
  // Arguments for `Node` constructor.
  struct Args {
    int64 id;
    string name;
    std::shared_ptr<Node> output;
  };

  using Factory = std::function<std::shared_ptr<Node>(Args)>;

  explicit Node(Args args)
      : id_(args.id), name_(args.name), output_(args.output.get()) {}

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

  // Returns the unique node ID.
  int64 id() const LOCKS_EXCLUDED(mu_) { return id_; }

  // Returns the node name.
  const string& name() const { return name_; }

  // Returns the node inputs.
  std::list<std::shared_ptr<Node>> inputs() const LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return inputs_;
  }

  // Returns the number of elements produced by the node.
  int64 num_elements() const LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return num_elements_;
  }

  // Returns the node output.
  Node* output() const { return output_; }

  // Returns the aggregate processing time.
  int64 processing_time() const LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return processing_time_;
  }

  // Records that the node produced an element.
  void record_element() LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    num_elements_++;
  }

  // Records that a node thread has started executing.
  void record_start(int64 time_nanos) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    work_start_[std::this_thread::get_id()] = time_nanos;
  }

  // Records that a node thread has stopped executing.
  void record_stop(int64 time_nanos) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    std::thread::id tid = std::this_thread::get_id();
    auto iter = work_start_.find(tid);
    if (iter != work_start_.end()) {
      processing_time_ += time_nanos - iter->second;
      work_start_.erase(iter);
    } else {
      LOG(WARNING)
          << "Encountered a stop event that was not preceded by a start event.";
    }
  }

  // Removes an input.
  void remove_input(std::shared_ptr<Node> input) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    inputs_.remove(input);
  }

  // Collects tunable parameters in the subtree rooted in this node.
  void CollectTunableParameters(
      std::vector<std::shared_ptr<Parameter>>* parameters) LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    for (auto& pair : parameters_) {
      if (pair.second->state->tunable) {
        parameters->push_back(pair.second);
      }
    }
    for (auto& input : inputs_) {
      input->CollectTunableParameters(parameters);
    }
  }

  // Returns the per-element output time for this node.
  int64 OutputTime(std::vector<int64>* input_times) const LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return OutputTimeLocked(input_times);
  }

  // Returns the per-element processing time spent in the subtree rooted in
  // this node.
  int64 ProcessingTime() const LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return ProcessingTimeLocked();
  }

  // Returns a copy of this node, making a deep copy of its inputs and a
  // shallow copy of its tunable parameters.
  //
  // The purpose for this method is to allow the model optimization logic to
  // operate over immutable state while allowing concurrent model updates.
  std::shared_ptr<Node> Snapshot(std::shared_ptr<Node> output)
      LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    std::shared_ptr<Node> result = Clone(output);
    result->processing_time_ = processing_time_;
    result->num_elements_ = num_elements_;
    result->parameters_ = parameters_;
    for (auto& input : inputs_) {
      result->add_input(input->Snapshot(result));
    }
    return result;
  }

 protected:
  // Creates a clone of this node.
  virtual std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const
      SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Returns the per-element processing time spent in this node.
  int64 NanosPerElementLocked() const SHARED_LOCKS_REQUIRED(mu_) {
    if (num_elements_ == 0) {
      return 0;
    }
    return static_cast<int64>(static_cast<double>(processing_time_) /
                              static_cast<double>(num_elements_));
  }

  // Returns the sum of per-element output time for the inputs of this node.
  int64 OutputTimeForInputs(std::vector<int64>* input_times) const
      SHARED_LOCKS_REQUIRED(mu_) {
    int64 sum = 0;
    for (auto& input : inputs_) {
      sum += input->OutputTime(input_times);
    }
    return sum;
  }

  // Returns the per-element output time for this node.
  virtual int64 OutputTimeLocked(std::vector<int64>* input_times) const
      SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Returns the sum of per-element processing time for the inputs of this node.
  //
  // TODO(jsimsa): use processing time history as a prior for future inputs
  int64 ProcessingTimeForInputs() const SHARED_LOCKS_REQUIRED(mu_) {
    int64 sum = 0;
    for (auto& input : inputs_) {
      sum += input->ProcessingTime();
    }
    return sum;
  }

  // Returns the per-element processing time spent in the subtree rooted in
  // this node.
  virtual int64 ProcessingTimeLocked() const SHARED_LOCKS_REQUIRED(mu_) = 0;

  mutable mutex mu_;
  const int64 id_;
  const string name_;
  int64 processing_time_ GUARDED_BY(mu_) = 0;
  int64 num_elements_ GUARDED_BY(mu_) = 0;
  std::map<std::thread::id, int64> work_start_ GUARDED_BY(mu_);
  std::map<string, std::shared_ptr<Parameter>> parameters_ GUARDED_BY(mu_);
  std::list<std::shared_ptr<Node>> inputs_ GUARDED_BY(mu_);

  // The reference to the output node is not owned so that that deletion of a
  // node results in recursive deletion of the subtree rooted in the node.
  Node* const output_;
};

// InterleaveMany is used to model datasets whose inputs are used to create
// datasets whose elements are then interleaved.
std::shared_ptr<Node> MakeInterleaveManyNode(Node::Args args);

// AsyncInterleaveMany nodes are the asynchronous version of InterleaveMany
// nodes.
std::shared_ptr<Node> MakeAsyncInterleaveManyNode(
    Node::Args args, std::vector<std::shared_ptr<Parameter>> parameters);

// KnownMany nodes model datasets that synchronously consume known number of
// input element per output element.
std::shared_ptr<Node> MakeKnownRatioNode(Node::Args args, double ratio);

// AsyncKnownRatio nodes are the asynchronous version of KnownRate nodes.
std::shared_ptr<Node> MakeAsyncKnownRatioNode(
    Node::Args args, double ratio,
    std::vector<std::shared_ptr<Parameter>> parameters);

// Source nodes represent data sources.
std::shared_ptr<Node> MakeSourceNode(Node::Args args);

// UnknownMany nodes represent datasets that synchronously consume an
// unknown number of input elements per output.
//
// Unlike KnownRatio nodes which expect the ratio between inputs and outputs is
// specified as a parameter, UnknownRatio estimates the ratio empirically.
std::shared_ptr<Node> MakeUnknownRatioNode(Node::Args args);

// Unknown nodes represent datasets for which we do not have a model. It acts
// as pass-through between inputs and output.
std::shared_ptr<Node> MakeUnknownNode(Node::Args args);

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

  // Adds a node with the given name and given output.
  std::shared_ptr<Node> AddNode(Node::Factory factory, const string& name,
                                const string& output_name) LOCKS_EXCLUDED(mu_);

  // Increments the processing time for the given node..
  void AddProcessingTime(const string& name, int64 delta) LOCKS_EXCLUDED(mu_);

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
  // Collects tunable parameters in the tree rooted in the given node.
  std::vector<std::shared_ptr<Parameter>> CollectTunableParameters(
      std::shared_ptr<Node> node);

  // Collects the output time for the given node.
  int64 OutputTime(std::shared_ptr<Node> node);

  // Collects the processing time for the given node.
  int64 ProcessingTime(std::shared_ptr<Node> node);

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
