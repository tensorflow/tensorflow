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

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <string>
// TODO(b/114492873): Move this include into core/platform.
#include <optional>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/model.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {
namespace model {

// A constant that can be used to enable auto-tuning.
constexpr int64_t kAutotune = -1;
constexpr char kParallelism[] = "parallelism";
constexpr char kBufferSize[] = "buffer_size";
constexpr char kCycleLength[] = "cycle_length";
constexpr char kDeterministic[] = "deterministic";
constexpr char kMaxBufferedElements[] = "max_buffered_elements";

// A key used to identify the input time of the model.
constexpr char kModelInputTimeKey[] = "model_input_time";

// Default share of available RAM that can be used by model's internal buffers.
constexpr double kRamBudgetShare = 0.5;

// Weight of the latest processing time used in computing the exponential moving
// average of processing time per element.
constexpr double kProcessingTimeEmaWeight = 0.1;

enum class TraversalOrder {
  BFS = 0,
  REVERSE_BFS = 1,
};

// Represents thread-safe state that can be shared between an input pipeline and
// the performance model.
struct SharedState {
 public:
  SharedState(int64_t value, std::shared_ptr<mutex> mu,
              std::shared_ptr<condition_variable> cond_var)
      : value(value),
        mu(std::move(mu)),
        cond_var(std::move(cond_var)),
        tunable(value == kAutotune) {}

  double value;
  const std::shared_ptr<mutex> mu;
  const std::shared_ptr<condition_variable> cond_var;
  const bool tunable;
};

// Represents a parameter.
struct Parameter {
  Parameter(const string& name, std::shared_ptr<SharedState> state, double min,
            double max)
      : name(name),
        // Sometimes non-autotune nodes (with `autotune_=false`) may contain
        // parameters (for example inputs of parallel interleave dataset which
        // are not in the current cycle). To avoid unrealistic situation
        // (say `buffer_size=-1` or `parallelism=-1`) in the optimization
        // computation, if the state value is `kAutotune=-1` (just to indicate
        // the `SharedState` is tunable), we initialize the parameter value to
        // be the minimal value of the state.
        value(state == nullptr || state->value == kAutotune ? min
                                                            : state->value),
        min(min),
        max(max),
        state(std::move(state)) {}

  explicit Parameter(const std::shared_ptr<Parameter> parameter)
      : name(parameter->name),
        value(parameter->value),
        min(parameter->min),
        max(parameter->max),
        state(parameter->state) {}

  // Human-readable name of the parameter.
  const string name;

  // Identifies the model value of the parameter. This can be different from
  // the actual value (e.g. during optimization search).
  double value;

  // Identifies the minimum value of the parameter.
  const double min;

  // Identifies the maximum value of the parameter.
  const double max;

  // Shared state of the parameter.
  std::shared_ptr<SharedState> state;
};

// Returns a new tunable parameter.
std::shared_ptr<Parameter> MakeParameter(const string& name,
                                         std::shared_ptr<SharedState> state,
                                         double min, double max);

// Returns a new non-tunable parameter.
std::shared_ptr<Parameter> MakeNonTunableParameter(const string& name,
                                                   double value);

// Class for managing the ram budget of an iterator. This is necessary for
// coordinating ram usage between the model-based autotuner and the legacy
// prefetch autotuner. Once the legacy autotuner is retired we can remove this
// class and move all ram budget management to the model autotuner.
class RamBudgetManager {
 public:
  explicit RamBudgetManager(int64_t budget) : budget_(budget) {
    if (budget <= 0) {
      LOG(WARNING) << "RAM budget is " << budget
                   << " which could prevent autotuner from properly adjusting "
                      "buffer sizes.";
    }
  }

  // Requests a new total memory allocation for the parts of the dataset
  // tuned by the model.
  //
  // The autotuner is expected to follow a pattern like
  //
  // int64_t budget = ram_budget_manager.AvailableModelRam();
  // NewModel potential_new_params = OptimizeModel(budget);
  // int64_t new_ram_used = potential_new_params.RamUsed();
  // if (ram_budget_manager.RequestModelAllocation(new_ram_used)) {
  //   ApplyModel(potential_new_params);
  // }
  //
  // Returns whether the request succeeded.
  bool RequestModelAllocation(int64_t total_bytes) {
    mutex_lock l(mu_);
    if (total_bytes > budget_ - legacy_prefetch_allocated_) {
      return false;
    }
    model_allocated_ = total_bytes;
    return true;
  }

  // Requests `delta_elements` allocated to the model where each element is of
  // size `element_size` bytes. `delta_elements` can be negative.
  // Returns the actual allocated delta elements.
  int64_t RequestModelBytes(int64_t delta_elements, double element_size) {
    if (delta_elements == 0) {
      return 0;
    }
    int64_t allocated_delta_elements = delta_elements;
    mutex_lock l(mu_);
    // If `delta_elements` is positive, allocate only up to the available
    // memory.
    if (delta_elements > 0) {
      int64_t max_delta_elements = static_cast<int64_t>(
          (budget_ - legacy_prefetch_allocated_ - model_allocated_) /
          element_size);
      if (max_delta_elements < 0) {
        return 0;
      }
      allocated_delta_elements = std::min(max_delta_elements, delta_elements);
    }
    model_allocated_ +=
        static_cast<int64_t>(allocated_delta_elements * element_size);
    return allocated_delta_elements;
  }

  // Requests `bytes` additional bytes for the purpose of legacy prefetch
  // autotuning.
  //
  // Unlike RequestModelAllocation, we use a delta number of bytes, since there
  // can only be one model per iterator but there may be multiple legacy
  // prefetch autotuners.
  //
  // Returns whether there were enough bytes left in the budget to serve the
  // request. If not, no bytes are allocated.
  bool RequestLegacyPrefetchBytes(int64_t delta_bytes) {
    mutex_lock l(mu_);
    if (delta_bytes > budget_ - legacy_prefetch_allocated_ - model_allocated_) {
      return false;
    }
    legacy_prefetch_allocated_ += delta_bytes;
    return true;
  }

  // The total number of bytes that the model could potentially use.
  int64_t AvailableModelRam() const {
    tf_shared_lock l(mu_);
    return budget_ - legacy_prefetch_allocated_;
  }

  void UpdateBudget(int64_t budget) {
    mutex_lock l(mu_);
    budget_ = budget;
    VLOG(2) << "Updated ram budget to " << budget;
  }

  std::string DebugString() {
    mutex_lock l(mu_);
    return absl::StrCat("RamBudgetManager: budget_: ", budget_,
                        " prefetch allocated: ", legacy_prefetch_allocated_,
                        " model allocated: ", model_allocated_);
  }

 private:
  mutable mutex mu_;
  int64_t budget_ TF_GUARDED_BY(mu_) = 0;
  // Number of bytes allocated by legacy prefetch autotuner.
  int64_t legacy_prefetch_allocated_ TF_GUARDED_BY(mu_) = 0;
  // Number of bytes allocated by the model.
  int64_t model_allocated_ TF_GUARDED_BY(mu_) = 0;
};

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
    int64_t id;
    string name;
    std::shared_ptr<Node> output;
  };

  using Factory = std::function<std::shared_ptr<Node>(Args)>;
  using NodeVector = std::vector<std::shared_ptr<Node>>;
  using NodePairList =
      std::list<std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>>>;
  using ModelParameters =
      std::vector<std::pair<string, std::shared_ptr<Parameter>>>;
  using NodeValues = absl::flat_hash_map<string, double>;
  using ParameterGradients =
      absl::flat_hash_map<std::pair<string, string>, double>;

  explicit Node(Args args)
      : id_(args.id),
        name_(std::move(args.name)),
        autotune_(true),
        buffered_bytes_(0),
        peak_buffered_bytes_(0),
        buffered_elements_(0),
        buffered_elements_low_(std::numeric_limits<int64_t>::max()),
        buffered_elements_high_(std::numeric_limits<int64_t>::min()),
        bytes_consumed_(0),
        bytes_produced_(0),
        num_elements_(0),
        processing_time_(0),
        record_metrics_(true),
        metrics_(name_),
        output_(args.output.get()),
        output_weak_ptr_(args.output) {}

  virtual ~Node() {
    // Clear the sub-nodes instead of relying on implicit shared pointer
    // destructor to avoid potential stack overflow when the tree is deep.
    std::deque<std::shared_ptr<Node>> queue;
    {
      mutex_lock l(mu_);
      while (!inputs_.empty()) {
        queue.push_back(inputs_.front());
        inputs_.pop_front();
      }
    }
    while (!queue.empty()) {
      auto node = queue.back();
      queue.pop_back();
      {
        mutex_lock l(node->mu_);
        while (!node->inputs_.empty()) {
          queue.push_back(node->inputs_.front());
          node->inputs_.pop_front();
        }
      }
    }

    FlushMetrics();
  }

  // Adds an input.
  void add_input(std::shared_ptr<Node> node) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    inputs_.push_back(node);
  }

  // Increments the aggregate processing time by the given delta.
  void add_processing_time(int64_t delta) TF_LOCKS_EXCLUDED(mu_) {
    processing_time_ += delta;
  }

  // Returns an indication whether autotuning is enabled for this node.
  bool autotune() const TF_LOCKS_EXCLUDED(mu_) { return autotune_; }

  // Returns the number of bytes stored in this node's buffer.
  int64_t buffered_bytes() const TF_LOCKS_EXCLUDED(mu_) {
    return buffered_bytes_;
  }

  // Returns the peak number of bytes stored in this node's buffer.
  int64_t peak_buffered_bytes() const TF_LOCKS_EXCLUDED(mu_) {
    return peak_buffered_bytes_;
  }

  // Returns the number of elements stored in this node's buffer.
  int64_t buffered_elements() const TF_LOCKS_EXCLUDED(mu_) {
    return buffered_elements_;
  }

  // Returns the low watermark of the number of elements stored in this node's
  // buffer. The watermarks are reset at the beginning of the execution time and
  // each time the buffer is upsized or downsized.
  int64_t buffered_elements_low() const TF_LOCKS_EXCLUDED(mu_) {
    return buffered_elements_low_;
  }

  // Returns the high watermark of the number of elements stored in this node's
  // buffer. The watermarks are reset at the beginning of the execution time and
  // each time the buffer is upsized or downsized.
  int64_t buffered_elements_high() const TF_LOCKS_EXCLUDED(mu_) {
    return buffered_elements_high_;
  }

  // Returns the number of bytes consumed by the node.
  int64_t bytes_consumed() const TF_LOCKS_EXCLUDED(mu_) {
    return bytes_consumed_;
  }

  // Returns the number of bytes produced by the node.
  int64_t bytes_produced() const TF_LOCKS_EXCLUDED(mu_) {
    return bytes_produced_;
  }

  // Indicates whether the node has tunable parameters.
  bool has_tunable_parameters() const TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    for (const auto& pair : parameters_) {
      if (pair.second->state->tunable) return true;
    }
    return false;
  }

  // Returns the unique node ID.
  int64_t id() const TF_LOCKS_EXCLUDED(mu_) { return id_; }

  // Returns the node inputs.
  std::list<std::shared_ptr<Node>> inputs() const TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return inputs_;
  }

  // Returns a longer node name that is guaranteed to be unique.
  string long_name() const { return strings::StrCat(name_, "(id:", id_, ")"); }

  // Returns the node name.
  const string& name() const { return name_; }

  // Returns the number of elements produced by the node.
  int64_t num_elements() const TF_LOCKS_EXCLUDED(mu_) { return num_elements_; }

  // Returns the node output.
  Node* output() const { return output_; }
  std::shared_ptr<Node> output_shared() { return output_weak_ptr_.lock(); }

  // Returns the parameter value.
  double parameter_value(const string& name) const TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return parameters_.at(name)->state->value;
  }

  // Returns the aggregate processing time.
  int64_t processing_time() const TF_LOCKS_EXCLUDED(mu_) {
    return processing_time_;
  }

  // Records that the node consumed the given number of bytes.
  void record_bytes_consumed(int64_t num_bytes) {
    bytes_consumed_ += num_bytes;
  }

  // Records that the node produced the given number of bytes.
  void record_bytes_produced(int64_t num_bytes) {
    bytes_produced_ += num_bytes;
  }

  // Records the change in this node's buffer.
  void record_buffer_event(int64_t bytes_delta, int64_t elements_delta) {
    buffered_bytes_ += bytes_delta;
    peak_buffered_bytes_.store(std::max(peak_buffered_bytes_, buffered_bytes_));
    buffered_elements_ += elements_delta;
    // There is no need to maintain watermarks for synchronous ops because we
    // will not upsize or downsize the buffers of synchronous ops.
    if (IsAsync()) {
      int64_t low_watermark =
          std::min(buffered_elements_low_, buffered_elements_);
      buffered_elements_low_ = low_watermark;
      int64_t high_watermark =
          std::max(buffered_elements_high_, buffered_elements_);
      buffered_elements_high_ = high_watermark;
    }
  }

  // Records that the node produced an element.
  void record_element() TF_LOCKS_EXCLUDED(mu_) {
    num_elements_++;
    {
      mutex_lock l(mu_);
      UpdateProcessingTimeEma();
    }
  }

  // Records that a node thread has started executing.
  void record_start(int64_t time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    DCHECK_EQ(work_start_, 0);
    work_start_ = time_nanos;
  }

  // Records that a node thread has stopped executing.
  void record_stop(int64_t time_nanos) TF_LOCKS_EXCLUDED(mu_) {
    // TODO(jsimsa): Use DCHECK_NE(work_start_, 0) here.
    if (work_start_ != 0) {
      processing_time_ += time_nanos - work_start_;
      work_start_ = 0;
    } else {
      VLOG(1) << "Encountered a stop event without a matching start event.";
    }
  }

  // Returns whether work is currently being recorded, i.e. whether we are
  // currently between a `record_start` and a `record_stop`.
  bool is_recording() TF_LOCKS_EXCLUDED(mu_) { return work_start_ > 0; }

  // Removes an input.
  void remove_input(std::shared_ptr<Node> input) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    inputs_.remove(input);
  }

  // Sets the value that determines whether autotuning is enabled for this node.
  void set_autotune(bool autotune) TF_LOCKS_EXCLUDED(mu_) {
    autotune_.store(autotune);
  }

  // Resets buffer watermarks to the current buffered elements.
  void ResetBufferWatermarks() {
    if (!IsAsync()) {
      return;
    }
    int64_t current_buffer_size = buffered_elements_;
    buffered_elements_low_ = current_buffer_size;
    buffered_elements_high_ = current_buffer_size;
  }

  // Returns true for asynchronous nodes; false otherwise.
  virtual bool IsAsync() const { return false; }

  // Returns the ratio of the node, which is defined as the number of elements
  // per input needed by the node to produce an element, e.g. batch size of a
  // `Batch`. It can be 0 if the ratio is unknown.
  virtual double Ratio() const { return 1.0; }

  // Computes the self time in nanoseconds of the node to produce one element.
  virtual double ComputeSelfTime() const;

  // Returns the parameter value if it exists, not ok status otherwise.
  absl::StatusOr<double> ParameterValue(const std::string& parameter_name) const
      TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    if (parameters_.contains(parameter_name)) {
      return parameters_.at(parameter_name)->value;
    }
    return errors::NotFound("Parameter ", parameter_name,
                            " was not found in model node ", long_name());
  }

  // Given the average time between events when the elements in the buffer are
  // produced (`producer_time`), the average time between events when elements
  // in the buffer are consumed (`consumer_time`) and the buffer size, the
  // method computes the expected time a consumer event will have to wait.
  //
  // The wait time is approximated as the product of the probability the buffer
  // will be empty and the time it takes to produce an element into the buffer.
  //
  // The formula used for computing the probability is derived by modeling the
  // problem as an M/M/1/K queue
  // (https://en.wikipedia.org/wiki/Birth%E2%80%93death_process#M/M/1/K_queue).
  //
  // Collects derivatives of `ComputeWaitTime` w.r.t `producer_time`,
  // `consumer_time' and `buffer_size` if the corresponding pointers are not
  // `nullptr`.
  static double ComputeWaitTime(double producer_time, double consumer_time,
                                double buffer_size,
                                double* producer_time_derivative,
                                double* consumer_time_derivative,
                                double* buffer_size_derivative);

  // Collects tunable parameters in the subtree rooted in this node.
  ModelParameters CollectTunableParameters() const TF_LOCKS_EXCLUDED(mu_);

  // Collects tunable parameters in this node.
  ModelParameters CollectNodeTunableParameters() const TF_LOCKS_EXCLUDED(mu_);

  // Returns a human-readable representation of this node.
  string DebugString() const TF_LOCKS_EXCLUDED(mu_);

  // Flushes the metrics recorded by this node.
  void FlushMetrics() TF_LOCKS_EXCLUDED(mu_);

  // Returns the per-element output time for this node and if `gradients` is not
  // `nullptr`, collects the output time gradient w.r.t. tunable parameters of
  // the subtree rooted in this node.
  double OutputTime(NodeValues* input_times,
                    ParameterGradients* gradients) const TF_LOCKS_EXCLUDED(mu_);

  // Returns a copy of this node, making a deep copy of its inputs and a
  // shallow copy of its tunable parameters.
  //
  // The purpose for this method is to allow the model optimization logic to
  // operate over immutable state while allowing concurrent model updates.
  std::shared_ptr<Node> Snapshot() const TF_LOCKS_EXCLUDED(mu_);

  // Returns the per-element processing time in nanoseconds spent in this node.
  double SelfProcessingTime() const TF_LOCKS_EXCLUDED(mu_);

  // Returns the total number of bytes buffered in all nodes in the subtree for
  // which autotuning is enabled.
  double TotalBufferedBytes() const TF_LOCKS_EXCLUDED(mu_);

  // Collects the total buffer limit of all nodes in the subtree for which
  // autotuning is enabled. This number represents the amount of memory that
  // would be used by the subtree nodes if all of their buffers were full.
  double TotalMaximumBufferedBytes() const TF_LOCKS_EXCLUDED(mu_);

  // Returns the per-element CPU time in nanoseconds spent in the subtree rooted
  // in this node. If `processing_times` is not `nullptr`, collects the
  // per-element CPU time spent in each node of the subtree.
  double TotalProcessingTime(NodeValues* processing_times)
      TF_LOCKS_EXCLUDED(mu_);

  // Produces a proto for this node. Does not produce a proto for input nodes.
  virtual Status ToProto(ModelProto::Node* node_proto) const;

  // Restores a node from the proto. Does not restore input nodes.
  static Status FromProto(ModelProto::Node node_proto,
                          std::shared_ptr<Node> output,
                          std::shared_ptr<Node>* node);

  // Returns a vector of nodes of the subtree rooted in this node. The nodes are
  // either in breadth-first search or reverse breadth-first search order
  // depending on the `order` argument. The nodes are collected based on the
  // results of the `collect_node` predicate: if the predicate returns `false`
  // for a given node, then the subtree rooted in this node is excluded. The
  // root node itself is not collected.
  NodeVector CollectNodes(TraversalOrder order,
                          bool collect_node(const std::shared_ptr<Node>)) const
      TF_LOCKS_EXCLUDED(mu_);

  // Downsizes buffer parameters of this node. Returns true if any buffer is
  // downsized.
  bool TryDownsizeBuffer();

  // Collects buffer parameters of this node that should be upsized.
  void CollectBufferParametersToUpsize(
      absl::flat_hash_map<Node*, Parameter*>& node_parameters);

  // Returns the average size of an element buffered in this node.
  double AverageBufferedElementSize() const {
    tf_shared_lock l(mu_);
    return AverageBufferedElementSizeLocked();
  }

  // Copies node's parameter state value to parameter value if the parameter
  // name matches `parameter_name`.
  void SyncStateValuesToParameterValues(const std::string& parameter_name);

 protected:
  // Used for (incrementally) recording metrics. The class is thread-safe.
  class Metrics {
   public:
    explicit Metrics(const string& name)
        : bytes_consumed_counter_(metrics::GetTFDataBytesConsumedCounter(name)),
          bytes_produced_counter_(metrics::GetTFDataBytesProducedCounter(name)),
          num_elements_counter_(metrics::GetTFDataElementsCounter(name)),
          recorded_bytes_consumed_(0),
          recorded_bytes_produced_(0),
          recorded_num_elements_(0) {}

    // Expects the total number of bytes consumed and records the delta since
    // last invocation.
    void record_bytes_consumed(int64_t total_bytes) {
      int64_t delta =
          total_bytes - recorded_bytes_consumed_.exchange(total_bytes);
      bytes_consumed_counter_->IncrementBy(delta);
    }

    // Expects the total number of bytes produced and records the delta since
    // last invocation.
    void record_bytes_produced(int64_t total_bytes) {
      int64_t delta =
          total_bytes - recorded_bytes_produced_.exchange(total_bytes);
      bytes_produced_counter_->IncrementBy(delta);
    }

    // Expects the total number of elements produced and records the delta since
    // last invocation.
    void record_num_elements(int64_t total_elements) {
      int64_t delta =
          total_elements - recorded_num_elements_.exchange(total_elements);
      num_elements_counter_->IncrementBy(delta);
    }

   private:
    monitoring::CounterCell* const bytes_consumed_counter_;
    monitoring::CounterCell* const bytes_produced_counter_;
    monitoring::CounterCell* const num_elements_counter_;
    std::atomic<int64_t> recorded_bytes_consumed_;
    std::atomic<int64_t> recorded_bytes_produced_;
    std::atomic<int64_t> recorded_num_elements_;
  };

  // Computes the exponential moving average of processing time per element.
  void UpdateProcessingTimeEma() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (previous_processing_time_ == 0) {
      if (num_elements_ > 0) {
        processing_time_ema_ =
            static_cast<double>(processing_time_) /
            static_cast<double>(num_elements_ + buffered_elements_);
      } else {
        processing_time_ema_ = static_cast<double>(processing_time_);
      }
    } else {
      processing_time_ema_ =
          (1.0 - kProcessingTimeEmaWeight) * processing_time_ema_ +
          kProcessingTimeEmaWeight *
              static_cast<double>(processing_time_ - previous_processing_time_);
    }
    previous_processing_time_ = processing_time_;
  }

  // Returns the number of inputs.
  int64_t num_inputs() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    int64_t num_inputs = 0;
    for (auto& input : inputs_) {
      // Inputs for which autotuning is disabled are excluded.
      if (input->autotune()) {
        ++num_inputs;
      }
    }
    return num_inputs;
  }

  // Creates a clone of this node.
  virtual std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const
      TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Returns the average size of an element buffered in this node.
  double AverageBufferedElementSizeLocked() const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Returns the sum of per-element output time for the tunable inputs of this
  // node.
  double OutputTimeForInputs(const NodeValues& output_times) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Returns the sum of output time gradient w.r.t. input time for the tunable
  // inputs of this node.
  double OutputTimeGradientsForInputs(const NodeValues& output_time_gradients)
      const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Computes the input time for this node and stores it in `input_times`.
  virtual void InputTimeLocked(NodeValues* input_times) const
      TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Computes the per-element output time for this node and stores it in
  // `output_times`. If `gradients` is not `nullptr`, computes the output time
  // gradient w.r.t. tunable parameters of the subtree rooted in this node and
  // stores it in `gradients`, also computes the output time gradient w.r.t.
  // input time and stores it in `output_time_gradients`.
  virtual void OutputTimeLocked(const NodeValues& input_times,
                                ParameterGradients* gradients,
                                NodeValues* output_times,
                                NodeValues* output_time_gradients) const
      TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  // Returns the sum of per-element processing time for the inputs of this node
  // by adding values for input nodes in `total_processing_times`. Processing
  // time for a given input is a weighted combination of a statistic based on
  // history of input processing time and the actual time. This is done to
  // improve accuracy of processing time estimation for newly created inputs.
  //
  // Uniform distribution of per-element processing times across different
  // inputs is assumed.
  double TotalProcessingTimeForInputs(const NodeValues& total_processing_times)
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Returns the per-element processing time spent in this node.
  double SelfProcessingTimeLocked() const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Computes the per-element CPU time spent in the subtree rooted in this node
  // and stores it in `total_processing_times`. If `processing_times` is not
  // `nullptr`, collects the per-element CPU time spent in each node of the
  // subtree.
  virtual void TotalProcessingTimeLocked(NodeValues* processing_times,
                                         NodeValues* total_processing_times)
      TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  // This is the locked version of the public `CollectNodes`.
  NodeVector CollectNodesLocked(TraversalOrder order,
                                bool collect_node(const std::shared_ptr<Node>))
      const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Collects tunable parameters in the subtree rooted in this node assuming
  // mutex locked.
  ModelParameters CollectTunableParametersLocked() const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Collect tunable parameters on the nodes which have recorded
  // elements.
  void CollectTunableParametersHelper(ModelParameters* parameters) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Build up debug string for the node and store in the debug strings map.
  void DebugStringHelper(absl::flat_hash_map<string, string>* debug_strings)
      const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Copy the node and add the (input, copy) pairs to the NodePairList.
  std::shared_ptr<Node> SnapshotHelper(std::shared_ptr<Node> cloned_output,
                                       NodePairList* node_pairs) const;

  // Compute total buffered bytes for the node and store in the total bytes map.
  void TotalBufferedBytesHelper(NodeValues* total_bytes) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Compute total maximum buffered bytes for the node and store in the total
  // bytes map.
  void TotalMaximumBufferedBytesHelper(NodeValues* total_bytes) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  // Compute and return the maximum buffered bytes on the node itself. By
  // default non-tunable nodes are assumed not to buffer any bytes, so the
  // tunable nodes as subclasses are expected to override this method to ensure
  // that the optimization algorithm respects the memory budget.
  virtual double MaximumBufferedBytes() const TF_SHARED_LOCKS_REQUIRED(mu_);

  // Restores node from the proto. Note that this is not done recursively, i.e.
  // input nodes are not restored.
  static Status FromProtoHelper(ModelProto::Node node_proto,
                                std::shared_ptr<Node> node);

  // Stores the time passed to the last call to `Node::record_start()` on the
  // current thread.
  //
  // NOTE: This thread-local variable is shared between all instances of `Node`
  // on which the same thread calls `record_start()` or `record_stop()`. It
  // relies on the invariant that at most one `Node` can be "active" on a
  // particular thread at any time. Therefore if `n->record_start()` is called
  // on thread `t`, then `n->record_stop()` must be called before another call
  // to `Node::record_start()` (for any node).
  static thread_local int64_t work_start_;  // Will be initialized to zero.

  mutable mutex mu_;
  const int64_t id_;
  const string name_;

  // Indicates whether the subtree rooted in this node should be included in
  // autotuning. In particular, if this is `false`, then the subtree is excluded
  // from computation of output time and processing time.
  std::atomic<bool> autotune_;
  std::atomic<int64_t> buffered_bytes_;
  std::atomic<int64_t> peak_buffered_bytes_;
  std::atomic<int64_t> buffered_elements_;
  std::atomic<int64_t> buffered_elements_low_;
  std::atomic<int64_t> buffered_elements_high_;
  std::atomic<int64_t> bytes_consumed_;
  std::atomic<int64_t> bytes_produced_;
  std::atomic<int64_t> num_elements_;
  std::atomic<int64_t> processing_time_;
  std::atomic<bool> record_metrics_;
  Metrics metrics_;
  absl::flat_hash_map<string, std::shared_ptr<Parameter>> parameters_
      TF_GUARDED_BY(mu_);

  // Statistic of inputs processing time history.
  double input_processing_time_sum_ = 0.0L;
  int64_t input_processing_time_count_ = 0;

  // Holds the previous processing time and the per element processing time
  // exponential moving average.
  int64_t previous_processing_time_ TF_GUARDED_BY(mu_) = 0;
  double processing_time_ema_ TF_GUARDED_BY(mu_) = 0.0;

  // Inputs of this node. These can represent an iterator created from the input
  // dataset but also other input iterators (e.g. created by the user-defined
  // functions of `flat_map` or `interleave`).
  std::list<std::shared_ptr<Node>> inputs_ TF_GUARDED_BY(mu_);

  // The reference to the output node is not owned so that deletion of a
  // node results in recursive deletion of the subtree rooted in the node.
  Node* const output_;
  std::weak_ptr<Node> output_weak_ptr_;
};

// InterleaveMany is used to model datasets whose inputs are used to create
// datasets whose elements are then interleaved.
std::shared_ptr<Node> MakeInterleaveManyNode(
    Node::Args args, std::vector<std::shared_ptr<Parameter>> parameters);

// AsyncInterleaveMany nodes are the asynchronous version of InterleaveMany
// nodes.
std::shared_ptr<Node> MakeAsyncInterleaveManyNode(
    Node::Args args, std::vector<std::shared_ptr<Parameter>> parameters);

// KnownMany nodes model datasets that synchronously consume known number of
// input element per output element.
std::shared_ptr<Node> MakeKnownRatioNode(Node::Args args, double ratio);

// AsyncKnownRatio nodes are the asynchronous version of KnownRate nodes.
std::shared_ptr<Node> MakeAsyncKnownRatioNode(
    Node::Args args, double ratio, double memory_ratio,
    std::vector<std::shared_ptr<Parameter>> parameters,
    bool is_legacy_prefetch_autotuned = false);

std::shared_ptr<Node> MakeAsyncKnownRatioNode(
    Node::Args args, double ratio,
    std::vector<std::shared_ptr<Parameter>> parameters,
    bool is_legacy_prefetch_autotuned = false);

// Source nodes represent data sources.
std::shared_ptr<Node> MakeSourceNode(Node::Args args);

// UnknownMany nodes represent datasets that synchronously consume an
// unknown number of input elements per output.
//
// Unlike KnownRatio nodes which expect the ratio between inputs and outputs is
// specified as a parameter, UnknownRatio estimates the ratio empirically.
std::shared_ptr<Node> MakeUnknownRatioNode(Node::Args args);

// AsyncUnknownRatio nodes are the asynchronous version of unknown ratio nodes.
std::shared_ptr<Node> MakeAsyncUnknownRatioNode(
    Node::Args args, std::vector<std::shared_ptr<Parameter>> parameters);

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
//
// The order of locks acquired is SharedState lock, Model lock, Node lock.
// SharedState lock is acquired first because it shares the same lock as the
// dataset iterator that contains it.
class Model {
 public:
  using OptimizationParams = ModelProto::OptimizationParams;
  using ModelParameters = Node::ModelParameters;
  using NodeValues = Node::NodeValues;
  using ParameterGradients = Node::ParameterGradients;

  explicit Model(std::optional<std::string> dataset_name);
  explicit Model() : Model(std::nullopt) {}
  ~Model();

  // Returns a pointer to the model's output node.
  std::shared_ptr<Node> output() const {
    mutex_lock l(mu_);
    return output_;
  }

  // Set the experiment that this job is part of.
  void AddExperiment(const std::string& experiment) {
    experiments_.insert(experiment);
  }

  // Adds a node with the given name and given parent.
  void AddNode(Node::Factory factory, const string& name,
               std::shared_ptr<Node> parent, std::shared_ptr<Node>* out_node)
      TF_LOCKS_EXCLUDED(mu_);

  // Returns a human-readable string representation of the model. This method
  // can be invoked automatically by monitoring gauges and to avoid frequent
  // recomputation, the implementation caches the result.
  std::string DebugString();

  // Uses the given algorithm and resource budgets to periodically perform the
  // autotuning optimization.
  //
  // `cpu_budget_func` can be used to provide the optimizer with up-to-date
  // values in cases where CPUs budgets may be changed by the runtime
  // dynamically.
  //
  // `ram_budget_func` is similar to `cpu_budget_func`. This lambda takes a
  // parameter that is the total number of bytes currently buffered by the
  // model.
  //
  // To terminate the execution of the optimization loop, the caller needs to
  // invoke `cancellation_mgr->StartCancel()`.
  Status OptimizeLoop(AutotuneAlgorithm algorithm,
                      std::function<int64_t()> cpu_budget_func,
                      double ram_budget_share,
                      std::optional<int64_t> fixed_ram_budget,
                      RamBudgetManager& ram_budget_manager,
                      CancellationManager* cancellation_manager);

  // Uses the given algorithm and resource budgets to perform the autotuning
  // optimization.
  void Optimize(AutotuneAlgorithm algorithm,
                std::function<int64_t()> cpu_budget_func,
                double ram_budget_share,
                std::optional<int64_t> fixed_ram_budget,
                double model_input_time, RamBudgetManager& ram_budget_manager,
                CancellationManager* cancellation_manager);

  // Optimizes buffers in the pipeline rooted at `snapshot`. It downsizes
  // buffers that are too large and upsizes buffers that are too small while
  // respecting the ram budget. If any node is downsized or upsized, the
  // watermarks of all nodes are reset to the buffered elements.
  void OptimizeBuffers(std::shared_ptr<Node> snapshot, int64_t ram_budget);

  // Collects the output time and if `gradients` is not `nullptr`, the output
  // time gradient w.r.t. tunable parameters of the subtree rooted in the given
  // node.
  double OutputTime(std::shared_ptr<Node> node, double model_input_time,
                    ParameterGradients* gradients);

  // Removes the given node.
  void RemoveNode(std::shared_ptr<Node> node) TF_LOCKS_EXCLUDED(mu_);

  // Produces a proto for this model.
  Status ToProto(ModelProto* model_proto);

  // Restores a model from the proto.
  static Status FromProto(ModelProto model_proto,
                          std::unique_ptr<Model>* model);

  // Saves this model with a given snapshot and its optimization parameters to a
  // file. Note that the file directory must already exist.
  Status Save(const string& fname, std::shared_ptr<Node> snapshot,
              const OptimizationParams& optimization_params);

  // Loads a model and its optimization parameters from a file with the given
  // name.
  static Status Load(const string& fname, std::unique_ptr<Model>* model,
                     OptimizationParams* optimization_params);

  // Records gap time between consecutive `GetNext()` calls.
  void RecordIteratorGapTime(uint64_t duration_usec);

  // Computes the target time in nsecs to use for `STAGE_BASED` autotune
  // algorithm. Returns 0 if there if there are not sufficient recorded iterator
  // gap times to produce a good estimate.
  double ComputeTargetTimeNsec();

  // Computes the target time in nsecs to use for estimating input bottlenecks.
  // Returns 0 if there are not sufficient recorded iterator gap times to
  // produce a good estimate.
  double ComputeExperimentalTargetTimeNsec();

  // Returns the time in nanoseconds it takes the pipeline to produce an
  // element, according to the latest model snapshot obtained from optimization.
  // Returns 0 if the model snapshot is empty or null. This may be caused by not
  // having executed an optimization round before.
  double ComputeSnapshotProcessingTimeNsec() const;

 private:
  // Determines whether optimization should stop given total processing time,
  // estimated output time, and estimated number of buffers bytes.
  using StopPredicate =
      std::function<bool(const ModelParameters&, double, double, double)>;

  static constexpr int64_t kOptimizationPeriodMinMs = 10;
  static constexpr int64_t kOptimizationPeriodMaxMs =
      60 * EnvTime::kSecondsToMillis;

  // Collects tunable parameters in the tree rooted in the given node, returning
  // a vector which contains pairs of node names and tunable parameters.
  ModelParameters CollectTunableParameters(std::shared_ptr<Node> node);

  // Copy parameter state values to parameter values if necessary.For some
  // nodes, the parameter state values are not tuned by Autotune and hence the
  // parameter values can be stale. We do not sync all parameters because it may
  // increase mutex contention with `GetNext()`.
  void MaybeSyncStateValuesToValues(std::shared_ptr<Node> snapshot);

  // Downsizes buffers that are too large for all nodes rooted at `snapshot`.
  // Returns true if any buffer is downsized.
  bool DownsizeBuffers(std::shared_ptr<Node> snapshot);

  // Upsizes buffers that are too small for all nodes rooted at `snapshot` while
  // respecting the ram budget. Returns true if any buffer is upsized.
  bool UpsizeBuffers(std::shared_ptr<Node> snapshot, int64_t ram_budget);

  // Reset buffer watermarks of all asynchronous nodes to their buffered
  // elements.
  void ResetBufferWatermarks();

  // Collects buffer parameters of all nodes in the model that should be
  // upsized.
  absl::flat_hash_map<Node*, Parameter*> CollectBufferParametersToUpsize(
      std::shared_ptr<Node> snapshot);

  // Flushes metrics recorded by the model.
  void FlushMetrics() TF_LOCKS_EXCLUDED(mu_);

  // This optimization algorithm starts by setting all tunable parallelism
  // parameters to the minimum value. It then improves current parameters by
  // making a step in the direction opposite to the gradient of `OutputTime` and
  // projecting resulting values on the feasible intervals. Improvement step is
  // repeated until either the output time improvement is smaller than threshold
  // value or the output time is less than the processing time needed to produce
  // an element divided by CPU budget.
  void OptimizeGradientDescent(std::shared_ptr<Node> snapshot,
                               const OptimizationParams& optimization_params,
                               CancellationManager* cancellation_manager);

  // Helper method for implementing hill-climb optimization that can be
  // parametrized by a predicate to use for stopping the optimization.
  void OptimizeHillClimbHelper(std::shared_ptr<Node> snapshot,
                               const OptimizationParams& optimization_params,
                               CancellationManager* cancellation_manager,
                               int64_t ram_budget,
                               RamBudgetManager& ram_budget_manager,
                               StopPredicate should_stop);

  // This optimization algorithm starts by setting all tunable parallelism
  // parameters to the minimum value. It then repeatedly identifies the
  // parameter whose increase in parallelism decreases the output time the most.
  // This process is repeated until all parameters reach their maximum values or
  // the projected output time is less than or equal to the processing time
  // needed to produce an element divided by CPU budget.
  void OptimizeHillClimb(std::shared_ptr<Node> snapshot,
                         const OptimizationParams& optimization_params,
                         CancellationManager* cancellation_manager,
                         RamBudgetManager& ram_budget_manager);

  // This optimization behaves similarly to the hill climb optimization but uses
  // a relaxed stoping condition, allowing the optimization to oversubscribe
  // CPU.
  void OptimizeMaxParallelism(std::shared_ptr<Node> snapshot,
                              const OptimizationParams& optimization_params,
                              CancellationManager* cancellation_manager,
                              RamBudgetManager& ram_budget_manager);

  // This optimization starts by setting all tunable parallelism parameters to
  // their minimum values. It then repeatedly increases the parallelism
  // parameter of the longest stage by 1 until either the longest stage is
  // faster than the target time or the memory or CPU budget is fully utilized.
  // TODO(b/226910071): The second part of this algorithm optimizes the buffer
  // sizes of parallel ops.
  void OptimizeStageBased(std::shared_ptr<Node> snapshot,
                          const OptimizationParams& optimization_params,
                          CancellationManager* cancellation_manager,
                          RamBudgetManager& ram_budget_manager);

  // This is the first part of the stage-based optimization that optimizes
  // tunable parallelism parameters for async interleave many nodes only. We
  // separately optimize async interleave many nodes more aggressively because
  // the variance of IO is difficult to predict.
  void OptimizeStageBasedAsyncInterleaveManyNodes(
      std::shared_ptr<Node> snapshot,
      const OptimizationParams& optimization_params,
      CancellationManager* cancellation_manager,
      RamBudgetManager& ram_budget_manager);

  // This is the second part of the stage-based optimization that optimizes
  // tunable parallelism parameters for all nodes other than async interleave
  // many nodes.
  void OptimizeStageBasedNonAsyncInterleaveManyNodes(
      std::shared_ptr<Node> snapshot, double target_time_nsec,
      const OptimizationParams& optimization_params,
      CancellationManager* cancellation_manager,
      RamBudgetManager& ram_budget_manager);

  // Determines if we should stop the gradient descent optimization iterations
  // based on number of increasable parameters, CPU budget, RAM budget and
  // current resource usage.
  bool ShouldStop(int64_t cpu_budget, int64_t ram_budget,
                  const ModelParameters& parameters,
                  const ModelParameters& parallelism_parameters,
                  const ModelParameters& buffer_size_parameters,
                  std::shared_ptr<Node> snapshot, bool* cpu_budget_reached);

  // Collects the processing time for the given node.
  double TotalProcessingTime(std::shared_ptr<Node> node);

  // Collects the total number of bytes buffered in all nodes in the subtree
  // rooted in the given node for which autotuning is enabled.
  double TotalBufferedBytes(std::shared_ptr<Node> node);

  // Collects the total buffer limit of all nodes in the subtree rooted in the
  // given node for which autotuning is enabled. This number represents the
  // amount of memory that would be used by the subtree nodes if all of their
  // buffers were full.
  double TotalMaximumBufferedBytes(std::shared_ptr<Node> node);

  std::optional<std::string> dataset_name_;
  // Used for coordination between different input pipeline threads. Exclusive
  // access is required only when adding or removing nodes. Concurrent access to
  // existing nodes is protected by a node mutex.
  mutable mutex mu_;
  // Used for coordinating the optimization loop and model modifications.
  condition_variable optimize_cond_var_;
  int64_t id_counter_ TF_GUARDED_BY(mu_) = 1;
  std::shared_ptr<Node> output_ TF_GUARDED_BY(mu_) = nullptr;

  // Determines the time the optimization loop should wait between
  // running optimizations.
  int64_t optimization_period_ms_ TF_GUARDED_BY(mu_);

  // Gauge cell that can be used to collect the state of the model.
  monitoring::GaugeCell<std::function<std::string()>>* model_gauge_cell_ =
      nullptr;
  // Used to synchronize metrics collection attempts against the model's
  // destruction.
  struct GuardedBool {
    explicit GuardedBool(bool val) : val(val) {}
    bool val TF_GUARDED_BY(mu);
    mutex mu;
  };
  std::shared_ptr<GuardedBool> safe_to_collect_metrics_;

  // Time use for rate limiting the recomputation of human-readable string
  // representation of the model.
  absl::Time cache_until_ = absl::InfinitePast();
  // Cached result of the `DebugString()` invocation used to implement rate
  // limiting of the computation.
  std::string cached_debug_string_ = "";
  // Used to coordinate gap time updates between different threads. Gap time is
  // the time between the completion of the previous `GetNext()` and the start
  // of the next `GetNext()`.
  mutable mutex gap_mu_;
  // Stores the latest gap times between consecutive `GetNext()`.
  std::deque<uint64_t> gap_times_usec_ TF_GUARDED_BY(gap_mu_);
  // The experiment that this job is part of.
  absl::flat_hash_set<std::string> experiments_;
  // Stores the optimization snapshot of the Model.
  std::shared_ptr<Node> snapshot_ TF_GUARDED_BY(mu_);
  // Stores the optimization parameters used by autotune.
  OptimizationParams optimization_params_ TF_GUARDED_BY(mu_);
  // Stores the model id in the string format
  std::string model_id_;
};

// Class to compute timing information for a model.
class ModelTiming {
 public:
  struct NodeTiming {
    // Pipeline ratio is the number of elements this node needs to produce in
    // order to produce an element at the root of the pipeline.
    double pipeline_ratio = 0.0;
    // The self time it takes this node to produce the elements needed to
    // produce one element of the root of the pipeline.
    double self_time_nsec = 0.0;
    // The total time it takes this node and the subtree rooted at this node to
    // produce the elements needed to produce one element at the root of the
    // pipeline.
    double total_time_nsec = 0.0;
  };

  explicit ModelTiming(std::shared_ptr<Node> root);

  // Returns the timing data for `node`.
  const NodeTiming* GetTiming(const Node* node) const;

  // Returns the root nodes of all stages.
  std::vector<std::shared_ptr<Node>> GetStageRoots() const;

  // Returns all the nodes of a stage given the stage root.
  std::vector<std::shared_ptr<Node>> GetStageNodes(
      std::shared_ptr<Node> stage_root) const;

  // Computes the total time for a node.
  void ComputeNodeTotalTime(const Node& node);

 private:
  // Computes the pipeline ratios of all nodes.
  void ComputePipelineRatios(const Node::NodeVector& bfs_nodes);

  // Computes the total time for all nodes. The `reverse_bfs_nodes` are assumed
  // to be a vector of model nodes in reversed BFS manner.
  void ComputeTotalTimes(const Node::NodeVector& reverse_bfs_nodes);

  // Computes the first input total time of an interleave node.
  double ComputeInterleaveManyFirstInputTotalTime(const Node& node);

  // Computes the total time of a node of any type other than async interleave.
  void ComputeNonAsyncInterleaveManyTotalTime(const Node& node);

  // Computes the total time of an async interleave node.
  void ComputeAsyncInterleaveManyTotalTime(const Node& node);
  // Computes the interleaved inputs' total time of an async interleave node.
  double ComputeAsyncInterleaveManyInterleavedInputsTotalTime(const Node& node);

  // Returns a vector of all nodes in the model. The nodes are either in
  // breadth-first search or reverse breadth-first search order depending on the
  // `order` argument. The nodes are collected based on the results of the
  // `collect_node` predicate: if the predicate returns `false` for a given
  // node, then the subtree rooted in this node is excluded. The root node
  // itself is not collected.
  Node::NodeVector CollectNodes(
      std::shared_ptr<Node> root, TraversalOrder order,
      bool collect_node(const std::shared_ptr<Node>)) const;

  // Stores a pointer to the root of a model.
  std::shared_ptr<Node> root_;

  // Holds a mapping from node to its timing node.
  absl::flat_hash_map<const Node*, NodeTiming> timing_nodes_;
};

}  // namespace model
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_MODEL_H_
