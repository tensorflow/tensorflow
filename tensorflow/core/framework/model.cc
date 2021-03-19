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

#include "tensorflow/core/framework/model.h"

#include <memory>

#include "absl/time/clock.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace data {
namespace model {

constexpr int64 Model::kOptimizationPeriodMinMs;
constexpr int64 Model::kOptimizationPeriodMaxMs;

namespace {

// Helper function for node traversal that doesn't skip any nodes.
inline bool IsAnyNode(const std::shared_ptr<Node> node) { return true; }

// Helper function for node traversal that filters out nodes for which
// autotuning is disabled.
inline bool IsAutotuneNode(const std::shared_ptr<Node> node) {
  return node->autotune();
}

// Wrapper for the square function to reduce verbosity.
inline double Square(double x) { return x * x; }

// Collects "essential" parallelism parameters and buffer size parameters in the
// tree rooted in the given node. Which parallelism parameters are essential is
// determined by the relative processing time spent in the corresponding
// transformation. The collected parameters are returned via maps that map node
// names to their respective parameters.
inline void CollectParameters(
    std::shared_ptr<Node> node,
    const absl::flat_hash_map<string, std::shared_ptr<Parameter>>& parameters,
    absl::flat_hash_map<string, std::shared_ptr<Parameter>>*
        parallelism_parameters,
    absl::flat_hash_map<string, std::shared_ptr<Parameter>>*
        buffer_size_parameters) {
  // Parallelism parameter is considered to be essential if the corresponding
  // transformations's processing time is greater than essential rate times the
  // average transformation self processing time.
  constexpr double kEssentialRate = 0.3L;

  absl::flat_hash_map<string, double> processing_times;
  double processing_time = node->TotalProcessingTime(&processing_times);
  double uniform_share =
      processing_time / static_cast<double>(processing_times.size());
  for (auto& pair : parameters) {
    if (pair.second->name == kParallelism &&
        processing_times[pair.first] > kEssentialRate * uniform_share) {
      parallelism_parameters->insert(pair);
    } else if (pair.second->name == kBufferSize) {
      buffer_size_parameters->insert(pair);
    }
  }
}

// Applies the gradient descent method once and updates the parameter values. If
// the new value is out of the range, bound it within the range between the
// minimal and maximum values.
inline void UpdateParameterValues(
    const absl::flat_hash_map<string, double>& gradients,
    absl::flat_hash_map<string, std::shared_ptr<Parameter>>* parameters) {
  // Gradient descent step size.
  constexpr double kDescentStep = 0.1L;
  double new_value;

  double max_abs_derivative = 1.0;
  for (auto& pair : *parameters) {
    if (std::round(pair.second->value) != pair.second->max) {
      auto* gradient = gtl::FindOrNull(gradients, pair.first);
      if (gradient) {
        max_abs_derivative = std::max(max_abs_derivative, std::abs(*gradient));
      }
    }
  }
  for (auto& pair : *parameters) {
    auto* gradient = gtl::FindOrNull(gradients, pair.first);
    if (gradient) {
      new_value =
          pair.second->value - kDescentStep * (*gradient) / max_abs_derivative;
      // Projection on a feasible interval.
      if (new_value > pair.second->max) {
        pair.second->value = pair.second->max;
      } else if (new_value < pair.second->min) {
        pair.second->value = pair.second->min;
      } else {
        pair.second->value = new_value;
      }
    }
  }
}

// Copies the parameter values (which are for optimization tuning) and updates
// the state values (which are for the input pipeline to follow).
inline void UpdateStateValues(
    absl::flat_hash_map<string, std::shared_ptr<Parameter>>* parameters) {
  for (auto& pair : *parameters) {
    auto& parameter = pair.second;
    VLOG(2) << "Setting tunable parameter " << pair.first << " to "
            << parameter->value;
    mutex_lock l(*parameter->state->mu);
    parameter->state->value = parameter->value;
    parameter->state->cond_var->notify_all();
  }
}

// The first input of InterleaveMany corresponds to the input dataset whose
// elements are used to create the (derived) input datasets whose elements are
// interleaved as output.
//
// TODO(jsimsa): model the first input
class InterleaveMany : public Node {
 public:
  using Node::Node;

  virtual ~InterleaveMany() {}

 protected:
  std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    return std::make_shared<InterleaveMany>(
        Args{id_, name_, std::move(output)});
  }

  void InputTimeLocked(absl::flat_hash_map<string, double>* input_times)
      const override TF_SHARED_LOCKS_REQUIRED(mu_) {
    double inherited_input_time;
    if (output_) {
      inherited_input_time = (*input_times)[output_->long_name()];
    } else {
      inherited_input_time = (*input_times)[kModelInputTimeKey];
    }

    if (num_inputs() <= 1) {
      (*input_times)[long_name()] = inherited_input_time;
      return;
    }
    // Here `inherited_input_time + SelfProcessingTimeLocked()` is the average
    // input time for InterleaveMany node to call one of the
    // `(num_inputs() - 1)` input nodes (except first input) to return an
    // element. Regardless of the `block_length` parameter of InterleaveMany
    // node, the average input time for any of the `(num_inputs() - 1)` input
    // nodes to be called is computed as:
    double input_time = (inherited_input_time + SelfProcessingTimeLocked()) *
                        static_cast<double>(num_inputs() - 1);
    (*input_times)[long_name()] = input_time;
  }

  // The output time is the sum of the self processing time and the average
  // output time of inputs comprising the interleave "cycle".
  void OutputTimeLocked(
      const absl::flat_hash_map<string, double>& input_times,
      absl::flat_hash_map<string, double>* gradients,
      absl::flat_hash_map<string, double>* output_times,
      absl::flat_hash_map<string, double>* output_time_gradients) const override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    double self_processing_time = SelfProcessingTimeLocked();
    if (num_inputs() <= 1) {
      (*output_times)[long_name()] = self_processing_time;
      if (gradients) {
        for (const auto& node :
             CollectNodes(TraversalOrder::REVERSE_BFS, IsAutotuneNode)) {
          gradients->erase(node->long_name());
        }
      }
      return;
    }

    double inputs_output_time =
        (OutputTimeForInputs(*output_times) -
         (*output_times)[inputs_.front()->long_name()]) /
        static_cast<double>(num_inputs() - 1);
    if (gradients) {
      for (const auto& node :
           CollectNodes(TraversalOrder::REVERSE_BFS, IsAutotuneNode)) {
        auto* gradient = gtl::FindOrNull(*gradients, node->long_name());
        if (gradient) {
          *gradient /= static_cast<double>(num_inputs() - 1);
        }
      }

      (*output_time_gradients)[long_name()] =
          OutputTimeGradientsForInputs(*output_time_gradients) -
          (*output_time_gradients)[inputs_.front()->long_name()];

      // Set derivatives w.r.t. tunable parameters of the subtree rooted in the
      // first input equal to 0 since its output time is excluded from
      // computations.
      absl::flat_hash_map<string, std::shared_ptr<Parameter>>
          first_input_parameters;
      inputs_.front()->CollectTunableParameters(&first_input_parameters);
      for (auto& pair : first_input_parameters) {
        (*gradients)[pair.first] = 0.0L;
      }
    }
    (*output_times)[long_name()] = self_processing_time + inputs_output_time;
  }

  // The processing time is the sum of the self processing time and the average
  // processing time of inputs comprising the interleave "cycle".
  void TotalProcessingTimeLocked(
      absl::flat_hash_map<string, double>* processing_times,
      absl::flat_hash_map<string, double>* total_processing_times) override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    double self_processing_time = SelfProcessingTimeLocked();
    if (processing_times) {
      (*processing_times)[long_name()] = self_processing_time;
    }
    if (num_inputs() <= 1) {
      (*total_processing_times)[long_name()] = self_processing_time;
      return;
    }
    double inputs_processing_time =
        (TotalProcessingTimeForInputs(*total_processing_times) -
         (*total_processing_times)[inputs_.front()->long_name()]) /
        static_cast<double>(num_inputs() - 1);
    (*total_processing_times)[long_name()] =
        self_processing_time + inputs_processing_time;
  }

  Status ToProto(ModelProto::Node* node_proto) const {
    TF_RETURN_IF_ERROR(Node::ToProto(node_proto));
    node_proto->set_node_class(NodeClass::INTERLEAVE_MANY);
    return Status::OK();
  }
};

// The first input of AsyncInterleaveMany corresponds to the input dataset whose
// elements are used to create the (derived) input datasets whose elements are
// interleaved as output.
//
// TODO(jsimsa): model the first input
class AsyncInterleaveMany : public Node {
 public:
  AsyncInterleaveMany(Node::Args args,
                      std::vector<std::shared_ptr<Parameter>> parameters)
      : Node(args) {
    for (auto& parameter : parameters) {
      parameters_[parameter->name] = std::move(parameter);
    }
  }

  virtual ~AsyncInterleaveMany() {}

 protected:
  std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    std::vector<std::shared_ptr<Parameter>> parameters;
    for (auto& pair : parameters_) {
      parameters.push_back(pair.second);
    }
    return std::make_shared<AsyncInterleaveMany>(
        Args{id_, name_, std::move(output)}, parameters);
  }

  void InputTimeLocked(absl::flat_hash_map<string, double>* input_times)
      const override TF_SHARED_LOCKS_REQUIRED(mu_) {
    double inherited_input_time;
    if (output_) {
      inherited_input_time = (*input_times)[output_->long_name()];
    } else {
      inherited_input_time = (*input_times)[kModelInputTimeKey];
    }

    if (num_inputs() <= 1) {
      (*input_times)[long_name()] = inherited_input_time;
      return;
    }
    // Here `inherited_input_time + SelfProcessingTimeLocked()` is the average
    // input time for AsyncInterleaveMany node to call one of the
    // `(num_inputs() - 1)` input nodes (except first input) to return an
    // element. Regardless of the `block_length` parameter of
    // AsyncInterleaveMany node, the average input time for any of the
    // `(num_inputs() - 1)` input nodes to be called is computed as:
    double input_time = (inherited_input_time + SelfProcessingTimeLocked()) *
                        static_cast<double>(num_inputs() - 1);
    (*input_times)[long_name()] = input_time;
  }

  // The output time is the sum of self processing time and expected wait time
  // from the buffer model estimated using
  // `ComputeWaitTime(producer_time, consumer_time, parallelism, ...)`, where
  // `producer_time` is the average output time of inputs comprising the
  // interleave "cycle" divided by `parallelism`, `consumer_time` is the
  // `input_time` specified through `input_times` divided by `num_inputs() - 1`,
  // and if the node has parallelism parameter, then `buffer_size` is derived
  // from `parallelism`.
  void OutputTimeLocked(
      const absl::flat_hash_map<string, double>& input_times,
      absl::flat_hash_map<string, double>* gradients,
      absl::flat_hash_map<string, double>* output_times,
      absl::flat_hash_map<string, double>* output_time_gradients) const override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    double self_processing_time = SelfProcessingTimeLocked();
    if (num_inputs() <= 1) {
      (*output_times)[long_name()] = self_processing_time;
      if (gradients) {
        for (const auto& node :
             CollectNodes(TraversalOrder::REVERSE_BFS, IsAutotuneNode)) {
          gradients->erase(node->long_name());
        }
      }
      return;
    }

    double output_time, wait_time, consumer_time, producer_time;
    double input_time = input_times.at(long_name());
    consumer_time = input_time / static_cast<double>(num_inputs() - 1);
    double parallelism = num_inputs() - 1;  // default to cycle length
    auto* parameter = gtl::FindOrNull(parameters_, kParallelism);
    if (parameter) {
      parallelism = std::min(parallelism, (*parameter)->value);
    }
    double output_time_for_inputs =
        OutputTimeForInputs(*output_times) -
        (*output_times)[inputs_.front()->long_name()];
    producer_time = output_time_for_inputs /
                    static_cast<double>(num_inputs() - 1) / parallelism;

    if (gradients) {
      double producer_time_der = 0.0L;
      double consumer_time_der = 0.0L;
      double buffer_size_der = 0.0L;
      wait_time = ComputeWaitTime(producer_time, consumer_time, parallelism,
                                  &producer_time_der, &consumer_time_der,
                                  &buffer_size_der);
      double inputs_time_der_sum =
          OutputTimeGradientsForInputs(*output_time_gradients);
      (*output_time_gradients)[long_name()] =
          consumer_time_der +
          producer_time_der * inputs_time_der_sum / parallelism;

      for (const auto& node :
           CollectNodes(TraversalOrder::REVERSE_BFS, IsAutotuneNode)) {
        auto* gradient = gtl::FindOrNull(*gradients, node->long_name());
        if (gradient) {
          *gradient *= (producer_time_der /
                        static_cast<double>(num_inputs() - 1) / parallelism);
        }
      }

      // Set derivatives w.r.t. tunable parameters of the subtree rooted in the
      // first input equal to 0 since its output time is excluded from
      // computations.
      absl::flat_hash_map<string, std::shared_ptr<Parameter>>
          first_input_parameters;
      inputs_.front()->CollectTunableParameters(&first_input_parameters);
      for (auto& pair : first_input_parameters) {
        (*gradients)[pair.first] = 0.0L;
      }
      // Add derivative w.r.t. own parallelism parameter.
      if (parameter && (*parameter)->state->tunable) {
        (*gradients)[long_name()] =
            buffer_size_der - producer_time_der * producer_time / parallelism;
      }
    } else {
      wait_time = ComputeWaitTime(producer_time, consumer_time, parallelism,
                                  /*producer_time_derivative=*/nullptr,
                                  /*consumer_time_derivative=*/nullptr,
                                  /*buffer_size_derivative=*/nullptr);
    }
    output_time = self_processing_time + wait_time;
    (*output_times)[long_name()] = output_time;
  }

  // The processing time is the sum of the self processing time and the average
  // processing time of inputs comprising the interleave "cycle".
  void TotalProcessingTimeLocked(
      absl::flat_hash_map<string, double>* processing_times,
      absl::flat_hash_map<string, double>* total_processing_times) override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    double self_processing_time = SelfProcessingTimeLocked();
    if (processing_times) {
      (*processing_times)[long_name()] = self_processing_time;
    }
    if (num_inputs() <= 1) {
      (*total_processing_times)[long_name()] = self_processing_time;
      return;
    }
    double inputs_processing_time =
        (TotalProcessingTimeForInputs(*total_processing_times) -
         (*total_processing_times)[inputs_.front()->long_name()]) /
        static_cast<double>(num_inputs() - 1);
    (*total_processing_times)[long_name()] =
        self_processing_time + inputs_processing_time;
  }

  double MaximumBufferedBytes() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    double result = 0;
    auto* parameter = gtl::FindOrNull(parameters_, kParallelism);
    if (parameter) {
      result += (*parameter)->value * AverageBufferedElementSize();
    }
    return result;
  }

  Status ToProto(ModelProto::Node* node_proto) const {
    TF_RETURN_IF_ERROR(Node::ToProto(node_proto));
    node_proto->set_node_class(NodeClass::ASYNC_INTERLEAVE_MANY);
    return Status::OK();
  }
};

class KnownRatio : public Node {
 public:
  KnownRatio(Node::Args args, double ratio) : Node(args), ratio_(ratio) {}

  virtual ~KnownRatio() {}

 protected:
  std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    return std::make_shared<KnownRatio>(Args{id_, name_, std::move(output)},
                                        ratio_);
  }

  // The input time is the sum of inherited input time and self processing time,
  // divided by `ratio_`.
  void InputTimeLocked(absl::flat_hash_map<string, double>* input_times)
      const override TF_SHARED_LOCKS_REQUIRED(mu_) {
    double inherited_input_time;
    if (output_) {
      inherited_input_time = (*input_times)[output_->long_name()];
    } else {
      inherited_input_time = (*input_times)[kModelInputTimeKey];
    }

    if (ratio_ == 0) {
      (*input_times)[long_name()] = inherited_input_time;
      return;
    }
    double input_time =
        (inherited_input_time + SelfProcessingTimeLocked()) / ratio_;
    (*input_times)[long_name()] = input_time;
  }

  // The output time is the sum of the self processing time and the product of
  // `ratio_` and the sum of output times of inputs.
  void OutputTimeLocked(
      const absl::flat_hash_map<string, double>& input_times,
      absl::flat_hash_map<string, double>* gradients,
      absl::flat_hash_map<string, double>* output_times,
      absl::flat_hash_map<string, double>* output_time_gradients) const override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    double self_processing_time = SelfProcessingTimeLocked();
    if (ratio_ == 0) {
      (*output_times)[long_name()] = self_processing_time;
      if (gradients) {
        for (const auto& node :
             CollectNodes(TraversalOrder::REVERSE_BFS, IsAutotuneNode)) {
          gradients->erase(node->long_name());
        }
      }
      return;
    }
    if (gradients) {
      for (const auto& node :
           CollectNodes(TraversalOrder::REVERSE_BFS, IsAutotuneNode)) {
        auto* gradient = gtl::FindOrNull(*gradients, node->long_name());
        if (gradient) {
          *gradient *= ratio_;
        }
      }
      (*output_time_gradients)[long_name()] =
          OutputTimeGradientsForInputs(*output_time_gradients);
    }
    double inputs_output_time = ratio_ * OutputTimeForInputs(*output_times);
    (*output_times)[long_name()] = self_processing_time + inputs_output_time;
  }

  // The processing time is the sum of the self processing time and the product
  // of `ratio_` and the sum of processing times of inputs.
  void TotalProcessingTimeLocked(
      absl::flat_hash_map<string, double>* processing_times,
      absl::flat_hash_map<string, double>* total_processing_times) override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    double self_processing_time = SelfProcessingTimeLocked();
    if (processing_times) {
      (*processing_times)[long_name()] = self_processing_time;
    }
    if (ratio_ == 0) {
      (*total_processing_times)[long_name()] = self_processing_time;
      return;
    }
    double inputs_processing_time =
        ratio_ * TotalProcessingTimeForInputs(*total_processing_times);
    (*total_processing_times)[long_name()] =
        self_processing_time + inputs_processing_time;
  }

  Status ToProto(ModelProto::Node* node_proto) const {
    TF_RETURN_IF_ERROR(Node::ToProto(node_proto));
    node_proto->set_node_class(NodeClass::KNOWN_RATIO);
    node_proto->set_ratio(ratio_);
    return Status::OK();
  }

 private:
  const double ratio_;
};

class AsyncKnownRatio : public Node {
 public:
  AsyncKnownRatio(Node::Args args, double ratio, double memory_ratio,
                  std::vector<std::shared_ptr<Parameter>> parameters)
      : Node(args), ratio_(ratio), memory_ratio_(memory_ratio) {
    for (auto& parameter : parameters) {
      parameters_[parameter->name] = std::move(parameter);
    }
  }

  virtual ~AsyncKnownRatio() {}

 protected:
  std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    std::vector<std::shared_ptr<Parameter>> parameters;
    for (auto& pair : parameters_) {
      parameters.push_back(pair.second);
    }
    return std::make_shared<AsyncKnownRatio>(
        Args{id_, name_, std::move(output)}, ratio_, memory_ratio_, parameters);
  }

  // The input time is the sum of inherited input time and parallelism adjusted
  // self processing time, divided by `ratio_`.
  void InputTimeLocked(absl::flat_hash_map<string, double>* input_times)
      const override TF_SHARED_LOCKS_REQUIRED(mu_) {
    double inherited_input_time;
    if (output_) {
      inherited_input_time = (*input_times)[output_->long_name()];
    } else {
      inherited_input_time = (*input_times)[kModelInputTimeKey];
    }
    double parallelism = 1.0;
    auto* parallelism_parameter = gtl::FindOrNull(parameters_, kParallelism);
    if (parallelism_parameter) {
      parallelism = (*parallelism_parameter)->value;
    }

    if (ratio_ == 0.0) {
      (*input_times)[long_name()] =
          inherited_input_time + SelfProcessingTimeLocked() / parallelism;
      return;
    }
    double input_time =
        (inherited_input_time + SelfProcessingTimeLocked() / parallelism) /
        ratio_;
    (*input_times)[long_name()] = input_time;
  }

  // The output time is the sum of parallelism adjusted self processing time and
  // expected wait time from the buffer model estimated using
  // `ComputeWaitTime(producer_time, consumer_time, parallelism, ...)`, where
  // `producer_time` is the product of `ratio_` and the sum of output times of
  // inputs, `consumer_time` is the product of `ratio_` and the `input_time`
  // specified through `input_times` (since for each element stored in the
  // buffer, the inputs need to be called `ratio_` times), and if the node has
  // parallelism parameter, then `buffer_size` is derived from `parallelism`.
  //
  // Current implementation assumes that there is at most 1 parameter per node.
  void OutputTimeLocked(
      const absl::flat_hash_map<string, double>& input_times,
      absl::flat_hash_map<string, double>* gradients,
      absl::flat_hash_map<string, double>* output_times,
      absl::flat_hash_map<string, double>* output_time_gradients) const override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    double parallelism = 1.0;
    double buffer_size = 0.0;
    auto* parallelism_parameter = gtl::FindOrNull(parameters_, kParallelism);
    auto* buffer_size_parameter = gtl::FindOrNull(parameters_, kBufferSize);
    if (parallelism_parameter) {
      parallelism = (*parallelism_parameter)->value;
      if (ratio_ == 0) {
        buffer_size = parallelism;
      } else {
        // Currently, MapAndBatch is the only transformation creates
        // AsyncKnownRatio nodes with ratio >= 1. For MapAndBatch, we create
        // `parallelism` threads to apply the function on elements from input
        // dataset, while one element in the buffer actually corresponds to
        // `ratio_` elements from input dataset. So we adjust the `buffer_size`
        // by dividing `ratio_`.
        buffer_size = parallelism / ratio_;
      }
    } else if (buffer_size_parameter) {
      buffer_size = (*buffer_size_parameter)->value;
    }
    double self_processing_time = SelfProcessingTimeLocked();
    double output_time, wait_time, consumer_time, producer_time;
    double input_time = input_times.at(long_name());

    if (ratio_ == 0) {
      consumer_time = input_time;
      producer_time = 0.0L;
      if (gradients) {
        for (const auto& node :
             CollectNodes(TraversalOrder::REVERSE_BFS, IsAutotuneNode)) {
          gradients->erase(node->long_name());
        }

        double producer_time_der = 0.0L;
        double consumer_time_der = 0.0L;
        double buffer_size_der = 0.0L;
        wait_time = ComputeWaitTime(producer_time, consumer_time, buffer_size,
                                    &producer_time_der, &consumer_time_der,
                                    &buffer_size_der);
        (*output_time_gradients)[long_name()] = consumer_time_der;
        if (parallelism_parameter && (*parallelism_parameter)->state->tunable) {
          (*gradients)[long_name()] = -(1.0L + consumer_time_der) *
                                          self_processing_time /
                                          Square(parallelism) +
                                      buffer_size_der;
        } else if (buffer_size_parameter &&
                   (*buffer_size_parameter)->state->tunable) {
          (*gradients)[long_name()] = buffer_size_der;
        }
      } else {
        wait_time = ComputeWaitTime(producer_time, consumer_time, buffer_size,
                                    /*producer_time_derivative=*/nullptr,
                                    /*consumer_time_derivative=*/nullptr,
                                    /*buffer_size_derivative=*/nullptr);
      }
      output_time = self_processing_time / parallelism + wait_time;
      (*output_times)[long_name()] = output_time;
      return;
    }

    consumer_time = input_time * ratio_;
    producer_time = ratio_ * OutputTimeForInputs(*output_times);
    if (gradients) {
      double producer_time_der = 0.0L;
      double consumer_time_der = 0.0L;
      double buffer_size_der = 0.0L;
      wait_time = ComputeWaitTime(producer_time, consumer_time, buffer_size,
                                  &producer_time_der, &consumer_time_der,
                                  &buffer_size_der);
      double inputs_time_der_sum =
          OutputTimeGradientsForInputs(*output_time_gradients);
      (*output_time_gradients)[long_name()] =
          consumer_time_der + producer_time_der * inputs_time_der_sum;

      for (const auto& node :
           CollectNodes(TraversalOrder::REVERSE_BFS, IsAutotuneNode)) {
        auto* gradient = gtl::FindOrNull(*gradients, node->long_name());
        if (gradient) {
          *gradient *= (ratio_ * producer_time_der);
        }
      }

      // Add derivative w.r.t. own parameter if it's tunable.
      if (parallelism_parameter && (*parallelism_parameter)->state->tunable) {
        (*gradients)[long_name()] = buffer_size_der / ratio_ -
                                    (1.0L + consumer_time_der +
                                     producer_time_der * inputs_time_der_sum) *
                                        self_processing_time /
                                        Square(parallelism);
      } else if (buffer_size_parameter &&
                 (*buffer_size_parameter)->state->tunable) {
        (*gradients)[long_name()] = buffer_size_der;
      }
    } else {
      wait_time = ComputeWaitTime(producer_time, consumer_time, buffer_size,
                                  /*producer_time_derivative=*/nullptr,
                                  /*consumer_time_derivative=*/nullptr,
                                  /*buffer_size_derivative=*/nullptr);
    }
    output_time = self_processing_time / parallelism + wait_time;
    (*output_times)[long_name()] = output_time;
  }

  // The processing time is the sum of the self processing time and the product
  // of `ratio_` and the sum of processing times of inputs.
  void TotalProcessingTimeLocked(
      absl::flat_hash_map<string, double>* processing_times,
      absl::flat_hash_map<string, double>* total_processing_times) override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    double self_processing_time = SelfProcessingTimeLocked();
    if (processing_times) {
      (*processing_times)[long_name()] = self_processing_time;
    }
    if (ratio_ == 0) {
      (*total_processing_times)[long_name()] = self_processing_time;
      return;
    }
    double inputs_processing_time =
        ratio_ * TotalProcessingTimeForInputs(*total_processing_times);
    (*total_processing_times)[long_name()] =
        self_processing_time + inputs_processing_time;
  }

  double MaximumBufferedBytes() const TF_SHARED_LOCKS_REQUIRED(mu_) {
    double result = 0;
    auto* parameter = gtl::FindOrNull(parameters_, kBufferSize);
    if (!parameter) {
      parameter = gtl::FindOrNull(parameters_, kParallelism);
    }

    if (parameter) {
      if (memory_ratio_ == 0) {
        result += (*parameter)->value * AverageBufferedElementSize();
      } else {
        // The estimation is currently not accurate for MapAndBatchDataset for
        // the maximum buffer size does not match `num_parallel_calls`
        // parameter.
        result +=
            (*parameter)->value * AverageBufferedElementSize() / memory_ratio_;
      }
    }
    return result;
  }

  Status ToProto(ModelProto::Node* node_proto) const {
    TF_RETURN_IF_ERROR(Node::ToProto(node_proto));
    node_proto->set_node_class(NodeClass::ASYNC_KNOWN_RATIO);
    node_proto->set_ratio(ratio_);
    node_proto->set_memory_ratio(memory_ratio_);
    return Status::OK();
  }

 private:
  // Identifies how many input elements need to be created to construct an
  // element for the dataset.
  //
  // Currently the value is 1 for PrefetchDataset and ParallelMapDataset,
  // batch_size for MapAndBatchDataset and ParallelBatchDataset.
  const double ratio_;
  // For parallelism nodes, identifies how many parallelism calls are introduced
  // by one buffered element. The value is defined to correctly estimate RAM
  // budget bound with given num_parallel_calls (or buffer_size) combined with
  // the estimated average size of buffered elements.
  const double memory_ratio_;
};

class UnknownRatio : public Node {
 public:
  using Node::Node;

  virtual ~UnknownRatio() {}

 protected:
  std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    return std::make_shared<UnknownRatio>(Args{id_, name_, std::move(output)});
  }

  // The input time is the sum of inherited input time and self processing time,
  // divided by the ratio estimate.
  void InputTimeLocked(absl::flat_hash_map<string, double>* input_times)
      const override TF_SHARED_LOCKS_REQUIRED(mu_) {
    double inherited_input_time;
    if (output_) {
      inherited_input_time = (*input_times)[output_->long_name()];
    } else {
      inherited_input_time = (*input_times)[kModelInputTimeKey];
    }

    if (num_elements_ == 0 || inputs_.empty() ||
        inputs_.front()->num_elements() == 0) {
      (*input_times)[long_name()] = inherited_input_time;
      return;
    }
    std::shared_ptr<Node> input = inputs_.front();
    double ratio = static_cast<double>(input->num_elements()) /
                   static_cast<double>(num_elements_);
    double input_time =
        (inherited_input_time + SelfProcessingTimeLocked()) / ratio;
    (*input_times)[long_name()] = input_time;
  }

  // The output time is the sum of the self processing time and the product of
  // the ratio estimate and the sum of output times of inputs.
  void OutputTimeLocked(
      const absl::flat_hash_map<string, double>& input_times,
      absl::flat_hash_map<string, double>* gradients,
      absl::flat_hash_map<string, double>* output_times,
      absl::flat_hash_map<string, double>* output_time_gradients) const override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    double self_processing_time = SelfProcessingTimeLocked();
    if (num_elements_ == 0 || inputs_.empty() ||
        inputs_.front()->num_elements() == 0) {
      (*output_times)[long_name()] = self_processing_time;
      if (gradients) {
        for (const auto& node :
             CollectNodes(TraversalOrder::REVERSE_BFS, IsAutotuneNode)) {
          gradients->erase(node->long_name());
        }
      }
      return;
    }
    // TODO(jsimsa): The current implementation assumes that the number of input
    // elements consumed per output is the same across all inputs.
    double ratio = static_cast<double>(inputs_.front()->num_elements()) /
                   static_cast<double>(num_elements_);
    if (gradients) {
      for (const auto& node :
           CollectNodes(TraversalOrder::REVERSE_BFS, IsAutotuneNode)) {
        auto* gradient = gtl::FindOrNull(*gradients, node->long_name());
        if (gradient) {
          *gradient *= ratio;
        }
      }
      (*output_time_gradients)[long_name()] =
          OutputTimeGradientsForInputs(*output_time_gradients);
    }
    double inputs_output_time = ratio * OutputTimeForInputs(*output_times);
    (*output_times)[long_name()] = self_processing_time + inputs_output_time;
  }

  // The processing time is the sum of the self processing time and the product
  // of the ratio estimate and the sum of processing times of inputs.
  void TotalProcessingTimeLocked(
      absl::flat_hash_map<string, double>* processing_times,
      absl::flat_hash_map<string, double>* total_processing_times) override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    double self_processing_time = SelfProcessingTimeLocked();
    if (processing_times) {
      (*processing_times)[long_name()] = self_processing_time;
    }
    if (inputs_.empty() || num_elements_ == 0) {
      (*total_processing_times)[long_name()] = self_processing_time;
      return;
    }
    // TODO(jsimsa): The current implementation assumes that the number of input
    // elements consumed per output is the same across all inputs.
    std::shared_ptr<Node> input = inputs_.front();
    double ratio = static_cast<double>(input->num_elements()) /
                   static_cast<double>(num_elements_);
    double inputs_processing_time =
        ratio * TotalProcessingTimeForInputs(*total_processing_times);
    (*total_processing_times)[long_name()] =
        self_processing_time + inputs_processing_time;
  }

  Status ToProto(ModelProto::Node* node_proto) const {
    TF_RETURN_IF_ERROR(Node::ToProto(node_proto));
    node_proto->set_node_class(NodeClass::UNKNOWN_RATIO);
    return Status::OK();
  }
};

class Unknown : public Node {
 public:
  using Node::Node;

  virtual ~Unknown() {}

 protected:
  std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    return std::make_shared<Unknown>(Args{id_, name_, std::move(output)});
  }

  // The input time is the inherited input time.
  void InputTimeLocked(absl::flat_hash_map<string, double>* input_times)
      const override TF_SHARED_LOCKS_REQUIRED(mu_) {
    double inherited_input_time;
    if (output_) {
      inherited_input_time = (*input_times)[output_->long_name()];
    } else {
      inherited_input_time = (*input_times)[kModelInputTimeKey];
    }
    (*input_times)[long_name()] = inherited_input_time;
  }

  // The output time is the sum of output times of inputs.
  void OutputTimeLocked(
      const absl::flat_hash_map<string, double>& input_times,
      absl::flat_hash_map<string, double>* gradients,
      absl::flat_hash_map<string, double>* output_times,
      absl::flat_hash_map<string, double>* output_time_gradients) const override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    (*output_times)[long_name()] = OutputTimeForInputs(*output_times);
    if (gradients) {
      (*output_time_gradients)[long_name()] =
          OutputTimeGradientsForInputs(*output_time_gradients);
    }
  }

  // The processing time is the sum of processing times of inputs.
  void TotalProcessingTimeLocked(
      absl::flat_hash_map<string, double>* processing_times,
      absl::flat_hash_map<string, double>* total_processing_times) override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    if (processing_times) {
      (*processing_times)[long_name()] = SelfProcessingTimeLocked();
    }
    (*total_processing_times)[long_name()] =
        TotalProcessingTimeForInputs(*total_processing_times);
  }

  Status ToProto(ModelProto::Node* node_proto) const {
    TF_RETURN_IF_ERROR(Node::ToProto(node_proto));
    node_proto->set_node_class(NodeClass::UNKNOWN);
    return Status::OK();
  }
};

}  // namespace

thread_local int64 Node::work_start_;

std::shared_ptr<Parameter> MakeParameter(const string& name,
                                         std::shared_ptr<SharedState> state,
                                         double min, double max) {
  return std::make_shared<Parameter>(name, state, min, max);
}

std::shared_ptr<Node> MakeInterleaveManyNode(Node::Args args) {
  return std::make_shared<InterleaveMany>(std::move(args));
}

std::shared_ptr<Node> MakeAsyncInterleaveManyNode(
    Node::Args args, std::vector<std::shared_ptr<Parameter>> parameters) {
  return std::make_shared<AsyncInterleaveMany>(std::move(args),
                                               std::move(parameters));
}

std::shared_ptr<Node> MakeKnownRatioNode(Node::Args args, double ratio) {
  return std::make_shared<KnownRatio>(std::move(args), ratio);
}

std::shared_ptr<Node> MakeAsyncKnownRatioNode(
    Node::Args args, double ratio, double memory_ratio,
    std::vector<std::shared_ptr<Parameter>> parameters) {
  return std::make_shared<AsyncKnownRatio>(std::move(args), ratio, memory_ratio,
                                           std::move(parameters));
}

std::shared_ptr<Node> MakeAsyncKnownRatioNode(
    Node::Args args, double ratio,
    std::vector<std::shared_ptr<Parameter>> parameters) {
  return MakeAsyncKnownRatioNode(std::move(args), /*ratio=*/ratio,
                                 /*memory_ratio=*/ratio, std::move(parameters));
}

std::shared_ptr<Node> MakeSourceNode(Node::Args args) {
  return MakeKnownRatioNode(std::move(args), 0);
}

std::shared_ptr<Node> MakeUnknownRatioNode(Node::Args args) {
  return std::make_shared<UnknownRatio>(std::move(args));
}

std::shared_ptr<Node> MakeUnknownNode(Node::Args args) {
  return std::make_shared<Unknown>(std::move(args));
}

double Node::ComputeWaitTime(const double& producer_time,
                             const double& consumer_time,
                             const double& buffer_size,
                             double* producer_time_derivative,
                             double* consumer_time_derivative,
                             double* buffer_size_derivative) {
  // If we set x=`consumer_time`, y=`producer_time`, n=`buffer_size`,
  // p=`p_buffer_empty`, T=`wait_time`, then we have:
  // if y = 0, then p = 0;
  // elif x = 0, then p = 1;
  // elif x = y, then p = 1 / (n+1);
  // else p = [1 - x/y] / [1 - power(x/y, n+1)].
  //
  // We also have T = p * y, and derivatives of T w.r.t. x, y, n are computed:
  // dT/dx = dp/dx * y,
  // dT/dy = p + dp/dy * y,
  // dT/dn = dp/dn * y.
  // Then the remaining work is to compute dp/dx, dp/dy, dp/dn by considering
  // different cases and substitute the values into above formulas.

  // Case 1: if producer is infinitely fast. The buffer will always be full.
  // Wait time will always be 0.
  if (producer_time == 0) {
    if (producer_time_derivative) {
      // Note a common error is `*producer_time_derivative = 0` since p=0 on the
      // line y=0 doesn't imply dp/dy = 0 there. Actually to compute dp/dy at
      // (x,0), we need to consider lim_{dy->0+} [p(x,dy)-p(x,0)] / dy, where
      // p(x,0)=0 and p(x,dy) = [1 - x/dy] / [1 - power(x/dy, n+1)].
      if (buffer_size == 0 || consumer_time == 0) {
        *producer_time_derivative = 1.0L;
      } else {
        *producer_time_derivative = 0.0L;
      }
    }
    if (consumer_time_derivative) {
      *consumer_time_derivative = 0.0L;
    }
    if (buffer_size_derivative) {
      *buffer_size_derivative = 0.0L;
    }
    return 0.0L;
  }

  // Case 2: if consumer is infinitely fast. Wait time is always the time to
  // produce an output.
  if (consumer_time == 0) {
    if (producer_time_derivative) {
      *producer_time_derivative = 1.0L;
    }
    if (consumer_time_derivative) {
      // Note a common error is `*consumer_time_derivative = 0` since p=1 on the
      // line x=0 doesn't imply dp/dx = 0 there. Actually to compute dp/dx at
      // (0,y), we need to consider lim_{dx->0+} [p(dx,y)-p(0,y)] / dx, where
      // p(0,y)=1, p(dx,y) = [1 - dx/y] / [1 - power(dx/y, n+1)] if y!=0.
      if (buffer_size == 0) {
        *consumer_time_derivative = 0.0L;
      } else {
        *consumer_time_derivative = -1.0L;
      }
    }
    if (buffer_size_derivative) {
      *buffer_size_derivative = 0.0L;
    }
    return producer_time;
  }

  // Case 3: the consumer and the producer are equally fast. Expected wait time
  // decreases linearly with the size of the buffer.
  if (consumer_time == producer_time) {
    const double p_buffer_empty = 1.0L / (buffer_size + 1.0L);
    const double p_buffer_empty_der =
        -buffer_size / (2.0L * buffer_size + 2.0L);
    if (producer_time_derivative) {
      // Note a common error is `*producer_time_derivative = p_buffer_empty`
      // since p=1/(n+1) on the line x=y doesn't imply dp/dy = 0 there. Actually
      // to compute dp/dy at (y,y), we need to consider
      // lim_{dy->0} [p(y,y+dy)-p(y,y)] / dy, where p(y,y)=1/(n+1),
      // p(y,y+dy) = [1 - y/(y+dy)] / [1 - power(y/(y+dy), n+1)].
      *producer_time_derivative = p_buffer_empty - p_buffer_empty_der;
    }
    if (consumer_time_derivative) {
      // Note a common error is `*consumer_time_derivative = 0` since
      // p=1/(n+1) on the line x=y doesn't imply dp/dx = 0 there. Actually to
      // compute dp/dx at (x,x), we need to consider
      // lim_{dx->0} [p(x+dx,x)-p(x,x)] / dx, where p(x,x)=1/(n+1),
      // p(x+dx,x) = [1 - (x+dx)/x] / [1 - power((x+dx)/x, n+1)].
      *consumer_time_derivative = p_buffer_empty_der;
    }
    if (buffer_size_derivative) {
      *buffer_size_derivative = -producer_time / Square(buffer_size + 1.0L);
    }
    return p_buffer_empty * producer_time;
  }

  // Case 4: the consumer is slower than the producer and neither is infinitely
  // fast. Case 4 and Case 5 actually follow same formula. Separate them for
  // numerical computation reasons.
  if (consumer_time > producer_time) {
    const double ratio = producer_time / consumer_time;
    const double ratio_pow = std::pow(ratio, buffer_size);
    const double p_buffer_empty =
        ratio_pow * (1.0L - ratio) / (1.0L - ratio * ratio_pow);
    const double p_buffer_empty_der =
        (buffer_size - (buffer_size + 1.0L) * ratio + ratio_pow * ratio) *
        ratio_pow / ratio / Square(1.0L - ratio_pow * ratio);
    if (producer_time_derivative) {
      *producer_time_derivative = p_buffer_empty + p_buffer_empty_der * ratio;
    }
    if (consumer_time_derivative) {
      *consumer_time_derivative = -p_buffer_empty_der * Square(ratio);
    }
    if (buffer_size_derivative) {
      *buffer_size_derivative = p_buffer_empty / (1.0L - ratio_pow * ratio) *
                                std::log(ratio) * producer_time;
    }
    return p_buffer_empty * producer_time;
  }

  // Case 5: the producer is slower than the consumer and neither is infinitely
  // fast.
  const double ratio = consumer_time / producer_time;
  const double ratio_pow = std::pow(ratio, buffer_size);
  const double p_buffer_empty = (1.0L - ratio) / (1.0L - ratio_pow * ratio);
  const double p_buffer_empty_der =
      ((buffer_size + 1.0L - buffer_size * ratio) * ratio_pow - 1.0L) /
      Square(1.0L - ratio_pow * ratio);
  if (producer_time_derivative) {
    *producer_time_derivative = p_buffer_empty - p_buffer_empty_der * ratio;
  }
  if (consumer_time_derivative) {
    *consumer_time_derivative = p_buffer_empty_der;
  }
  if (buffer_size_derivative) {
    *buffer_size_derivative = p_buffer_empty / (1.0L - ratio_pow * ratio) *
                              ratio_pow * ratio * std::log(ratio) *
                              producer_time;
  }
  return p_buffer_empty * producer_time;
}

void Node::CollectTunableParameters(
    absl::flat_hash_map<string, std::shared_ptr<Parameter>>* parameters) const {
  tf_shared_lock l(mu_);
  // Collect tunable parameters from the leaves of the nodes tree to the root.
  for (const auto& node :
       CollectNodes(TraversalOrder::REVERSE_BFS, IsAutotuneNode)) {
    tf_shared_lock l(node->mu_);
    node->CollectTunableParametersHelper(parameters);
  }
  CollectTunableParametersHelper(parameters);
}

string Node::DebugString() const {
  absl::flat_hash_map<string, string> debug_strings;
  tf_shared_lock l(mu_);
  // Build up the debug string from the leaves of the nodes tree to the root.
  for (const auto& node :
       CollectNodes(TraversalOrder::REVERSE_BFS, IsAnyNode)) {
    tf_shared_lock l(node->mu_);
    node->DebugStringHelper(&debug_strings);
  }
  DebugStringHelper(&debug_strings);

  return debug_strings[long_name()];
}

void Node::FlushMetrics() {
  if (!record_metrics_) {
    return;
  }
  metrics_.record_bytes_consumed(bytes_consumed_);
  metrics_.record_bytes_produced(bytes_produced_);
  metrics_.record_num_elements(num_elements_);
}

double Node::OutputTime(absl::flat_hash_map<string, double>* input_times,
                        absl::flat_hash_map<string, double>* gradients) const {
  // To store the output time gradient w.r.t. input time (if `gradients` is not
  // `nullptr`) and the output time for each node.
  absl::flat_hash_map<string, double> output_time_gradients, output_times;
  tf_shared_lock l(mu_);
  auto nodes = CollectNodes(TraversalOrder::BFS, IsAutotuneNode);

  // Computes and stores input time for each node from the root to leaves of the
  // nodes tree.
  InputTimeLocked(input_times);
  for (const auto& node : nodes) {
    tf_shared_lock l(node->mu_);
    node->InputTimeLocked(input_times);
  }

  std::reverse(nodes.begin(), nodes.end());
  // Computes and stores the output time and output time gradient w.r.t. input
  // time (if `gradients` is not `nullptr`) for each node from leaves of the
  // nodes tree to the root.
  for (const auto& node : nodes) {
    tf_shared_lock l(node->mu_);
    node->OutputTimeLocked(*input_times, gradients, &output_times,
                           &output_time_gradients);
  }
  OutputTimeLocked(*input_times, gradients, &output_times,
                   &output_time_gradients);

  return output_times[long_name()];
}

std::shared_ptr<Node> Node::Snapshot() const {
  NodePairList node_pairs;
  auto result = SnapshotHelper(nullptr, &node_pairs);

  while (!node_pairs.empty()) {
    auto node_pair = node_pairs.front();
    node_pairs.pop_front();
    std::shared_ptr<Node> current = node_pair.first,
                          cloned_output = node_pair.second;
    cloned_output->add_input(
        current->SnapshotHelper(cloned_output, &node_pairs));
  }
  return result;
}

double Node::SelfProcessingTime() const {
  tf_shared_lock l(mu_);
  return SelfProcessingTimeLocked();
}

double Node::TotalBufferedBytes() const {
  absl::flat_hash_map<string, double> total_bytes;
  tf_shared_lock l(mu_);
  // Compute total buffered bytes from the leaves of the nodes tree to the root.
  for (const auto& node :
       CollectNodes(TraversalOrder::REVERSE_BFS, IsAnyNode)) {
    tf_shared_lock l(node->mu_);
    node->TotalBufferedBytesHelper(&total_bytes);
  }
  TotalBufferedBytesHelper(&total_bytes);

  return total_bytes[long_name()];
}

double Node::TotalMaximumBufferedBytes() const {
  absl::flat_hash_map<string, double> total_bytes;
  tf_shared_lock l(mu_);
  // Compute total maximum buffered bytes from the leaves of the nodes tree
  // to the root.
  for (const auto& node :
       CollectNodes(TraversalOrder::REVERSE_BFS, IsAnyNode)) {
    tf_shared_lock l(node->mu_);
    node->TotalMaximumBufferedBytesHelper(&total_bytes);
  }
  TotalMaximumBufferedBytesHelper(&total_bytes);

  return total_bytes[long_name()];
}

double Node::TotalProcessingTime(
    absl::flat_hash_map<string, double>* processing_times) {
  // Create a hash map to store the per-element CPU time spent in the subtree
  // rooted in each node.
  absl::flat_hash_map<string, double> total_processing_times;
  tf_shared_lock l(mu_);

  // Computes per-element CPU time spent in the subtree rooted in the node from
  // the leaves of the nodes tree to the root.
  for (const auto& node :
       CollectNodes(TraversalOrder::REVERSE_BFS, IsAutotuneNode)) {
    tf_shared_lock l(node->mu_);
    node->TotalProcessingTimeLocked(processing_times, &total_processing_times);
  }
  TotalProcessingTimeLocked(processing_times, &total_processing_times);

  return total_processing_times[long_name()];
}

double Node::AverageBufferedElementSize() const {
  DCHECK_GE(num_elements_, 0);
  DCHECK_GE(buffered_elements_, 0);
  if (num_elements_ <= 0) {
    if (buffered_elements_ <= 0) {
      // If there are no produced elements or buffered elements recorded, return
      // 0.
      return 0;
    }
    // If there are no produced elements but some buffered elements, return the
    // average size of all buffered elements.
    return static_cast<double>(buffered_bytes_) /
           static_cast<double>(buffered_elements_);
  }

  if (buffered_elements_ <= 0) {
    // If there are no buffered elements but some produced elements, return the
    // average size of all produced elements.
    return static_cast<double>(bytes_produced_) /
           static_cast<double>(num_elements_);
  }

  // Otherwise, return the mean value of average size of all produced elements
  // and average size of all buffered elements.
  return (static_cast<double>(bytes_produced_) /
              static_cast<double>(num_elements_) +
          static_cast<double>(buffered_bytes_) /
              static_cast<double>(buffered_elements_)) /
         2.0;
}

double Node::OutputTimeForInputs(
    const absl::flat_hash_map<string, double>& output_times) const {
  double sum = 0;
  for (auto& input : inputs_) {
    // Inputs for which autotuning is disabled are excluded.
    if (input->autotune()) {
      sum += output_times.at(input->long_name());
    }
  }
  return sum;
}

double Node::OutputTimeGradientsForInputs(
    const absl::flat_hash_map<string, double>& output_time_gradients) const {
  double sum = 0;
  for (auto& input : inputs_) {
    // Inputs for which autotuning is disabled are excluded.
    if (input->autotune()) {
      sum +=
          gtl::FindWithDefault(output_time_gradients, input->long_name(), 0.0L);
    }
  }
  return sum;
}

double Node::TotalProcessingTimeForInputs(
    const absl::flat_hash_map<string, double>& total_processing_times) {
  // If the number of elements produced by an input is smaller than this
  // constant, then its processing time is estimated using a weighted average
  // of the empirical processing time and processing time history.
  constexpr int kNumElementsThreshold = 30;

  // Identifies the minimum number of input processing times to collect
  // before the processing time history is used as a prior.
  constexpr int kCountThreshold = 30;

  double sum = 0;
  for (auto& input : inputs_) {
    // Inputs for which autotuning is disabled are excluded.
    if (input->autotune()) {
      double input_processing_time =
          total_processing_times.at(input->long_name());
      int64 num_elements = input->num_elements();
      if (num_elements < kNumElementsThreshold) {
        if (input_processing_time_count_ < kCountThreshold) {
          sum += input_processing_time;
        } else {
          // The fewer elements the input has produced so far, the more weight
          // is assigned to the prior to reduce volatility.
          double prior_weight = 1.0L / static_cast<double>(2 << num_elements);
          double prior =
              input_processing_time_sum_ / input_processing_time_count_;
          sum += (1.0L - prior_weight) * input_processing_time +
                 prior_weight * prior;
        }
      } else {
        sum += input_processing_time;
        input_processing_time_count_++;
        input_processing_time_sum_ += input_processing_time;
      }
    }
  }
  return sum;
}

double Node::SelfProcessingTimeLocked() const {
  if (num_elements_ == 0) {
    return 0;
  }
  return static_cast<double>(processing_time_) /
         static_cast<double>(num_elements_);
}

Node::NodeVector Node::CollectNodes(
    TraversalOrder order, bool collect_node(const std::shared_ptr<Node>)) const
    TF_SHARED_LOCKS_REQUIRED(mu_) {
  NodeVector node_vector;
  std::list<std::shared_ptr<Node>> temp_list;

  for (auto& input : inputs_) {
    if (collect_node(input)) {
      node_vector.push_back(input);
      temp_list.push_back(input);
    }
  }

  while (!temp_list.empty()) {
    auto cur_node = temp_list.front();
    temp_list.pop_front();
    tf_shared_lock l(cur_node->mu_);
    for (auto& input : cur_node->inputs_) {
      if (collect_node(input)) {
        node_vector.push_back(input);
        temp_list.push_back(input);
      }
    }
  }

  if (order == TraversalOrder::REVERSE_BFS) {
    std::reverse(node_vector.begin(), node_vector.end());
  }
  return node_vector;
}

void Node::CollectTunableParametersHelper(
    absl::flat_hash_map<string, std::shared_ptr<Parameter>>* parameters) const
    TF_SHARED_LOCKS_REQUIRED(mu_) {
  // If autotune is turned off or there are no elements recorded, we don't
  // collect the parameters on the node.
  if (!autotune_ || num_elements_ <= 0) {
    return;
  }
  for (auto& pair : parameters_) {
    if (pair.second->state->tunable) {
      parameters->insert(std::make_pair(long_name(), pair.second));
    }
  }
}

void Node::DebugStringHelper(absl::flat_hash_map<string, string>* debug_strings)
    const TF_SHARED_LOCKS_REQUIRED(mu_) {
  string result;
  strings::StrAppend(&result, long_name(), ":\n");
  strings::StrAppend(&result, "  autotune=", autotune_.load(), "\n");
  strings::StrAppend(&result, "  buffered_bytes=", buffered_bytes_.load(),
                     "\n");
  strings::StrAppend(&result, "  buffered_elements=", buffered_elements_.load(),
                     "\n");
  strings::StrAppend(&result, "  bytes_consumed=", bytes_consumed_.load(),
                     "\n");
  strings::StrAppend(&result, "  bytes_produced=", bytes_produced_.load(),
                     "\n");
  strings::StrAppend(&result, "  processing_time=", processing_time_.load(),
                     "\n");
  strings::StrAppend(&result, "  num_elements=", num_elements_.load(), "\n");
  string inputs;
  for (auto& input : inputs_) {
    strings::StrAppend(&inputs, input->long_name(), ",");
  }
  strings::StrAppend(&result, "  inputs={", inputs, "}\n");
  for (auto& input : inputs_) {
    strings::StrAppend(&result, debug_strings->at(input->long_name()));
  }
  debug_strings->insert(std::make_pair(long_name(), result));
}

std::shared_ptr<Node> Node::SnapshotHelper(
    std::shared_ptr<Node> cloned_output, Node::NodePairList* node_pairs) const {
  tf_shared_lock l(mu_);

  // Clone current node(`this`), also set clone of its output node
  // (`cloned_output`) to be the output node of the cloned node
  // (`cloned_current`).
  std::shared_ptr<Node> cloned_current = Clone(cloned_output);
  {
    cloned_current->autotune_.store(autotune_);
    cloned_current->buffered_bytes_.store(buffered_bytes_);
    cloned_current->buffered_elements_.store(buffered_elements_);
    cloned_current->bytes_consumed_.store(bytes_consumed_);
    cloned_current->bytes_produced_.store(bytes_produced_);
    cloned_current->num_elements_.store(num_elements_);
    cloned_current->record_metrics_.store(false);
    cloned_current->processing_time_.store(processing_time_);
    mutex_lock l2(cloned_current->mu_);
    cloned_current->parameters_ = parameters_;
  }

  for (auto& input : inputs_) {
    node_pairs->push_back(std::make_pair(input, cloned_current));
  }
  return cloned_current;
}

void Node::TotalBufferedBytesHelper(
    absl::flat_hash_map<string, double>* total_bytes) const
    TF_SHARED_LOCKS_REQUIRED(mu_) {
  if (!autotune_) {
    total_bytes->insert(std::make_pair(long_name(), 0));
    return;
  }

  double result = 0;
  auto* parameter = gtl::FindOrNull(parameters_, kBufferSize);
  if (!parameter) {
    parameter = gtl::FindOrNull(parameters_, kParallelism);
  }
  if (parameter) {
    result = buffered_bytes_;
  }
  for (auto& input : inputs_) {
    result += total_bytes->at(input->long_name());
  }
  total_bytes->insert(std::make_pair(long_name(), result));
}

void Node::TotalMaximumBufferedBytesHelper(
    absl::flat_hash_map<string, double>* total_bytes) const
    TF_SHARED_LOCKS_REQUIRED(mu_) {
  if (!autotune_) {
    total_bytes->insert(std::make_pair(long_name(), 0));
    return;
  }

  double result = MaximumBufferedBytes();
  for (auto& input : inputs_) {
    result += total_bytes->at(input->long_name());
  }
  total_bytes->insert(std::make_pair(long_name(), result));
}

double Node::MaximumBufferedBytes() const TF_SHARED_LOCKS_REQUIRED(mu_) {
  return 0;
}

Status Node::ToProto(ModelProto::Node* node_proto) const {
  tf_shared_lock l(mu_);
  node_proto->set_id(id_);
  node_proto->set_name(name_);
  node_proto->set_autotune(autotune_);
  node_proto->set_buffered_bytes(buffered_bytes_);
  node_proto->set_buffered_elements(buffered_elements_);
  node_proto->set_bytes_consumed(bytes_consumed_);
  node_proto->set_bytes_produced(bytes_produced_);
  node_proto->set_num_elements(num_elements_);
  node_proto->set_processing_time(processing_time_);
  node_proto->set_record_metrics(record_metrics_);

  // Produce protos for all parameters.
  for (auto const& parameter : parameters_) {
    ModelProto::Node::Parameter* parameter_proto = node_proto->add_parameters();
    parameter_proto->set_name(parameter.first);
    parameter_proto->set_value(parameter.second->value);
    parameter_proto->set_min(parameter.second->min);
    parameter_proto->set_max(parameter.second->max);
    parameter_proto->set_state_value(parameter.second->state->value);
    parameter_proto->set_tunable(parameter.second->state->tunable);
  }

  // Produce protos for all inputs.
  for (auto const& input : inputs_) {
    ModelProto::Node* input_proto = node_proto->add_inputs();
    TF_RETURN_IF_ERROR(input->ToProto(input_proto));
  }
  return Status::OK();
}

Status Node::FromProtoHelper(ModelProto::Node node_proto,
                             std::shared_ptr<Node> node) {
  tf_shared_lock l(node->mu_);
  node->autotune_.store(node_proto.autotune());
  node->buffered_bytes_.store(node_proto.buffered_bytes());
  node->buffered_elements_.store(node_proto.buffered_elements());
  node->bytes_consumed_.store(node_proto.bytes_consumed());
  node->bytes_produced_.store(node_proto.bytes_produced());
  node->num_elements_.store(node_proto.num_elements());
  node->processing_time_.store(node_proto.processing_time());
  node->record_metrics_.store(node_proto.record_metrics());

  // Restore parameters.
  int64 num_parameters = node_proto.parameters_size();
  for (int i = 0; i < num_parameters; i++) {
    const ModelProto::Node::Parameter& parameter_proto =
        node_proto.parameters(i);
    std::shared_ptr<SharedState> state;
    if (parameter_proto.tunable()) {
      state =
          std::make_shared<SharedState>(kAutotune, std::make_shared<mutex>(),
                                        std::make_shared<condition_variable>());
      state->value = parameter_proto.state_value();
    } else {
      state = std::make_shared<SharedState>(
          parameter_proto.state_value(), std::make_shared<mutex>(),
          std::make_shared<condition_variable>());
    }
    node->parameters_[parameter_proto.name()] =
        MakeParameter(parameter_proto.name(), state, parameter_proto.min(),
                      parameter_proto.max());
  }
  return Status::OK();
}

Status Node::FromProto(ModelProto::Node node_proto,
                       std::shared_ptr<Node> output,
                       std::shared_ptr<Node>* node) {
  // Note that parameters are restored in `FromProtoHelper`.
  Args args = {node_proto.id(), node_proto.name(), std::move(output)};
  std::shared_ptr<Node> restored_node;
  switch (node_proto.node_class()) {
    case NodeClass::INTERLEAVE_MANY:
      restored_node = std::make_shared<InterleaveMany>(args);
      break;
    case NodeClass::ASYNC_INTERLEAVE_MANY:
      restored_node = std::make_shared<AsyncInterleaveMany>(
          args, /*parameters=*/std::vector<std::shared_ptr<Parameter>>());
      break;
    case NodeClass::KNOWN_RATIO:
      restored_node = std::make_shared<KnownRatio>(args, node_proto.ratio());
      break;
    case NodeClass::ASYNC_KNOWN_RATIO:
      restored_node = std::make_shared<AsyncKnownRatio>(
          args, node_proto.ratio(), node_proto.memory_ratio(),
          /*parameters=*/std::vector<std::shared_ptr<Parameter>>());
      break;
    case NodeClass::UNKNOWN_RATIO:
      restored_node = std::make_shared<UnknownRatio>(args);
      break;
    default:
      restored_node = std::make_shared<Unknown>(args);
  }
  TF_RETURN_IF_ERROR(FromProtoHelper(node_proto, restored_node));

  // Restore all input nodes as well.
  int64 num_inputs = node_proto.inputs_size();
  for (int64 i = 0; i < num_inputs; i++) {
    const ModelProto::Node& input_proto = node_proto.inputs(i);
    std::shared_ptr<Node> input;
    TF_RETURN_IF_ERROR(FromProto(input_proto, restored_node, &input));
    restored_node->add_input(input);
  }
  (*node) = std::move(restored_node);
  return Status::OK();
}

void Model::AddNode(Node::Factory factory, const string& name,
                    std::shared_ptr<Node> parent,
                    std::shared_ptr<Node>* out_node) {
  // The name captures the sequence of iterators joined by `::`. We only use the
  // last element of the sequence as the name node.
  auto node_name = str_util::Split(name, ':', str_util::SkipEmpty()).back();
  mutex_lock l(mu_);
  std::shared_ptr<Node> node = factory({id_counter_++, node_name, parent});
  if (!output_) {
    output_ = node;
  }
  if (parent) {
    VLOG(3) << "Adding " << node->long_name() << " as input for "
            << parent->long_name();
    parent->add_input(node);
  } else {
    VLOG(3) << "Adding " << node->long_name();
  }
  collect_resource_usage_ =
      collect_resource_usage_ || node->has_tunable_parameters();
  *out_node = std::move(node);
  // TODO(jsimsa): Reset the optimization period when a node is added so that
  // autotuning adapts to changes to the input pipeline faster. Initial attempt
  // to enable this functionality caused a regression (see b/179812091).
}

void Model::FlushMetrics() {
  std::deque<std::shared_ptr<Node>> queue;
  {
    tf_shared_lock l(mu_);
    if (output_) queue.push_back(output_);
  }
  while (!queue.empty()) {
    auto node = queue.front();
    queue.pop_front();
    node->FlushMetrics();
    for (auto input : node->inputs()) {
      queue.push_back(input);
    }
  }
}

void Model::Optimize(AutotuneAlgorithm algorithm, int64 cpu_budget,
                     int64 ram_budget, double model_input_time,
                     CancellationManager* cancellation_manager) {
  std::shared_ptr<Node> snapshot;
  {
    tf_shared_lock lock(mu_);
    snapshot = output_->Snapshot();
  }
  OptimizationParams optimization_params;
  optimization_params.set_algorithm(algorithm);
  optimization_params.set_cpu_budget(cpu_budget);
  optimization_params.set_ram_budget(ram_budget);
  optimization_params.set_model_input_time(model_input_time);
  switch (algorithm) {
    case AutotuneAlgorithm::HILL_CLIMB:
      OptimizeHillClimb(snapshot, optimization_params, cancellation_manager);
      break;
    case AutotuneAlgorithm::GRADIENT_DESCENT:
      OptimizeGradientDescent(snapshot, optimization_params,
                              cancellation_manager);
      break;
    default:
      VLOG(2) << "Autotuning algorithm was not recognized. Aborting "
                 "optimization.";
      return;
  }
  if (!save_dir_.empty()) {
    mutex_lock lock(mu_);
    Status status = EnsureSaveLoopThreadStarted();
    if (status.ok() && save_buffer_.size() < kMaxNumBufferedOptimizeArgs) {
      save_buffer_.push_back(std::make_pair(snapshot, optimization_params));
      save_cond_var_.notify_all();
    } else if (save_buffer_.size() >= kMaxNumBufferedOptimizeArgs) {
      VLOG(3) << "Saved snapshots buffer is full. Current snapshot and "
                 "optimization parameters will not be saved.";
    }
  }
}

void Model::RemoveNode(std::shared_ptr<Node> node) {
  mutex_lock l(mu_);
  if (node) {
    if (node->output()) {
      node->output()->remove_input(node);
    }
    VLOG(3) << "Removing " << node->long_name();
  }
}

absl::flat_hash_map<string, std::shared_ptr<Parameter>>
Model::CollectTunableParameters(std::shared_ptr<Node> node) {
  absl::flat_hash_map<string, std::shared_ptr<Parameter>> parameters;
  node->CollectTunableParameters(&parameters);
  return parameters;
}

bool Model::ShouldStop(
    int64 cpu_budget, int64 ram_budget,
    const absl::flat_hash_map<string, std::shared_ptr<Parameter>>& parameters,
    const absl::flat_hash_map<string, std::shared_ptr<Parameter>>&
        parallelism_parameters,
    const absl::flat_hash_map<string, std::shared_ptr<Parameter>>&
        buffer_size_parameters,
    std::shared_ptr<Node> snapshot, bool* cpu_budget_reached) {
  if (!(*cpu_budget_reached)) {
    // If those essential transformations' parallelism reaches the CPU
    // budget, we will only tune the buffer size parameters in future
    // iterations.
    int64 model_parallelism = 0;
    for (auto& pair : parallelism_parameters) {
      model_parallelism += std::round(pair.second->value);
    }
    *cpu_budget_reached = (model_parallelism > cpu_budget);
  }

  bool all_max = true;
  for (auto& pair :
       (*cpu_budget_reached ? buffer_size_parameters : parameters)) {
    if (std::round(pair.second->value) < pair.second->max) {
      all_max = false;
      break;
    }
  }

  // If all parameters have reached their maximum values or RAM budget is
  // reached, we stop the iterations.
  return all_max || TotalMaximumBufferedBytes(snapshot) > ram_budget;
}

// TODO(jsimsa): Add support for tracking and using the model input time.
Status Model::OptimizeLoop(AutotuneAlgorithm algorithm, int64 cpu_budget,
                           int64 ram_budget,
                           CancellationManager* cancellation_manager) {
  std::function<void()> unused;
  TF_RETURN_IF_ERROR(RegisterCancellationCallback(
      cancellation_manager,
      [this]() {
        mutex_lock l(mu_);
        optimize_cond_var_.notify_all();
      },
      /*deregister_fn=*/&unused));

  int64 last_optimization_ms = 0;
  int64 current_time_ms = EnvTime::NowMicros() / EnvTime::kMillisToMicros;
  while (true) {
    {
      mutex_lock l(mu_);
      while (!cancellation_manager->IsCancelled() &&
             last_optimization_ms + optimization_period_ms_ > current_time_ms) {
        auto wait_ms =
            last_optimization_ms + optimization_period_ms_ - current_time_ms;
        VLOG(2) << "Waiting for " << wait_ms << " ms.";
        optimize_cond_var_.wait_for(l, std::chrono::milliseconds(wait_ms));
        current_time_ms = EnvTime::NowMicros() / EnvTime::kMillisToMicros;
      }
      if (cancellation_manager->IsCancelled()) {
        return Status::OK();
      }
    }

    int64 start_ms = EnvTime::NowMicros() / EnvTime::kMillisToMicros;
    Optimize(algorithm, cpu_budget, ram_budget, /*model_input_time=*/0,
             cancellation_manager);
    int64 end_ms = EnvTime::NowMicros() / EnvTime::kMillisToMicros;
    VLOG(2) << "Optimized for " << end_ms - start_ms << " ms.";

    // Exponentially increase the period of running the optimization
    // until a threshold is reached.
    {
      mutex_lock l(mu_);
      optimization_period_ms_ =
          std::min(optimization_period_ms_ << 1, kOptimizationPeriodMaxMs);
    }
    current_time_ms = EnvTime::NowMicros() / EnvTime::kMillisToMicros;
    last_optimization_ms = current_time_ms;
    FlushMetrics();
  }
}

void Model::OptimizeGradientDescent(
    std::shared_ptr<Node> snapshot,
    const OptimizationParams& optimization_params,
    CancellationManager* cancellation_manager) {
  VLOG(2) << "Starting optimization of tunable parameters with Gradient "
             "Descent.";
  auto parameters = CollectTunableParameters(snapshot);
  if (parameters.empty()) {
    VLOG(2) << "The Gradient Descent optimization is terminated since no node "
               "with tunable parameters has recorded elements.";
    return;
  }
  VLOG(2) << "Number of tunable parameters: " << parameters.size();

  // The maps of "essential" parallelism parameters and buffer size parameters.
  absl::flat_hash_map<string, std::shared_ptr<Parameter>>
      parallelism_parameters, buffer_size_parameters;
  CollectParameters(snapshot, parameters, &parallelism_parameters,
                    &buffer_size_parameters);

  // Initialize the parameter values to minimal before tuning.
  for (auto& pair : parameters) {
    pair.second->value = pair.second->min;
  }

  // Optimization is stopped once the `OutputTime` improvement is smaller than
  // this value.
  constexpr double kOptimizationPrecision = 100.0L;

  // Maximum number of iterations for optimization.
  constexpr int64 kMaxIterations = 1000;

  double output_time = 0;
  double new_output_time;

  // When the CPU budget is reached, the parallelism parameter values are fixed
  // and we only increase the buffer size parameters.
  bool cpu_budget_reached = false;

  for (int i = 0; i < kMaxIterations; ++i) {
    if (cancellation_manager->IsCancelled() ||
        ShouldStop(optimization_params.cpu_budget(),
                   optimization_params.ram_budget(), parameters,
                   parallelism_parameters, buffer_size_parameters, snapshot,
                   &cpu_budget_reached)) {
      break;
    }
    absl::flat_hash_map<string, double> gradients;
    new_output_time = OutputTime(
        snapshot, optimization_params.model_input_time(), &gradients);
    // We also terminate once the improvement of the output latency is too
    // small.
    if (std::abs(output_time - new_output_time) < kOptimizationPrecision) {
      break;
    }

    UpdateParameterValues(
        gradients, &(cpu_budget_reached ? buffer_size_parameters : parameters));
    output_time = new_output_time;
  }

  for (auto& pair : parameters) {
    pair.second->value = std::round(pair.second->value);
  }
  UpdateStateValues(&parameters);
}

void Model::OptimizeHillClimb(std::shared_ptr<Node> snapshot,
                              const OptimizationParams& optimization_params,
                              CancellationManager* cancellation_manager) {
  VLOG(2) << "Starting optimization of tunable parameters with Hill Climb.";
  const double processing_time = TotalProcessingTime(snapshot);
  auto parameters = CollectTunableParameters(snapshot);
  if (parameters.empty()) {
    VLOG(2) << "The Hill Climb optimization is terminated since no node with "
               "tunable parameters has recorded elements.";
    return;
  }
  VLOG(2) << "Number of tunable parameters: " << parameters.size();

  // Buffer size parameter will only be incremented if the output latency
  // improvement is greater than this constant.
  constexpr double kBufferSizeMinDelta = 1.0L;

  // Initialize the parameter values to minimal before tuning.
  for (auto& pair : parameters) {
    pair.second->value = pair.second->min;
  }
  while (!cancellation_manager->IsCancelled()) {
    const double output_time =
        OutputTime(snapshot, optimization_params.model_input_time(),
                   /*gradients=*/nullptr);
    bool all_max = true;
    for (auto& pair : parameters) {
      if (pair.second->value < pair.second->max) {
        all_max = false;
        break;
      }
    }
    if (output_time < processing_time / optimization_params.cpu_budget() ||
        all_max ||
        TotalMaximumBufferedBytes(snapshot) >
            optimization_params.ram_budget()) {
      break;
    }
    double best_delta = -1.0L;
    Parameter* best_parameter = nullptr;
    for (auto& pair : parameters) {
      if (pair.second->value >= pair.second->max) {
        continue;
      }
      pair.second->value++;
      double new_output_time =
          OutputTime(snapshot, optimization_params.model_input_time(),
                     /*gradients=*/nullptr);
      double delta = output_time - new_output_time;
      if (delta > best_delta &&
          (delta > kBufferSizeMinDelta || pair.second->name != kBufferSize)) {
        best_delta = delta;
        best_parameter = pair.second.get();
      }
      pair.second->value--;
    }
    if (!best_parameter) {
      VLOG(2) << "Failed to find a tunable parameter that would further "
                 "decrease the output time. This means that the autotuning "
                 "optimization got stuck in a local maximum. The optimization "
                 "attempt will terminate early.";
      break;
    }
    best_parameter->value++;
  }
  UpdateStateValues(&parameters);
}

double Model::OutputTime(std::shared_ptr<Node> node, double model_input_time,
                         absl::flat_hash_map<string, double>* gradients) {
  // To store the input time for each node.
  absl::flat_hash_map<string, double> input_times = {
      {kModelInputTimeKey, model_input_time}};

  // TODO(jsimsa): Now that we are accounting for buffer size in wait time
  // computation, assuming that the input is infinitely fast will result in
  // inaccurate estimates of the output latency.
  //
  // We should compute the output latency as a fix-point of the following
  // equation: `output_time = node(OutputTime(input_times(1, output_time))`.

  return node->OutputTime(&input_times, gradients);
}

double Model::TotalBufferedBytes(std::shared_ptr<Node> node) {
  return node->TotalBufferedBytes();
}

double Model::TotalMaximumBufferedBytes(std::shared_ptr<Node> node) {
  return node->TotalMaximumBufferedBytes();
}

double Model::TotalProcessingTime(std::shared_ptr<Node> node) {
  return node->TotalProcessingTime(/*processing_times=*/nullptr);
}

Status Model::ToProto(ModelProto* model_proto) {
  ModelProto::Node* output_proto = model_proto->mutable_output();
  tf_shared_lock lock(mu_);
  TF_RETURN_IF_ERROR(output_->ToProto(output_proto));
  model_proto->set_id_counter(id_counter_);
  model_proto->set_collect_resource_usage(collect_resource_usage_);
  return Status::OK();
}

Status Model::FromProto(ModelProto model_proto, std::unique_ptr<Model>* model) {
  std::unique_ptr<Model> restored_model = std::make_unique<Model>();
  std::shared_ptr<Node> output;
  TF_RETURN_IF_ERROR(
      Node::FromProto(model_proto.output(), /*output=*/nullptr, &output));
  mutex_lock lock(restored_model->mu_);
  restored_model->output_ = output;
  restored_model->id_counter_ = model_proto.id_counter();
  restored_model->collect_resource_usage_.store(
      model_proto.collect_resource_usage());
  *model = std::move(restored_model);
  return Status::OK();
}

Status Model::Save(const string& fname, std::shared_ptr<Node> snapshot,
                   const OptimizationParams& optimization_params) {
  ModelProto model_proto;
  std::unique_ptr<Model> model_snapshot = std::make_unique<Model>();
  {
    mutex_lock lock(model_snapshot->mu_);
    model_snapshot->output_ = std::move(snapshot);
    model_snapshot->id_counter_ = id_counter_;
    model_snapshot->collect_resource_usage_.store(collect_resource_usage_);
  }
  TF_RETURN_IF_ERROR(model_snapshot->ToProto(&model_proto));
  OptimizationParams* saved_optimization_params =
      model_proto.mutable_optimization_params();
  *saved_optimization_params = optimization_params;
  return WriteBinaryProto(Env::Default(), fname, model_proto);
}

Status Model::Load(const string& fname, std::unique_ptr<Model>* model,
                   OptimizationParams* optimization_params) {
  ModelProto model_proto;
  TF_RETURN_IF_ERROR(ReadBinaryProto(Env::Default(), fname, &model_proto));
  TF_RETURN_IF_ERROR(FromProto(model_proto, model));
  const OptimizationParams restored_optimization_params =
      model_proto.optimization_params();
  *optimization_params = restored_optimization_params;
  return Status::OK();
}

Status Model::EnsureSaveLoopThreadStarted() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  if (!save_thread_) {
    save_thread_ = absl::WrapUnique(
        Env::Default()->StartThread({}, "tf_data_model_save", [this]() {
          Status status = SaveLoop();
          if (!status.ok()) {
            VLOG(2) << "Model save loop failed: " << status.ToString();
          }
        }));
  }
  return Status::OK();
}

Status Model::SaveLoop() {
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(save_dir_));
  while (true) {
    std::pair<std::shared_ptr<Node>, OptimizationParams> to_save;
    {
      mutex_lock l(mu_);
      while (!save_thread_cancelled_ && save_buffer_.empty()) {
        save_cond_var_.wait(l);
      }
      if (save_thread_cancelled_) {
        return Status::OK();
      }
      to_save = save_buffer_.front();
      save_buffer_.pop_front();
    }
    string model_name =
        absl::StrCat("autotune_model_",
                     Hash64Combine(static_cast<uint64>(EnvTime::NowMicros()),
                                   reinterpret_cast<uint64>(this)));
    string fname = io::JoinPath(save_dir_, model_name);
    TF_RETURN_IF_ERROR(Save(fname, to_save.first, to_save.second));
    VLOG(2) << "Model was saved as " << fname;
  }
}

}  // namespace model
}  // namespace data
}  // namespace tensorflow
