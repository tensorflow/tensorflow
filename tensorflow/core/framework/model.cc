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
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace data {
namespace model {
namespace {

// Wrapper for the square function to reduce verbosity.
inline double Square(double x) { return x * x; }

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
    double old_input_time;
    if (output_) {
      old_input_time = (*input_times)[output_->long_name()];
    } else {
      old_input_time = gtl::FindWithDefault(*input_times, kInputTimeKey, 0.0L);
    }

    if (num_inputs() <= 1) {
      (*input_times)[long_name()] = old_input_time;
      return;
    }
    double new_input_time =
        old_input_time +
        SelfProcessingTimeLocked() * static_cast<double>(num_inputs() - 1);
    (*input_times)[long_name()] = new_input_time;
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
        for (const auto& node : CollectNodes(TraversalOrder::REVERSE_BFS)) {
          gradients->erase(node->long_name());
        }
      }
      return;
    }

    double output_time = (OutputTimeForInputs(*output_times) -
                          (*output_times)[inputs_.front()->long_name()]) /
                         static_cast<double>(num_inputs() - 1);
    if (gradients) {
      for (const auto& node : CollectNodes(TraversalOrder::REVERSE_BFS)) {
        auto* gradient = gtl::FindOrNull(*gradients, node->long_name());
        if (gradient) {
          *gradient /= static_cast<double>(num_inputs() - 1);
        }
      }

      (*output_time_gradients)[long_name()] =
          (OutputTimeGradientsForInputs(*output_time_gradients) -
           (*output_time_gradients)[inputs_.front()->long_name()]) /
          static_cast<double>(num_inputs() - 1);

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
    (*output_times)[long_name()] = self_processing_time + output_time;
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
    double processing_time =
        (TotalProcessingTimeForInputs(*total_processing_times) -
         (*total_processing_times)[inputs_.front()->long_name()]) /
        static_cast<double>(num_inputs() - 1);
    (*total_processing_times)[long_name()] =
        self_processing_time + processing_time;
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
    double input_time;

    if (num_inputs() <= 1) {
      if (output_) {
        input_time = (*input_times)[output_->long_name()];
      } else {
        input_time = gtl::FindWithDefault(*input_times, kInputTimeKey, 0.0L);
      }
    } else {
      input_time =
          SelfProcessingTimeLocked() * static_cast<double>(num_inputs() - 1);
    }
    (*input_times)[long_name()] = input_time;
  }

  // The output time is estimated using `ComputeWaitTime(output_time,
  // input_time, parallelism, ...)`, where `output_time` is the sum of the
  // self-processing time and the average output time of inputs comprising the
  // interleave "cycle", `input_time` is specified through `input_times` and
  // `buffer_size` is derived from parallelism.
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
        for (const auto& node : CollectNodes(TraversalOrder::REVERSE_BFS)) {
          gradients->erase(node->long_name());
        }
      }
      return;
    }

    double input_time;
    if (output_) {
      input_time = input_times.at(output_->long_name());
    } else {
      input_time = gtl::FindWithDefault(input_times, kInputTimeKey, 0.0L);
    }

    double parallelism = num_inputs() - 1;  // default to cycle length
    auto* parameter = gtl::FindOrNull(parameters_, kParallelism);
    if (parameter) {
      parallelism = std::min(parallelism, (*parameter)->value);
    }

    double output_time_for_inputs =
        OutputTimeForInputs(*output_times) -
        (*output_times)[inputs_.front()->long_name()];
    double output_time = output_time_for_inputs /
                         static_cast<double>(num_inputs() - 1) / parallelism;
    double result;

    if (gradients) {
      double output_time_der = 0.0L;
      double input_time_der = 0.0L;
      double buffer_size_der = 0.0L;
      result = ComputeWaitTime(self_processing_time + output_time, input_time,
                               parallelism, &output_time_der, &input_time_der,
                               &buffer_size_der);
      (*output_time_gradients)[long_name()] = input_time_der;
      double parallelism_der = -output_time_for_inputs /
                               static_cast<double>(num_inputs() - 1) /
                               Square(parallelism);

      for (const auto& node : CollectNodes(TraversalOrder::REVERSE_BFS)) {
        auto* gradient = gtl::FindOrNull(*gradients, node->long_name());
        if (gradient) {
          *gradient *= (output_time_der /
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
            output_time_der * parallelism_der + buffer_size_der;
      }
    } else {
      result = ComputeWaitTime(self_processing_time + output_time, input_time,
                               parallelism,
                               /*output_time_derivative=*/nullptr,
                               /*input_time_derivative=*/nullptr,
                               /*buffer_size_derivative=*/nullptr);
    }
    (*output_times)[long_name()] = result;
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
    double processing_time =
        (TotalProcessingTimeForInputs(*total_processing_times) -
         (*total_processing_times)[inputs_.front()->long_name()]) /
        static_cast<double>(num_inputs() - 1);
    (*total_processing_times)[long_name()] =
        self_processing_time + processing_time;
  }
};

class KnownRatio : public Node {
 public:
  KnownRatio(Node::Args args, int64 ratio) : Node(args), ratio_(ratio) {}

  virtual ~KnownRatio() {}

 protected:
  std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    return std::make_shared<KnownRatio>(Args{id_, name_, std::move(output)},
                                        ratio_);
  }

  void InputTimeLocked(absl::flat_hash_map<string, double>* input_times)
      const override TF_SHARED_LOCKS_REQUIRED(mu_) {
    double old_input_time;
    if (output_) {
      old_input_time = (*input_times)[output_->long_name()];
    } else {
      old_input_time = gtl::FindWithDefault(*input_times, kInputTimeKey, 0.0L);
    }

    if (ratio_ == 0) {
      (*input_times)[long_name()] = old_input_time;
      return;
    }
    double new_input_time =
        (old_input_time + SelfProcessingTimeLocked()) / ratio_;
    (*input_times)[long_name()] = new_input_time;
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
        for (const auto& node : CollectNodes(TraversalOrder::REVERSE_BFS)) {
          gradients->erase(node->long_name());
        }
      }
      return;
    }
    double result =
        self_processing_time + ratio_ * OutputTimeForInputs(*output_times);
    if (gradients) {
      for (const auto& node : CollectNodes(TraversalOrder::REVERSE_BFS)) {
        auto* gradient = gtl::FindOrNull(*gradients, node->long_name());
        if (gradient) {
          *gradient *= ratio_;
        }
      }
      (*output_time_gradients)[long_name()] =
          OutputTimeGradientsForInputs(*output_time_gradients);
    }
    (*output_times)[long_name()] = result;
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
    double processing_time =
        ratio_ * TotalProcessingTimeForInputs(*total_processing_times);
    (*total_processing_times)[long_name()] =
        self_processing_time + processing_time;
  }

 private:
  const double ratio_;
};

class AsyncKnownRatio : public Node {
 public:
  AsyncKnownRatio(Node::Args args, double ratio,
                  std::vector<std::shared_ptr<Parameter>> parameters)
      : Node(args), ratio_(ratio) {
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
        Args{id_, name_, std::move(output)}, ratio_, parameters);
  }

  void InputTimeLocked(absl::flat_hash_map<string, double>* input_times)
      const override TF_SHARED_LOCKS_REQUIRED(mu_) {
    double input_time;

    if (ratio_ == 0.0) {
      if (output_) {
        input_time = (*input_times)[output_->long_name()];
      } else {
        input_time = gtl::FindWithDefault(*input_times, kInputTimeKey, 0.0L);
      }
      (*input_times)[long_name()] = input_time;
      return;
    }

    double parallelism = 1.0;
    auto* parallelism_parameter = gtl::FindOrNull(parameters_, kParallelism);
    if (parallelism_parameter) {
      parallelism = (*parallelism_parameter)->value;
    }
    input_time = SelfProcessingTimeLocked() / ratio_ / parallelism;
    (*input_times)[long_name()] = input_time;
  }

  // The output time is estimated using `ComputeWaitTime(output_time,
  // input_time, parallelism, ...)`, where `output_time` is the sum of the self
  // processing time and the product of `ratio_` and the sum of output times of
  // inputs, `input_time` is specified through `input_times` and if the node
  // has parallelism parameter, then `buffer_size` is derived from parallelism.
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
      buffer_size = parallelism;
    } else if (buffer_size_parameter) {
      buffer_size = (*buffer_size_parameter)->value;
    }
    double self_processing_time = SelfProcessingTimeLocked();
    double result;
    double input_time;
    if (output_) {
      input_time = input_times.at(output_->long_name());
    } else {
      input_time = gtl::FindWithDefault(input_times, kInputTimeKey, 0.0L);
    }

    if (ratio_ == 0.0) {
      double output_time = self_processing_time / parallelism;
      if (gradients) {
        for (const auto& node : CollectNodes(TraversalOrder::REVERSE_BFS)) {
          gradients->erase(node->long_name());
        }

        double output_time_der = 0.0L;
        double input_time_der = 0.0L;
        double buffer_size_der = 0.0L;
        result = ComputeWaitTime(output_time, input_time, buffer_size,
                                 &output_time_der, &input_time_der,
                                 &buffer_size_der);
        (*output_time_gradients)[long_name()] = input_time_der;
        // Add derivative w.r.t. own parameter if it's tunable.
        if (parallelism_parameter && (*parallelism_parameter)->state->tunable) {
          (*gradients)[long_name()] =
              -output_time_der * self_processing_time / Square(parallelism) +
              buffer_size_der;
        } else if (buffer_size_parameter &&
                   (*buffer_size_parameter)->state->tunable) {
          (*gradients)[long_name()] = buffer_size_der;
        }
      } else {
        result = ComputeWaitTime(output_time, input_time, buffer_size,
                                 /*output_time_derivative=*/nullptr,
                                 /*input_time_derivative=*/nullptr,
                                 /*buffer_size_derivative=*/nullptr);
      }
      (*output_times)[long_name()] = result;
      return;
    }

    double output_time = self_processing_time / parallelism +
                         ratio_ * OutputTimeForInputs(*output_times);
    if (gradients) {
      double output_time_der = 0.0L;
      double input_time_der = 0.0L;
      double buffer_size_der = 0.0L;
      result =
          ComputeWaitTime(output_time, input_time, buffer_size,
                          &output_time_der, &input_time_der, &buffer_size_der);
      (*output_time_gradients)[long_name()] = input_time_der;

      for (const auto& node : CollectNodes(TraversalOrder::REVERSE_BFS)) {
        auto* gradient = gtl::FindOrNull(*gradients, node->long_name());
        if (gradient) {
          *gradient *= (ratio_ * output_time_der);
        }
      }

      // Add derivative w.r.t. own parameter if it's tunable.
      if (parallelism_parameter && (*parallelism_parameter)->state->tunable) {
        double inputs_time_der_sum =
            OutputTimeGradientsForInputs(*output_time_gradients);
        (*gradients)[long_name()] =
            -output_time_der * self_processing_time / Square(parallelism) +
            buffer_size_der -
            output_time_der * inputs_time_der_sum * self_processing_time /
                Square(parallelism);
      } else if (buffer_size_parameter &&
                 (*buffer_size_parameter)->state->tunable) {
        (*gradients)[long_name()] = buffer_size_der;
      }
    } else {
      result = ComputeWaitTime(output_time, input_time, buffer_size,
                               /*output_time_derivative=*/nullptr,
                               /*input_time_derivative=*/nullptr,
                               /*buffer_size_derivative=*/nullptr);
    }
    (*output_times)[long_name()] = result;
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
    double processing_time =
        ratio_ * TotalProcessingTimeForInputs(*total_processing_times);
    (*total_processing_times)[long_name()] =
        self_processing_time + processing_time;
  }

 private:
  const double ratio_;
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

  void InputTimeLocked(absl::flat_hash_map<string, double>* input_times)
      const override TF_SHARED_LOCKS_REQUIRED(mu_) {
    double old_input_time;
    if (output_) {
      old_input_time = (*input_times)[output_->long_name()];
    } else {
      old_input_time = gtl::FindWithDefault(*input_times, kInputTimeKey, 0.0L);
    }

    if (num_elements_ == 0 || inputs_.empty() ||
        inputs_.front()->num_elements() == 0) {
      (*input_times)[long_name()] = old_input_time;
      return;
    }
    std::shared_ptr<Node> input = inputs_.front();
    double ratio = static_cast<double>(input->num_elements()) /
                   static_cast<double>(num_elements_);
    double new_input_time =
        (old_input_time + SelfProcessingTimeLocked()) / ratio;
    (*input_times)[long_name()] = new_input_time;
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
        for (const auto& node : CollectNodes(TraversalOrder::REVERSE_BFS)) {
          gradients->erase(node->long_name());
        }
      }
      return;
    }
    // TODO(jsimsa): The current implementation assumes that the number of input
    // elements consumed per output is the same across all inputs.
    double ratio = static_cast<double>(inputs_.front()->num_elements()) /
                   static_cast<double>(num_elements_);
    double result =
        self_processing_time + ratio * OutputTimeForInputs(*output_times);
    if (gradients) {
      for (const auto& node : CollectNodes(TraversalOrder::REVERSE_BFS)) {
        auto* gradient = gtl::FindOrNull(*gradients, node->long_name());
        if (gradient) {
          *gradient *= ratio;
        }
      }
      (*output_time_gradients)[long_name()] =
          OutputTimeGradientsForInputs(*output_time_gradients);
    }
    (*output_times)[long_name()] = result;
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
    double processing_time =
        ratio * TotalProcessingTimeForInputs(*total_processing_times);
    (*total_processing_times)[long_name()] =
        self_processing_time + processing_time;
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

  void InputTimeLocked(absl::flat_hash_map<string, double>* input_times)
      const override TF_SHARED_LOCKS_REQUIRED(mu_) {
    double input_time;
    if (output_) {
      input_time = (*input_times)[output_->long_name()];
    } else {
      input_time = gtl::FindWithDefault(*input_times, kInputTimeKey, 0.0L);
    }
    (*input_times)[long_name()] = input_time;
  }

  // The output time is the sum of output times of inputs.
  void OutputTimeLocked(
      const absl::flat_hash_map<string, double>& input_times,
      absl::flat_hash_map<string, double>* gradients,
      absl::flat_hash_map<string, double>* output_times,
      absl::flat_hash_map<string, double>* output_time_gradients) const override
      TF_SHARED_LOCKS_REQUIRED(mu_) {
    double result = OutputTimeForInputs(*output_times);
    (*output_times)[long_name()] = result;
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
    double processing_time =
        TotalProcessingTimeForInputs(*total_processing_times);
    (*total_processing_times)[long_name()] = processing_time;
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
    Node::Args args, double ratio,
    std::vector<std::shared_ptr<Parameter>> parameters) {
  return std::make_shared<AsyncKnownRatio>(std::move(args), ratio,
                                           std::move(parameters));
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

double Node::ComputeWaitTime(const double& output_time,
                             const double& input_time,
                             const double& buffer_size,
                             double* output_time_derivative,
                             double* input_time_derivative,
                             double* buffer_size_derivative) {
  // If we set x=`input_time`, y=`output_time`, n=`buffer_size`,
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
  if (output_time == 0) {
    if (output_time_derivative) {
      // Note a common error is `*output_time_derivative = 0` since p=0 on the
      // line y=0 doesn't imply dp/dy = 0 there. Actually to compute dp/dy at
      // (x,0), we need to consider lim_{dy->0+} [p(x,dy)-p(x,0)] / dy, where
      // p(x,0)=0 and p(x,dy) = [1 - x/dy] / [1 - power(x/dy, n+1)].
      if (buffer_size == 0 || input_time == 0) {
        *output_time_derivative = 1.0L;
      } else {
        *output_time_derivative = 0.0L;
      }
    }
    if (input_time_derivative) {
      *input_time_derivative = 0.0L;
    }
    if (buffer_size_derivative) {
      *buffer_size_derivative = 0.0L;
    }
    return 0.0L;
  }

  // Case 2: if consumer is infinitely fast. Wait time is always the time to
  // produce an output.
  if (input_time == 0) {
    if (output_time_derivative) {
      *output_time_derivative = 1.0L;
    }
    if (input_time_derivative) {
      // Note a common error is `*input_time_derivative = 0` since p=1 on the
      // line x=0 doesn't imply dp/dx = 0 there. Actually to compute dp/dx at
      // (0,y), we need to consider lim_{dx->0+} [p(dx,y)-p(0,y)] / dx, where
      // p(0,y)=1, p(dx,y) = [1 - dx/y] / [1 - power(dx/y, n+1)] if y!=0.
      if (buffer_size == 0) {
        *input_time_derivative = 0.0L;
      } else {
        *input_time_derivative = -1.0L;
      }
    }
    if (buffer_size_derivative) {
      *buffer_size_derivative = 0.0L;
    }
    return output_time;
  }

  // Case 3: the consumer and the producer are equally fast. Expected wait time
  // decreases linearly with the size of the buffer.
  if (input_time == output_time) {
    const double p_buffer_empty = 1.0L / (buffer_size + 1.0L);
    const double p_buffer_empty_der =
        -buffer_size / (2.0L * buffer_size + 2.0L);
    if (output_time_derivative) {
      // Note a common error is `*output_time_derivative = p_buffer_empty` since
      // p=1/(n+1) on the line x=y doesn't imply dp/dy = 0 there. Actually to
      // compute dp/dy at (y,y), we need to consider
      // lim_{dy->0} [p(y,y+dy)-p(y,y)] / dy, where p(y,y)=1/(n+1),
      // p(y,y+dy) = [1 - y/(y+dy)] / [1 - power(y/(y+dy), n+1)].
      *output_time_derivative = p_buffer_empty - p_buffer_empty_der;
    }
    if (input_time_derivative) {
      // Note a common error is `*input_time_derivative = 0` since
      // p=1/(n+1) on the line x=y doesn't imply dp/dx = 0 there. Actually to
      // compute dp/dx at (x,x), we need to consider
      // lim_{dx->0} [p(x+dx,x)-p(x,x)] / dx, where p(x,x)=1/(n+1),
      // p(x+dx,x) = [1 - (x+dx)/x] / [1 - power((x+dx)/x, n+1)].
      *input_time_derivative = p_buffer_empty_der;
    }
    if (buffer_size_derivative) {
      *buffer_size_derivative = -output_time / Square(buffer_size + 1.0L);
    }
    return p_buffer_empty * output_time;
  }

  // Case 4: the consumer is slower than the producer and neither is infinitely
  // fast. Case 4 and Case 5 actually follow same formula. Separate them for
  // numerical computation reasons.
  if (input_time > output_time) {
    const double ratio = output_time / input_time;
    const double ratio_pow = std::pow(ratio, buffer_size);
    const double p_buffer_empty =
        ratio_pow * (1.0L - ratio) / (1.0L - ratio * ratio_pow);
    const double p_buffer_empty_der =
        (buffer_size - (buffer_size + 1.0L) * ratio + ratio_pow * ratio) *
        ratio_pow / ratio / Square(1.0L - ratio_pow * ratio);
    if (output_time_derivative) {
      *output_time_derivative = p_buffer_empty + p_buffer_empty_der * ratio;
    }
    if (input_time_derivative) {
      *input_time_derivative = -p_buffer_empty_der * Square(ratio);
    }
    if (buffer_size_derivative) {
      *buffer_size_derivative = p_buffer_empty / (1.0L - ratio_pow * ratio) *
                                std::log(ratio) * output_time;
    }
    return p_buffer_empty * output_time;
  }

  // Case 5: the producer is slower than the consumer and neither is infinitely
  // fast.
  const double ratio = input_time / output_time;
  const double ratio_pow = std::pow(ratio, buffer_size);
  const double p_buffer_empty = (1.0L - ratio) / (1.0L - ratio_pow * ratio);
  const double p_buffer_empty_der =
      ((buffer_size + 1.0L - buffer_size * ratio) * ratio_pow - 1.0L) /
      Square(1.0L - ratio_pow * ratio);
  if (output_time_derivative) {
    *output_time_derivative = p_buffer_empty - p_buffer_empty_der * ratio;
  }
  if (input_time_derivative) {
    *input_time_derivative = p_buffer_empty_der;
  }
  if (buffer_size_derivative) {
    *buffer_size_derivative = p_buffer_empty / (1.0L - ratio_pow * ratio) *
                              ratio_pow * ratio * std::log(ratio) * output_time;
  }
  return p_buffer_empty * output_time;
}

void Node::CollectTunableParameters(
    absl::flat_hash_map<string, std::shared_ptr<Parameter>>* parameters) const {
  tf_shared_lock l(mu_);
  // Collect tunable parameters from the leaves of the nodes tree to the root.
  for (const auto& node : CollectNodes(TraversalOrder::REVERSE_BFS)) {
    tf_shared_lock l(node->mu_);
    node->CollectTunableParametersHelper(parameters);
  }
  CollectTunableParametersHelper(parameters);
}

string Node::DebugString() const {
  absl::flat_hash_map<string, string> debug_strings;
  tf_shared_lock l(mu_);
  // Build up the debug string from the leaves of the nodes tree to the root.
  for (const auto& node : CollectNodes(TraversalOrder::REVERSE_BFS)) {
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
  auto nodes = CollectNodes(TraversalOrder::BFS);

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

std::shared_ptr<Node> Node::Snapshot(std::shared_ptr<Node> output) const {
  NodePairList node_pairs;
  auto result = SnapshotHelper(output, &node_pairs);

  while (!node_pairs.empty()) {
    auto node_pair = node_pairs.front();
    node_pairs.pop_front();
    std::shared_ptr<Node> input_node = node_pair.first,
                          parent_node_copy = node_pair.second;
    parent_node_copy->add_input(
        input_node->SnapshotHelper(parent_node_copy, &node_pairs));
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
  for (const auto& node : CollectNodes(TraversalOrder::REVERSE_BFS)) {
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
  for (const auto& node : CollectNodes(TraversalOrder::REVERSE_BFS)) {
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
  for (const auto& node : CollectNodes(TraversalOrder::REVERSE_BFS)) {
    tf_shared_lock l(node->mu_);
    node->TotalProcessingTimeLocked(processing_times, &total_processing_times);
  }
  TotalProcessingTimeLocked(processing_times, &total_processing_times);

  return total_processing_times[long_name()];
}

double Node::AverageBufferedElementSize() const {
  if (buffered_elements_ == 0) {
    return 0;
  }
  return static_cast<double>(buffered_bytes_) /
         static_cast<double>(buffered_elements_);
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

Node::NodeVector Node::CollectNodes(TraversalOrder order) const
    TF_SHARED_LOCKS_REQUIRED(mu_) {
  NodeVector node_vector;
  std::list<std::shared_ptr<Node>> temp_list;

  {
    for (auto& input : inputs_) {
      node_vector.push_back(input);
      temp_list.push_back(input);
    }
  }

  while (!temp_list.empty()) {
    auto cur_node = temp_list.front();
    temp_list.pop_front();
    {
      tf_shared_lock l(cur_node->mu_);
      for (auto& input : cur_node->inputs_) {
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
  if (!autotune_) {
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
    std::shared_ptr<Node> clone_base, Node::NodePairList* node_pairs) const {
  tf_shared_lock l(mu_);
  std::shared_ptr<Node> result_node = Clone(clone_base);
  {
    result_node->autotune_.store(autotune_);
    result_node->buffered_bytes_.store(buffered_bytes_);
    result_node->buffered_elements_.store(buffered_elements_);
    result_node->bytes_consumed_.store(bytes_consumed_);
    result_node->bytes_produced_.store(bytes_produced_);
    result_node->num_elements_.store(num_elements_);
    result_node->record_metrics_.store(false);
    result_node->processing_time_.store(processing_time_);
    mutex_lock l2(result_node->mu_);
    result_node->parameters_ = parameters_;
  }

  for (auto& input : inputs_) {
    node_pairs->push_back(std::make_pair(input, result_node));
  }
  return result_node;
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

  double result = 0;
  auto* parameter = gtl::FindOrNull(parameters_, kBufferSize);
  if (!parameter) {
    parameter = gtl::FindOrNull(parameters_, kParallelism);
  }
  if (parameter) {
    result = (*parameter)->value * AverageBufferedElementSize();
  }
  for (auto& input : inputs_) {
    result += total_bytes->at(input->long_name());
  }
  total_bytes->insert(std::make_pair(long_name(), result));
}

void Model::AddNode(Node::Factory factory, const string& name,
                    const string& output_name,
                    std::shared_ptr<Node>* out_node) {
  // The name captures the sequence of iterators joined by `::`. We use the full
  // sequence as the key in the lookup table, but only the last element of the
  // sequence as the name node.
  std::vector<string> tokens =
      str_util::Split(name, ':', str_util::SkipEmpty());
  // The output name might contain an index. We need to strip it to make it
  // possible for the model to successfully identify the output node.
  string sanitized_output_name = output_name;
  if (str_util::EndsWith(output_name, "]")) {
    sanitized_output_name = output_name.substr(0, output_name.rfind('['));
  }
  std::shared_ptr<Node> output;
  mutex_lock l(mu_);
  auto it = lookup_table_.find(sanitized_output_name);
  if (it != lookup_table_.end()) {
    output = it->second;
  }
  std::shared_ptr<Node> node = factory({id_counter_++, tokens.back(), output});
  if (!output_) {
    output_ = node;
  }
  if (output) {
    VLOG(3) << "Adding " << node->long_name() << " as input for "
            << output->long_name();
    output->add_input(node);
  } else {
    VLOG(3) << "Adding " << node->long_name();
  }
  collect_resource_usage_ =
      collect_resource_usage_ || node->has_tunable_parameters();
  lookup_table_.insert(std::make_pair(name, node));
  *out_node = node;
}

void Model::FlushMetrics() {
  tf_shared_lock l(mu_);
  for (const auto& pair : lookup_table_) {
    pair.second->FlushMetrics();
  }
}

void Model::Optimize(AutotuneAlgorithm algorithm, int64 cpu_budget,
                     int64 ram_budget) {
  switch (algorithm) {
    case AutotuneAlgorithm::HILL_CLIMB:
      OptimizeHillClimb(cpu_budget, ram_budget);
      break;
    case AutotuneAlgorithm::GRADIENT_DESCENT:
      OptimizeGradientDescent(cpu_budget, ram_budget);
      break;
  }
}

void Model::RemoveNode(const string& name) {
  mutex_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (node) {
    if ((*node)->output()) {
      (*node)->output()->remove_input(*node);
    }
    VLOG(3) << "Removing " << (*node)->long_name();
  }
  lookup_table_.erase(name);
}

absl::flat_hash_map<string, std::shared_ptr<Parameter>>
Model::CollectTunableParameters(std::shared_ptr<Node> node) {
  absl::flat_hash_map<string, std::shared_ptr<Parameter>> parameters;
  node->CollectTunableParameters(&parameters);
  return parameters;
}

absl::flat_hash_map<string, std::shared_ptr<Parameter>>
Model::CollectEssentialParallelism(std::shared_ptr<Node> node) {
  // Parallelism parameter is considered to be essential if the corresponding
  // transformations's processing time is greater than essential rate times the
  // average transformation self processing time.
  constexpr double kEssentialRate = 0.3L;

  absl::flat_hash_map<string, std::shared_ptr<Parameter>> parameters;
  node->CollectTunableParameters(&parameters);
  absl::flat_hash_map<string, double> processing_times;
  double processing_time = node->TotalProcessingTime(&processing_times);
  double uniform_share =
      processing_time / static_cast<double>(processing_times.size());
  absl::flat_hash_map<string, std::shared_ptr<Parameter>> essential_parameters;
  for (auto& pair : parameters) {
    if (pair.second->name == kParallelism &&
        processing_times[pair.first] > kEssentialRate * uniform_share) {
      essential_parameters.insert(pair);
    }
  }
  return essential_parameters;
}

void Model::OptimizeGradientDescent(int64 cpu_budget, int64 ram_budget) {
  std::shared_ptr<Node> snapshot;
  {
    tf_shared_lock lock(mu_);
    snapshot = output_->Snapshot(nullptr);
  }
  VLOG(2) << "Starting optimization of tunable parameters with GradientDescent";
  auto parameters = CollectTunableParameters(snapshot);
  auto essential_parameters = CollectEssentialParallelism(snapshot);
  // We add the number of model's buffered bytes because it is excluded from the
  // memory budget, but it is included in the maximum number of buffered bytes.
  ram_budget += TotalBufferedBytes(snapshot);
  for (auto& pair : parameters) {
    pair.second->value = pair.second->min;
  }
  // Gradient descent step size.
  constexpr double kDescentStep = 0.1L;

  // Optimization is stopped once the `OutputTime` improvement is smaller than
  // this value.
  constexpr double kOptimizationPrecision = 100.0L;

  // Maximum number of iterations for optimization.
  constexpr int64 kMaxIterations = 1000;

  double output_time = 0;
  double new_output_time;
  double new_value;
  for (int i = 0; i < kMaxIterations; ++i) {
    absl::flat_hash_map<string, double> gradients;
    new_output_time = OutputTime(snapshot, &gradients);
    int64 model_parallelism = 0;
    for (auto& pair : essential_parameters) {
      model_parallelism += std::round(pair.second->value);
    }
    // We terminate once the improvement of the output latency is too small or
    // the essential transformations' parallelism reaches the CPU budget or the
    // worst-case total buffer size exceeds the memory budget.
    if (std::abs(output_time - new_output_time) < kOptimizationPrecision ||
        model_parallelism > cpu_budget ||
        TotalMaximumBufferedBytes(snapshot) > ram_budget) {
      break;
    }
    double max_abs_derivative = 1.0;
    for (auto& pair : parameters) {
      if (pair.second->value != pair.second->max) {
        max_abs_derivative =
            std::max(max_abs_derivative, std::abs(gradients[pair.first]));
      }
    }
    for (auto& pair : parameters) {
      new_value = pair.second->value -
                  kDescentStep * gradients[pair.first] / max_abs_derivative;
      // Projection on a feasible interval.
      if (new_value > pair.second->max) {
        pair.second->value = pair.second->max;
      } else if (new_value < pair.second->min) {
        pair.second->value = pair.second->min;
      } else {
        pair.second->value = new_value;
      }
    }
    output_time = new_output_time;
  }
  VLOG(2) << "Number of tunable parameters: " << parameters.size();
  for (auto& pair : parameters) {
    pair.second->value = std::round(pair.second->value);
    auto& parameter = pair.second;
    VLOG(2) << "Setting tunable parameter " << pair.first << " to "
            << parameter->value;
    mutex_lock l(*parameter->state->mu);
    parameter->state->value = parameter->value;
    parameter->state->cond_var->notify_all();
  }
}

void Model::OptimizeHillClimb(int64 cpu_budget, int64 ram_budget) {
  std::shared_ptr<Node> snapshot;
  {
    tf_shared_lock lock(mu_);
    snapshot = output_->Snapshot(nullptr);
  }
  VLOG(2) << "Starting optimization of tunable parameters with HillClimb";
  const double processing_time = TotalProcessingTime(snapshot);
  auto parameters = CollectTunableParameters(snapshot);
  // We add the number of model's buffered bytes because it is excluded from the
  // memory budget, but it is included in the maximum number of buffered bytes.
  ram_budget += TotalBufferedBytes(snapshot);
  // Buffer size parameter will only be incremented if the output latency
  // improvement is greater than this constant.
  constexpr double kBufferSizeMinDelta = 1.0L;

  for (auto& pair : parameters) {
    pair.second->value = pair.second->min;
  }
  while (true) {
    const double output_time = OutputTime(snapshot, /*gradients=*/nullptr);
    bool all_max = true;
    for (auto& pair : parameters) {
      if (pair.second->value < pair.second->max) {
        all_max = false;
        break;
      }
    }
    if (output_time < processing_time / cpu_budget || all_max ||
        TotalMaximumBufferedBytes(snapshot) > ram_budget) {
      break;
    }
    double best_delta = -1.0L;
    Parameter* best_parameter = nullptr;
    for (auto& pair : parameters) {
      if (pair.second->value == pair.second->max) {
        continue;
      }
      pair.second->value++;
      double new_output_time = OutputTime(snapshot, /*gradients=*/nullptr);
      double delta = output_time - new_output_time;
      if (delta > best_delta &&
          (delta > kBufferSizeMinDelta || pair.second->name != kBufferSize)) {
        best_delta = delta;
        best_parameter = pair.second.get();
      }
      pair.second->value--;
    }
    if (!best_parameter) {
      VLOG(2) << "Failed to find a tunable parameter that would decrease the "
                 "output time. This means that the autotuning optimization got "
                 "stuck in a local maximum. The optimization attempt will be "
                 "aborted.";
      return;
    }
    best_parameter->value++;
  }
  VLOG(2) << "Number of tunable parameters: " << parameters.size();
  for (auto& pair : parameters) {
    auto& parameter = pair.second;
    VLOG(2) << "Setting tunable parameter " << pair.first << " to "
            << parameter->value;
    mutex_lock l(*parameter->state->mu);
    parameter->state->value = parameter->value;
    parameter->state->cond_var->notify_all();
  }
}

double Model::OutputTime(std::shared_ptr<Node> node,
                         absl::flat_hash_map<string, double>* gradients) {
  // To store the input time for each node.
  absl::flat_hash_map<string, double> input_times;

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

}  // namespace model
}  // namespace data
}  // namespace tensorflow
