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

// Key of the derivative w.r.t. the last input time in the gradient of
// `OutputTime`.
constexpr char kInputTimeDerivativeKey[] = "last_input_time";

// Wrapper for the square function to reduce verbosity.
inline double Square(double x) { return x * x; }

// Given the average time between output events (`output_time`), the average
// time between input events (`input_time`) and the buffer size, the method
// computes the expected time an input event will have to wait.
//
// The wait time is approximated as the product of the probability the buffer
// will be empty and the time it takes to produce an element into the buffer.
//
// The formula used for computing the probability is derived by modeling the
// problem as an M/M/1/K queue
// (https://en.wikipedia.org/wiki/Birth%E2%80%93death_process#M/M/1/K_queue).
//
// Collects derivatives of `ComputeWaitTime` w.r.t `output_time`, `input_time'
// and `buffer_size` if the corresponding pointers are not `nullptr`.
double ComputeWaitTime(double output_time, double input_time,
                       double buffer_size, double* output_time_derivative,
                       double* input_time_derivative,
                       double* buffer_size_derivative) {
  // Case 0: either the producer or the consumer are infinitely fast. Wait time
  // is the time to produce an output.
  if (output_time == 0 || input_time == 0) {
    if (output_time_derivative) {
      *output_time_derivative = 1.0L;
    }
    if (input_time_derivative) {
      *input_time_derivative = 0.0L;
    }
    if (buffer_size_derivative) {
      *buffer_size_derivative = 0.0L;
    }
    return output_time;
  }
  // Case 1: the consumer is slower than the producer. Wait time is 0 since the
  // buffer will be full in the long run.
  if (input_time > output_time) {
    if (output_time_derivative) {
      *output_time_derivative = 0.0L;
    }
    if (input_time_derivative) {
      *input_time_derivative = 0.0L;
    }
    if (buffer_size_derivative) {
      *buffer_size_derivative = 0.0L;
    }
    return 0;
  }
  // Case 2: the consumer and the producer are equally fast. Expected wait time
  // decreases linearly with the size of the buffer.
  if (input_time == output_time) {
    const double p_buffer_empty = 1.0L / (buffer_size + 1.0L);
    if (output_time_derivative) {
      *output_time_derivative = p_buffer_empty;
    }
    if (input_time_derivative) {
      *input_time_derivative = 0.0L;
    }
    if (buffer_size_derivative) {
      const double p_buffer_empty_der = -1.0L / Square(buffer_size + 1.0L);
      *buffer_size_derivative = p_buffer_empty_der * output_time;
    }
    return p_buffer_empty * output_time;
  }
  // Case 3: the producer is slower than the consumer and neither is infinitely
  // fast.
  const double alpha = 1.0L / input_time;
  const double beta = 1.0L / output_time;
  const double ratio_pow = std::pow((beta / alpha), (buffer_size + 1.0L));
  const double p_buffer_empty = (1.0L - beta / alpha) / (1.0L - ratio_pow);
  if (output_time_derivative) {
    *output_time_derivative =
        (1.0L - ratio_pow -
         (output_time - input_time) * (buffer_size + 1.0L) * ratio_pow /
             output_time) /
        Square(1.0L - ratio_pow);
  }
  if (input_time_derivative) {
    *input_time_derivative =
        (ratio_pow - 1.0L +
         (buffer_size + 1.0L) * ratio_pow * (alpha / beta - 1.0L)) /
        Square(1.0L - ratio_pow);
  }
  if (buffer_size_derivative) {
    const double p_buffer_empty_der = (1.0L - beta / alpha) * ratio_pow *
                                      std::log(beta / alpha) /
                                      Square(1.0L - ratio_pow);
    *buffer_size_derivative = p_buffer_empty_der * output_time;
  }

  return p_buffer_empty * output_time;
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
      SHARED_LOCKS_REQUIRED(mu_) {
    return std::make_shared<InterleaveMany>(
        Args{id_, name_, std::move(output)});
  }

  // The output time is the sum of the self processing time and the average
  // output time of inputs comprising the interleave "cycle".
  double OutputTimeLocked(std::vector<double>* input_times,
                          std::map<string, double>* gradient) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    if (num_inputs() <= 1) {
      return SelfProcessingTimeLocked();
    }
    double delta = SelfProcessingTimeLocked() * (num_inputs() - 1);
    input_times->back() += delta;
    auto cleanup = gtl::MakeCleanup(
        [input_times, delta]() { input_times->back() -= delta; });
    double output_time;
    if (gradient) {
      std::map<string, double> inputs_gradient;
      output_time =
          (OutputTimeForInputs(input_times, &inputs_gradient) -
           inputs_.front()->OutputTime(input_times, /*gradient=*/nullptr)) /
          static_cast<double>(num_inputs() - 1);
      for (auto& pair : inputs_gradient) {
        (*gradient)[pair.first] =
            pair.second / static_cast<double>(num_inputs() - 1);
      }
      auto last_input_time_der =
          gtl::FindWithDefault(*gradient, kInputTimeDerivativeKey, 0.0L);
      (*gradient)[kInputTimeDerivativeKey] =
          last_input_time_der + inputs_gradient[kInputTimeDerivativeKey] /
                                    static_cast<double>(num_inputs() - 1);
      // Set derivatives w.r.t. tunable parameters of the subtree rooted in the
      // first input equal to 0 since its output time is excluded from
      // computations.
      std::map<string, std::shared_ptr<Parameter>> first_input_parameters;
      inputs_.front()->CollectTunableParameters(&first_input_parameters);
      for (auto& pair : first_input_parameters) {
        (*gradient)[pair.first] = 0.0L;
      }
    } else {
      output_time =
          (OutputTimeForInputs(input_times, /*gradient=*/nullptr) -
           inputs_.front()->OutputTime(input_times, /*gradient=*/nullptr)) /
          static_cast<double>(num_inputs() - 1);
    }
    return SelfProcessingTimeLocked() + output_time;
  }

  // The processing time is the sum of the self processing time and the average
  // processing time of inputs comprising the interleave "cycle".
  double TotalProcessingTimeLocked(std::map<string, double>* processing_times)
      override SHARED_LOCKS_REQUIRED(mu_) {
    double self_processing_time = SelfProcessingTimeLocked();
    if (processing_times) {
      (*processing_times)[long_name()] = self_processing_time;
    }
    if (num_inputs() <= 1) {
      return self_processing_time;
    }
    double processing_time =
        (TotalProcessingTimeForInputs(processing_times) -
         inputs_.front()->TotalProcessingTime(/*processing_times=*/nullptr)) /
        static_cast<double>(num_inputs() - 1);
    return self_processing_time + processing_time;
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
      SHARED_LOCKS_REQUIRED(mu_) {
    std::vector<std::shared_ptr<Parameter>> parameters;
    for (auto& pair : parameters_) {
      parameters.push_back(pair.second);
    }
    return std::make_shared<AsyncInterleaveMany>(
        Args{id_, name_, std::move(output)}, parameters);
  }

  // The output time is estimated using `ComputeWaitTime(output_time,
  // input_time, parallelism, ...)`, where `output_time` is the sum of the
  // self-processing time and the average output time of inputs comprising the
  // interleave "cycle", `input_time` is specified through `input_times` and
  // `buffer_size` is derived from parallelism.
  double OutputTimeLocked(std::vector<double>* input_times,
                          std::map<string, double>* gradient) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    if (num_inputs() <= 1) {
      return SelfProcessingTimeLocked();
    }
    double old_input_time = input_times->back();
    double new_input_time =
        SelfProcessingTimeLocked() * static_cast<double>(num_inputs() - 1);
    input_times->push_back(new_input_time);
    auto cleanup =
        gtl::MakeCleanup([input_times]() { input_times->pop_back(); });
    double parallelism = num_inputs() - 1;  // default to cycle length
    auto* parameter = gtl::FindOrNull(parameters_, kParallelism);
    if (parameter) {
      parallelism = std::min(parallelism, (*parameter)->value);
    }
    if (gradient) {
      std::map<string, double> inputs_gradient;
      double output_time_for_inputs =
          OutputTimeForInputs(input_times, &inputs_gradient) -
          inputs_.front()->OutputTime(input_times, /*gradient=*/nullptr);
      double output_time = output_time_for_inputs /
                           static_cast<double>(num_inputs() - 1) / parallelism;
      double output_time_der = 0.0L;
      double input_time_der = 0.0L;
      double buffer_size_der = 0.0L;
      double result = ComputeWaitTime(
          SelfProcessingTimeLocked() + output_time, old_input_time, parallelism,
          &output_time_der, &input_time_der, &buffer_size_der);
      auto last_input_time_der =
          gtl::FindWithDefault(*gradient, kInputTimeDerivativeKey, 0.0L);
      (*gradient)[kInputTimeDerivativeKey] =
          last_input_time_der + input_time_der;
      double parallelism_der = -output_time_for_inputs /
                               static_cast<double>(num_inputs() - 1) /
                               Square(parallelism);
      for (auto& pair : inputs_gradient) {
        if (pair.first != kInputTimeDerivativeKey) {
          (*gradient)[pair.first] = output_time_der * pair.second /
                                    static_cast<double>(num_inputs() - 1) /
                                    parallelism;
        }
      }
      // Set derivatives w.r.t. tunable parameters of the subtree rooted in the
      // first input equal to 0 since its output time is excluded from
      // computations.
      std::map<string, std::shared_ptr<Parameter>> first_input_parameters;
      inputs_.front()->CollectTunableParameters(&first_input_parameters);
      for (auto& pair : first_input_parameters) {
        (*gradient)[pair.first] = 0.0L;
      }
      // Add derivative w.r.t. own parallelism parameter.
      if (parameter && (*parameter)->state->tunable) {
        (*gradient)[long_name()] =
            output_time_der * parallelism_der + buffer_size_der;
      }
      return result;
    }
    double output_time =
        (OutputTimeForInputs(input_times, /*gradient=*/nullptr) -
         inputs_.front()->OutputTime(input_times, /*gradient=*/nullptr)) /
        static_cast<double>(num_inputs() - 1) / parallelism;
    return ComputeWaitTime(
        SelfProcessingTimeLocked() + output_time, old_input_time, parallelism,
        /*output_time_derivative=*/nullptr,
        /*input_time_derivative=*/nullptr, /*buffer_size_derivative=*/nullptr);
  }

  // The processing time is the sum of the self processing time and the average
  // processing time of inputs comprising the interleave "cycle".
  double TotalProcessingTimeLocked(std::map<string, double>* processing_times)
      override SHARED_LOCKS_REQUIRED(mu_) {
    double self_processing_time = SelfProcessingTimeLocked();
    if (processing_times) {
      (*processing_times)[long_name()] = self_processing_time;
    }
    if (num_inputs() <= 1) {
      return self_processing_time;
    }
    double processing_time =
        TotalProcessingTimeForInputs(processing_times) -
        inputs_.front()->TotalProcessingTime(/*processing_times=*/nullptr);
    return self_processing_time +
           processing_time / static_cast<double>(num_inputs() - 1);
  }
};

class KnownRatio : public Node {
 public:
  KnownRatio(Node::Args args, int64 ratio) : Node(args), ratio_(ratio) {}

  virtual ~KnownRatio() {}

 protected:
  std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    return std::make_shared<KnownRatio>(Args{id_, name_, std::move(output)},
                                        ratio_);
  }

  // The output time is the sum of the self processing time and the product of
  // `ratio_` and the sum of output times of inputs.
  double OutputTimeLocked(std::vector<double>* input_times,
                          std::map<string, double>* gradient) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    if (ratio_ == 0) {
      return SelfProcessingTimeLocked();
    }
    double old_input_time = input_times->back();
    input_times->back() +=
        (old_input_time + SelfProcessingTimeLocked()) / ratio_;
    auto cleanup = gtl::MakeCleanup([input_times, old_input_time]() {
      input_times->back() = old_input_time;
    });
    double result;
    if (gradient) {
      std::map<string, double> inputs_gradient;
      result = SelfProcessingTimeLocked() +
               ratio_ * OutputTimeForInputs(input_times, &inputs_gradient);
      auto last_input_time_der =
          gtl::FindWithDefault(*gradient, kInputTimeDerivativeKey, 0.0L);
      (*gradient)[kInputTimeDerivativeKey] =
          last_input_time_der + ratio_ *
                                    inputs_gradient[kInputTimeDerivativeKey] *
                                    (1.0L + 1.0L / ratio_);
      for (auto& pair : inputs_gradient) {
        if (pair.first != kInputTimeDerivativeKey) {
          (*gradient)[pair.first] = pair.second * ratio_;
        }
      }
    } else {
      result = SelfProcessingTimeLocked() +
               ratio_ * OutputTimeForInputs(input_times, /*gradient=*/nullptr);
    }
    return result;
  }

  // The processing time is the sum of the self processing time and the product
  // of `ratio_` and the sum of processing times of inputs.
  double TotalProcessingTimeLocked(std::map<string, double>* processing_times)
      override SHARED_LOCKS_REQUIRED(mu_) {
    double self_processing_time = SelfProcessingTimeLocked();
    if (processing_times) {
      (*processing_times)[long_name()] = self_processing_time;
    }
    return self_processing_time +
           ratio_ * TotalProcessingTimeForInputs(processing_times);
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
      SHARED_LOCKS_REQUIRED(mu_) {
    std::vector<std::shared_ptr<Parameter>> parameters;
    for (auto& pair : parameters_) {
      parameters.push_back(pair.second);
    }
    return std::make_shared<AsyncKnownRatio>(
        Args{id_, name_, std::move(output)}, ratio_, parameters);
  }

  // The output time is estimated using `ComputeWaitTime(output_time,
  // input_time, parallelism, ...)`, where `output_time` is the sum of the self
  // processing time and the product of `ratio_` and the sum of output times of
  // inputs, `input_time` is specified through `input_times` and if the node
  // has parallelism parameter, then `buffer_size` is derived from parallelism.
  //
  // Current implementation assumes that there is at most 1 parameter per node.
  double OutputTimeLocked(std::vector<double>* input_times,
                          std::map<string, double>* gradient) const override
      SHARED_LOCKS_REQUIRED(mu_) {
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
    if (ratio_ == 0.0) {
      double output_time = self_processing_time / parallelism;
      if (gradient) {
        double output_time_der = 0.0L;
        double input_time_der = 0.0L;
        double buffer_size_der = 0.0L;
        double result = ComputeWaitTime(output_time, input_times->back(),
                                        buffer_size, &output_time_der,
                                        &input_time_der, &buffer_size_der);
        auto last_input_time_der =
            gtl::FindWithDefault(*gradient, kInputTimeDerivativeKey, 0.0L);
        (*gradient)[kInputTimeDerivativeKey] =
            last_input_time_der + input_time_der;
        // Add derivative w.r.t. own parameter if it's tunable.
        if (parallelism_parameter && (*parallelism_parameter)->state->tunable) {
          (*gradient)[long_name()] =
              -output_time_der * self_processing_time / Square(parallelism) +
              buffer_size_der;
        } else if (buffer_size_parameter &&
                   (*buffer_size_parameter)->state->tunable) {
          (*gradient)[long_name()] = buffer_size_der;
        }
        return result;
      }
      return ComputeWaitTime(output_time, input_times->back(), buffer_size,
                             /*output_time_derivative=*/nullptr,
                             /*input_time_derivative=*/nullptr,
                             /*buffer_size_derivative=*/nullptr);
    }
    double old_input_time = input_times->back();
    double new_input_time = self_processing_time / ratio_ / parallelism;
    input_times->push_back(new_input_time);
    auto cleanup =
        gtl::MakeCleanup([input_times]() { input_times->pop_back(); });
    if (gradient) {
      std::map<string, double> inputs_gradient;
      double output_time_der = 0.0L;
      double input_time_der = 0.0L;
      double buffer_size_der = 0.0L;
      double output_time =
          self_processing_time / parallelism +
          ratio_ * OutputTimeForInputs(input_times, &inputs_gradient);
      double result =
          ComputeWaitTime(output_time, old_input_time, buffer_size,
                          &output_time_der, &input_time_der, &buffer_size_der);
      auto last_input_time_der =
          gtl::FindWithDefault(*gradient, kInputTimeDerivativeKey, 0.0L);
      (*gradient)[kInputTimeDerivativeKey] =
          last_input_time_der + input_time_der;
      for (auto& pair : inputs_gradient) {
        if (pair.first != kInputTimeDerivativeKey) {
          (*gradient)[pair.first] = pair.second * ratio_ * output_time_der;
        }
      }
      // Add derivative w.r.t. own parameter if it's tunable.
      if (parallelism_parameter && (*parallelism_parameter)->state->tunable) {
        (*gradient)[long_name()] =
            -output_time_der * self_processing_time / Square(parallelism) +
            buffer_size_der -
            output_time_der * inputs_gradient[kInputTimeDerivativeKey] *
                self_processing_time / Square(parallelism);
      } else if (buffer_size_parameter &&
                 (*buffer_size_parameter)->state->tunable) {
        (*gradient)[long_name()] = buffer_size_der;
      }
      return result;
    }
    double output_time =
        self_processing_time / parallelism +
        ratio_ * OutputTimeForInputs(input_times, /*gradient=*/nullptr);
    return ComputeWaitTime(output_time, old_input_time, buffer_size,
                           /*output_time_derivative=*/nullptr,
                           /*input_time_derivative=*/nullptr,
                           /*buffer_size_derivative=*/nullptr);
  }

  // The processing time is the sum of the self processing time and the product
  // of `ratio_` and the sum of processing times of inputs.
  double TotalProcessingTimeLocked(std::map<string, double>* processing_times)
      override SHARED_LOCKS_REQUIRED(mu_) {
    double self_processing_time = SelfProcessingTimeLocked();
    if (processing_times) {
      (*processing_times)[long_name()] = self_processing_time;
    }
    return self_processing_time +
           ratio_ * TotalProcessingTimeForInputs(processing_times);
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
      SHARED_LOCKS_REQUIRED(mu_) {
    return std::make_shared<UnknownRatio>(Args{id_, name_, std::move(output)});
  }

  // The output time is the sum of the self processing time and the product of
  // the ratio estimate and the sum of output times of inputs.
  double OutputTimeLocked(std::vector<double>* input_times,
                          std::map<string, double>* gradient) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    if (num_elements_ == 0 || inputs_.empty() ||
        inputs_.front()->num_elements() == 0) {
      return SelfProcessingTimeLocked();
    }
    // TODO(jsimsa): The current implementation assumes that the number of input
    // elements consumed per output is the same across all inputs.
    std::shared_ptr<Node> input = inputs_.front();
    double ratio = static_cast<double>(input->num_elements()) /
                   static_cast<double>(num_elements_);
    double old_input_time = input_times->back();
    input_times->back() = (old_input_time + SelfProcessingTimeLocked()) / ratio;
    auto cleanup = gtl::MakeCleanup([input_times, old_input_time]() {
      input_times->back() = old_input_time;
    });
    if (gradient) {
      std::map<string, double> inputs_gradient;
      double result =
          SelfProcessingTimeLocked() +
          ratio * OutputTimeForInputs(input_times, &inputs_gradient);
      auto last_input_time_der =
          gtl::FindWithDefault(*gradient, kInputTimeDerivativeKey, 0.0L);
      (*gradient)[kInputTimeDerivativeKey] =
          last_input_time_der +
          inputs_gradient[kInputTimeDerivativeKey] / ratio;
      for (auto& pair : inputs_gradient) {
        if (pair.first != kInputTimeDerivativeKey) {
          (*gradient)[pair.first] = pair.second * ratio;
        }
      }
      return result;
    }
    return SelfProcessingTimeLocked() +
           ratio * OutputTimeForInputs(input_times, /*gradient=*/nullptr);
  }

  // The processing time is the sum of the self processing time and the product
  // of the ratio estimate and the sum of processing times of inputs.
  double TotalProcessingTimeLocked(std::map<string, double>* processing_times)
      override SHARED_LOCKS_REQUIRED(mu_) {
    double self_processing_time = SelfProcessingTimeLocked();
    if (processing_times) {
      (*processing_times)[long_name()] = self_processing_time;
    }
    if (inputs_.empty() || num_elements_ == 0) {
      return self_processing_time;
    }
    // TODO(jsimsa): The current implementation assumes that the number of input
    // elements consumed per output is the same across all inputs.
    std::shared_ptr<Node> input = inputs_.front();
    double ratio = static_cast<double>(input->num_elements()) /
                   static_cast<double>(num_elements_);
    return self_processing_time +
           ratio * TotalProcessingTimeForInputs(processing_times);
  }
};

class Unknown : public Node {
 public:
  using Node::Node;

  virtual ~Unknown() {}

 protected:
  std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    return std::make_shared<Unknown>(Args{id_, name_, std::move(output)});
  }

  // The output time is the sum of output times of inputs.
  double OutputTimeLocked(std::vector<double>* input_times,
                          std::map<string, double>* gradient) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    return OutputTimeForInputs(input_times, gradient);
  }

  // The processing time is the sum of processing times of inputs.
  double TotalProcessingTimeLocked(std::map<string, double>* processing_times)
      override SHARED_LOCKS_REQUIRED(mu_) {
    return TotalProcessingTimeForInputs(processing_times);
  }
};

}  // namespace

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

std::shared_ptr<Node> Model::AddNode(Node::Factory factory, const string& name,
                                     const string& output_name) {
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
  return node;
}

void Model::AddProcessingTime(const string& name, int64 delta) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (node) {
    (*node)->add_processing_time(delta);
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

void Model::RecordElement(const string& name) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (node) {
    (*node)->record_element();
  }
}

int64 Model::NumElements(const string& name) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (node) {
    return (*node)->num_elements();
  }
  return 0;
}

void Model::RecordStart(const string& name, bool stop_output) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (collect_resource_usage_ && node) {
    int64 now_nanos = absl::GetCurrentTimeNanos();
    if (stop_output && (*node)->output()) {
      (*node)->output()->record_stop(now_nanos);
    }
    (*node)->record_start(now_nanos);
  }
}

void Model::RecordStop(const string& name, bool start_output) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (collect_resource_usage_ && node) {
    int64 now_nanos = absl::GetCurrentTimeNanos();
    (*node)->record_stop(now_nanos);
    if (start_output && (*node)->output()) {
      (*node)->output()->record_start(now_nanos);
    }
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
    remove_node_hook_(*node);
  }
  lookup_table_.erase(name);
}

std::map<string, std::shared_ptr<Parameter>> Model::CollectTunableParameters(
    std::shared_ptr<Node> node) {
  std::map<string, std::shared_ptr<Parameter>> parameters;
  node->CollectTunableParameters(&parameters);
  return parameters;
}

std::map<string, std::shared_ptr<Parameter>> Model::CollectEssentialParallelism(
    std::shared_ptr<Node> node) {
  // Parallelism parameter is considered to be essential if the coressponding
  // transformations's processing time is greater than essential rate times the
  // average transformation self processing time.
  constexpr double kEssentialRate = 0.3L;

  std::map<string, std::shared_ptr<Parameter>> parameters;
  node->CollectTunableParameters(&parameters);
  std::map<string, double> processing_times;
  double processing_time = node->TotalProcessingTime(&processing_times);
  double uniform_share =
      processing_time / static_cast<double>(processing_times.size());
  std::map<string, std::shared_ptr<Parameter>> essential_parameters;
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
    std::map<string, double> gradient;
    new_output_time = OutputTime(snapshot, &gradient);
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
            std::max(max_abs_derivative, std::abs(gradient[pair.first]));
      }
    }
    for (auto& pair : parameters) {
      new_value = pair.second->value -
                  kDescentStep * gradient[pair.first] / max_abs_derivative;
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
    const double output_time = OutputTime(snapshot, /*gradient=*/nullptr);
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
      double new_output_time = OutputTime(snapshot, /*gradient=*/nullptr);
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
                         std::map<string, double>* gradient) {
  std::vector<double> input_times(1, 0);
  // TODO(jsimsa): Now that we are accounting for buffer size in wait time
  // computation, assuming that the input is infinitely fast will result in
  // inaccurate estimates of the output latency.
  //
  // We should compute the output latency as a fix-point of the following
  // equation: `output_time = node(OutputTime(input_times(1, output_time))`.
  return node->OutputTime(&input_times, gradient);
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
