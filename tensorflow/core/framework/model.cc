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

namespace tensorflow {
namespace data {
namespace model {

std::shared_ptr<Parameter> MakeParameter(const string& name,
                                         std::shared_ptr<SharedState> state,
                                         int64 min, int64 max) {
  return std::make_shared<Parameter>(name, state, min, max);
}

namespace {

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

  int64 OutputTimeLocked(std::vector<int64>* input_times) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    if (inputs_.size() <= 1) {
      return NanosPerElementLocked();
    }
    int64 delta = NanosPerElementLocked() * (inputs_.size() - 1);
    input_times->back() += delta;
    auto cleanup = gtl::MakeCleanup(
        [input_times, delta]() { input_times->back() -= delta; });
    int64 output_time =
        static_cast<double>(OutputTimeForInputs(input_times) -
                            inputs_.front()->OutputTime(input_times)) /
        static_cast<double>(inputs_.size() - 1);
    return NanosPerElementLocked() + output_time;
  }

  int64 ProcessingTimeLocked() const override SHARED_LOCKS_REQUIRED(mu_) {
    if (inputs_.size() <= 1) {
      return NanosPerElementLocked();
    }
    int64 processing_time =
        static_cast<double>(ProcessingTimeForInputs() -
                            inputs_.front()->ProcessingTime()) /
        static_cast<double>(inputs_.size() - 1);
    return NanosPerElementLocked() + processing_time;
  }
};

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

  int64 OutputTimeLocked(std::vector<int64>* input_times) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    if (inputs_.size() <= 1) {
      return NanosPerElementLocked();
    }
    int64 old_input_time = input_times->back();
    int64 new_input_time = static_cast<double>(NanosPerElementLocked()) *
                           static_cast<double>(inputs_.size() - 1);
    input_times->push_back(new_input_time);
    auto cleanup =
        gtl::MakeCleanup([input_times]() { input_times->pop_back(); });
    double parallelism = inputs_.size() - 1;  // default to cycle length
    if (auto* parameter = gtl::FindOrNull(parameters_, "parallelism")) {
      parallelism = std::min(static_cast<int>(parallelism),
                             static_cast<int>((*parameter)->value));
    }
    int64 output_time =
        static_cast<double>(OutputTimeForInputs(input_times) -
                            inputs_.front()->OutputTime(input_times)) /
        static_cast<double>(inputs_.size() - 1) / parallelism;
    return std::max(0LL,
                    NanosPerElementLocked() + output_time - old_input_time);
  }

  int64 ProcessingTimeLocked() const override SHARED_LOCKS_REQUIRED(mu_) {
    if (inputs_.size() <= 1) {
      return NanosPerElementLocked();
    }
    int64 processing_time =
        ProcessingTimeForInputs() - inputs_.front()->ProcessingTime();
    return NanosPerElementLocked() +
           static_cast<double>(processing_time) /
               static_cast<double>(inputs_.size() - 1);
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

  int64 OutputTimeLocked(std::vector<int64>* input_times) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    if (ratio_ == 0) {
      return NanosPerElementLocked();
    }
    int64 old_input_time = input_times->back();
    input_times->back() += static_cast<int64>(
        static_cast<double>(old_input_time + NanosPerElementLocked()) / ratio_);
    auto cleanup = gtl::MakeCleanup([input_times, old_input_time]() {
      input_times->back() = old_input_time;
    });
    return NanosPerElementLocked() + ratio_ * OutputTimeForInputs(input_times);
  }

  int64 ProcessingTimeLocked() const override SHARED_LOCKS_REQUIRED(mu_) {
    return NanosPerElementLocked() + ratio_ * ProcessingTimeForInputs();
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

  int64 OutputTimeLocked(std::vector<int64>* input_times) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    double parallelism = 1.0;
    if (auto* parameter = gtl::FindOrNull(parameters_, "parallelism")) {
      parallelism = (*parameter)->value;
    }
    if (ratio_ == 0.0) {
      int64 output_time =
          static_cast<double>(NanosPerElementLocked()) / parallelism;
      return std::max(0LL, output_time - input_times->back());
    }
    int64 old_input_time = input_times->back();
    int64 new_input_time = static_cast<int64>(
        static_cast<double>(NanosPerElementLocked()) / ratio_ / parallelism);
    input_times->push_back(new_input_time);
    auto cleanup =
        gtl::MakeCleanup([input_times]() { input_times->pop_back(); });
    int64 output_time = static_cast<int64>(
        static_cast<double>(NanosPerElementLocked()) / parallelism +
        ratio_ * OutputTimeForInputs(input_times));
    return std::max(0LL, output_time - old_input_time);
  }

  int64 ProcessingTimeLocked() const override SHARED_LOCKS_REQUIRED(mu_) {
    return NanosPerElementLocked() + ratio_ * ProcessingTimeForInputs();
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

  int64 OutputTimeLocked(std::vector<int64>* input_times) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    if (num_elements_ == 0 || inputs_.empty() ||
        inputs_.front()->num_elements() == 0) {
      return NanosPerElementLocked();
    }
    // TODO(jsimsa): The current implementation assumes that the number of input
    // elements consumed per output is the same across all inputs.
    std::shared_ptr<Node> input = inputs_.front();
    double ratio = static_cast<double>(input->num_elements()) /
                   static_cast<double>(num_elements_);
    int64 old_input_time = input_times->back();
    input_times->back() =
        static_cast<double>(old_input_time + NanosPerElementLocked()) / ratio;
    auto cleanup = gtl::MakeCleanup([input_times, old_input_time]() {
      input_times->back() = old_input_time;
    });
    return NanosPerElementLocked() +
           static_cast<int64>(
               ratio * static_cast<double>(OutputTimeForInputs(input_times)));
  }

  int64 ProcessingTimeLocked() const override SHARED_LOCKS_REQUIRED(mu_) {
    if (inputs_.empty() || num_elements_ == 0) {
      return NanosPerElementLocked();
    }
    // TODO(jsimsa): The current implementation that the number of input
    // elements consumed per output is the same across all inputs.
    std::shared_ptr<Node> input = inputs_.front();
    double ratio = static_cast<double>(input->num_elements()) /
                   static_cast<double>(num_elements_);
    return NanosPerElementLocked() +
           static_cast<int64>(ratio *
                              static_cast<double>(ProcessingTimeForInputs()));
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

  int64 OutputTimeLocked(std::vector<int64>* input_times) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    return OutputTimeForInputs(input_times);
  }

  int64 ProcessingTimeLocked() const override SHARED_LOCKS_REQUIRED(mu_) {
    return ProcessingTimeForInputs();
  }
};

}  // namespace

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
    output->add_input(node);
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

// The optimization algorithm starts by setting all tunable parallelism
// parameters to 1. It then repeatedly identifies the parameter whose increase
// in parallelism decreases the output time the most. This process is repeated
// until all parameters reach their maximum values or the projected output time
// is less than or equal to the processing time needed to produce an element
// divided by CPU budget.
void Model::Optimize(int64 cpu_budget) {
  std::shared_ptr<Node> snapshot;
  {
    tf_shared_lock lock(mu_);
    snapshot = output_->Snapshot(nullptr);
  }
  const int64 processing_time = ProcessingTime(snapshot);
  auto parameters = CollectTunableParameters(snapshot);
  for (auto& parameter : parameters) {
    parameter->value = 1;
  }
  while (true) {
    const int64 output_time = OutputTime(snapshot);
    bool all_max = true;
    for (auto& parameter : parameters) {
      if (parameter->value < parameter->max) {
        all_max = false;
        break;
      }
    }
    if (output_time < processing_time / cpu_budget || all_max) {
      break;
    }
    int64 best_delta = -1;
    Parameter* best_parameter = nullptr;
    for (auto& parameter : parameters) {
      if (parameter->value == parameter->max) {
        continue;
      }
      parameter->value++;
      int64 delta = output_time - OutputTime(snapshot);
      if (delta > best_delta) {
        best_delta = delta;
        best_parameter = parameter.get();
      }
      parameter->value--;
    }
    if (!best_parameter) {
      // This should never happen because we are using a model snapshot and
      // the output time is monotonically decreasing w.r.t. parallelism.
      LOG(WARNING) << "Failed to find a tunable parameter that would "
                      "decrease the output time, aborting the current "
                      "optimization attempt.";
      return;
    }
    best_parameter->value++;
  }
  VLOG(2) << "Number of tunable parameters: " << parameters.size();
  for (auto& parameter : parameters) {
    VLOG(2) << "Setting tunable parameter: " << parameter->value;
    mutex_lock l(*parameter->state->mu);
    parameter->state->value = parameter->value;
    parameter->state->cond_var->notify_all();
  }
}

void Model::RecordElement(const string& name) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (node) {
    (*node)->record_element();
  }
}

void Model::RecordStart(const string& name, bool stop_output) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (collect_resource_usage_ && node) {
    int64 now_nanos = Env::Default()->NowNanos();
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
    int64 now_nanos = Env::Default()->NowNanos();
    (*node)->record_stop(now_nanos);
    if (start_output && (*node)->output()) {
      (*node)->output()->record_start(now_nanos);
    }
  }
}

void Model::RemoveNode(const string& name) {
  mutex_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (node && (*node)->output()) {
    (*node)->output()->remove_input(*node);
  }
  lookup_table_.erase(name);
}

std::vector<std::shared_ptr<Parameter>> Model::CollectTunableParameters(
    std::shared_ptr<Node> node) {
  std::vector<std::shared_ptr<Parameter>> parameters;
  node->CollectTunableParameters(&parameters);
  return parameters;
}

int64 Model::OutputTime(std::shared_ptr<Node> node) {
  std::vector<int64> input_times(1, 0);
  return node->OutputTime(&input_times);
}

int64 Model::ProcessingTime(std::shared_ptr<Node> node) {
  return node->ProcessingTime();
}

}  // namespace model
}  // namespace data
}  // namespace tensorflow
