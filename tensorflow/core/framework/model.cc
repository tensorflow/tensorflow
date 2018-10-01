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

// TODO(jsimsa): Use `Node` subclassing instead of types and node statements.
void Model::Node::CollectTunables(
    std::vector<std::shared_ptr<Node::Tunable>>* tunables) {
  tf_shared_lock l(mu_);
  for (auto input : inputs_) {
    input->CollectTunables(tunables);
  }
  switch (type_) {
    case Type::MAP_AND_BATCH:
    case Type::PARALLEL_INTERLEAVE_V2:
    case Type::PARALLEL_MAP: {
      if (auto* tunable_param =
              gtl::FindOrNull(tunable_params_, "parallelism")) {
        tunables->push_back(*tunable_param);
      }
      return;
    }
    default:
      return;
  }
}

int64 Model::Node::GetParameterValue(const string& name) {
  if (auto* tunable_param = gtl::FindOrNull(tunable_params_, name)) {
    return (*tunable_param)->value;
  }
  return constant_params_[name];
}

int64 Model::Node::ProcessingTimeLocked() {
  switch (type_) {
    case Type::BATCH:
    case Type::MAP_AND_BATCH:
    case Type::PADDED_BATCH: {
      int64 batch_size = GetParameterValue("batch_size");
      return NanosPerElementLocked() + batch_size * ProcessingTimeForInputs();
    }
    case Type::FILTER: {
      std::shared_ptr<Node> input = inputs_.front();
      double ratio = static_cast<double>(input->num_elements()) /
                     static_cast<double>(num_elements_);
      return NanosPerElementLocked() +
             static_cast<int64>(ratio *
                                static_cast<double>(ProcessingTimeForInputs()));
    }
    case Type::FLAT_MAP:
    case Type::INTERLEAVE:
    case Type::PARALLEL_INTERLEAVE:
    case Type::PARALLEL_INTERLEAVE_V2: {
      // TODO(jsimsa): model the first input
      // TODO(jsimsa): use processing time history as a prior for future inputs
      if (inputs_.size() <= 1) {
        return NanosPerElementLocked();
      }
      int64 processing_time =
          ProcessingTimeForInputs() - inputs_.front()->ProcessingTime();
      return NanosPerElementLocked() +
             static_cast<double>(processing_time) /
                 static_cast<double>(inputs_.size() - 1);
    }
    case Type::CACHE:
    case Type::CONCATENATE:
    case Type::MAP:
    case Type::PARALLEL_MAP:
    case Type::PREFETCH:
      // TODO(jsimsa): use processing time history as a prior for future inputs
    case Type::REPEAT:
    case Type::SHUFFLE:
    case Type::SKIP:
    case Type::TAKE:
    case Type::ZIP: {
      return NanosPerElementLocked() + ProcessingTimeForInputs();
    }
    default:
      return NanosPerElementLocked();
  }
}

int64 Model::Node::OutputTimeLocked(std::vector<int64>* input_times) {
  switch (type_) {
    case Type::BATCH:
    case Type::PADDED_BATCH: {
      double batch_size = GetParameterValue("batch_size");
      int64 old_value = (*input_times)[input_times->size() - 1];
      (*input_times)[input_times->size() - 1] = static_cast<int64>(
          static_cast<double>(old_value + NanosPerElementLocked()) /
          batch_size);
      auto cleanup = gtl::MakeCleanup([input_times, old_value]() {
        (*input_times)[input_times->size() - 1] = old_value;
      });
      return NanosPerElementLocked() +
             batch_size * OutputTimeForInputs(input_times);
    }
    case Type::FILTER: {
      std::shared_ptr<Node> input = inputs_.front();
      int64 old_value = (*input_times)[input_times->size() - 1];
      double ratio = static_cast<double>(input->num_elements()) /
                     static_cast<double>(num_elements_);
      (*input_times)[input_times->size() - 1] = static_cast<int64>(
          static_cast<double>(old_value + NanosPerElementLocked()) / ratio);
      auto cleanup = gtl::MakeCleanup([input_times, old_value]() {
        (*input_times)[input_times->size() - 1] = old_value;
      });
      return NanosPerElementLocked() +
             static_cast<int64>(
                 static_cast<double>(OutputTimeForInputs(input_times)) * ratio);
    }
    case Type::FLAT_MAP:
    case Type::INTERLEAVE: {
      // TODO(jsimsa): model the first input
      // TODO(jsimsa): use cycle length metadata instead of `inputs_.size() - 1`
      if (inputs_.size() <= 1) {
        return NanosPerElementLocked();
      }
      int64 delta =
          static_cast<int64>(static_cast<double>(NanosPerElementLocked()) *
                             static_cast<double>(inputs_.size() - 1));
      (*input_times)[input_times->size() - 1] += delta;
      auto cleanup = gtl::MakeCleanup([input_times, delta]() {
        (*input_times)[input_times->size() - 1] -= delta;
      });
      int64 output_time = OutputTimeForInputs(input_times) -
                          inputs_.front()->OutputTime(input_times);
      return NanosPerElementLocked() +
             static_cast<double>(output_time) /
                 static_cast<double>(inputs_.size() - 1);
    }
    case Type::MAP_AND_BATCH: {
      double batch_size = GetParameterValue("batch_size");
      double parallelism = GetParameterValue("parallelism");
      int64 delta =
          static_cast<int64>(static_cast<double>(NanosPerElementLocked()) /
                             (batch_size * parallelism));
      input_times->push_back(delta);
      auto cleanup =
          gtl::MakeCleanup([input_times]() { input_times->pop_back(); });
      int64 output_time = static_cast<int64>(
          static_cast<double>(NanosPerElementLocked()) / parallelism +
          batch_size * OutputTimeForInputs(input_times));
      return std::max(0LL,
                      output_time - input_times->at(input_times->size() - 2));
    }
    case Type::PARALLEL_INTERLEAVE: {
      // TODO(jsimsa): model the first input
      if (inputs_.size() <= 1) {
        return NanosPerElementLocked();
      }
      int64 delta = static_cast<double>(NanosPerElementLocked()) *
                    static_cast<double>(inputs_.size() - 1);
      input_times->push_back(delta);
      auto cleanup =
          gtl::MakeCleanup([input_times]() { input_times->pop_back(); });
      int64 inputs_output_time = OutputTimeForInputs(input_times) -
                                 inputs_.front()->OutputTime(input_times);
      double parallelism = GetParameterValue("parallelism");
      int64 output_time =
          NanosPerElementLocked() + ((static_cast<double>(inputs_output_time) /
                                      static_cast<double>(inputs_.size() - 1)) /
                                     parallelism);
      return std::max(0LL,
                      output_time - input_times->at(input_times->size() - 2));
    }
    case Type::PARALLEL_INTERLEAVE_V2: {
      // TODO(jsimsa): model the first input
      if (inputs_.size() <= 1) {
        return NanosPerElementLocked();
      }
      int64 delta = static_cast<double>(NanosPerElementLocked()) *
                    static_cast<double>(inputs_.size() - 1);
      input_times->push_back(delta);
      auto cleanup =
          gtl::MakeCleanup([input_times]() { input_times->pop_back(); });
      int64 inputs_output_time = OutputTimeForInputs(input_times) -
                                 inputs_.front()->OutputTime(input_times);
      double parallelism =
          std::min(static_cast<int>(GetParameterValue("cycle_length")),
                   static_cast<int>(GetParameterValue("parallelism")));
      int64 output_time =
          NanosPerElementLocked() + ((static_cast<double>(inputs_output_time) /
                                      static_cast<double>(inputs_.size() - 1)) /
                                     parallelism);
      return std::max(0LL,
                      output_time - input_times->at(input_times->size() - 2));
    }
    case Type::PARALLEL_MAP: {
      double parallelism =
          std::min(port::NumSchedulableCPUs(),
                   static_cast<int>(GetParameterValue("parallelism")));
      int64 delta = static_cast<int64>(
          static_cast<double>(NanosPerElementLocked()) / parallelism);
      input_times->push_back(delta);
      auto cleanup =
          gtl::MakeCleanup([input_times]() { input_times->pop_back(); });
      int64 output_time =
          static_cast<double>(NanosPerElementLocked()) / parallelism +
          OutputTimeForInputs(input_times);
      return std::max(0LL,
                      output_time - input_times->at(input_times->size() - 2));
    }
    case Type::PREFETCH: {
      int64 delta = NanosPerElementLocked();
      input_times->push_back(delta);
      auto cleanup =
          gtl::MakeCleanup([input_times]() { input_times->pop_back(); });
      return std::max(0LL, NanosPerElementLocked() +
                               OutputTimeForInputs(input_times) -
                               input_times->at(input_times->size() - 2));
    }
    case Type::CACHE:
    case Type::CONCATENATE:
    case Type::MAP:
    case Type::REPEAT:
    case Type::SHUFFLE:
    case Type::SKIP:
    case Type::TAKE:
    case Type::ZIP: {
      int64 delta = NanosPerElementLocked();
      (*input_times)[input_times->size() - 1] += delta;
      auto cleanup = gtl::MakeCleanup([input_times, delta]() {
        (*input_times)[input_times->size() - 1] -= delta;
      });
      return NanosPerElementLocked() + OutputTimeForInputs(input_times);
    }
    default:
      return NanosPerElementLocked();
  }
}

void Model::AddConstantParameter(const string& node_name,
                                 const string& parameter_name, int64 value) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, node_name);
  if (node) {
    (*node)->add_constant_param(parameter_name, value);
  }
}

void Model::AddNode(const string& name, const string& output_name) {
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
  std::shared_ptr<Node> node(new Node(id_counter_++, tokens.back(), output));
  if (!output_) {
    output_ = node;
  }
  if (output) {
    output->add_input(node);
  }
  lookup_table_.insert(std::make_pair(name, node));
}

void Model::AddProcessingTime(const string& name, int64 delta) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (node) {
    (*node)->add_processing_time(delta);
  }
}

void Model::AddTunableParameter(const string& node_name,
                                const string& parameter_name,
                                std::atomic<int64>* value, int64 min, int64 max,
                                condition_variable* cond_var) {
  tf_shared_lock l(mu_);
  auto node = *gtl::FindOrNull(lookup_table_, node_name);
  DCHECK(node);
  node->add_tunable_param(parameter_name, value, min, max, cond_var);
}

// The optimization algorithm starts by setting all tunable parallelism
// parameters to 1. It then repeatedly identifies the parameter whose increase
// in parallelism decreases the output time the most. This process is repeated
// until all parameters reach their maximum values or the projected output time
// is less than or equal to the processing time needed to produce an element
// divided by CPU budget.
void Model::Optimize(int64 cpu_budget) {
  tf_shared_lock lock(mu_);
  std::vector<std::shared_ptr<Model::Node::Tunable>> tunables;
  const int64 processing_time = ProcessingTime();
  tunables = CollectTunables();
  for (auto tunable : tunables) {
    tunable->value = 1;
  }
  while (true) {
    const int64 output_time = OutputTime();
    bool all_tunables = true;
    for (auto& tunable : tunables) {
      if (tunable->value < tunable->max) {
        all_tunables = false;
        break;
      }
    }
    if (output_time < processing_time / cpu_budget || all_tunables) {
      break;
    }
    int64 best_delta = -1;
    Model::Node::Tunable* best_tunable = nullptr;
    for (auto& tunable : tunables) {
      if (tunable->value == tunable->max) {
        continue;
      }
      tunable->value++;
      int64 delta = output_time - OutputTime();
      if (delta > best_delta) {
        best_delta = delta;
        best_tunable = tunable.get();
      }
      tunable->value--;
    }
    if (!best_tunable) {
      // NOTE: This can happen because we are performing the optimization
      // while the model data is changing. If this becomes an issue, we should
      // look into performing the optimization using a model snapshot.
      break;
    }
    best_tunable->value++;
  }
  VLOG(2) << "Number of knobs: " << tunables.size();
  for (auto& tunable : tunables) {
    VLOG(2) << "Setting tunable parameter: " << tunable->value;
    tunable->value_ptr->store(tunable->value);
    if (tunable->cond_var) {
      tunable->cond_var->notify_all();
    }
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
  if (node) {
    if (stop_output && (*node)->output()) {
      (*node)->output()->record_stop();
    }
    (*node)->record_start();
  }
}

void Model::RecordStop(const string& name, bool start_output) {
  tf_shared_lock l(mu_);
  auto node = gtl::FindOrNull(lookup_table_, name);
  if (node) {
    (*node)->record_stop();
    if (start_output && (*node)->output()) {
      (*node)->output()->record_start();
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

std::vector<std::shared_ptr<Model::Node::Tunable>> Model::CollectTunables() {
  std::vector<std::shared_ptr<Model::Node::Tunable>> tunables;
  output_->CollectTunables(&tunables);
  return tunables;
}

int64 Model::OutputTime() {
  std::vector<int64> input_times(1, 0);
  return output_->OutputTime(&input_times);
}

int64 Model::ProcessingTime() { return output_->ProcessingTime(); }

}  // namespace model
}  // namespace data
}  // namespace tensorflow
