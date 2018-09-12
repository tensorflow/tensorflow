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

#include "tensorflow/core/framework/model.h"

namespace tensorflow {
namespace data {
namespace model {

// TODO(jsimsa): Use `Node` subclassing instead of types and node statements.
void Node::CollectKnobs(std::vector<Node::Knob>* knobs) {
  mutex_lock l(mu_);
  switch (type_) {
    case Type::PARALLEL_INTERLEAVE_V2: {
      for (auto input : inputs_) {
        input->CollectKnobs(knobs);
      }
      int64 processing_time = static_cast<int64>(
          static_cast<double>(ProcessingTimeLocked() -
                              inputs_.front()->ProcessingTime()) /
          static_cast<double>(inputs_.size() - 1));
      knobs->emplace_back(
          Node::Knob{this, processing_time, metadata_["parallelism"]});
      return;
    }
    case Type::MAP_AND_BATCH:
    case Type::PARALLEL_MAP: {
      for (auto input : inputs_) {
        input->CollectKnobs(knobs);
      }
      knobs->emplace_back(
          Node::Knob{this, NanosPerElementLocked(), metadata_["parallelism"]});
      return;
    }
    case Type::BATCH:
    case Type::CACHE:
    case Type::CONCATENATE:
    case Type::FILTER:
    case Type::FLAT_MAP:
    case Type::INTERLEAVE:
    case Type::MAP:
    case Type::PADDED_BATCH:
    case Type::PARALLEL_INTERLEAVE:
    case Type::PREFETCH:
    case Type::REPEAT:
    case Type::SHUFFLE:
    case Type::SKIP:
    case Type::TAKE:
    case Type::ZIP: {
      for (auto input : inputs_) {
        input->CollectKnobs(knobs);
      }
      return;
    }
    default:
      return;
  }
}

int64 Node::ProcessingTimeLocked() {
  switch (type_) {
    case Type::BATCH:
    case Type::MAP_AND_BATCH:
    case Type::PADDED_BATCH: {
      int64 batch_size = metadata_["batch_size"];
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

int64 Node::OutputTimeLocked(std::vector<int64>* input_times) {
  switch (type_) {
    case Type::BATCH:
    case Type::PADDED_BATCH: {
      double batch_size = metadata_["batch_size"];
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
      double batch_size = metadata_["batch_size"];
      double parallelism = metadata_["parallelism"];
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
    case Type::PARALLEL_INTERLEAVE:
    case Type::PARALLEL_INTERLEAVE_V2: {
      // TODO(jsimsa): model the first input
      if (inputs_.size() <= 1) {
        return NanosPerElementLocked();
      }
      int64 delta =
          static_cast<int64>(static_cast<double>(NanosPerElementLocked()) *
                             static_cast<double>(inputs_.size() - 1));
      input_times->push_back(delta);
      auto cleanup =
          gtl::MakeCleanup([input_times]() { input_times->pop_back(); });
      int64 inputs_output_time = OutputTimeForInputs(input_times) -
                                 inputs_.front()->OutputTime(input_times);
      double parallelism = std::min(port::NumSchedulableCPUs(),
                                    static_cast<int>(metadata_["parallelism"]));
      int64 output_time =
          NanosPerElementLocked() + ((static_cast<double>(inputs_output_time) /
                                      static_cast<double>(inputs_.size() - 1)) /
                                     parallelism);
      return std::max(0LL,
                      output_time - input_times->at(input_times->size() - 2));
    }
    case Type::PARALLEL_MAP: {
      double parallelism = std::min(port::NumSchedulableCPUs(),
                                    static_cast<int>(metadata_["parallelism"]));
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

Model::Model(const proto::Model& model_proto) {
  id_counter_ = model_proto.id_counter();
  std::map<int64, std::shared_ptr<Node>> lookup_table;
  for (auto node_proto : model_proto.node()) {
    std::shared_ptr<Node> node(new Node(node_proto));
    lookup_table[node_proto.id()] = node;
  }
  for (auto node_proto : model_proto.node()) {
    std::shared_ptr<Node> node = lookup_table[node_proto.id()];
    for (int64 id : node_proto.input()) {
      node->add_input(lookup_table[id]);
    }
    node->set_output(lookup_table[node_proto.output()]);
  }
  output_ = lookup_table[model_proto.output()];
}

std::shared_ptr<Node> Model::AddNode(const string& name,
                                     const string& output_name) {
  mutex_lock l(mu_);
  std::shared_ptr<Node> output;
  auto it = lookup_table_.find(output_name);
  if (it != lookup_table_.end()) {
    output = it->second;
  }
  std::shared_ptr<Node> node(new Node(id_counter_++, output));
  if (!output_) {
    output_ = node;
  }
  if (output) {
    output->add_input(node);
  }
  lookup_table_.insert(std::make_pair(name, node));
  return node;
}

std::shared_ptr<Node> Model::LookupNode(const string& name) {
  tf_shared_lock l(mu_);
  std::shared_ptr<Node> result;
  auto it = lookup_table_.find(name);
  if (it != lookup_table_.end()) {
    result = it->second;
  }
  return result;
}

void Model::Optimize() {
  mutex_lock l(mu_);
  int64 processing_time = ProcessingTime();
  int64 num_cpus = port::NumSchedulableCPUs();
  std::vector<Node::Knob> knobs = CollectKnobs();
  // The optimization algorithm starts by setting all parallelism knobs to 1. It
  // then repeatedly identifies the knob that, when turned up by 1, decreases
  // the output time the most. This process is repeated until all knobs reach
  // the number of schedulable CPUs or the projected output time is less than or
  // equal to the processing time needed to produce an element divided by the
  // number of schedulable CPUs.
  for (auto& knob : knobs) {
    LOG(INFO) << knob.node->name() << " " << knob.processing_time;
    knob.value = 1;
    knob.node->set_metadata("parallelism", knob.value);
  }
  while (true) {
    int64 output_time = OutputTime();
    bool all_knobs = true;
    for (auto knob : knobs) {
      if (knob.value < num_cpus) {
        all_knobs = false;
        break;
      }
    }
    if (output_time < processing_time / num_cpus || all_knobs) {
      break;
    }
    int64 best_delta = -1;
    int best_knob = -1;
    for (int i = 0; i < knobs.size(); ++i) {
      if (knobs[i].value == num_cpus) {
        continue;
      }
      knobs[i].node->set_metadata("parallelism", knobs[i].value + 1);
      int64 delta = output_time - OutputTime();
      if (delta > best_delta) {
        best_delta = delta;
        best_knob = i;
      }
      knobs[i].node->set_metadata("parallelism", knobs[i].value);
    }
    knobs[best_knob].value++;
    knobs[best_knob].node->set_metadata("parallelism", knobs[best_knob].value);
  }
  for (auto knob : knobs) {
    LOG(INFO) << knob.node->name() << " " << knob.value;
  }
  LOG(INFO) << "output time: " << OutputTime();
  LOG(INFO) << "processing time: " << ProcessingTime();
}

void Model::OutputToFile() {
  proto::Model model_proto;
  ToProto(&model_proto);
  string filename;
  Env::Default()->LocalTempFilename(&filename);
  TF_CHECK_OK(WriteStringToFile(Env::Default(), filename,
                                model_proto.SerializeAsString()));
  LOG(INFO) << filename;
}

void Model::RemoveNode(const string& prefix) {
  mutex_lock l(mu_);
  lookup_table_.erase(prefix);
}

void Model::ToProto(proto::Model* model_proto) {
  mutex_lock l(mu_);
  model_proto->set_id_counter(id_counter_);
  model_proto->set_output(output_->id());
  AddNodeToProto(output_, model_proto);
}

// static
void Model::AddNodeToProto(const std::shared_ptr<Node>& node,
                           proto::Model* model_proto) {
  proto::Node* node_proto = model_proto->add_node();
  node->ToProto(node_proto);
  for (const std::shared_ptr<Node>& input : node->inputs()) {
    AddNodeToProto(input, model_proto);
  }
}

std::vector<Node::Knob> Model::CollectKnobs() {
  std::vector<Node::Knob> knobs;
  output_->CollectKnobs(&knobs);
  return knobs;
}

int64 Model::OutputTime() {
  std::vector<int64> input_times(1, 0);
  return output_->OutputTime(&input_times);
}

int64 Model::ProcessingTime() { return output_->ProcessingTime(); }

}  // namespace model
}  // namespace data
}  // namespace tensorflow
