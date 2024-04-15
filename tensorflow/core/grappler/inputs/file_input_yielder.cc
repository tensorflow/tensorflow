/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/inputs/file_input_yielder.h"

#include <memory>
#include <unordered_set>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace grappler {

FileInputYielder::FileInputYielder(const std::vector<string>& filenames,
                                   size_t max_iterations)
    : filenames_(filenames),
      current_file_(0),
      current_iteration_(0),
      max_iterations_(max_iterations),
      bad_inputs_(0) {
  CHECK_GT(filenames.size(), 0) << "List of filenames is empty.";
}

bool FileInputYielder::NextItem(GrapplerItem* item) {
  if (filenames_.size() == bad_inputs_) {
    // All the input files are bad, give up.
    return false;
  }

  if (current_file_ >= filenames_.size()) {
    if (current_iteration_ >= max_iterations_) {
      return false;
    } else {
      ++current_iteration_;
      current_file_ = 0;
      bad_inputs_ = 0;
    }
  }

  const string& filename = filenames_[current_file_];
  ++current_file_;

  if (!Env::Default()->FileExists(filename).ok()) {
    LOG(WARNING) << "Skipping non existent file " << filename;
    // Attempt to process the next item on the list
    bad_inputs_ += 1;
    return NextItem(item);
  }

  LOG(INFO) << "Loading model from " << filename;

  MetaGraphDef metagraph;
  Status s = ReadBinaryProto(Env::Default(), filename, &metagraph);
  if (!s.ok()) {
    s = ReadTextProto(Env::Default(), filename, &metagraph);
  }
  if (!s.ok()) {
    LOG(WARNING) << "Failed to read MetaGraphDef from " << filename << ": "
                 << s.ToString();
    // Attempt to process the next item on the list
    bad_inputs_ += 1;
    return NextItem(item);
  }

  if (metagraph.collection_def().count("train_op") == 0 ||
      !metagraph.collection_def().at("train_op").has_node_list() ||
      metagraph.collection_def().at("train_op").node_list().value_size() == 0) {
    LOG(ERROR) << "No train op specified";
    bad_inputs_ += 1;
    metagraph = MetaGraphDef();
    return NextItem(item);
  } else {
    std::unordered_set<string> train_ops;
    for (const string& val :
         metagraph.collection_def().at("train_op").node_list().value()) {
      train_ops.insert(NodeName(val));
    }
    std::unordered_set<string> train_ops_found;
    for (auto& node : metagraph.graph_def().node()) {
      if (train_ops.find(node.name()) != train_ops.end()) {
        train_ops_found.insert(node.name());
      }
    }
    if (train_ops_found.size() != train_ops.size()) {
      for (const auto& train_op : train_ops) {
        if (train_ops_found.find(train_op) != train_ops_found.end()) {
          LOG(ERROR) << "Non existent train op specified: " << train_op;
        }
      }
      bad_inputs_ += 1;
      metagraph = MetaGraphDef();
      return NextItem(item);
    }
  }

  const string id =
      strings::StrCat(Fingerprint64(metagraph.SerializeAsString()));

  ItemConfig cfg;
  std::unique_ptr<GrapplerItem> new_item =
      GrapplerItemFromMetaGraphDef(id, metagraph, cfg);
  if (new_item == nullptr) {
    bad_inputs_ += 1;
    metagraph = MetaGraphDef();
    return NextItem(item);
  }

  *item = std::move(*new_item);
  return true;
}

}  // end namespace grappler
}  // end namespace tensorflow
