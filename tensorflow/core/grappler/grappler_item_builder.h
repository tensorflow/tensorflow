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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPPLER_ITEM_BUILDER_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPPLER_ITEM_BUILDER_H_

#include <memory>
#include <set>
#include <string>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {

class MetaGraphDef;

namespace grappler {

struct ItemConfig {
  ItemConfig() {}

  // If true, ignore all user specified node placement.
  bool ignore_user_placement = true;
  // If true, ignore all user specified colocation attributes.
  bool ignore_colocation = true;
  // Dimension to use if a placeholder node has an _output_shapes attribute with
  // a dimension of -1.
  int placeholder_unknown_output_shape_dim = -1;
  // If true, erases all "_noinline" attributes from user-defined functions.
  // Has no effect if "inline_functions" is disabled.
  bool erase_noinline_attributes = false;
  // If non-empty, override the directory of asset paths.
  string assets_directory_override;
  // If true, runs ModelPruner on the graph.
  bool prune_graph = false;
  // Override feed nodes list.
  std::set<string> feed_nodes;
  // Override fetch nodes list.
  std::set<string> fetch_nodes;

  // Configs for graph optimizations from common_runtime. This is NOT Grappler
  // function optimizer. When Grappler is invoked at runtime, it is typically
  // running after common_runtime pass.
  //
  // If true, does L1 optimizations.
  bool apply_optimizations = false;
  // If true, does function inlining.
  bool inline_functions = false;
};

// Method for optimizing the graph def (including function inlining and other
// optimizations). This is optimizations from common_runtime, NOT Grappler
// function optimizer.
Status RuntimeGraphOptimizer(const GraphDef& graph_def_arg,
                             GraphDef* output_graph_def, const ItemConfig& cfg);

// Factory method for creating a GrapplerItem from a MetaGraphDef.
// Returns nullptr if the given meta_graph cannot be converted.
std::unique_ptr<GrapplerItem> GrapplerItemFromMetaGraphDef(
    const string& id, const MetaGraphDef& meta_graph, const ItemConfig& cfg);

// Factory method for creating a GrapplerItem from a file
// containing a MetaGraphDef in either binary or text format.
// Returns nullptr if the given meta_graph cannot be converted.
std::unique_ptr<GrapplerItem> GrapplerItemFromMetaGraphDefFile(
    const string& id, const string& meta_graph_file, const ItemConfig& cfg);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPPLER_ITEM_BUILDER_H_
