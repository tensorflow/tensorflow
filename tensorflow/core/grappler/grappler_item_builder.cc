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
#include "tensorflow/core/grappler/grappler_item_builder.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variable.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/inputs/utils.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace grappler {

namespace {

void InitializeTensor(DataType type, Tensor* tensor) {
  const int period = 7;
  if (type == DT_FLOAT) {
    auto flat = tensor->flat<float>();
    // Populate numbers 0, 0.1, 0.2, ..., 0.5, 0.6, 0, 0.1, 0.2, ...
    for (int i = 0; i < flat.size(); i++) {
      flat(i) = static_cast<float>(i % period) / 10.0f;
    }
  } else if (type == DT_INT64) {
    auto flat = tensor->flat<int64>();
    // Populate numbers 0, 1, 2, ..., 5, 6, 0, 1, 2, ...
    for (int i = 0; i < flat.size(); i++) {
      flat(i) = i % period;
    }
  } else {
    memset(const_cast<char*>(tensor->tensor_data().data()), 0,
           tensor->tensor_data().size());
  }
}

// Optimize the graph def (including function inlining and other optimizations).
// This is a temporary change that optimizes the graph in context of a single
// gpu machine. Down the line, we may want to make grappler_item_builder aware
// of the cluster type (E.g: single cpu, multiple gpu, etc)  being simulated in
// order to get the correct session options and environment, and performing the
// correct optimizations.
Status OptimizeGraph(const GraphDef& graph_def, GraphDef* output_graph_def,
                     const ItemConfig& cfg) {
  if (!cfg.apply_optimizations && !cfg.inline_functions) {
    return Status::OK();
  }

  // Create a session option for a single GPU device.
  SessionOptions options;

  // Inline all functions.
  GraphDef inlined_graph_def(graph_def);

  // Instantiate all variables for function library runtime creation.
  std::vector<Device*> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
      options, "/job:localhost/replica:0/task:0", &devices));
  std::unique_ptr<DeviceMgr> dvc_mgr(new DeviceMgr(devices));
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             inlined_graph_def.library());
  Env* env = Env::Default();

  // Optimizer options: L1 and inlining. L1 is default.
  OptimizerOptions* optimizer_opts =
      options.config.mutable_graph_options()->mutable_optimizer_options();
  if (cfg.apply_optimizations) {
    optimizer_opts->set_opt_level(::tensorflow::OptimizerOptions_Level_L1);
  } else {
    optimizer_opts->set_opt_level(::tensorflow::OptimizerOptions_Level_L0);
  }
  optimizer_opts->set_do_function_inlining(cfg.inline_functions);

  // Create the function library runtime.
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(dvc_mgr.get(), env,
                                        inlined_graph_def.versions().producer(),
                                        &function_library, *optimizer_opts));
  FunctionLibraryRuntime* flr = pflr->GetFLR(devices[0]->name());

  // Create the GraphOptimizer to optimize the graph def.
  GraphConstructorOptions graph_ctor_opts;
  graph_ctor_opts.allow_internal_ops = true;
  graph_ctor_opts.expect_device_spec = false;
  std::unique_ptr<Graph> graphptr(new Graph(function_library));
  // Populate default attrs to the NodeDefs in the GraphDef.
  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&inlined_graph_def,
                                               *graphptr->op_registry(), 0));

  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(graph_ctor_opts, inlined_graph_def,
                                            graphptr.get()));

  // Optimize the graph.
  GraphOptimizer optimizer(*optimizer_opts);
  optimizer.Optimize(flr, env, devices[0], &graphptr, /*shape_map=*/nullptr);
  graphptr->ToGraphDef(output_graph_def);

  return Status::OK();
}
}  // namespace

// static
std::unique_ptr<GrapplerItem> GrapplerItemFromMetaGraphDef(
    const string& id, const MetaGraphDef& meta_graph, const ItemConfig& cfg) {
  if (id.empty()) {
    LOG(ERROR) << "id must be non-empty.";
    return nullptr;
  }
  std::unique_ptr<GrapplerItem> new_item(new GrapplerItem());
  new_item->id = id;
  new_item->graph = meta_graph.graph_def();

  // Attempt to detect the fetch node(s).
  if (meta_graph.collection_def().count("train_op") > 0) {
    const CollectionDef& nodes = meta_graph.collection_def().at("train_op");
    if (nodes.has_node_list()) {
      for (const auto& node : nodes.node_list().value()) {
        const string name = NodeName(node);
        if (name.empty()) {
          LOG(ERROR) << "Invalid fetch node name " << node
                     << ", skipping this input";
          return nullptr;
        }
        LOG(INFO) << "Will use fetch node " << name;
        new_item->fetch.push_back(name);
      }
    }
  }
  if (new_item->fetch.empty()) {
    LOG(ERROR) << "Failed to detect the fetch node(s), skipping this input";
    return nullptr;
  }

  for (auto& node : *new_item->graph.mutable_node()) {
    if (IsPlaceholder(node)) {
      if (node.attr().count("dtype") == 0) {
        LOG(ERROR) << "Unknown type for placeholder " << node.name()
                   << ", skipping this input";
        return nullptr;
      }
      DataType type = node.attr().at("dtype").type();

      if (node.attr().count("shape") == 0) {
        LOG(INFO) << "Unknown shape for placeholder " << node.name()
                  << ", skipping this input";
        return nullptr;
      }

      // Replace all unknown dimensions in the placeholder's tensorshape proto
      // with cfg.placeholder_unknown_output_shape_dim and create a tensorshape
      // from it. We do this because in newer protos, the input placeholder
      // shape is not empty if the shape is partially defined.
      TensorShape shape;
      TensorShapeProto shape_proto;
      std::vector<int32> dims;
      for (const auto& dim_proto : node.attr().at("shape").shape().dim()) {
        if (cfg.placeholder_unknown_output_shape_dim >= 0 &&
            dim_proto.size() == -1) {
          dims.push_back(cfg.placeholder_unknown_output_shape_dim);
          shape_proto.add_dim()->set_size(
              cfg.placeholder_unknown_output_shape_dim);
        } else {
          dims.push_back(std::max<int32>(1, dim_proto.size()));
          shape_proto.add_dim()->set_size(dim_proto.size());
        }
      }
      Status make_shape_status =
          TensorShapeUtils::MakeShape(dims.data(), dims.size(), &shape);
      if (!make_shape_status.ok()) {
        LOG(ERROR) << "Invalid shape for placeholder " << node.name() << ": "
                   << make_shape_status << ", skipping this input";
        return nullptr;
      }

      // Some placeholder nodes have a mis-match between the node
      // attribute "shape" and a different node attribute "_output_shapes".
      // Specifically, a shape with shape.dims() == 0 could indicate either
      // a scalar or an unknown shape. In those cases, we check _output_shapes
      // for additional information.
      // This case is observed in the bnmt graphs. Have not observed any
      // cases where there was more than 1 _output_shapes, so limit it
      // to cases where there is only 1 _output_shapes.
      // We only do this if cfg.placeholder_unknown_output_shape_dim has
      // been set to avoid crashing non-BNMT graphs.
      if ((cfg.placeholder_unknown_output_shape_dim >= 0) &&
          (shape.dims() == 0) && (node.attr().count("_output_shapes") == 1) &&
          (node.attr().at("_output_shapes").list().shape(0).dim_size() != 0)) {
        shape.Clear();
        shape_proto.clear_dim();
        for (int dim_i = 0;
             dim_i <
             node.attr().at("_output_shapes").list().shape(0).dim_size();
             dim_i++) {
          const ::tensorflow::TensorShapeProto_Dim dim =
              node.attr().at("_output_shapes").list().shape(0).dim(dim_i);
          if (dim.size() == -1) {
            shape.AddDim(cfg.placeholder_unknown_output_shape_dim);
            shape_proto.add_dim()->set_size(
                cfg.placeholder_unknown_output_shape_dim);
          } else {
            int size = node.attr()
                           .at("_output_shapes")
                           .list()
                           .shape(0)
                           .dim(dim_i)
                           .size();
            shape.AddDim(size);
            shape_proto.add_dim()->set_size(size);
          }
        }
      }
      Tensor fake_input(type, shape);
      InitializeTensor(type, &fake_input);
      new_item->feed.emplace_back(node.name(), fake_input);
      // Set the shape of the node in the graph. This is needed for statically
      // inferring shapes and is a no-op when dynamically inferring shapes as
      // the Placeholder shape will match the shape passed from new_item->feed.
      *(node.mutable_attr()->at("shape").mutable_shape()) = shape_proto;
    }

    // Erase the recorded result of any previous shape inference to start again
    // from scratch.
    node.mutable_attr()->erase("_output_shapes");

    // Delete user specified placement if requested.
    if (cfg.ignore_user_placement) {
      node.clear_device();
    }
    // Delete colocation constraints if requested.
    if (cfg.ignore_colocation) {
      auto attr = node.mutable_attr();
      auto it = attr->find("_class");
      if (it != attr->end()) {
        attr->erase(it);
      }
    }
  }

  for (const string& var_collection :
       {"variables", "local_variables", "model_variables",
        "trainable_variables"}) {
    if (meta_graph.collection_def().count(var_collection) == 0) {
      continue;
    }
    const CollectionDef& vars = meta_graph.collection_def().at(var_collection);
    for (const auto& raw_var : vars.bytes_list().value()) {
      VariableDef var;
      var.ParseFromString(raw_var);
      if (!var.initializer_name().empty()) {
        new_item->init_ops.push_back(var.initializer_name());
      }
    }
  }

  if (meta_graph.collection_def().count("table_initializer") > 0) {
    const CollectionDef& inits =
        meta_graph.collection_def().at("table_initializer");
    if (inits.has_node_list()) {
      for (const auto& node : inits.node_list().value()) {
        new_item->init_ops.push_back(node);
        // Tables are initialized from files, which can take a long time. Add 30
        // minutes to the initialization time for each table to avoid timing
        // out.
        // TODO(bsteiner): adjust the timeout based on the file size.
        new_item->expected_init_time += 30 * 60;
      }
    }
  }

  if (meta_graph.collection_def().count("queue_runners") > 0) {
    const CollectionDef& vars = meta_graph.collection_def().at("queue_runners");
    for (const auto& raw : vars.bytes_list().value()) {
      QueueRunnerDef queue_runner;
      if (!queue_runner.ParseFromString(raw)) {
        LOG(ERROR) << "Could parse queue_runners, skipping this input";
        return nullptr;
      }
      if (queue_runner.cancel_op_name().empty()) {
        LOG(ERROR) << "Queue without a cancel op, skipping this input";
        return nullptr;
      }
      new_item->queue_runners.push_back(queue_runner);
    }
  }

  // Make sure we still can access the input files (aka "asset_filepaths") since
  // these might have been moved or deleted, the cns cell might have been shut
  // down, or we might be running as a user who does not have access to the
  // files.
  if (meta_graph.collection_def().count("asset_filepaths") > 0) {
    const CollectionDef& file_paths =
        meta_graph.collection_def().at("asset_filepaths");
    std::vector<string> paths;
    for (const auto& raw_path : file_paths.bytes_list().value()) {
      paths.push_back(raw_path);
    }
    if (!FilesExist(paths, nullptr)) {
      LOG(ERROR)
          << "Can't access one or more of the asset files, skipping this input";
      return nullptr;
    }
  }

  if (meta_graph.collection_def().count("savers") > 0) {
    const CollectionDef& savers = meta_graph.collection_def().at("savers");
    for (const auto& raw : savers.bytes_list().value()) {
      SaverDef saver;
      // Skip bad savers since we don't need saves/restores to be able to run a
      // graph.
      if (!saver.ParseFromString(raw)) {
        continue;
      }
      if (saver.filename_tensor_name().empty()) {
        continue;
      }
      new_item->save_op = saver.save_tensor_name();
      new_item->restore_op = saver.restore_op_name();
      new_item->save_restore_loc_tensor = saver.filename_tensor_name();
      // Only use the first saver since it's not clear what to do if there's
      // more than one.
      break;
    }
  } else {
    const SaverDef& saver = meta_graph.saver_def();
    new_item->save_op = saver.save_tensor_name();
    new_item->restore_op = saver.restore_op_name();
    new_item->save_restore_loc_tensor = saver.filename_tensor_name();
  }

  // Optimize the graph (function inlining, l1 optimizations, etc).
  Status optimize_status =
      OptimizeGraph(new_item->graph, &new_item->graph, cfg);
  if (!optimize_status.ok()) {
    LOG(ERROR) << "Graph preprocessing failed: " << optimize_status;
    return nullptr;
  }

  // Validate feed, fetch and init nodes
  std::unordered_set<string> nodes;
  for (const auto& node : new_item->graph.node()) {
    nodes.insert(node.name());
  }
  for (const auto& feed : new_item->feed) {
    if (nodes.find(feed.first) == nodes.end()) {
      LOG(ERROR) << "Feed node " << feed.first << " doesn't exist in graph";
      return nullptr;
    }
  }
  for (const auto& fetch : new_item->fetch) {
    if (nodes.find(fetch) == nodes.end()) {
      LOG(ERROR) << "Fetch node " << fetch << " doesn't exist in graph";
      return nullptr;
    }
  }
  for (const auto& init : new_item->init_ops) {
    if (nodes.find(init) == nodes.end()) {
      LOG(ERROR) << "Init node " << init << " doesn't exist in graph";
      return nullptr;
    }
  }
  return new_item;
}

}  // end namespace grappler
}  // end namespace tensorflow
