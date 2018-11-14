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
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variable.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/inputs/utils.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/protobuf_internal.h"
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
  } else if (type != DT_STRING && type != DT_RESOURCE && type != DT_VARIANT) {
    // DT_STRING, DT_RESOURCE and DT_VARIANT are not simple types according to
    // is_simple_type<> in tensorflow/core/framework/type_traits.h, and
    // Allocator will run non-trivial constructor/destructor for a Tensor with
    // one of these types, so we should not memset its buffer.
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
Status OptimizeGraph(const GraphDef& graph_def_arg, GraphDef* output_graph_def,
                     const ItemConfig& cfg) {
  if (!cfg.apply_optimizations && !cfg.erase_noinline_attributes) {
    return Status::OK();
  }

  // Create a session option for a single GPU device.
  SessionOptions options;

  // Make a local copy of graph def, because we need to change some things.
  GraphDef graph_def(graph_def_arg);

  if (cfg.erase_noinline_attributes) {
    // TF optimizer doesn't inline functions with "_noinline" attribute,
    // so let's go over the function library and erase it.
    for (auto& func : *graph_def.mutable_library()->mutable_function()) {
      func.mutable_attr()->erase("_noinline");
    }
  }

  // Instantiate all variables for function library runtime creation.
  std::vector<Device*> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
      options, "/job:localhost/replica:0/task:0", &devices));
  std::unique_ptr<DeviceMgr> dvc_mgr(new DeviceMgr(devices));
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             graph_def.library());
  Env* env = Env::Default();

  // Optimizer options: L1 and inlining. L1 is default.
  OptimizerOptions* optimizer_opts =
      options.config.mutable_graph_options()->mutable_optimizer_options();
  if (cfg.apply_optimizations) {
    optimizer_opts->set_opt_level(::tensorflow::OptimizerOptions_Level_L1);
  } else {
    optimizer_opts->set_opt_level(::tensorflow::OptimizerOptions_Level_L0);
  }

  // Create the function library runtime.
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(dvc_mgr.get(), env,
                                        graph_def.versions().producer(),
                                        &function_library, *optimizer_opts));
  FunctionLibraryRuntime* flr = pflr->GetFLR(devices[0]->name());

  // Create the GraphOptimizer to optimize the graph def.
  GraphConstructorOptions graph_ctor_opts;
  graph_ctor_opts.allow_internal_ops = true;
  graph_ctor_opts.expect_device_spec = false;
  std::unique_ptr<Graph> graphptr(new Graph(function_library));

  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(graph_ctor_opts, graph_def, graphptr.get()));

  // Optimize the graph.
  ::tensorflow::GraphOptimizer optimizer(*optimizer_opts);
  optimizer.Optimize(flr, env, devices[0], &graphptr, /*shape_map=*/nullptr);
  graphptr->ToGraphDef(output_graph_def);

  // The default values of attributes might have been stripped by the optimizer.
  // Add them back.
  return AddDefaultAttrsToGraphDef(output_graph_def, *graphptr->op_registry(),
                                   0, true);
}

// Applies the same graph pruning logic to the graph as Session.Run in TF.
// If the returned status is not OK, item state may be inconsistent.
Status PruneGraph(GrapplerItem* item) {
  ModelPruner pruner;
  GraphDef pruned_graph;
  Cluster* cluster = nullptr;  // ModelPruner doesn't check cluster.
  TF_RETURN_IF_ERROR(pruner.Optimize(cluster, *item, &pruned_graph));
  item->graph = std::move(pruned_graph);
  return Status::OK();
}

// Replace any unknown dimensions in a shape with
// cfg.placeholder_unknown_output_shape_dim if it is no less than 0.
Status ReplaceUnknownShapeDim(const ItemConfig& cfg,
                              const TensorShapeProto& shape_pb_in,
                              TensorShapeProto* shape_pb_out,
                              TensorShape* shape_out) {
  std::vector<int32> dims;
  for (const auto& dim_proto : shape_pb_in.dim()) {
    if (cfg.placeholder_unknown_output_shape_dim >= 0 &&
        dim_proto.size() == -1) {
      dims.push_back(cfg.placeholder_unknown_output_shape_dim);
      shape_pb_out->add_dim()->set_size(
          cfg.placeholder_unknown_output_shape_dim);
    } else {
      dims.push_back(std::max<int32>(1, dim_proto.size()));
      shape_pb_out->add_dim()->set_size(dim_proto.size());
    }
  }
  return TensorShapeUtils::MakeShape(dims.data(), dims.size(), shape_out);
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

  // Fill in feed nodes from config, if any provided.
  for (const auto& feed_node : cfg.feed_nodes) {
    const string feed_name = NodeName(feed_node);
    new_item->feed.emplace_back(feed_name, Tensor());
  }
  for (const auto& fetch_node : cfg.fetch_nodes) {
    new_item->fetch.emplace_back(NodeName(fetch_node));
  }

  // Attempt to detect the fetch node(s) if they were not set explicitly.
  if (new_item->fetch.empty() &&
      meta_graph.collection_def().count("train_op") > 0) {
    const CollectionDef& nodes = meta_graph.collection_def().at("train_op");
    if (nodes.has_node_list()) {
      for (const auto& node : nodes.node_list().value()) {
        new_item->fetch.push_back(NodeName(node));
      }
    }
  }

  // Detect feed and fetch nodes from signature defs. Signatures may share same
  // inputs or outputs.
  std::unordered_set<string> signature_feed_nodes;
  std::unordered_set<string> signature_fetch_nodes;
  for (const auto& name_and_signature : meta_graph.signature_def()) {
    for (const auto& name_and_input : name_and_signature.second.inputs()) {
      const TensorInfo& input = name_and_input.second;
      if (input.has_coo_sparse()) {
        // Define the shapes following the comment of CooSparse.
        // TODO(yuefengz): we probably want to use different dim values for the
        // three tensors of a SparseTensor.
        int64 dim = std::max(1, cfg.placeholder_unknown_output_shape_dim);
        TensorShape shape_1d({dim});
        TensorShape shape_2d({dim, dim});

        if (gtl::InsertIfNotPresent(
                &signature_feed_nodes,
                NodeName(input.coo_sparse().values_tensor_name()))) {
          Tensor value_tensor(input.dtype(), shape_1d);
          InitializeTensor(input.dtype(), &value_tensor);
          new_item->feed.emplace_back(
              NodeName(input.coo_sparse().values_tensor_name()), value_tensor);
        }
        if (gtl::InsertIfNotPresent(
                &signature_feed_nodes,
                NodeName(input.coo_sparse().indices_tensor_name()))) {
          Tensor indices_tensor(DT_INT64, shape_2d);
          InitializeTensor(input.dtype(), &indices_tensor);
          new_item->feed.emplace_back(
              NodeName(input.coo_sparse().indices_tensor_name()),
              indices_tensor);
        }
        if (gtl::InsertIfNotPresent(
                &signature_feed_nodes,
                NodeName(input.coo_sparse().dense_shape_tensor_name()))) {
          Tensor dense_shape_tensor(DT_INT64, shape_1d);
          InitializeTensor(input.dtype(), &dense_shape_tensor);
          new_item->feed.emplace_back(
              NodeName(input.coo_sparse().dense_shape_tensor_name()),
              dense_shape_tensor);
        }
      } else {
        if (gtl::InsertIfNotPresent(&signature_feed_nodes,
                                    NodeName(input.name()))) {
          TensorShape shape;
          TensorShapeProto shape_proto;
          Status s = ReplaceUnknownShapeDim(cfg, input.tensor_shape(),
                                            &shape_proto, &shape);
          if (!s.ok()) {
            LOG(ERROR) << "Invalid shape for signature input " << input.name()
                       << ": " << s << ", skipping this input";
            return nullptr;
          }

          Tensor fake_input(input.dtype(), shape);
          InitializeTensor(input.dtype(), &fake_input);
          new_item->feed.emplace_back(NodeName(input.name()), fake_input);
        }
      }
    }
    for (const auto& name_and_output : name_and_signature.second.outputs()) {
      const TensorInfo& output = name_and_output.second;
      if (output.has_coo_sparse()) {
        if (gtl::InsertIfNotPresent(
                &signature_fetch_nodes,
                NodeName(output.coo_sparse().values_tensor_name()))) {
          new_item->fetch.push_back(
              NodeName(output.coo_sparse().values_tensor_name()));
        }
        if (gtl::InsertIfNotPresent(
                &signature_fetch_nodes,
                NodeName(output.coo_sparse().indices_tensor_name()))) {
          new_item->fetch.push_back(
              NodeName(output.coo_sparse().indices_tensor_name()));
        }
        if (gtl::InsertIfNotPresent(
                &signature_fetch_nodes,
                NodeName(output.coo_sparse().dense_shape_tensor_name()))) {
          new_item->fetch.push_back(
              NodeName(output.coo_sparse().dense_shape_tensor_name()));
        }
      } else {
        if (gtl::InsertIfNotPresent(&signature_fetch_nodes,
                                    NodeName(output.name()))) {
          new_item->fetch.push_back(NodeName(output.name()));
        }
      }
    }
  }

  for (const auto& feed : new_item->feed) {
    if (feed.first.empty()) {
      LOG(ERROR) << "Invalid feed node name skipping this input";
      return nullptr;
    } else {
      VLOG(1) << "Will use feed node " << feed.first;
    }
  }

  for (const auto& fetch : new_item->fetch) {
    if (fetch.empty()) {
      LOG(ERROR) << "Invalid fetch node name skipping this input";
      return nullptr;
    } else {
      VLOG(1) << "Will use fetch node " << fetch;
    }
  }

  if (new_item->fetch.empty()) {
    LOG(ERROR) << "Failed to detect the fetch node(s), skipping this input";
    return nullptr;
  }

  // TODO(yuefengz): consider handling saved_model_main_op and legacy_init_op.
  // The reason why they are difficult to handle is because they may not intend
  // to initialize all variables that are required to run fetch nodes. We may
  // have to run restore op first.

  // Try to find initializers from variables and tables as init ops.
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
        new_item->init_ops.push_back(NodeName(var.initializer_name()));
      }
    }
  }

  if (meta_graph.collection_def().count("table_initializer") > 0) {
    const CollectionDef& inits =
        meta_graph.collection_def().at("table_initializer");
    if (inits.has_node_list()) {
      for (const auto& node : inits.node_list().value()) {
        new_item->init_ops.push_back(NodeName(node));
        // Tables are initialized from files, which can take a long time. Add
        // 30 minutes to the initialization time for each table to avoid
        // timing out.
        // TODO(bsteiner): adjust the timeout based on the file size.
        new_item->expected_init_time += 30 * 60;
      }
    }
  }

  // We keep the mapping from asset node to asset files. This should have been
  // used as feed but since asset node is usually a constant node, we will fill
  // the values of these constant nodes with their actual asset file paths.
  std::unordered_map<string, string> asset_node_to_value;

  // Assets file may have changed their directory, we assemble their new paths
  // if assets_directory_override is set. We also make sure we still can
  // access these asset files.
  if (!cfg.assets_directory_override.empty()) {
    if (meta_graph.collection_def().count("saved_model_assets") > 0) {
      const CollectionDef& collection =
          meta_graph.collection_def().at("saved_model_assets");
      const auto& any_assets = collection.any_list().value();
      for (const auto& any_asset : any_assets) {
        AssetFileDef asset_file_def;
        if (!ParseAny(any_asset, &asset_file_def, "tensorflow.AssetFileDef")
                 .ok()) {
          LOG(ERROR) << "Failed to parse AssetFile.";
          continue;
        }
        string asset_filepath = io::JoinPath(cfg.assets_directory_override,
                                             asset_file_def.filename());
        if (!FilesExist({asset_filepath}, nullptr)) {
          LOG(ERROR) << "Can't access one or more of the asset files "
                     << asset_filepath << ", skipping this input";
          return nullptr;
        }
        asset_node_to_value[NodeName(asset_file_def.tensor_info().name())] =
            asset_filepath;
      }
    }
  } else if (meta_graph.collection_def().count("asset_filepaths") > 0) {
    const CollectionDef& file_paths =
        meta_graph.collection_def().at("asset_filepaths");
    std::vector<string> paths;
    for (const auto& raw_path : file_paths.bytes_list().value()) {
      paths.push_back(raw_path);
    }
    if (!FilesExist(paths, nullptr)) {
      LOG(ERROR) << "Can't access one or more of the asset files, skipping "
                    "this input";
      return nullptr;
    }
  }

  if (meta_graph.collection_def().count("queue_runners") > 0) {
    const CollectionDef& vars = meta_graph.collection_def().at("queue_runners");
    for (const auto& raw : vars.bytes_list().value()) {
      QueueRunnerDef queue_runner;
      if (!queue_runner.ParseFromString(raw)) {
        LOG(ERROR) << "Could not parse queue_runners, skipping this input";
        return nullptr;
      }
      if (queue_runner.cancel_op_name().empty()) {
        LOG(ERROR) << "Queue without a cancel op, skipping this input";
        return nullptr;
      }
      new_item->queue_runners.push_back(queue_runner);
    }
  }

  // Add each node referenced in a collection to the list of nodes to keep.
  for (const auto& col : meta_graph.collection_def()) {
    const CollectionDef& collection = col.second;
    for (const string& node : collection.node_list().value()) {
      new_item->keep_ops.push_back(NodeName(node));
    }
  }

  for (auto& node : *new_item->graph.mutable_node()) {
    if (IsPlaceholder(node) && node.op() != "PlaceholderWithDefault") {
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
      Status make_shape_status = ReplaceUnknownShapeDim(
          cfg, node.attr().at("shape").shape(), &shape_proto, &shape);
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
          (shape.dims() == 0) && (node.attr().count("_output_shapes") == 1)) {
        const auto& output_shapes =
            node.attr().at("_output_shapes").list().shape(0);

        if (output_shapes.dim_size() != 0) {
          shape.Clear();
          shape_proto.clear_dim();

          for (const auto& dim : output_shapes.dim()) {
            auto size = dim.size();
            if (size == -1) size = cfg.placeholder_unknown_output_shape_dim;
            shape.AddDim(size);
            shape_proto.add_dim()->set_size(size);
          }
        }
      }

      Tensor fake_input(type, shape);
      InitializeTensor(type, &fake_input);

      if (cfg.feed_nodes.empty()) {
        // No specific feed nodes were given. Assume all placeholders are fed.
        if (signature_feed_nodes.count(node.name()) == 0) {
          new_item->feed.emplace_back(node.name(), fake_input);
        }
      } else if (cfg.feed_nodes.count(node.name()) > 0) {
        // If specific feed nodes were given, only update their tensors.
        auto it = find_if(new_item->feed.begin(), new_item->feed.end(),
                          [&node](std::pair<string, Tensor>& f) {
                            return f.first == node.name();
                          });
        QCHECK(it != new_item->feed.end());
        it->second = fake_input;
      }

      // Set the shape of the node in the graph. This is needed for statically
      // inferring shapes and is a no-op when dynamically inferring shapes as
      // the Placeholder shape will match the shape passed from new_item->feed.
      *(node.mutable_attr()->at("shape").mutable_shape()) = shape_proto;
    } else if (IsConstant(node)) {
      auto it = asset_node_to_value.find(node.name());
      if (it != asset_node_to_value.end()) {
        auto iter = node.mutable_attr()->find("value");
        if (iter == node.attr().end()) {
          LOG(ERROR) << "Value attribute expected in const op for asset files";
          return nullptr;
        }
        if (!iter->second.has_tensor() ||
            iter->second.tensor().string_val_size() != 1) {
          LOG(INFO) << "Unexected AttrValue proto: "
                    << iter->second.DebugString();
          return nullptr;
        }
        LOG(INFO) << "Using asset file " << it->second << " for node "
                  << node.name();
        *(iter->second.mutable_tensor()->mutable_string_val(0)) = it->second;
      }
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

  // Instantiate all the missing attributes with their default values.
  Status attr_status = AddDefaultAttrsToGraphDef(
      &new_item->graph,
      FunctionLibraryDefinition(OpRegistry::Global(),
                                new_item->graph.library()),
      0, true);
  if (!attr_status.ok()) {
    LOG(ERROR) << "Failed to instantiate default attribute values: "
               << attr_status.error_message();
    return nullptr;
  }

  // Optimize the graph (function inlining, l1 optimizations, etc).
  VLOG(1) << "Number of nodes in graph before OptimizeGraph: "
          << new_item->graph.node_size();
  Status optimize_status =
      OptimizeGraph(new_item->graph, &new_item->graph, cfg);
  if (!optimize_status.ok()) {
    LOG(ERROR) << "Graph preprocessing failed: " << optimize_status;
    return nullptr;
  }
  VLOG(1) << "Number of nodes in graph after OptimizeGraph: "
          << new_item->graph.node_size();

  if (cfg.prune_graph) {
    VLOG(1) << "Pruning graph...";
    auto status = PruneGraph(new_item.get());
    if (!status.ok()) {
      LOG(ERROR) << "Pruning failed: " << status.error_message();
      return nullptr;
    }
    VLOG(1) << "Number of nodes in graph after pruning: "
            << new_item->graph.node_size();
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

std::unique_ptr<GrapplerItem> GrapplerItemFromMetaGraphDefFile(
    const string& id, const string& meta_graph_file, const ItemConfig& cfg) {
  MetaGraphDef meta_graph;
  if (!ReadMetaGraphDefFromFile(meta_graph_file, &meta_graph).ok()) {
    return nullptr;
  }
  return GrapplerItemFromMetaGraphDef(id, meta_graph, cfg);
}

}  // end namespace grappler
}  // end namespace tensorflow
