/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_mlir_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "xla/hlo/builder/xla_computation.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

namespace {

// A fake device to simulate the presence of a CPU.
class FakeDevice : public Device {
 public:
  explicit FakeDevice(const DeviceAttributes& device_attributes)
      : Device(nullptr, device_attributes) {}

  absl::Status Sync() override {
    return errors::Unimplemented("FakeDevice::Sync()");
  }
};

// Translates the graph input information from tf2xla:::Config to
// GraphImportConfig.
absl::Status ConvertInputInfo(
    const tf2xla::Config& config,
    const std::unordered_map<std::string, std::string>& feed_name_remap,
    GraphImportConfig* specs) {
  std::vector<std::string> array_names;
  std::vector<std::string> data_types;
  std::vector<std::optional<std::vector<int>>> shapes;
  for (const tf2xla::Feed& feed : config.feed()) {
    std::string place_holder_name =
        feed_name_remap.at(TensorIdToString(feed.id()));
    array_names.push_back(place_holder_name);
    data_types.push_back(
        feed.type() == DT_INVALID ? "" : DataType_Name(feed.type()));
    if (feed.shape().unknown_rank()) {
      shapes.push_back(std::nullopt);
      continue;
    }
    std::vector<int> dims;
    dims.reserve(feed.shape().dim_size());
    absl::c_for_each(feed.shape().dim(), [&](const TensorShapeProto::Dim d) {
      dims.push_back(d.size());
    });
    shapes.push_back(dims);
  }

  return ParseInputArrayInfo(array_names, data_types, shapes, &specs->inputs);
}

// Translates the graph output information from tf2xla:::Config to
// GraphImportConfig.
absl::Status ConvertOutputInfo(const tf2xla::Config& config,
                               GraphImportConfig* specs) {
  std::vector<std::string> array_names;
  for (const tf2xla::Fetch& fetch : config.fetch()) {
    array_names.push_back(fetch.id().node_name());
  }

  return ParseOutputArrayInfo(array_names, &specs->outputs);
}

}  // namespace

absl::Status ConvertGraphDefToXlaViaMlir(
    GraphDef graph_def, const tf2xla::Config& config,
    xla::XlaComputation* computation, absl::string_view debug_info_filename,
    absl::string_view debug_info_path_begin_marker) {
  // AddPlaceholdersForFeeds prepares for PruneGraphDefInto and serves two
  // purposes: (1) It creates a placeholder node for each feed, so that
  // PruneGraphDefInfo can prune away the node containing the feed. (2) It
  // is also a workaround for b/149029125. It replaces a feed representation
  // with a placeholder node that contains a single output.
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), graph_def.library());
  std::unique_ptr<Graph> graph(new Graph(flib_def));
  std::unordered_map<string, string> feed_name_remap;
  TF_RETURN_IF_ERROR(AddPlaceholdersForFeeds(config, graph->op_registry(),
                                             &feed_name_remap, &graph_def));

  // TODO(b/149024678): remove this workaround after the ticket is fixed.
  //   Prune the GraphDef because MLIR importer doesn't allow unknown ops in
  //   graph nodes even the nodes are not needed for computing the outputs.
  GraphDef pruned_graph_def;
  TF_RETURN_IF_ERROR(PruneGraphDefInto(config, graph_def, &pruned_graph_def));

  GraphImportConfig specs;
  specs.prune_unused_nodes = false;
  specs.convert_legacy_fed_inputs = false;
  specs.graph_as_function = false;
  specs.upgrade_legacy = true;
  TF_RETURN_IF_ERROR(ConvertInputInfo(config, feed_name_remap, &specs));
  TF_RETURN_IF_ERROR(ConvertOutputInfo(config, &specs));

  GraphDebugInfo debug_info;
  if (!debug_info_filename.empty()) {
    TF_RETURN_IF_ERROR(LoadProtoFromFile(debug_info_filename, &debug_info));

    if (!debug_info_path_begin_marker.empty()) {
      for (size_t i = 0, e = debug_info.files_size(); i < e; ++i) {
        std::string* file_name = debug_info.mutable_files(i);
        size_t location =
            file_name->rfind(std::string(debug_info_path_begin_marker));
        if (location != std::string::npos) {
          *file_name = file_name->substr(location +
                                         debug_info_path_begin_marker.length());
        }
      }
    }
  }

  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      ConvertGraphdefToMlir(pruned_graph_def, debug_info, specs, &context));

  // Construct a CPU device and add the device to the operations.
  DeviceSet device_set;
  DeviceAttributes attr;
  attr.set_name("/job:localhost/replica:0/task:0/device:CPU:0");
  attr.set_device_type(DeviceType("CPU").type());
  FakeDevice device(attr);
  device_set.AddDevice(&device);
  AddDevicesToOp(*module, &device_set);

  TF_RETURN_IF_ERROR(mlir::TF::RunBridgeWithStandardPipeline(
      *module, /*enable_logging=*/VLOG_IS_ON(1), /*enable_inliner=*/true));

  // Convert the MLIR module to XLA computation. If the input graph can't be
  // lowered down to a single graph node with a single island by the previous
  // step, this step will return an error.
  return ConvertMLIRToXlaComputation(
      *module, /*device_type=*/"XLA_CPU_JIT", computation,
      /*use_tuple_args=*/false, /*prefer_tf2xla=*/false,
      /*return_tuple=*/true);
}

}  // namespace tensorflow
