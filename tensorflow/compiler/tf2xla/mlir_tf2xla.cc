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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"
#include "tensorflow/compiler/mlir/xla/mlir_hlo_to_hlo.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"

namespace tensorflow {

namespace {

// A fake device to simulate the presence of a CPU.
class FakeDevice : public Device {
 public:
  explicit FakeDevice(const DeviceAttributes& device_attributes)
      : Device(nullptr, device_attributes) {}

  Status Sync() override { return errors::Unimplemented("FakeDevice::Sync()"); }
};

// Translates the graph input information from tf2xla:::Config to
// GraphImportConfig.
Status ConvertInputInfo(const tf2xla::Config& config,
                        GraphImportConfig* specs) {
  std::vector<std::string> array_names;
  std::vector<std::string> data_types;
  std::vector<std::vector<int>> shapes;
  for (const tf2xla::Feed& feed : config.feed()) {
    array_names.push_back(feed.id().node_name());
    if (feed.type() != DT_INVALID) {
      data_types.push_back(DataType_Name(feed.type()));
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
Status ConvertOutputInfo(const tf2xla::Config& config,
                         GraphImportConfig* specs) {
  std::vector<std::string> array_names;
  for (const tf2xla::Fetch& fetch : config.fetch()) {
    array_names.push_back(fetch.id().node_name());
  }

  return ParseOutputArrayInfo(array_names, &specs->outputs);
}

}  // namespace

Status ConvertGraphDefToXlaViaMlir(const GraphDef& graph_def,
                                   const tf2xla::Config& config,
                                   xla::XlaComputation* computation) {
  GraphDebugInfo debug_info;
  mlir::MLIRContext context;
  GraphImportConfig specs;
  specs.prune_unused_nodes = false;
  specs.convert_legacy_fed_inputs = false;
  specs.graph_as_function = false;
  specs.upgrade_legacy = false;
  TF_RETURN_IF_ERROR(ConvertInputInfo(config, &specs));
  TF_RETURN_IF_ERROR(ConvertOutputInfo(config, &specs));

  TF_ASSIGN_OR_RETURN(
      mlir::OwningModuleRef module,
      ConvertGraphdefToMlir(graph_def, debug_info, specs, &context));

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
  return ConvertMLIRToXlaComputation(*module, computation,
                                     /*use_tuple_args=*/false,
                                     /*always_return_tuple=*/true);
}

}  // namespace tensorflow
