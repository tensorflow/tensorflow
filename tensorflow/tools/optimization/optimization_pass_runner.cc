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
// This file creates a library that can run any registered optimization pass.
// The binary that uses this will be run in a form similar to:
// ./optimization_pass_runner  --input_file_path=/tmp/input.pbtxt
// --output_file_path=/tmp/output.pbtxt
// --optimization_pass=NameOfGraphOptimizationPass
#include "tensorflow/tools/optimization/optimization_pass_runner.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {

namespace {
// A fake device used to populate a DeviceSet.
class FakeDevice : public Device {
 private:
  explicit FakeDevice(const DeviceAttributes& device_attributes)
      : Device(nullptr, device_attributes) {}

 public:
  Status Sync() override;
  static std::unique_ptr<Device> Make(const string& name, const string& type);
};

Status FakeDevice::Sync() {
  return errors::Unimplemented("FakeDevice::Sync()");
}

std::unique_ptr<Device> FakeDevice::Make(const string& name,
                                         const string& type) {
  DeviceAttributes device_attributes;
  device_attributes.set_name(name);
  device_attributes.set_device_type(DeviceType(type).type());
  return std::unique_ptr<Device>(new FakeDevice(device_attributes));
}
}  // namespace

Status OptimizationPassRunner::RunMain(int argc, char** argv) {
  string input_file_path;
  string output_file_path;
  string optimization_pass;

  const std::vector<Flag> flag_list = {
      Flag("input_file_path", &input_file_path, "Location of the input graph."),
      Flag("output_file_path", &output_file_path,
           "Location to write the resulting graph."),
      // For now only a single optimization pass can be run.
      Flag("optimization_pass", &optimization_pass,
           "Which optimization pass to run."),
  };
  if (!Flags::Parse(&argc, argv, flag_list)) {
    return errors::FailedPrecondition("Invalid flags passed");
  }
  port::InitMain(argv[0], &argc, &argv);

  if (input_file_path.empty()) {
    return errors::FailedPrecondition("input_file_path is a required flag.");
  }
  if (output_file_path.empty()) {
    return errors::FailedPrecondition("output_file_path is a required flag.");
  }
  if (optimization_pass.empty()) {
    return errors::FailedPrecondition("optimization_pass is a required flag.");
  }

  // Turn on XLA Auto-Jit.
  auto session_options = absl::make_unique<SessionOptions>();
  session_options->config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(jit_level_);
  FunctionDefLibrary flib;
  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());

  GraphOptimizationPassOptions options;
  options.session_options = session_options.release();
  options.graph = &graph;
  options.flib_def =
      new FunctionLibraryDefinition((*options.graph)->op_registry(), flib);

  // Grab the data
  GraphDef graphdef;
  GraphConstructorOptions graph_opts;
  graph_opts.expect_device_spec = true;
  graph_opts.allow_internal_ops = true;
  TF_RETURN_IF_ERROR(ReadTextProto(Env::Default(), input_file_path, &graphdef));
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(graph_opts, graphdef, options.graph->get()));

  // Add all devices that were previously configured with AddDevice.
  DeviceSet device_set;
  for (auto& device : devices_) {
    device_set.AddDevice(device.get());
  }
  options.device_set = &device_set;

  Status result = errors::NotFound(
      "An OptimizationPass was not found with the desired name.");

  // Run the optimization pass specified by the command line flag.
  for (const auto& groups_and_passes :
       OptimizationPassRegistry::Global()->groups()) {
    for (const auto& phase_and_passes : groups_and_passes.second) {
      for (const auto& pass : phase_and_passes.second) {
        if (pass->name() == optimization_pass) {
          result = pass->Run(options);
        }
      }
    }
  }

  TF_RETURN_IF_ERROR(result);

  // Write out the result.
  options.graph->get()->ToGraphDef(&graphdef);
  TF_RETURN_IF_ERROR(
      WriteTextProto(Env::Default(), output_file_path, graphdef));
  return Status::OK();
}

Status OptimizationPassRunner::SetJitLevel(
    OptimizerOptions::GlobalJitLevel jit_level) {
  jit_level_ = jit_level;
  return Status::OK();
}

Status OptimizationPassRunner::AddDevice(const string& name,
                                         const string& type) {
  devices_.push_back(FakeDevice::Make(name, type));
  return Status::OK();
}

}  // namespace tensorflow
