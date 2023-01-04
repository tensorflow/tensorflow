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

#include "tensorflow/compiler/jit/mark_for_compilation_pass_test_helper.h"

#include "tensorflow/compiler/jit/cluster_scoping_pass.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
/*static*/ Status MarkForCompilationPassTestHelper::MarkForCompilation(
    std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def,
    MarkForCompilationPassTestHelper::Options options) {
  // Assign all unassigned nodes to the CPU device.
  static const char* kCpuDevice = "/job:localhost/replica:0/task:0/cpu:0";
  for (Node* n : (*graph)->nodes()) {
    if (n->assigned_device_name().empty()) {
      n->set_assigned_device_name(kCpuDevice);
    }
  }

  SessionOptions session_options;
  if (options.enable_global_jit) {
    session_options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_global_jit_level(OptimizerOptions::ON_2);
  }

  // Call AddDevices to register the XLA devices.
  //
  // It may be worth refactoring out XlaOpRegistry::RegisterCompilationDevice to
  // make this more direct, but probably not worth it solely for this test.
  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(session_options, "", &devices));

  GraphOptimizationPassOptions opt_options;
  opt_options.graph = graph;
  opt_options.session_options = &session_options;
  opt_options.flib_def = flib_def;

  if (options.enable_cluster_scoping) {
    ClusterScopingPass cluster_scoping_pass;
    TF_RETURN_IF_ERROR(cluster_scoping_pass.Run(opt_options));
  }

  MarkForCompilationPass mark_for_compilation_pass;
  return mark_for_compilation_pass.RunForTest(
      opt_options,
      /*disable_deadness_analysis=*/options.disable_deadness_analysis,
      /*deterministic_cluster_names=*/options.deterministic_cluster_names);
}

/*static*/ Status MarkForCompilationPassTestHelper::MarkForCompilation(
    std::unique_ptr<Graph>* graph,
    MarkForCompilationPassTestHelper::Options options) {
  FunctionDefLibrary flib;
  FunctionLibraryDefinition flib_def((*graph)->op_registry(), flib);
  return MarkForCompilation(graph, &flib_def, options);
}
}  // namespace tensorflow
