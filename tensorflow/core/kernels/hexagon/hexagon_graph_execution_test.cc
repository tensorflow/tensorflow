/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
// Before calling this test program, download a model as follows.
// $ curl https://storage.googleapis.com/download.tensorflow.org/models/tensorflow_inception_v3_stripped_optimized_quantized.pb \
// -o /tmp/tensorflow_inception_v3_stripped_optimized_quantized.pb
// adb push /tmp/tensorflow_inception_v3_stripped_optimized_quantized.pb \
// /data/local/tmp

#include <memory>

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/hexagon/graph_transferer.h"
#include "tensorflow/core/kernels/hexagon/hexagon_control_wrapper.h"
#include "tensorflow/core/kernels/hexagon/hexagon_ops_definitions.h"
#include "tensorflow/core/kernels/hexagon/i_graph_transfer_ops_definitions.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// CAVEAT: This test only runs when you specify hexagon library using
// makefile.
// TODO(satok): Make this generic so that this can run without any
// additionanl steps.
#ifdef USE_HEXAGON_LIBS
TEST(GraphTransferer, RunInceptionV3OnHexagonExample) {
  const string filename =
      "/data/local/tmp/tensorflow_inception_v3_stripped_optimized_quantized.pb";
  const IGraphTransferOpsDefinitions* ops_definitions =
      &HexagonOpsDefinitions::getInstance();
  std::vector<GraphTransferer::InputNodeInfo> input_node_info_list = {
      GraphTransferer::InputNodeInfo{"Mul",
                                     Tensor{DT_FLOAT, {1, 299, 299, 3}}}};
  std::vector<string> output_node_names = {"softmax"};
  const bool is_text_proto = false;

  GraphTransferer::OutputTensorInfo output_tensor_info;
  GraphTransferer gt;
  gt.EnableStrictCheckMode(false);
  Status status = gt.LoadGraphFromProtoFile(
      *ops_definitions, filename, input_node_info_list, output_node_names,
      is_text_proto, true /* dry_run_for_unknown_shape */, &output_tensor_info);
  ASSERT_TRUE(status.ok()) << status;

  HexagonControlWrapper hexagon_control_wrapper;
  const int version = hexagon_control_wrapper.GetVersion();
  ASSERT_GE(version, 1);
  LOG(INFO) << "Hexagon controller version is " << version;

  // 1. Initialize hexagon
  hexagon_control_wrapper.Init();

  // 2. Setup graph in hexagon
  hexagon_control_wrapper.SetupGraph(gt);

  // 3. Fill input node's output
  hexagon_control_wrapper.FillInputNode("Mul", {});

  // 4. Execute graph
  hexagon_control_wrapper.ExecuteGraph();

  // 5. Read output node's outputs
  std::vector<ISocControlWrapper::ByteArray> outputs;
  hexagon_control_wrapper.ReadOutputNode("softmax", &outputs);

  // 6. Teardown graph in hexagon
  hexagon_control_wrapper.TeardownGraph();

  // 7. Finalize hexagon
  hexagon_control_wrapper.Finalize();
}
#endif

}  // namespace tensorflow
