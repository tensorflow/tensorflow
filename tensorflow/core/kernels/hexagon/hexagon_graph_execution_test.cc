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
// $ curl
// https://storage.googleapis.com/download.tensorflow.org/models/imagenet_comp_graph_label_strings.txt
// -o /tmp/imagenet_comp_graph_label_strings.txt
// adb push /tmp/imagenet_comp_graph_label_strings.txt /data/local/tmp

#include <memory>

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/hexagon/graph_transfer_utils.h"
#include "tensorflow/core/kernels/hexagon/graph_transferer.h"
#include "tensorflow/core/kernels/hexagon/hexagon_control_wrapper.h"
#include "tensorflow/core/kernels/hexagon/hexagon_ops_definitions.h"
#include "tensorflow/core/kernels/hexagon/i_graph_transfer_ops_definitions.h"
#include "tensorflow/core/kernels/hexagon/i_soc_control_wrapper.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/profile_utils/clock_cycle_profiler.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

using ByteArray = ISocControlWrapper::ByteArray;

const bool DBG_DUMP_FLOAT_DATA = false;
const int WIDTH = 299;
const int HEIGHT = 299;
const int DEPTH = 3;
const int EXPECTED_FIRST_RESULT_ID = 59;
const int EXECUTION_REPEAT_COUNT = 3;

static void DumpTop10Results(
    const std::vector<ISocControlWrapper::ByteArray>& outputs) {
  CHECK(outputs.size() == 1);
  const int byte_size = std::get<1>(outputs.at(0));
  const int element_count = byte_size / sizeof(float);
  const float* float_array =
      reinterpret_cast<float*>(std::get<0>(outputs.at(0)));
  const string label_filename =
      "/data/local/tmp/imagenet_comp_graph_label_strings.txt";
  string label_str;
  TF_CHECK_OK(ReadFileToString(Env::Default(), label_filename, &label_str));
  std::vector<string> labels = str_util::Split(label_str, '\n');
  GraphTransferUtils::DumpTopNFloatResults(
      float_array, labels.data(),
      std::min(element_count, static_cast<int>(labels.size())),
      10 /* show top_n results */);
}

static void CheckFirstResult(
    const std::vector<ISocControlWrapper::ByteArray>& outputs,
    const int expected_first_id) {
  EXPECT_GE(outputs.size(), 1);
  const int byte_size = std::get<1>(outputs.at(0));
  const int element_count = byte_size / sizeof(float);
  const float* float_array =
      reinterpret_cast<float*>(std::get<0>(outputs.at(0)));
  EXPECT_GE(element_count, 1);
  std::vector<string> labels(element_count);
  std::priority_queue<std::tuple<float, int, string>> queue =
      GraphTransferUtils::GetTopNFloatResults(float_array, labels.data(),
                                              element_count);
  const std::tuple<float, int, string>& entry = queue.top();
  EXPECT_EQ(expected_first_id, std::get<1>(entry));
}

// CAVEAT: This test only runs when you specify hexagon library using
// makefile.
// TODO(satok): Make this generic so that this can run without any
// additionanl steps.
#ifdef USE_HEXAGON_LIBS
TEST(GraphTransferer, RunInceptionV3OnHexagonExample) {
  const string image_filename = "/data/local/tmp/img_299x299.bmp";
  const string model_filename =
      "/data/local/tmp/tensorflow_inception_v3_stripped_optimized_quantized.pb";
  const IGraphTransferOpsDefinitions* ops_definitions =
      &HexagonOpsDefinitions::getInstance();
  std::vector<GraphTransferer::InputNodeInfo> input_node_info_list = {
      GraphTransferer::InputNodeInfo{
          "Mul", Tensor{DT_FLOAT, {1, WIDTH, HEIGHT, DEPTH}}}};
  std::vector<string> output_node_names = {"softmax"};
  const bool is_text_proto = false;

  GraphTransferer::OutputTensorInfo output_tensor_info;
  GraphTransferer gt;
  gt.EnableStrictCheckMode(false);
  Status status = gt.LoadGraphFromProtoFile(
      *ops_definitions, model_filename, input_node_info_list, output_node_names,
      is_text_proto, true /* dry_run_for_unknown_shape */, &output_tensor_info);
  ASSERT_TRUE(status.ok()) << status;

  HexagonControlWrapper hexagon_control_wrapper;
  const int version = hexagon_control_wrapper.GetVersion();
  ASSERT_GE(version, 1);
  LOG(INFO) << "Hexagon controller version is " << version;

  // Read the data from the bitmap file into memory
  string bmp;
  TF_CHECK_OK(ReadFileToString(Env::Default(), image_filename, &bmp));
  const int fsize = bmp.size();
  LOG(INFO) << "Read " << image_filename << ", size = " << fsize << "bytes";
  const int64 pixel_count = WIDTH * HEIGHT * DEPTH;
  CHECK(fsize >= 22 /* pos of height */ + sizeof(int));
  CHECK(bmp.data() != nullptr);
  uint8* const img_bytes = bit_cast<uint8*>(bmp.data());
  const int header_size = *(reinterpret_cast<int*>(img_bytes + 10));
  LOG(INFO) << "header size = " << header_size;
  const int size = *(reinterpret_cast<int*>(img_bytes + 14));
  LOG(INFO) << "image size = " << size;
  const int width = *(reinterpret_cast<int*>(img_bytes + 18));
  LOG(INFO) << "width = " << width;
  const int height = *(reinterpret_cast<int*>(img_bytes + 22));
  LOG(INFO) << "height = " << height;
  CHECK(fsize >= (WIDTH + 1) * WIDTH * 3 + header_size);

  uint8* const bmp_pixels = &img_bytes[header_size];

  std::vector<float> img_floats(pixel_count);
  int src_pixel_index = 0;
  CHECK(pixel_count % 3 == 0);
  for (int i = 0; i < pixel_count / 3; ++i) {
    const int src_pos = 3 * src_pixel_index;
    const int dst_pos = 3 * i;
    ++src_pixel_index;
    CHECK(src_pos + 2 + header_size < fsize);
    CHECK(dst_pos + 2 < pixel_count);
    // Convert (B, G, R) in bitmap to (R, G, B)
    img_floats[dst_pos] =
        (static_cast<float>(bmp_pixels[src_pos + 2]) - 128.0f) / 128.0f;
    img_floats[dst_pos + 1] =
        (static_cast<float>(bmp_pixels[src_pos + 1]) - 128.0f) / 128.0f;
    img_floats[dst_pos + 2] =
        (static_cast<float>(bmp_pixels[src_pos]) - 128.0f) / 128.0f;
    if (DBG_DUMP_FLOAT_DATA) {
      LOG(INFO) << i << " (" << img_floats[dst_pos] << ", "
                << img_floats[dst_pos + 1] << ", " << img_floats[dst_pos + 2]
                << ") (" << static_cast<int>(bmp_pixels[src_pos + 2]) << ", "
                << static_cast<int>(bmp_pixels[src_pos + 1]) << ", "
                << static_cast<int>(bmp_pixels[src_pos]) << ")";
    }
    if (src_pixel_index % (WIDTH + 1) == (WIDTH - 1)) {
      // skip bmp padding
      ++src_pixel_index;
    }
  }
  const ByteArray ba =
      std::make_tuple(reinterpret_cast<uint8*>(img_floats.data()),
                      pixel_count * sizeof(float), DT_FLOAT);

  // 1. Initialize hexagon
  hexagon_control_wrapper.Init();

  // 2. Setup graph in hexagon
  hexagon_control_wrapper.SetupGraph(gt);

  // 3. Fill input node's output
  hexagon_control_wrapper.FillInputNode("Mul", ba);

  // 4. Execute graph
  profile_utils::CpuUtils::EnableClockCycleProfiling(true);
  ClockCycleProfiler prof;
  for (int i = 0; i < EXECUTION_REPEAT_COUNT; ++i) {
    prof.Start();
    hexagon_control_wrapper.ExecuteGraph();
    prof.Stop();
  }

  // 5-1. Read output node's outputs
  std::vector<ISocControlWrapper::ByteArray> outputs;
  hexagon_control_wrapper.ReadOutputNode("softmax", &outputs);

  // 5-2. Dump results
  DumpTop10Results(outputs);
  CheckFirstResult(outputs, EXPECTED_FIRST_RESULT_ID);
  prof.DumpStatistics("Graph Execution");

  // 6. Teardown graph in hexagon
  hexagon_control_wrapper.TeardownGraph();

  // 7. Finalize hexagon
  hexagon_control_wrapper.Finalize();
}
#endif

}  // namespace tensorflow
