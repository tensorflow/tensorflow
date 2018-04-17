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
/* Before calling this test program, download a model as follows.
$ curl
https://storage.googleapis.com/download.tensorflow.org/models/tensorflow_inception_v3_stripped_optimized_quantized.pb
\ -o /tmp/tensorflow_inception_v3_stripped_optimized_quantized.pb
$ adb push /tmp/tensorflow_inception_v3_stripped_optimized_quantized.pb \
/data/local/tmp
$ curl
https://storage.googleapis.com/download.tensorflow.org/models/imagenet_comp_graph_label_strings.txt
-o /tmp/imagenet_comp_graph_label_strings.txt
adb push /tmp/imagenet_comp_graph_label_strings.txt /data/local/tmp
*/

// define EIGEN_USE_THREADS to include quantization_utils.h
#define EIGEN_USE_THREADS

#include <memory>

#include "tensorflow/core/framework/graph_transfer_info.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/hexagon/graph_transfer_utils.h"
#include "tensorflow/core/kernels/hexagon/graph_transferer.h"
#include "tensorflow/core/kernels/hexagon/hexagon_control_wrapper.h"
#include "tensorflow/core/kernels/hexagon/hexagon_ops_definitions.h"
#include "tensorflow/core/kernels/i_remote_fused_graph_executor.h"
#include "tensorflow/core/kernels/i_remote_fused_graph_ops_definitions.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/profile_utils/clock_cycle_profiler.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

using ByteArray = HexagonControlWrapper::ByteArray;

constexpr const char* const IMAGE_FILENAME = "/data/local/tmp/img_299x299.bmp";
constexpr const char* const MODEL_FILENAME =
    "/data/local/tmp/tensorflow_inception_v3_stripped_optimized_quantized.pb";
constexpr const char* const MODEL_WITH_QUANTIZED_INPUT_FILENAME =
    "/data/local/tmp/"
    "tensorflow_inception_v3_stripped_optimized_quantized_with_quantized_input."
    "pb";
constexpr const char* const FUSED_MODEL_FILENAME =
    "/data/local/tmp/"
    "tensorflow_inception_v3_stripped_optimized_quantized_fused_hexagon.pb";
constexpr const char* const REMOTE_FUSED_GRAPH_EXECUTE_NODE_NAME =
    "remote_fused_graph_execute_node";
constexpr bool USE_SHAPE_INFERENCE = false;

const bool DBG_DUMP_FLOAT_DATA = false;
const int WIDTH = 299;
const int HEIGHT = 299;
const int DEPTH = 3;
const int EXPECTED_FIRST_RESULT_ID = 59;
const int EXECUTION_REPEAT_COUNT = 10;

static void CheckHexagonControllerVersion() {
  HexagonControlWrapper hexagon_control_wrapper;
  const int version = hexagon_control_wrapper.GetVersion();
  ASSERT_GE(version, 1);
  LOG(INFO) << "Hexagon controller version is " << version;
}

static void DumpTop10Results(const int byte_size,
                             const float* const float_array) {
  const int element_count = byte_size / sizeof(float);
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

static void DumpTop10Results(const std::vector<ByteArray>& outputs) {
  CHECK(outputs.size() == 1);
  const int byte_size = std::get<1>(outputs.at(0));
  const float* float_array =
      reinterpret_cast<float*>(std::get<0>(outputs.at(0)));
  DumpTop10Results(byte_size, float_array);
}

static void CheckFirstResult(const std::vector<ByteArray>& outputs,
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

static void LoadImage(std::vector<float>* img_floats_ptr) {
  CHECK(img_floats_ptr != nullptr);
  std::vector<float>& img_floats = *img_floats_ptr;
  // Read the data from the bitmap file into memory
  string bmp;
  TF_CHECK_OK(ReadFileToString(Env::Default(), IMAGE_FILENAME, &bmp));
  const int fsize = bmp.size();
  LOG(INFO) << "Read " << IMAGE_FILENAME << ", size = " << fsize << "bytes";
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

  img_floats.resize(pixel_count);
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
}

static void QuantizeImage(const std::vector<float>& float_vec,
                          std::vector<quint8>* quint8_vec) {
  quint8_vec->resize(float_vec.size());
  for (int i = 0; i < float_vec.size(); ++i) {
    quint8_vec->at(i) = FloatToQuantized<quint8>(float_vec[i], -1.0f, 1.0f);
  }
}

static Tensor BuildImageTensor(const std::vector<float>& img_floats) {
  LOG(INFO) << "Loading image finished.";
  Tensor img_tensor(DT_FLOAT, {1, WIDTH, HEIGHT, DEPTH});
  CHECK_EQ(WIDTH * HEIGHT * DEPTH, img_floats.size());
  CHECK_EQ(img_tensor.TotalBytes(), img_floats.size() * sizeof(float));
  LOG(INFO) << "Copy data to tensor.";
  std::memcpy(img_tensor.flat<float>().data(), img_floats.data(),
              img_tensor.TotalBytes());
  return img_tensor;
}

static Tensor BuildQuantizedImageTensor(
    const std::vector<quint8>& quantized_img) {
  LOG(INFO) << "Loading image finished.";
  Tensor img_tensor(DT_QUINT8, {1, WIDTH, HEIGHT, DEPTH});
  CHECK_EQ(WIDTH * HEIGHT * DEPTH, quantized_img.size());
  CHECK_EQ(img_tensor.TotalBytes(), quantized_img.size() * sizeof(quint8));
  LOG(INFO) << "Copy data to tensor.";
  std::memcpy(img_tensor.flat<quint8>().data(), quantized_img.data(),
              img_tensor.TotalBytes());
  return img_tensor;
}

/* static */ RemoteFusedGraphExecuteInfo
BuildRemoteFusedGraphExecuteInfoWithGraphTransferInfo(
    const GraphTransferInfo& graph_transfer_info) {
  RemoteFusedGraphExecuteInfo execute_info;
  execute_info.set_executor_name("build_hexagon_remote_fused_graph_executor");
  for (const GraphTransferGraphInputNodeInfo& input :
       graph_transfer_info.graph_input_node_info()) {
    execute_info.add_graph_input_node_name(input.name());
    RemoteFusedGraphExecuteInfo::TensorShapeTypeProto& tensor_shape_type =
        *execute_info.add_default_graph_input_tensor_shape();
    tensor_shape_type.set_dtype(input.dtype());
    TensorShapeProto& tensor_shape_proto = *tensor_shape_type.mutable_shape();
    for (const int64 dim : input.shape()) {
      tensor_shape_proto.add_dim()->set_size(dim);
    }
  }

  for (const GraphTransferGraphOutputNodeInfo& output :
       graph_transfer_info.graph_output_node_info()) {
    execute_info.add_graph_output_node_name(output.name());
    RemoteFusedGraphExecuteInfo::TensorShapeTypeProto& tensor_shape_type =
        *execute_info.add_default_graph_output_tensor_shape();
    tensor_shape_type.set_dtype(output.dtype());
    TensorShapeProto& tensor_shape_proto = *tensor_shape_type.mutable_shape();
    for (const int64 dim : output.shape()) {
      tensor_shape_proto.add_dim()->set_size(dim);
    }
  }

  execute_info.set_serialized_executor_parameters(
      graph_transfer_info.SerializeAsString());
  return execute_info;
}

static void RunInferenceByHexagonControlWrapper(const GraphTransferer& gt,
                                                const Tensor& img_tensor) {
  const RemoteFusedGraphExecuteInfo execute_info =
      BuildRemoteFusedGraphExecuteInfoWithGraphTransferInfo(
          gt.GetGraphTransferInfo());

  HexagonControlWrapper hexagon_control_wrapper;
  // 1. Initialize hexagon
  hexagon_control_wrapper.Init(execute_info);

  // 2. Setup graph in hexagon
  hexagon_control_wrapper.SetupGraph();

  // 3. Fill input node's output
  hexagon_control_wrapper.FillInputNode("Mul", img_tensor);

  // 4. Execute graph
  const int64 start_time_us = Env::Default()->NowMicros();
  for (int i = 0; i < EXECUTION_REPEAT_COUNT; ++i) {
    hexagon_control_wrapper.ExecuteGraph();
  }
  const int64 end_time_us = Env::Default()->NowMicros();

  // 5-1. Read output node's outputs
  std::vector<ByteArray> outputs;
  hexagon_control_wrapper.ReadOutputNode("softmax", &outputs);

  // 5-2. Dump results
  DumpTop10Results(outputs);
  CheckFirstResult(outputs, EXPECTED_FIRST_RESULT_ID);
  LOG(INFO) << "Average execution time = "
            << (end_time_us - start_time_us) / EXECUTION_REPEAT_COUNT << "us";

  // 6. Teardown graph in hexagon
  hexagon_control_wrapper.TeardownGraph();

  // 7. Finalize hexagon
  hexagon_control_wrapper.Finalize();
}

static void RunFusedGraph(const GraphDef& fused_graph_def) {
  // Setup input tensor
  std::vector<float> img_floats;
  LoadImage(&img_floats);

  LOG(INFO) << "Ioading image finished.";
  const Tensor img_tensor = BuildImageTensor(img_floats);

  // Setup session
  std::vector<Tensor> output_tensors;
  SessionOptions session_options;
  session_options.env = Env::Default();
  std::unique_ptr<Session> session =
      std::unique_ptr<Session>(NewSession(session_options));
  TF_ASSERT_OK(session->Create(fused_graph_def));

  // Setup session arguments
  RunOptions run_options;
  run_options.set_trace_level(RunOptions::FULL_TRACE);
  RunMetadata run_metadata;

  std::vector<std::pair<string, tensorflow::Tensor>> input_tensors;
  input_tensors.emplace_back("Mul", img_tensor);
  std::vector<string> output_node_names;
  output_node_names.emplace_back(REMOTE_FUSED_GRAPH_EXECUTE_NODE_NAME);

  LOG(INFO) << "Run graph";
  // Run inference with all node as output
  TF_ASSERT_OK(session->Run(run_options, input_tensors, output_node_names, {},
                            &output_tensors, &run_metadata));
  ASSERT_EQ(1, output_tensors.size());
  const Tensor& output_tensor = output_tensors.at(0);
  LOG(INFO) << "Output byte size = " << output_tensor.TotalBytes();
  LOG(INFO) << "Output shape = " << output_tensor.shape().DebugString();
  DumpTop10Results(
      output_tensor.TotalBytes(),
      reinterpret_cast<const float*>(output_tensor.flat<float>().data()));
}

static void CompareGraphTransferInfo(const GraphTransferInfo& gfi0,
                                     const GraphTransferInfo& gfi1) {
  LOG(INFO) << "(1) node count: " << gfi1.node_info_size() << ", "
            << gfi1.const_node_info_size();

  // 1. check node_info
  ASSERT_EQ(gfi0.node_info_size(), gfi1.node_info_size());
  for (int i = 0; i < gfi0.node_info_size(); ++i) {
    const GraphTransferNodeInfo& ni0 = gfi0.node_info(i);
    const GraphTransferNodeInfo& ni1 = gfi1.node_info(i);
    EXPECT_EQ(ni0.DebugString(), ni1.DebugString());
    EXPECT_EQ(ni0.ByteSizeLong(), ni1.ByteSizeLong());
  }

  // 2. check const_node_info
  ASSERT_EQ(gfi0.const_node_info_size(), gfi1.const_node_info_size());
  for (int i = 0; i < gfi0.const_node_info_size(); ++i) {
    const GraphTransferConstNodeInfo& cni0 = gfi0.const_node_info(i);
    const GraphTransferConstNodeInfo& cni1 = gfi1.const_node_info(i);
    ASSERT_EQ(cni0.shape_size(), cni1.shape_size());
    for (int j = 0; j < cni0.shape_size(); ++j) {
      EXPECT_EQ(cni0.shape(j), cni1.shape(j));
    }
    EXPECT_EQ(cni0.ByteSizeLong(), cni1.ByteSizeLong());
    EXPECT_EQ(cni0.DebugString(), cni1.DebugString());
  }

  // 3. check node_input_info
  ASSERT_EQ(gfi0.node_input_info_size(), gfi1.node_input_info_size());
  for (int i = 0; i < gfi0.node_input_info_size(); ++i) {
    const GraphTransferNodeInputInfo& nii0 = gfi0.node_input_info(i);
    const GraphTransferNodeInputInfo& nii1 = gfi1.node_input_info(i);
    EXPECT_EQ(nii0.ByteSizeLong(), nii1.ByteSizeLong());
    EXPECT_EQ(nii0.DebugString(), nii1.DebugString());
  }

  // 4. check node_output_info
  ASSERT_EQ(gfi0.node_output_info_size(), gfi1.node_output_info_size());
  for (int i = 0; i < gfi0.node_output_info_size(); ++i) {
    const GraphTransferNodeOutputInfo& noi0 = gfi0.node_output_info(i);
    const GraphTransferNodeOutputInfo& noi1 = gfi1.node_output_info(i);
    ASSERT_EQ(noi0.max_byte_size_size(), noi1.max_byte_size_size());
    for (int j = 0; j < noi0.max_byte_size_size(); ++j) {
      EXPECT_EQ(noi0.max_byte_size(j), noi1.max_byte_size(j));
    }
    EXPECT_EQ(noi0.ByteSizeLong(), noi1.ByteSizeLong());
    EXPECT_EQ(noi0.DebugString(), noi1.DebugString());
  }

  // 5. check graph_input_node_info
  ASSERT_EQ(gfi0.graph_input_node_info_size(),
            gfi1.graph_input_node_info_size());
  for (int i = 0; i < gfi0.graph_input_node_info_size(); ++i) {
    const GraphTransferGraphInputNodeInfo& gini0 =
        gfi0.graph_input_node_info(i);
    const GraphTransferGraphInputNodeInfo& gini1 =
        gfi0.graph_input_node_info(i);
    EXPECT_EQ(gini0.ByteSizeLong(), gini1.ByteSizeLong());
    EXPECT_EQ(gini0.DebugString(), gini1.DebugString());
  }

  // 6. check graph_output_node_info
  ASSERT_EQ(gfi0.graph_output_node_info_size(),
            gfi1.graph_output_node_info_size());
  for (int i = 0; i < gfi0.graph_output_node_info_size(); ++i) {
    const GraphTransferGraphOutputNodeInfo& goni0 =
        gfi0.graph_output_node_info(i);
    const GraphTransferGraphOutputNodeInfo& goni1 =
        gfi0.graph_output_node_info(i);
    EXPECT_EQ(goni0.ByteSizeLong(), goni1.ByteSizeLong());
    EXPECT_EQ(goni0.DebugString(), goni1.DebugString());
  }
}

// CAVEAT: This test only runs when you specify hexagon library using
// makefile.
// CAVEAT: This test is disabled by default because hexagon can keep only
// two inception graphs on memory which are allocated by other two tests.
// Memory of these graphs are not released until process is killed right now.
// TODO(satok): Figure out how to release memory on hexagon without process
// termination.
#ifdef USE_HEXAGON_LIBS
TEST(GraphTransferer,
     DISABLED_RunInceptionV3OnHexagonExampleWithHexagonWrapper) {
  LOG(INFO) << "Run inception v3 on hexagon with hexagon controller";
  CheckHexagonControllerVersion();

  const IRemoteFusedGraphOpsDefinitions* ops_definitions =
      &HexagonOpsDefinitions::getInstance();
  std::vector<std::pair<string, Tensor>> inputs;
  inputs.emplace_back("Mul", Tensor(DT_FLOAT, {1, WIDTH, HEIGHT, DEPTH}));
  std::vector<string> output_node_names = {"softmax"};

  GraphTransferer gt;
  gt.EnableStrictCheckMode(false);
  profile_utils::CpuUtils::EnableClockCycleProfiling(true);
  ClockCycleProfiler prof;
  prof.Start();
  Status status = gt.LoadGraphFromProtoFile(
      *ops_definitions, MODEL_FILENAME, inputs, output_node_names,
      false,  // is_text_proto
      false,  // shape_inference_for_unknown_shape
      true    // dry_run_for_unknown_shape
  );
  ASSERT_TRUE(status.ok()) << status;
  prof.Stop();
  prof.DumpStatistics("LoadGraphFromProtoFile");

  std::vector<float> img_floats;
  LoadImage(&img_floats);
  const Tensor img_tensor = BuildImageTensor(img_floats);
  RunInferenceByHexagonControlWrapper(gt, img_tensor);
}

TEST(GraphTransferer,
     DISABLED_RunInceptionV3OnHexagonExampleWithHexagonWrapperQuantizedInput) {
  LOG(INFO) << "Run inception v3 on hexagon with hexagon controller "
            << "with quantized input";
  CheckHexagonControllerVersion();

  const IRemoteFusedGraphOpsDefinitions* ops_definitions =
      &HexagonOpsDefinitions::getInstance();
  std::vector<std::pair<string, Tensor>> inputs;
  inputs.emplace_back("Mul", Tensor(DT_QUINT8, {1, WIDTH, HEIGHT, DEPTH}));
  std::vector<string> output_node_names = {"softmax"};

  GraphTransferer gt;
  gt.EnableStrictCheckMode(false);
  profile_utils::CpuUtils::EnableClockCycleProfiling(true);
  ClockCycleProfiler prof;
  prof.Start();
  Status status = gt.LoadGraphFromProtoFile(
      *ops_definitions, MODEL_WITH_QUANTIZED_INPUT_FILENAME, inputs,
      output_node_names,
      /*is_text_proto=*/false,
      /*shape_inference_for_unknown_shape=*/false,
      /*dry_run_for_unknown_shape=*/true);
  ASSERT_TRUE(status.ok()) << status;
  prof.Stop();
  prof.DumpStatistics("LoadGraphFromProtoFile");

  std::vector<float> img_floats;
  LoadImage(&img_floats);
  std::vector<quint8> quantized_img;
  QuantizeImage(img_floats, &quantized_img);
  const Tensor img_tensor = BuildQuantizedImageTensor(quantized_img);
  RunInferenceByHexagonControlWrapper(gt, img_tensor);
}

TEST(GraphTransferer,
     DISABLED_RunInceptionV3OnHexagonExampleWithHexagonWrapperShapeInference) {
  LOG(INFO) << "Run inception v3 on hexagon with hexagon controller";
  CheckHexagonControllerVersion();

  const IRemoteFusedGraphOpsDefinitions* ops_definitions =
      &HexagonOpsDefinitions::getInstance();
  std::vector<std::pair<string, Tensor>> inputs;
  inputs.emplace_back("Mul", Tensor(DT_FLOAT, {1, WIDTH, HEIGHT, DEPTH}));
  std::vector<string> output_node_names = {"softmax"};

  GraphTransferer gt;
  gt.EnableStrictCheckMode(false);
  profile_utils::CpuUtils::EnableClockCycleProfiling(true);
  ClockCycleProfiler prof;
  prof.Start();
  Status status = gt.LoadGraphFromProtoFile(
      *ops_definitions, MODEL_FILENAME, inputs, output_node_names,
      false,  // is_text_proto
      true,   // shape_inference_for_unknown_shape
      false   // dry_run_for_unknown_shape
  );
  ASSERT_TRUE(status.ok()) << status;
  prof.Stop();
  prof.DumpStatistics("LoadGraphFromProtoFile");

  std::vector<float> img_floats;
  LoadImage(&img_floats);
  const Tensor img_tensor = BuildImageTensor(img_floats);
  RunInferenceByHexagonControlWrapper(gt, img_tensor);
}

TEST(GraphTransferer, RunInceptionV3OnHexagonExampleWithTfRuntime) {
  LOG(INFO) << "Fuse and run inception v3 on hexagon with tf runtime";
  CheckHexagonControllerVersion();

  const IRemoteFusedGraphOpsDefinitions* ops_definitions =
      &HexagonOpsDefinitions::getInstance();
  std::vector<std::pair<string, Tensor>> inputs;
  inputs.emplace_back("Mul", Tensor(DT_FLOAT, {1, WIDTH, HEIGHT, DEPTH}));
  std::vector<string> outputs = {"softmax"};

  std::vector<float> img_floats;
  LoadImage(&img_floats);

  LOG(INFO) << "Ioading image finished.";

  GraphDef graph_def;
  Status status = ReadBinaryProto(Env::Default(), MODEL_FILENAME, &graph_def);

  ASSERT_TRUE(status.ok());

  LOG(INFO) << "Build fused graph";
  GraphDef fused_graph_def = GraphTransferUtils::BuildFusedGraphDef(
      HexagonOpsDefinitions::getInstance(),
      REMOTE_FUSED_GRAPH_EXECUTE_NODE_NAME, inputs, outputs, &graph_def);

  RunFusedGraph(fused_graph_def);
}

TEST(GraphTransferer, DISABLED_RunInceptionV3OnHexagonExampleWithFusedGraph) {
  LOG(INFO) << "Run inception v3 with fused graph";
  CheckHexagonControllerVersion();

  GraphDef fused_graph_def;
  Status status =
      ReadBinaryProto(Env::Default(), FUSED_MODEL_FILENAME, &fused_graph_def);
  RunFusedGraph(fused_graph_def);
}

TEST(GraphTransferer, DISABLED_CheckShapeInferencePerformance) {
  CheckHexagonControllerVersion();
  profile_utils::CpuUtils::EnableClockCycleProfiling(true);

  const IRemoteFusedGraphOpsDefinitions* ops_definitions =
      &HexagonOpsDefinitions::getInstance();
  std::vector<std::pair<string, Tensor>> inputs;
  inputs.emplace_back("Mul", Tensor(DT_FLOAT, {1, WIDTH, HEIGHT, DEPTH}));
  std::vector<string> output_node_names = {"softmax"};

  RemoteFusedGraphExecuteUtils::TensorShapeMap output_tensor_info0;
  GraphTransferer gt0;
  gt0.EnableStrictCheckMode(false);
  ClockCycleProfiler prof0;
  prof0.Start();
  Status status = gt0.LoadGraphFromProtoFile(
      *ops_definitions, MODEL_FILENAME, inputs, output_node_names,
      false,  // is_text_proto
      false,  // shape_inference_for_unknown_shape
      true    // dry_run_for_unknown_shape
  );
  const GraphTransferInfo& gfi0 = gt0.GetGraphTransferInfo();

  ASSERT_TRUE(status.ok());
  prof0.Stop();
  prof0.DumpStatistics("Estimate shape by dryrun");

  LOG(INFO) << "(0) node count: " << gfi0.node_info_size() << ", "
            << gfi0.const_node_info_size();

  RemoteFusedGraphExecuteUtils::TensorShapeMap output_tensor_info1;
  GraphTransferer gt1;
  gt1.EnableStrictCheckMode(true);
  ClockCycleProfiler prof1;
  prof1.Start();
  status = gt1.LoadGraphFromProtoFile(
      *ops_definitions, MODEL_FILENAME, inputs, output_node_names,
      false,  // is_text_proto
      true,   // shape_inference_for_unknown_shape
      false   // dry_run_for_unknown_shape
  );
  const GraphTransferInfo& gfi1 = gt1.GetGraphTransferInfo();

  ASSERT_TRUE(status.ok());
  prof1.Stop();
  prof1.DumpStatistics("Estiame shape by shape inference");

  CompareGraphTransferInfo(gfi0, gfi1);

  const RemoteFusedGraphExecuteInfo ei0 =
      BuildRemoteFusedGraphExecuteInfoWithGraphTransferInfo(gfi0);
  const RemoteFusedGraphExecuteInfo ei1 =
      BuildRemoteFusedGraphExecuteInfoWithGraphTransferInfo(gfi1);

  GraphTransferInfo rgfi0;
  rgfi0.ParseFromString(ei0.serialized_executor_parameters());
  GraphTransferInfo rgfi1;
  rgfi1.ParseFromString(ei1.serialized_executor_parameters());

  CompareGraphTransferInfo(rgfi0, rgfi1);
  CompareGraphTransferInfo(gfi0, rgfi0);
  CompareGraphTransferInfo(gfi1, rgfi1);
}
#endif

}  // namespace tensorflow
