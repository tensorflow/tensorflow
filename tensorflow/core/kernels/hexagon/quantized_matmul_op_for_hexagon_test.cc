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
// Tests in this file are designed to evaluate hexagon DSP operations.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

#ifdef USE_HEXAGON_LIBS
#include "tensorflow/core/platform/hexagon/soc_interface.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#endif

namespace tensorflow {

class QuantizedMatMulOpForHexagonTest : public OpsTestBase {
 protected:
  void SetUp() final {
#ifdef USE_HEXAGON_LIBS
    profile_utils::CpuUtils::EnableClockCycleProfiling(true);
    LOG(INFO) << "Hexagon libs are linked (wrapper version = "
              << soc_interface_GetWrapperVersion()
              << ", hexagon binary version = "
              << soc_interface_GetSocControllerVersion() << ")";
    LOG(INFO) << "Cpu frequency = "
              << profile_utils::CpuUtils::GetCycleCounterFrequency();
#else
    LOG(WARNING) << "Hexagon libs are not linked.";
#endif
  }
};

// Shows some statistics of hexagon dsp using hexagon specific APIs
#ifdef USE_HEXAGON_LIBS
TEST_F(QuantizedMatMulOpForHexagonTest, EvaluateSharedLibOverhead) {
  const uint64 overhead_shared_lib_start =
      profile_utils::CpuUtils::GetCurrentClockCycle();
  const int wrapper_version = soc_interface_GetWrapperVersion();
  const uint64 overhead_shared_lib_end =
      profile_utils::CpuUtils::GetCurrentClockCycle();
  const uint64 overhead_shared_lib_diff =
      (overhead_shared_lib_end - overhead_shared_lib_start);
  const uint64 overhead_hexagon_rpc_start =
      profile_utils::CpuUtils::GetCurrentClockCycle();
  const int hexagon_binary_version = soc_interface_GetSocControllerVersion();
  const uint64 overhead_hexagon_rpc_end =
      profile_utils::CpuUtils::GetCurrentClockCycle();
  const uint64 overhead_hexagon_rpc_diff =
      (overhead_hexagon_rpc_end - overhead_hexagon_rpc_start);
  LOG(INFO) << "Shared lib (ver = " << wrapper_version << ") overhead is "
            << overhead_shared_lib_diff << " cycles, time = "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   profile_utils::CpuUtils::ConvertClockCycleToTime(
                       overhead_shared_lib_diff))
                   .count()
            << " usec";
  LOG(INFO) << "hexagon rpc (ver = " << hexagon_binary_version
            << ") overhead is " << overhead_hexagon_rpc_diff
            << " cycles, time = "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   profile_utils::CpuUtils::ConvertClockCycleToTime(
                       overhead_hexagon_rpc_diff))
                   .count()
            << " usec";
}
#endif

// Runs two small matrices through the operator, and leaves all the parameters
// at their default values.
// This test is a sample to execute matmul on hexagon.
TEST_F(QuantizedMatMulOpForHexagonTest, Small_NoParams) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_mat_mul_op", "QuantizedMatMul")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("Toutput", DataTypeToEnum<qint32>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // A matrix is:
  // |  1 |  2 |  3 |
  // |  4 |  5 |  6 |
  AddInputFromArray<quint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // B matrix is:
  // |  7 |  8 |  9 | 10 |
  // | 11 | 12 | 13 | 14 |
  // | 15 | 16 | 17 | 18 |
  AddInputFromArray<quint8>(TensorShape({3, 4}),
                            {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  AddInputFromArray<float>(TensorShape({1}), {0});
  AddInputFromArray<float>(TensorShape({1}), {255.0f});
  AddInputFromArray<float>(TensorShape({1}), {0});
  AddInputFromArray<float>(TensorShape({1}), {255.0f});

  TF_ASSERT_OK(RunOpKernel());
  // Here are the results we expect, from hand calculations:
  // (1 * 7) + (2 * 11) + (3 * 15) = 74
  // (1 * 8) + (2 * 12) + (3 * 16) = 80
  // (1 * 9) + (2 * 13) + (3 * 17) = 86
  // (1 * 10) + (2 * 14) + (3 * 18) = 92
  // (4 * 7) + (5 * 11) + (6 * 15) = 173
  // (4 * 8) + (5 * 12) + (6 * 16) = 188
  // (4 * 9) + (5 * 13) + (6 * 17) = 203
  // (4 * 10) + (5 * 14) + (6 * 18) = 218
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 4}));
  test::FillValues<qint32>(&expected, {74, 80, 86, 92, 173, 188, 203, 218});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

}  // namespace tensorflow
