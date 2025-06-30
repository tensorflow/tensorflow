/* Copyright 2025 The OpenXLA Authors.
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
#include <stdlib.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {
class NvshmemGpuCollectivesTest : public ::testing::Test {};

static const char* test_binary_name;

absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> CompileExecutable(
    absl::string_view program, xla::PjRtClient& client,
    xla::CompileOptions compile_options = xla::CompileOptions()) {
  TF_ASSIGN_OR_RETURN(auto hlo_module,
                      ParseAndReturnUnverifiedModule(program, {}));

  xla::XlaComputation xla_computation(hlo_module->ToProto());
  return client.CompileAndLoad(xla_computation, compile_options);
}

absl::StatusOr<std::string> GetDataTypeString(xla::PrimitiveType data_type) {
  switch (data_type) {
    case xla::PrimitiveType::F32:
      return "f32";
    case xla::PrimitiveType::F64:
      return "f64";
    case xla::PrimitiveType::BF16:
      return "bf16";
    case xla::PrimitiveType::F16:
      return "f16";
    case xla::PrimitiveType::U32:
      return "u32";
    case xla::PrimitiveType::U64:
      return "u64";
    case xla::PrimitiveType::S32:
      return "s32";
    case xla::PrimitiveType::S64:
      return "s64";
    default:
      return absl::InvalidArgumentError("Invalida data type.");
  }
}

void RunNvshmemTest(PrimitiveType data_type, absl::string_view test_case) {
  const int num_ranks = 2;
  tsl::SubProcess child[num_ranks];
  for (int rank_id = 0; rank_id < num_ranks; ++rank_id) {
    std::vector<std::string> argv;
    argv.push_back(test_binary_name);
    argv.push_back(absl::StrFormat("--rank_id=%d", rank_id));
    argv.push_back(absl::StrFormat("--num_ranks=%d", num_ranks));
    argv.push_back(absl::StrFormat("--input_data_type=%d", (int)data_type));
    argv.push_back(absl::StrFormat("--test_case=%s", test_case));
    argv.push_back(absl::StrFormat("--v=1"));
    child[rank_id].SetProgram(test_binary_name, argv);
    child[rank_id].SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
    child[rank_id].SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
    ASSERT_TRUE(child[rank_id].Start()) << "rank " << rank_id;
  }
  for (int rank_id = 0; rank_id < num_ranks; ++rank_id) {
    std::string stdout_str;
    std::string stderr_str;
    int child_status =
        child[rank_id].Communicate(nullptr, &stdout_str, &stderr_str);
    EXPECT_EQ(child_status, 0) << " rank " << rank_id << "\nstdout:\n"
                               << stdout_str << "\nstderr:\n"
                               << stderr_str;
  }
}

TEST(NvshmemGpuCollectivesTest, NvshmemCollectivePermuteFloat) {
  RunNvshmemTest(PrimitiveType::F32, "collective_permute");
}

TEST(NvshmemGpuCollectivesTest, NvshmemSendRecvFloat) {
  RunNvshmemTest(PrimitiveType::F32, "send_recv");
}

absl::Status NvshmemCollectiveTestBody(int rank_id, int num_ranks,
                                       int input_data_type,
                                       absl::string_view test_case) {
  xla::PrimitiveType data_type = (xla::PrimitiveType)input_data_type;
  std::unique_ptr<xla::DistributedRuntimeService> service;
  if (rank_id == 0) {
    xla::CoordinationServiceImpl::Options service_options;
    service_options.num_nodes = num_ranks;
    TF_ASSIGN_OR_RETURN(service, xla::GetDistributedRuntimeService(
                                     "[::]:12345", service_options));
  }

  xla::DistributedRuntimeClient::Options distributed_options;
  distributed_options.node_id = rank_id;
  distributed_options.init_timeout = absl::Seconds(120);
  auto distributed_client =
      GetDistributedRuntimeClient("127.0.0.1:12345", distributed_options);
  TF_QCHECK_OK(distributed_client->Connect());
  GpuClientOptions client_options;
  client_options.node_id = rank_id;
  client_options.allowed_devices = {rank_id};
  client_options.num_nodes = num_ranks;
  client_options.kv_store = GetDistributedKeyValueStore(distributed_client,
                                                        /*key_prefix=*/"gpu:");
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                      GetStreamExecutorGpuClient(client_options));

  xla::CompileOptions options;
  options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_experimental_enable_nvshmem(true);
  options.executable_build_options.set_run_backend_only(true);
  options.executable_build_options.set_use_spmd_partitioning(false);
  options.executable_build_options.set_num_replicas(num_ranks);
  TF_ASSIGN_OR_RETURN(std::string data_type_str, GetDataTypeString(data_type));
  std::string kProgram;

  if (test_case == "collective_permute") {
    kProgram = absl::StrFormat(R"(
        HloModule NvshmemCollectivePermute
        ENTRY test_computation {
          data = %s[] constant(42)
          start = (%s[], %s[]) collective-permute-start(data),
                source_target_pairs={{0,1},{1,0}},
                backend_config={"collective_backend_config":{"backend":"NVSHMEM"}}
          ROOT done = %s[] collective-permute-done(start)
        })",
                               data_type_str, data_type_str, data_type_str,
                               data_type_str);
  } else if (test_case == "send_recv") {
    kProgram = absl::StrFormat(R"(
        HloModule NvshmemSendRecv
        ENTRY test_computation {
          data = %s[] constant(42)
          after-all = token[] after-all()
          recv = (%s[], %s[], token[]) recv(after-all), channel_id=0, frontend_attributes={_xla_send_recv_source_target_pairs="{{0,1}}"}, backend_config={"collective_backend_config":{"backend":"NVSHMEM"}}
          send = (%s[], %s[], token[]) send(data, after-all), channel_id=0, control-predecessors={recv}, frontend_attributes={_xla_send_recv_source_target_pairs="{{0,1}}"}, backend_config={"collective_backend_config":{"backend":"NVSHMEM"}}
          recv-done = (%s[], token[]) recv-done(recv), channel_id=0
          recv-data = %s[] get-tuple-element(recv-done), index=0
          send-done = token[] send-done(send), channel_id=0, control-predecessors={recv}
          ROOT result = %s[] copy(recv-data)
        })",
                               data_type_str, data_type_str, data_type_str,
                               data_type_str, data_type_str, data_type_str,
                               data_type_str, data_type_str);
  }
  TF_ASSIGN_OR_RETURN(auto executable,
                      CompileExecutable(kProgram, *client, options));
  TF_ASSIGN_OR_RETURN(auto hlo_modules, executable->GetHloModules());
  TF_ASSIGN_OR_RETURN(auto result, executable->Execute({{}}, ExecuteOptions()));
  std::vector<std::unique_ptr<xla::PjRtBuffer>>& result_buffers = result[0];
  TF_ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> literal,
                      result_buffers[0]->ToLiteralSync());

  if (test_case == "collective_permute") {
    switch (data_type) {
      case xla::PrimitiveType::F32: {
        TF_RET_CHECK(literal->data<float>()[0] == 42.0f);
        break;
      }
      case xla::PrimitiveType::F64: {
        TF_RET_CHECK(literal->data<double>()[0] == 42.0);
        break;
      }
      case xla::PrimitiveType::BF16: {
        TF_RET_CHECK(literal->data<Eigen::bfloat16>()[0] ==
                     Eigen::bfloat16(42));
        break;
      }
      case xla::PrimitiveType::F16: {
        TF_RET_CHECK(literal->data<Eigen::half>()[0] == Eigen::half(42));
        break;
      }
      case xla::PrimitiveType::U32: {
        TF_RET_CHECK(literal->data<uint32_t>()[0] == 42);
        break;
      }
      case xla::PrimitiveType::U64: {
        TF_RET_CHECK(literal->data<uint64_t>()[0] == 42);
        break;
      }
      case xla::PrimitiveType::S32: {
        TF_RET_CHECK(literal->data<int32_t>()[0] == 42);
        break;
      }
      case xla::PrimitiveType::S64: {
        TF_RET_CHECK(literal->data<int64_t>()[0] == 42);
        break;
      }
      default:
        return absl::InvalidArgumentError("Invalid data type.");
    }
  } else if (test_case == "send_recv" && rank_id == 1) {
    switch (data_type) {
      case xla::PrimitiveType::F32: {
        float value = literal->data<float>()[0];
        TF_RET_CHECK(value == 42.0f)
            << "Value mismatch: got " << value << ", expected 42.0f";
        break;
      }
      case xla::PrimitiveType::F64: {
        double value = literal->data<double>()[0];
        TF_RET_CHECK(value == 42.0)
            << "Value mismatch: got " << value << ", expected 42.0";
        break;
      }
      case xla::PrimitiveType::BF16: {
        float value = static_cast<float>(literal->data<Eigen::bfloat16>()[0]);
        TF_RET_CHECK(literal->data<Eigen::bfloat16>()[0] == Eigen::bfloat16(42))
            << "Value mismatch: got " << value << ", expected 42";
        break;
      }
      case xla::PrimitiveType::F16: {
        float value = static_cast<float>(literal->data<Eigen::half>()[0]);
        TF_RET_CHECK(literal->data<Eigen::half>()[0] == Eigen::half(42))
            << "Value mismatch: got " << value << ", expected 42";
        break;
      }
      case xla::PrimitiveType::U32: {
        uint32_t value = literal->data<uint32_t>()[0];
        TF_RET_CHECK(value == 42)
            << "Value mismatch: got " << value << ", expected 42";
        break;
      }
      case xla::PrimitiveType::U64: {
        uint64_t value = literal->data<uint64_t>()[0];
        TF_RET_CHECK(value == 42)
            << "Value mismatch: got " << value << ", expected 42";
        break;
      }
      case xla::PrimitiveType::S32: {
        int32_t value = literal->data<int32_t>()[0];
        TF_RET_CHECK(value == 42)
            << "Value mismatch: got " << value << ", expected 42";
        break;
      }
      case xla::PrimitiveType::S64: {
        int64_t value = literal->data<int64_t>()[0];
        TF_RET_CHECK(value == 42)
            << "Value mismatch: got " << value << ", expected 42";
        break;
      }
      default:
        return absl::InvalidArgumentError("Invalid data type.");
    }
  }

  VLOG(1) << "Rank " << rank_id << " completed successfully";

  return absl::OkStatus();
}
}  // namespace
}  // namespace xla

int main(int argc, char* argv[]) {
  // Save name of binary so that it may invoke itself.
  xla::test_binary_name = argv[0];
  int rank_id = -1;
  int num_ranks = -1;
  int input_data_type = (int)xla::PrimitiveType::F32;
  std::string test_case = "all_reduce";  // Add test_case parameter
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("rank_id", &rank_id, "Rank ID for nvshmem collective test."),
      tsl::Flag("num_ranks", &num_ranks,
                "Total number of ranks for nvshmem collective test."),
      tsl::Flag("input_data_type", &input_data_type,
                "Data type to test for nvshmem collective test."),
      tsl::Flag("test_case", &test_case,
                "Test case to run (collective_permute, send_recv)."),
  };
  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  testing::InitGoogleTest(&argc, argv);
  if (rank_id >= 0) {
    return xla::NvshmemCollectiveTestBody(rank_id, num_ranks, input_data_type,
                                          test_case)
        .raw_code();
  }
  return RUN_ALL_TESTS();
}
