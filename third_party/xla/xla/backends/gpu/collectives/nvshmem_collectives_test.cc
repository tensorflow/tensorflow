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

#include "xla/backends/gpu/collectives/nvshmem_collectives.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "xla/debug_options_flags.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/util/command_line_flags.h"

namespace xla::gpu {
namespace {

// Tests that NVSHMEM library can be loaded and initialized.
TEST(NvshmemTest, Initialization) {
  const int num_nodes = 2;
  tsl::SubProcess child[num_nodes];
  for (int node_id = 0; node_id < num_nodes; ++node_id) {
    std::vector<std::string> argv;
    argv.push_back("nvshmem_test");
    argv.push_back(absl::StrFormat("--node_id=%d", node_id));
    argv.push_back(absl::StrFormat("--num_nodes=%d", num_nodes));
    child[node_id].SetProgram("/proc/self/exe", argv);
    child[node_id].SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
    child[node_id].SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
    ASSERT_TRUE(child[node_id].Start()) << "node " << node_id;
  }
  for (int node_id = 0; node_id < num_nodes; ++node_id) {
    std::string stdout_str;
    std::string stderr_str;
    int child_status =
        child[node_id].Communicate(nullptr, &stdout_str, &stderr_str);
    EXPECT_EQ(child_status, 0) << " node " << node_id << "\nstdout:\n"
                               << stdout_str << "\nstderr:\n"
                               << stderr_str;
  }
}

absl::Status InitializationTestBody(const int node_id, const int num_nodes) {
  std::unique_ptr<xla::DistributedRuntimeService> service;
  if (node_id == 0) {
    xla::CoordinationServiceImpl::Options service_options;
    service_options.num_nodes = num_nodes;
    TF_ASSIGN_OR_RETURN(service, xla::GetDistributedRuntimeService(
                                     "[::]:12345", service_options));
  }

  xla::DistributedRuntimeClient::Options distributed_options;
  distributed_options.node_id = node_id;
  distributed_options.init_timeout = absl::Seconds(120);
  auto distributed_client =
      GetDistributedRuntimeClient("127.0.0.1:12345", distributed_options);
  TF_QCHECK_OK(distributed_client->Connect());
  auto kv_store =
      GetDistributedKeyValueStore(distributed_client, /*key_prefix=*/"gpu:");

  NvshmemCollectives::Default()->SetEnvInfo(node_id, num_nodes, 1, kv_store);
  cudaSetDevice(node_id);
  TF_ASSIGN_OR_RETURN(void* ptr, NvshmemCollectives::Default()->Allocate(1024));
  TF_RET_CHECK(ptr != nullptr);
  TF_RETURN_IF_ERROR(NvshmemCollectives::Default()->Deallocate(ptr));
  return absl::OkStatus();
}

}  // namespace
}  // namespace xla::gpu

int main(int argc, char* argv[]) {
  // Save name of binary so that it may invoke itself.
  int node_id = -1;
  int num_nodes = -1;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("node_id", &node_id, "Node ID for Initialization test."),
      tsl::Flag("num_nodes", &num_nodes,
                "Number of nodes for Initialization test."),
  };
  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  testing::InitGoogleTest(&argc, argv);
  if (node_id >= 0) {
    absl::Status result = xla::gpu::InitializationTestBody(node_id, num_nodes);
    if (!result.ok()) {
      LOG(ERROR) << result;
    }
    return result.raw_code();
  }
  return RUN_ALL_TESTS();
}
