/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/client/executable_build_options.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/compile_options.pb.h"
#include "xla/service/computation_placer.h"
#include "xla/service/test_compilation_environment.pb.h"
#include "xla/shape.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

using ::tsl::proto_testing::EqualsProto;

// In order to use TestCompilationEnvironment* with CompilationEnvironments, we
// must define ProcessNewEnv for them.
std::unique_ptr<tsl::protobuf::Message> ProcessNewEnv(
    std::unique_ptr<tsl::protobuf::Message> msg) {
  std::unique_ptr<test::TestCompilationEnvironment1> env(
      tensorflow::down_cast<test::TestCompilationEnvironment1*>(msg.release()));
  return env;
}

TEST(ExecutableBuildOptionsTest, ProtoRoundTripWorks) {
  ExecutableBuildOptionsProto p;
  p.set_device_ordinal(1);

  // Set result_layout to an array shape.
  p.mutable_result_layout()->set_element_type(PrimitiveType::F32);
  p.mutable_result_layout()->add_dimensions(2);
  p.mutable_result_layout()->add_is_dynamic_dimension(true);

  {
    CompilationEnvironments::RegisterProcessNewEnvFn(
        test::TestCompilationEnvironment1::descriptor(), ProcessNewEnv);
    CompilationEnvironments envs;
    auto env1 = std::make_unique<test::TestCompilationEnvironment1>();
    env1->set_some_flag(10);
    TF_ASSERT_OK(envs.AddEnv(std::move(env1)));
    p.mutable_comp_envs()->MergeFrom(envs.ToProto());
  }
  p.mutable_debug_options()->set_xla_gpu_force_compilation_parallelism(3);
  p.set_num_replicas(4);
  p.set_num_partitions(5);
  p.set_use_spmd_partitioning(true);
  p.set_use_auto_spmd_partitioning(true);
  p.set_exec_time_optimization_effort(6.0);
  p.set_memory_fitting_effort(7.0);
  p.set_deduplicate_hlo(true);
  DeviceAssignment{1, 1}.Serialize(p.mutable_device_assignment());
  p.mutable_device_assignment()->set_replica_count(1);
  p.set_alias_passthrough_params(true);
  p.set_run_backend_only(true);
  p.add_allow_spmd_sharding_propagation_to_parameters(true);
  p.add_allow_spmd_sharding_propagation_to_output(true);
  p.mutable_fdo_profile()->append("a");
  p.set_device_memory_size(10);
  p.add_auto_spmd_partitioning_mesh_shape(11);
  p.add_auto_spmd_partitioning_mesh_ids(12);
  p.set_use_shardy_partitioner(true);
  p.set_process_index(13);
  p.set_process_count(14);

  TF_ASSERT_OK_AND_ASSIGN(const ExecutableBuildOptions options,
                          ExecutableBuildOptionsFromProto(p));
  TF_ASSERT_OK_AND_ASSIGN(const ExecutableBuildOptionsProto p2,
                          options.ToProto());
  EXPECT_THAT(p2, EqualsProto(p));
}

TEST(ExecutableBuildOptionsTest, SerializationFailsOnNonSerializableFields) {
  {
    ExecutableBuildOptions options;
    tsl::thread::ThreadPool pool{tsl::Env::Default(), tsl::ThreadOptions(), "",
                                 1};
    options.set_compile_thread_pool(&pool);
    EXPECT_THAT(options.ToProto(),
                tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    ExecutableBuildOptions options;
    options.set_layout_canonicalization_callback(
        [](const HloModule&)
            -> absl::StatusOr<std::pair<std::vector<Shape>, Shape>> {
          return std::make_pair(std::vector<Shape>(), Shape());
        });
    EXPECT_THAT(options.ToProto(),
                tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

}  // namespace
}  // namespace xla
