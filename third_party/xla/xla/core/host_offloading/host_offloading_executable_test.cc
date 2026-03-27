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

#include "xla/core/host_offloading/host_offloading_executable.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/ffi.h"
#include "xla/backends/cpu/nanort/nanort_client.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/core/host_offloading/host_offloading_buffer.h"
#include "xla/core/host_offloading/host_offloading_executable.pb.h"
#include "xla/core/host_offloading/host_offloading_nanort_executable.h"
#include "xla/core/host_offloading/host_offloading_pjrt_executable.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla {
namespace {

absl::StatusOr<std::unique_ptr<HostOffloadingExecutable>> CompileFromString(
    absl::string_view str,
    HostOffloadingExecutableProto::ExecutableType executable_type) {
  HloModuleConfig config;
  TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnUnverifiedModule(str));

  HostOffloadingExecutableProto executable_proto;
  *executable_proto.mutable_hlo_module() = module->ToProto();
  executable_proto.set_executable_type(executable_type);

  switch (executable_type) {
    case HostOffloadingExecutableProto::EXECUTABLE_TYPE_NANORT: {
      xla::cpu::NanoRtClient client;
      XlaComputation computation(module->ToProto());
      TF_ASSIGN_OR_RETURN(auto executable, client.Compile(computation));
      TF_ASSIGN_OR_RETURN(auto aot_compilation_result,
                          client.Export(executable.get()));

      xla::cpu::CpuAotCompilationResult* cpu_aot_compilation_result =
          absl::down_cast<cpu::CpuAotCompilationResult*>(
              aot_compilation_result.get());

      *executable_proto.mutable_aot_compilation_result() =
          cpu_aot_compilation_result->proto();
      return HostOffloadingNanoRtExecutable::LoadFromProto(executable_proto);
    }

    case HostOffloadingExecutableProto::EXECUTABLE_TYPE_PJRT:
      return HostOffloadingPjRtExecutable::LoadFromProto(executable_proto);
    default:
      return absl::InvalidArgumentError(
          "Unsupported executable type: " +
          HostOffloadingExecutableProto::ExecutableType_Name(executable_type));
  }
}

HostOffloadingExecutable::ExecuteOptions EmptyExecuteOptions() {
  return {.launch_id = 0};
}

class HostOffloadingRuntimeExecutableTest
    : public ::testing::TestWithParam<
          HostOffloadingExecutableProto::ExecutableType> {};

using ::testing::ElementsAreArray;

TEST_P(HostOffloadingRuntimeExecutableTest, NonAliasedOutput) {
  std::string str = R"(
    HloModule add

    ENTRY %main {
      %p0 = f32[4] parameter(0)
      ROOT %add = f32[4] add(%p0, %p0)
    }
  )";

  HostOffloadingExecutableProto::ExecutableType
      host_offloading_executable_type = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(
      auto computation,
      CompileFromString(str, host_offloading_executable_type));

  Shape shape = ShapeUtil::MakeShape(xla::PrimitiveType::F32, {4});

  auto input_literal = LiteralUtil::CreateR1<float>({1., 2., 3., 4.});
  auto result_literal = LiteralUtil::CreateR1<float>({0., 0., 0., 0.});

  std::vector<ShapeTree<HostOffloadingBuffer>> parameters = {
      ShapeTree<HostOffloadingBuffer>(
          shape, HostOffloadingBuffer(input_literal.data<float>())),
  };
  ShapeTree<HostOffloadingBuffer> result(
      shape, HostOffloadingBuffer(result_literal.data<float>()));

  auto execute_event =
      computation->Execute(parameters, result, EmptyExecuteOptions());
  tsl::BlockUntilReady(execute_event);
  EXPECT_FALSE(execute_event.IsError());
  EXPECT_THAT(result_literal.data<float>(), ElementsAreArray({2, 4, 6, 8}));
}

TEST_P(HostOffloadingRuntimeExecutableTest, AliasedOutput) {
  std::string str = R"(
    HloModule add, input_output_alias={ {}: (1, {}, must-alias) }

    ENTRY %main {
      %p0 = f32[4] parameter(0)
      %p1 = f32[4] parameter(1)
      ROOT %add = f32[4] add(%p0, %p0)
    }
  )";

  HostOffloadingExecutableProto::ExecutableType
      host_offloading_executable_type = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(
      auto computation,
      CompileFromString(str, host_offloading_executable_type));

  Shape shape = ShapeUtil::MakeShape(xla::PrimitiveType::F32, {4});

  auto input_literal = LiteralUtil::CreateR1<float>({1., 2., 3., 4.});
  auto result_literal = LiteralUtil::CreateR1<float>({0., 0., 0., 0.});

  std::vector<ShapeTree<HostOffloadingBuffer>> parameters = {
      ShapeTree<HostOffloadingBuffer>(
          shape, HostOffloadingBuffer(input_literal.data<float>())),
      ShapeTree<HostOffloadingBuffer>(
          shape, HostOffloadingBuffer(result_literal.data<float>())),
  };
  ShapeTree<HostOffloadingBuffer> result(
      shape, HostOffloadingBuffer(result_literal.data<float>()));

  auto execute_event =
      computation->Execute(parameters, result, EmptyExecuteOptions());
  tsl::BlockUntilReady(execute_event);
  EXPECT_FALSE(execute_event.IsError());
  EXPECT_THAT(result_literal.data<float>(), ElementsAreArray({2, 4, 6, 8}));
}

TEST_P(HostOffloadingRuntimeExecutableTest, TwoOutputsOneAliased) {
  std::string str = R"(
    HloModule add, input_output_alias={ {0}: (1, {}, must-alias) }

    ENTRY %main {
      %p0 = f32[4] parameter(0)
      %p1 = f32[4] parameter(1)
      %add = f32[4] add(%p0, %p0)
      %mul = f32[4] multiply(%p0, %p0)
      ROOT %tuple = (f32[4], f32[4]) tuple(%add, %mul)
    }
  )";

  HostOffloadingExecutableProto::ExecutableType
      host_offloading_executable_type = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(
      auto computation,
      CompileFromString(str, host_offloading_executable_type));

  Shape shape = ShapeUtil::MakeShape(xla::PrimitiveType::F32, {4});

  auto input_literal = LiteralUtil::CreateR1<float>({1., 2., 3., 4.});
  auto result0_literal = LiteralUtil::CreateR1<float>({0., 0., 0., 0.});
  auto result1_literal = LiteralUtil::CreateR1<float>({0., 0., 0., 0.});

  std::vector<ShapeTree<HostOffloadingBuffer>> parameters = {
      ShapeTree<HostOffloadingBuffer>(
          shape, HostOffloadingBuffer(input_literal.data<float>())),
      ShapeTree<HostOffloadingBuffer>(
          shape, HostOffloadingBuffer(result0_literal.data<float>())),
  };

  const bool is_pjrt_executable =
      host_offloading_executable_type ==
      HostOffloadingExecutableProto::EXECUTABLE_TYPE_PJRT;

  // PJRT uses aliasing to return the first result.
  ShapeTree<HostOffloadingBuffer> result;
  if (is_pjrt_executable) {
    result = ShapeTree<HostOffloadingBuffer>(
        shape, HostOffloadingBuffer(result1_literal.data<float>()));
  } else {  // NanoRt executable
    result = ShapeTree<HostOffloadingBuffer>(
        ShapeUtil::MakeTupleShape({shape, shape}));
    *result.mutable_element({0}) =
        HostOffloadingBuffer(result0_literal.data<float>());
    *result.mutable_element({1}) =
        HostOffloadingBuffer(result1_literal.data<float>());
  }

  auto execute_event =
      computation->Execute(parameters, result, EmptyExecuteOptions());
  tsl::BlockUntilReady(execute_event);
  EXPECT_FALSE(execute_event.IsError());
  EXPECT_THAT(result0_literal.data<float>(), ElementsAreArray({2, 4, 6, 8}));
  EXPECT_THAT(result1_literal.data<float>(), ElementsAreArray({1, 4, 9, 16}));
}

TEST_P(HostOffloadingRuntimeExecutableTest, NonAliasedTupleOutput) {
  std::string str = R"(
    HloModule add

    ENTRY %main {
      %p0 = f32[4] parameter(0)
      %add = f32[4] add(%p0, %p0)
      %mul = f32[4] multiply(%p0, %p0)
      ROOT %tuple = (f32[4], f32[4]) tuple(%add, %mul)
    }
  )";

  HostOffloadingExecutableProto::ExecutableType
      host_offloading_executable_type = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(
      auto computation,
      CompileFromString(str, host_offloading_executable_type));

  Shape shape = ShapeUtil::MakeShape(xla::PrimitiveType::F32, {4});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});

  auto input_literal = LiteralUtil::CreateR1<float>({1., 2., 3., 4.});
  auto result0_literal = LiteralUtil::CreateR1<float>({0., 0., 0., 0.});
  auto result1_literal = LiteralUtil::CreateR1<float>({0., 0., 0., 0.});

  std::vector<ShapeTree<HostOffloadingBuffer>> parameters = {
      ShapeTree<HostOffloadingBuffer>(
          shape, HostOffloadingBuffer(input_literal.data<float>())),
  };
  ShapeTree<HostOffloadingBuffer> result(tuple_shape);
  *result.mutable_element({0}) =
      HostOffloadingBuffer(result0_literal.data<float>());
  *result.mutable_element({1}) =
      HostOffloadingBuffer(result1_literal.data<float>());

  auto execute_event =
      computation->Execute(parameters, result, EmptyExecuteOptions());
  tsl::BlockUntilReady(execute_event);
  EXPECT_FALSE(execute_event.IsError());
  EXPECT_THAT(result0_literal.data<float>(), ElementsAreArray({2, 4, 6, 8}));
  EXPECT_THAT(result1_literal.data<float>(), ElementsAreArray({1, 4, 9, 16}));
}

TEST_P(HostOffloadingRuntimeExecutableTest, TupleParameter) {
  std::string str = R"(
    HloModule add

    ENTRY %main {
      %p0 = ((f32[4], f32[4]), f32[4]) parameter(0)
      %t0 = (f32[4], f32[4]) get-tuple-element(%p0), index=0
      %v0 = f32[4] get-tuple-element(%t0), index=0
      %v1 = f32[4] get-tuple-element(%t0), index=1
      %v2 = f32[4] get-tuple-element(%p0), index=1
      %add0 = f32[4] add(%v0, %v1)
      ROOT %add1 = f32[4] add(%add0, %v2)
    }
  )";

  HostOffloadingExecutableProto::ExecutableType
      host_offloading_executable_type = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(
      auto computation,
      CompileFromString(str, host_offloading_executable_type));

  Shape shape = ShapeUtil::MakeShape(xla::PrimitiveType::F32, {4});
  Shape tuple_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({shape, shape}), shape});

  auto input0_literal = LiteralUtil::CreateR1<float>({1., 2., 3., 4.});
  auto input1_literal = LiteralUtil::CreateR1<float>({2., 3., 4., 5.});
  auto input2_literal = LiteralUtil::CreateR1<float>({3., 4., 5., 6.});
  auto result_literal = LiteralUtil::CreateR1<float>({0., 0., 0., 0.});

  std::vector<ShapeTree<HostOffloadingBuffer>> parameters = {
      ShapeTree<HostOffloadingBuffer>(tuple_shape)};
  *parameters[0].mutable_element({0, 0}) =
      HostOffloadingBuffer(input0_literal.data<float>());
  *parameters[0].mutable_element({0, 1}) =
      HostOffloadingBuffer(input1_literal.data<float>());
  *parameters[0].mutable_element({1}) =
      HostOffloadingBuffer(input2_literal.data<float>());

  ShapeTree<HostOffloadingBuffer> result(
      shape, HostOffloadingBuffer(result_literal.data<float>()));

  auto execute_event =
      computation->Execute(parameters, result, EmptyExecuteOptions());
  tsl::BlockUntilReady(execute_event);
  EXPECT_FALSE(execute_event.IsError());
  EXPECT_THAT(result_literal.data<float>(), ElementsAreArray({6, 9, 12, 15}));
}

TEST_P(HostOffloadingRuntimeExecutableTest, TupleParameterWithAliasedOutput) {
  std::string str = R"(
    HloModule add, input_output_alias={ {}: (0, {1}, must-alias) }

    ENTRY %main {
      %p0 = ((f32[4], f32[4]), f32[4]) parameter(0)
      %t0 = (f32[4], f32[4]) get-tuple-element(%p0), index=0
      %v0 = f32[4] get-tuple-element(%t0), index=0
      %v1 = f32[4] get-tuple-element(%t0), index=1
      %v2 = f32[4] get-tuple-element(%p0), index=1
      %add0 = f32[4] add(%v0, %v1)
      ROOT %add1 = f32[4] add(%add0, %v2)
    }
  )";

  HostOffloadingExecutableProto::ExecutableType
      host_offloading_executable_type = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(
      auto computation,
      CompileFromString(str, host_offloading_executable_type));

  Shape shape = ShapeUtil::MakeShape(xla::PrimitiveType::F32, {4});
  Shape tuple_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({shape, shape}), shape});

  auto input0_literal = LiteralUtil::CreateR1<float>({1., 2., 3., 4.});
  auto input1_literal = LiteralUtil::CreateR1<float>({2., 3., 4., 5.});
  auto result_literal = LiteralUtil::CreateR1<float>({0., 0., 0., 0.});

  std::vector<ShapeTree<HostOffloadingBuffer>> parameters = {
      ShapeTree<HostOffloadingBuffer>(tuple_shape)};
  *parameters[0].mutable_element({0, 0}) =
      HostOffloadingBuffer(input0_literal.data<float>());
  *parameters[0].mutable_element({0, 1}) =
      HostOffloadingBuffer(input1_literal.data<float>());
  *parameters[0].mutable_element({1}) =
      HostOffloadingBuffer(result_literal.data<float>());

  ShapeTree<HostOffloadingBuffer> result(
      shape, HostOffloadingBuffer(result_literal.data<float>()));

  auto execute_event =
      computation->Execute(parameters, result, EmptyExecuteOptions());
  tsl::BlockUntilReady(execute_event);
  EXPECT_FALSE(execute_event.IsError());
  EXPECT_THAT(result_literal.data<float>(), ElementsAreArray({3, 5, 7, 9}));
}

constexpr int32_t kDummyFFIResult = 12345;

absl::Status CustomCallUsingThreadPool(
    ffi::Result<ffi::BufferR0<PrimitiveType::S32>> dummy_result,
    const Eigen::ThreadPoolDevice* pool) {
  if (pool == nullptr) {
    return absl::InvalidArgumentError(
        "FFI call expected to be called with a threadpool.");
  }
  dummy_result->typed_data()[0] = kDummyFFIResult;
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kCustomCallUsingThreadPool, CustomCallUsingThreadPool,
                       xla::ffi::Ffi::Bind()
                           .Ret<ffi::BufferR0<PrimitiveType::S32>>()
                           .Ctx<xla::ffi::IntraOpThreadPool>());

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "CustomCallUsingThreadPool",
                         "Host", kCustomCallUsingThreadPool);

TEST_P(HostOffloadingRuntimeExecutableTest, FfiWithThreadpool) {
  std::string hlo = R"(
    HloModule module

    ENTRY custom_call {
      ROOT custom-call = s32[] custom-call(),
        custom_call_target="CustomCallUsingThreadPool",
        api_version=API_VERSION_TYPED_FFI
    }
  )";

  HostOffloadingExecutableProto::ExecutableType
      host_offloading_executable_type = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(
      auto computation,
      CompileFromString(hlo, host_offloading_executable_type));

  Shape shape = ShapeUtil::MakeShape(xla::PrimitiveType::S32, {});
  auto result_literal = LiteralUtil::CreateR0<int32_t>(0);

  ShapeTree<HostOffloadingBuffer> result(
      shape, HostOffloadingBuffer(result_literal.data<int32_t>()));

  auto execute_event = computation->Execute({}, result, EmptyExecuteOptions());
  tsl::BlockUntilReady(execute_event);
  EXPECT_FALSE(execute_event.IsError());
  EXPECT_THAT(result_literal.data<int32_t>(),
              ElementsAreArray({kDummyFFIResult}));
}

TEST_P(HostOffloadingRuntimeExecutableTest, Int4) {
  constexpr absl::string_view hlo = R"(
    HloModule jit_f, entry_computation_layout={(s4[4]{0}, s4[4]{0})->s4[4]{0}}

    ENTRY %main.4 (Arg_0.1: s4[4], Arg_1.2: s4[4]) -> s4[4] {
      %Arg_0.1 = s4[4]{0} parameter(0)
      %Arg_1.2 = s4[4]{0} parameter(1)
      ROOT %add.3 = s4[4]{0} add(%Arg_0.1, %Arg_1.2)
    }
  )";
  HostOffloadingExecutableProto::ExecutableType
      host_offloading_executable_type = GetParam();

  if (host_offloading_executable_type ==
      HostOffloadingExecutableProto::EXECUTABLE_TYPE_PJRT) {
    // TODO(basioli): Problem is probably in host_offloading_pjrt_executable fix
    // it and enable this test.
    GTEST_SKIP() << "Int4 is not supported in PJRT executable";
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto computation,
      CompileFromString(hlo, host_offloading_executable_type));

  Shape shape = ShapeUtil::MakeShape(xla::PrimitiveType::S4, {4});
  shape.mutable_layout()->set_element_size_in_bits(4);

  auto input0_literal = LiteralUtil::CreateR1<uint8_t>(
      {(1 << 4) | 1, (1 << 4) | 1});  // {1, 1, 1, 1} for int4
  auto input1_literal = LiteralUtil::CreateR1<uint8_t>(
      {(1 << 4) | 1, (1 << 4) | 1});  // {1, 1, 1, 1} for int4
  auto result_literal = LiteralUtil::CreateR1<uint8_t>({0, 0});

  std::vector<ShapeTree<HostOffloadingBuffer>> parameters = {
      ShapeTree<HostOffloadingBuffer>(
          shape, HostOffloadingBuffer(input0_literal.untyped_data(),
                                      input0_literal.size_bytes())),
      ShapeTree<HostOffloadingBuffer>(
          shape, HostOffloadingBuffer(input1_literal.untyped_data(),
                                      input1_literal.size_bytes())),
  };

  ShapeTree<HostOffloadingBuffer> result(
      shape, HostOffloadingBuffer(result_literal.untyped_data(),
                                  result_literal.size_bytes()));

  auto execute_event =
      computation->Execute(parameters, result, EmptyExecuteOptions());
  tsl::BlockUntilReady(execute_event);
  EXPECT_FALSE(execute_event.IsError());
  // {2, 2, 2, 2} for int4
  EXPECT_THAT(result_literal.data<uint8_t>(),
              ElementsAreArray({(2 << 4) | 2, (2 << 4) | 2}));
}

INSTANTIATE_TEST_SUITE_P(
    HostOffloadingRuntimeExecutableParameters,
    HostOffloadingRuntimeExecutableTest,
    ::testing::Values(HostOffloadingExecutableProto::EXECUTABLE_TYPE_NANORT,
                      HostOffloadingExecutableProto::EXECUTABLE_TYPE_PJRT),
    [](const testing::TestParamInfo<
        HostOffloadingExecutableProto::ExecutableType>& info) {
      return HostOffloadingExecutableProto::ExecutableType_Name(info.param);
    });

TEST(HostOffloadingNanortTest, DeviceAssignment) {
  std::string str = R"(
    HloModule add

    ENTRY %main {
      %p0 = f32[4] parameter(0)
      ROOT %add = f32[4] add(%p0, %p0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(
      auto computation,
      CompileFromString(str,
                        HostOffloadingExecutableProto::EXECUTABLE_TYPE_NANORT));

  auto host_offloading_nanort_executable =
      absl::down_cast<HostOffloadingNanoRtExecutable*>(computation.get());
  ASSERT_NE(host_offloading_nanort_executable, nullptr);
  ASSERT_NE(host_offloading_nanort_executable->device_assignment(), nullptr);

  // NOTE: Default device assignment has 1 replica and 1 computation.s
  ASSERT_EQ(
      host_offloading_nanort_executable->device_assignment()->replica_count(),
      1);
  ASSERT_EQ(host_offloading_nanort_executable->device_assignment()
                ->computation_count(),
            1);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

void BM_HostOffloadingExecutableAddScalars(
    benchmark::State& state,
    HostOffloadingExecutableProto::ExecutableType executable_type) {
  const std::string hlo = R"(
        HloModule add

        ENTRY e {
          p0 = f32[] parameter(0)
          p1 = f32[] parameter(1)
          ROOT add = f32[] add(p0, p1)
          }
          )";

  TF_ASSERT_OK_AND_ASSIGN(auto computation,
                          CompileFromString(hlo, executable_type));

  auto input_0 = LiteralUtil::CreateR0<float>(3);
  auto input_1 = LiteralUtil::CreateR0<float>(2);
  auto result_literal = LiteralUtil::CreateR0<float>(0);

  Shape shape = ShapeUtil::MakeShape(xla::PrimitiveType::F32, {});

  std::vector<ShapeTree<HostOffloadingBuffer>> parameters = {
      ShapeTree<HostOffloadingBuffer>(
          shape, HostOffloadingBuffer(input_0.data<float>())),
      ShapeTree<HostOffloadingBuffer>(
          shape, HostOffloadingBuffer(input_1.data<float>()))};

  ShapeTree<HostOffloadingBuffer> result(
      shape, HostOffloadingBuffer(result_literal.data<float>()));

  for (auto _ : state) {
    auto execute_event =
        computation->Execute(parameters, result, EmptyExecuteOptions());
    tsl::BlockUntilReady(execute_event);
    EXPECT_FALSE(execute_event.IsError());
  }
}

BENCHMARK_CAPTURE(BM_HostOffloadingExecutableAddScalars, nanort,
                  HostOffloadingExecutableProto::EXECUTABLE_TYPE_NANORT);
BENCHMARK_CAPTURE(BM_HostOffloadingExecutableAddScalars, pjrt,
                  HostOffloadingExecutableProto::EXECUTABLE_TYPE_PJRT);

}  // namespace
}  // namespace xla
