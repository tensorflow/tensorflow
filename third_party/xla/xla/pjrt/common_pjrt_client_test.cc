/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/common_pjrt_client.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"  // IWYU pragma: keep
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/primitive_util.h"
#include "xla/runtime/device_id.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/scoped_memory_debug_annotation.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

TEST(TransferIncarnationsTest, EmptySnapshotReturnsNullopt) {
  ASSERT_OK_AND_ASSIGN(auto transfer_incarnations,
                       TransferIncarnations(/*src_process_index=*/1,
                                            /*dst_process_index=*/2, {}));
  EXPECT_FALSE(transfer_incarnations.has_value());
}

TEST(TransferIncarnationsTest, SnapshotKeepsSourceThenDestination) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(2), IncarnationId(20)},
      {TaskId(1), IncarnationId(10)},
  };

  ASSERT_OK_AND_ASSIGN(
      auto forward,
      TransferIncarnations(/*src_process_index=*/2, /*dst_process_index=*/1,
                           incarnations));
  ASSERT_TRUE(forward.has_value());
  EXPECT_THAT(*forward, ElementsAre(IncarnationId(20), IncarnationId(10)));

  ASSERT_OK_AND_ASSIGN(
      auto swapped,
      TransferIncarnations(/*src_process_index=*/1, /*dst_process_index=*/2,
                           incarnations));
  ASSERT_TRUE(swapped.has_value());
  EXPECT_THAT(*swapped, ElementsAre(IncarnationId(10), IncarnationId(20)));
}

TEST(TransferIncarnationsTest, SameTaskKeepsBothIncarnations) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(3), IncarnationId(7)},
  };

  ASSERT_OK_AND_ASSIGN(
      auto transfer_incarnations,
      TransferIncarnations(/*src_process_index=*/3, /*dst_process_index=*/3,
                           incarnations));
  ASSERT_TRUE(transfer_incarnations.has_value());
  EXPECT_THAT(*transfer_incarnations,
              ElementsAre(IncarnationId(7), IncarnationId(7)));
}

TEST(TransferIncarnationsTest, MissingTaskFails) {
  absl::flat_hash_map<TaskId, IncarnationId> incarnations = {
      {TaskId(1), IncarnationId(10)},
  };

  EXPECT_THAT(TransferIncarnations(/*src_process_index=*/1,
                                   /*dst_process_index=*/2, incarnations),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

template <typename T>
Literal MakePatternLiteral(const Shape& shape, absl::Span<const T> palette) {
  Literal literal(shape);
  absl::Span<T> data = literal.data<T>();
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = palette[i % palette.size()];
  }
  return literal;
}

std::vector<uint8_t> PackLiteralBytes(const Literal& literal) {
  const PrimitiveType element_type = literal.shape().element_type();
  const int bit_width = primitive_util::BitWidth(element_type);
  const size_t num_elements =
      static_cast<size_t>(ShapeUtil::ElementsIn(literal.shape()));
  const size_t packed_bytes = CeilOfRatio<size_t>(num_elements, 8 / bit_width);
  std::vector<uint8_t> packed(packed_bytes, 0);
  absl::Span<const char> src(static_cast<const char*>(literal.untyped_data()),
                             num_elements);
  absl::Span<char> dst(reinterpret_cast<char*>(packed.data()), packed.size());
  primitive_util::PackIntN(element_type, src, dst);
  return packed;
}

TEST(CommonPjRtClientDelinearizeTest, SubBytePackedMatchingMinorToMajor) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  auto* common_client = absl::down_cast<CommonPjRtClient*>(client.get());
  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());

  const std::vector<s4> s4_palette = {
      s4(-8), s4(-7), s4(-6), s4(-5), s4(-4), s4(-3), s4(-2), s4(-1),
      s4(0),  s4(1),  s4(2),  s4(3),  s4(4),  s4(5),  s4(6),  s4(7)};
  const std::vector<u4> u4_palette = {
      u4(0), u4(1), u4(2),  u4(3),  u4(4),  u4(5),  u4(6),  u4(7),
      u4(8), u4(9), u4(10), u4(11), u4(12), u4(13), u4(14), u4(15)};
  const std::vector<std::vector<int64_t>> dims_cases = {
      {}, {0}, {1}, {3, 5}, {4, 4}};

  for (const std::vector<int64_t>& dims : dims_cases) {
    for (int64_t mem_space :
         {Layout::kDefaultMemorySpace, Layout::kHostMemorySpace}) {
      Shape s4_host_shape = ShapeUtil::MakeShape(S4, dims);
      Literal expected_s4 = MakePatternLiteral<s4>(s4_host_shape, s4_palette);
      std::vector<uint8_t> packed_s4 = PackLiteralBytes(expected_s4);
      Shape s4_device_shape = s4_host_shape;
      s4_device_shape.mutable_layout()->set_element_size_in_bits(4);
      s4_device_shape.mutable_layout()->set_memory_space(mem_space);
      Literal actual_s4(s4_host_shape);
      EXPECT_THAT(common_client->Delinearize(packed_s4, s4_device_shape,
                                             &actual_s4, memory_space),
                  IsOk());
      EXPECT_EQ(actual_s4, expected_s4);

      if (ShapeUtil::ElementsIn(s4_host_shape) % 2 == 1) {
        // Non-zero high-nibble padding in the final packed byte must be masked.
        packed_s4.back() |= 0xF0;
        Literal actual_padded_s4(s4_host_shape);
        EXPECT_THAT(common_client->Delinearize(packed_s4, s4_device_shape,
                                               &actual_padded_s4, memory_space),
                    IsOk());
        EXPECT_EQ(actual_padded_s4, expected_s4);
      }

      Shape u4_host_shape = ShapeUtil::MakeShape(U4, dims);
      Literal expected_u4 = MakePatternLiteral<u4>(u4_host_shape, u4_palette);
      std::vector<uint8_t> packed_u4 = PackLiteralBytes(expected_u4);
      Shape u4_device_shape = u4_host_shape;
      u4_device_shape.mutable_layout()->set_element_size_in_bits(4);
      u4_device_shape.mutable_layout()->set_memory_space(mem_space);
      Literal actual_u4(u4_host_shape);
      EXPECT_THAT(common_client->Delinearize(packed_u4, u4_device_shape,
                                             &actual_u4, memory_space),
                  IsOk());
      EXPECT_EQ(actual_u4, expected_u4);
    }
  }

  const std::vector<s2> s2_palette = {s2(-2), s2(-1), s2(0), s2(1)};
  Shape s2_host_shape = ShapeUtil::MakeShape(S2, {3, 3});
  Literal expected_s2 = MakePatternLiteral<s2>(s2_host_shape, s2_palette);
  std::vector<uint8_t> packed_s2 = PackLiteralBytes(expected_s2);
  Shape s2_device_shape = s2_host_shape;
  s2_device_shape.mutable_layout()->set_element_size_in_bits(2);
  Literal actual_s2(s2_host_shape);
  EXPECT_THAT(common_client->Delinearize(packed_s2, s2_device_shape, &actual_s2,
                                         memory_space),
              IsOk());
  EXPECT_EQ(actual_s2, expected_s2);

  const std::vector<u2> u2_palette = {u2(0), u2(1), u2(2), u2(3)};
  Shape u2_host_shape = ShapeUtil::MakeShape(U2, {3, 3});
  Literal expected_u2 = MakePatternLiteral<u2>(u2_host_shape, u2_palette);
  std::vector<uint8_t> packed_u2 = PackLiteralBytes(expected_u2);
  Shape u2_device_shape = u2_host_shape;
  u2_device_shape.mutable_layout()->set_element_size_in_bits(2);
  Literal actual_u2(u2_host_shape);
  EXPECT_THAT(common_client->Delinearize(packed_u2, u2_device_shape, &actual_u2,
                                         memory_space),
              IsOk());
  EXPECT_EQ(actual_u2, expected_u2);
}

TEST(CommonPjRtClientDelinearizeTest,
     SubBytePackedNonDefaultMatchingMinorToMajor) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  auto* common_client = absl::down_cast<CommonPjRtClient*>(client.get());
  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());

  Shape col_major_shape = ShapeUtil::MakeShapeWithDenseLayout(
      S4, {3, 5}, /*minor_to_major=*/{0, 1});
  const std::vector<s4> palette = {s4(-8), s4(-5), s4(-1), s4(0),
                                   s4(2),  s4(4),  s4(7)};
  Literal expected = MakePatternLiteral<s4>(col_major_shape, palette);
  std::vector<uint8_t> packed = PackLiteralBytes(expected);

  Shape device_shape = col_major_shape;
  device_shape.mutable_layout()->set_element_size_in_bits(4);
  device_shape.mutable_layout()->set_memory_space(Layout::kHostMemorySpace);

  Literal actual(col_major_shape);
  EXPECT_THAT(
      common_client->Delinearize(packed, device_shape, &actual, memory_space),
      IsOk());
  EXPECT_EQ(actual, expected);
}

TEST(CommonPjRtClientDelinearizeTest, SubBytePackedTransposedMinorToMajor) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  auto* common_client = absl::down_cast<CommonPjRtClient*>(client.get());
  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());

  Literal row_major = LiteralUtil::CreateR2<s4>(
      {{s4(-8), s4(-3), s4(1)}, {s4(2), s4(5), s4(7)}});
  std::vector<uint8_t> packed = PackLiteralBytes(row_major);

  Shape device_shape = ShapeUtil::MakeShapeWithDenseLayout(
      S4, {2, 3}, /*minor_to_major=*/{1, 0});
  device_shape.mutable_layout()->set_element_size_in_bits(4);

  Literal actual(ShapeUtil::MakeShapeWithDenseLayout(
      S4, {2, 3}, /*minor_to_major=*/{0, 1}));
  EXPECT_THAT(
      common_client->Delinearize(packed, device_shape, &actual, memory_space),
      IsOk());
  // In {0, 1} layout, dimension 0 is minor: (0,0), (1,0), (0,1), (1,1), ...
  EXPECT_THAT(actual.data<s4>(),
              ElementsAre(s4(-8), s4(2), s4(-3), s4(5), s4(1), s4(7)));
}

TEST(CommonPjRtClientDelinearizeTest, HostMemorySpaceMatchingMinorToMajor) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  auto* common_client = absl::down_cast<CommonPjRtClient*>(client.get());
  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());

  Literal expected =
      LiteralUtil::CreateR2<float>({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
  Shape device_shape = expected.shape();
  device_shape.mutable_layout()->set_memory_space(Layout::kHostMemorySpace);

  absl::Span<const uint8_t> src_bytes(
      static_cast<const uint8_t*>(expected.untyped_data()),
      expected.size_bytes());
  Literal actual(expected.shape());
  EXPECT_THAT(common_client->Delinearize(src_bytes, device_shape, &actual,
                                         memory_space),
              IsOk());
  EXPECT_EQ(actual, expected);

  Shape empty_shape = ShapeUtil::MakeShape(F32, {0});
  Shape empty_device_shape = empty_shape;
  empty_device_shape.mutable_layout()->set_memory_space(
      Layout::kHostMemorySpace);
  Literal actual_empty(empty_shape);
  EXPECT_THAT(common_client->Delinearize(absl::Span<const uint8_t>(),
                                         empty_device_shape, &actual_empty,
                                         memory_space),
              IsOk());
  EXPECT_EQ(actual_empty, Literal(empty_shape));
}

TEST(CommonPjRtClientDelinearizeTest,
     SubByteHostMemoryForDeviceManagerRoundTrip) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  PjRtHostMemoryForDeviceManager* manager =
      client->GetPjRtHostMemoryForDeviceManager();
  ASSERT_NE(manager, nullptr);

  const std::vector<s4> palette = {s4(-8), s4(-5), s4(-1), s4(0),
                                   s4(2),  s4(4),  s4(7)};
  Shape host_shape = ShapeUtil::MakeShape(S4, {3, 5});
  Shape device_shape = host_shape;
  device_shape.mutable_layout()->set_element_size_in_bits(4);
  Literal expected = MakePatternLiteral<s4>(host_shape, palette);

  ASSERT_OK_AND_ASSIGN(
      PjRtChunk device_chunk,
      manager->ToDeviceLayout(expected.untyped_data(), expected.size_bytes(),
                              host_shape, device_shape));
  Literal actual(host_shape);
  EXPECT_THAT(manager->ToHostLayout(device_chunk.data(), device_chunk.size(),
                                    device_shape, actual.untyped_data(),
                                    actual.size_bytes(), host_shape),
              IsOk());
  EXPECT_EQ(actual, expected);
}

TEST(CommonPjRtClientDelinearizeTest,
     SubByteBufferFromHostLiteralToLiteralRoundTrip) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  const std::vector<s4> palette = {s4(-8), s4(-5), s4(-1), s4(0),
                                   s4(2),  s4(4),  s4(7)};
  Shape host_shape = ShapeUtil::MakeShape(S4, {3, 5});
  Literal expected = MakePatternLiteral<s4>(host_shape, palette);

  for (PjRtMemorySpace* memory_space : client->memory_spaces()) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtBuffer> buffer,
                         client->BufferFromHostLiteral(expected, memory_space));
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> actual,
                         buffer->ToLiteral().Await());
    EXPECT_EQ(*actual, expected);

    Layout col_major_device_layout = LayoutUtil::MakeLayout({0, 1});
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtBuffer> col_major_buffer,
                         client->BufferFromHostLiteral(
                             expected, memory_space, &col_major_device_layout));
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> col_major_to_default,
                         col_major_buffer->ToLiteral().Await());
    EXPECT_EQ(*col_major_to_default, expected);

    Literal col_major_literal(ShapeUtil::MakeShapeWithDenseLayout(
        S4, {3, 5}, /*minor_to_major=*/{0, 1}));
    ASSERT_OK(col_major_buffer->ToLiteral(&col_major_literal).Await());
    EXPECT_EQ(col_major_literal, expected.Relayout(col_major_device_layout));
  }
}

TEST(CommonPjRtClientExecuteTest,
     MultiOutputBatchingPreservesAliasingMemorySpacesAndAnnotations) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions{}));
  PjRtDevice* device = client->addressable_devices()[0];
  ASSERT_OK_AND_ASSIGN(PjRtMemorySpace * default_space,
                       device->default_memory_space());
  PjRtMemorySpace* pinned_host_space = nullptr;
  for (PjRtMemorySpace* ms : device->memory_spaces()) {
    if (ms->kind() == "pinned_host") {
      pinned_host_space = ms;
      break;
    }
  }
  ASSERT_NE(pinned_host_space, nullptr);

  constexpr absl::string_view kProgram = R"hlo(
    HloModule multi_output_batching, input_output_alias={ {1}: (0, {}, must-alias) }

    ENTRY main {
      p0 = s32[4]{0} parameter(0)
      p1 = s32[2]{0} parameter(1)
      c1 = s32[4]{0} constant({1, 2, 3, 4})
      c10 = s32[2]{0} constant({10, 20})
      out1 = s32[4]{0} add(p0, c1)
      host0 = s32[2]{0:S(5)} copy(p1)
      c10_host = s32[2]{0:S(5)} copy(c10)
      out0 = s32[2]{0:S(5)} add(host0, c10_host)
      out2 = s32[4]{0} multiply(out1, c1)
      out3 = s32[2]{0:S(5)} subtract(out0, c10_host)
      ROOT tuple = (s32[2]{0:S(5)}, s32[4]{0}, s32[4]{0}, s32[2]{0:S(5)})
          tuple(out0, out1, out2, out3)
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnUnverifiedModule(kProgram, {}));
  XlaComputation computation(hlo_module->ToProto());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                       client->CompileAndLoad(computation, CompileOptions{}));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> arg0,
      client->BufferFromHostLiteral(
          LiteralUtil::CreateR1<int32_t>({10, 20, 30, 40}), default_space));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> arg1,
      client->BufferFromHostLiteral(LiteralUtil::CreateR1<int32_t>({5, 6}),
                                    default_space));

  tsl::profiler::ScopedMemoryDebugAnnotation outer_anno(
      "outer_op", /*step_id=*/42, "outer_region", /*data_type=*/7,
      []() { return std::string("outer_shape"); });

  ExecuteOptions execute_options = {};
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<PjRtBuffer>> results,
                       executable->ExecuteSharded({arg0.get(), arg1.get()},
                                                  device, execute_options));

  const tsl::profiler::MemoryDebugAnnotation& restored_anno =
      tsl::profiler::ScopedMemoryDebugAnnotation::CurrentAnnotation();
  EXPECT_EQ(restored_anno.pending_op_name, "outer_op");
  EXPECT_EQ(restored_anno.pending_step_id, 42);
  EXPECT_EQ(restored_anno.pending_region_type, "outer_region");
  EXPECT_EQ(restored_anno.pending_data_type, 7);
  EXPECT_EQ(restored_anno.pending_shape_func(0), "outer_shape");

  ASSERT_EQ(results.size(), 4);
  EXPECT_TRUE(arg0->IsDeleted());
  EXPECT_FALSE(arg1->IsDeleted());
  EXPECT_EQ(results[0]->memory_space(), pinned_host_space);
  EXPECT_EQ(results[1]->memory_space(), default_space);
  EXPECT_EQ(results[2]->memory_space(), default_space);
  EXPECT_EQ(results[3]->memory_space(), pinned_host_space);

  // Deleting individual outputs leaves remaining outputs in both memory spaces
  // intact.
  results[0]->Delete();
  results[1]->Delete();

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> out2,
                       results[2]->ToLiteral().Await());
  EXPECT_THAT(out2->data<int32_t>(), ElementsAre(11, 44, 99, 176));

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> out3,
                       results[3]->ToLiteral().Await());
  EXPECT_THAT(out3->data<int32_t>(), ElementsAre(5, 6));
}

// NOLINTBEGIN(misc-include-cleaner): benchmark/benchmark.h is the public
// umbrella header across internal and OSS Google Benchmark versions.
void BM_DelinearizeInt4(benchmark::State& state) {
  const int64_t num_elements = state.range(0);
  absl::StatusOr<std::unique_ptr<PjRtClient>> client =
      GetXlaPjrtCpuClient(CpuClientOptions{});
  CHECK_OK(client);
  auto* common_client = absl::down_cast<CommonPjRtClient*>(client->get());
  absl::StatusOr<PjRtMemorySpace*> memory_space =
      (*client)->addressable_devices()[0]->default_memory_space();
  CHECK_OK(memory_space);

  Shape host_shape = ShapeUtil::MakeShape(S4, {num_elements});
  Shape device_shape = host_shape;
  device_shape.mutable_layout()->set_element_size_in_bits(4);

  const size_t packed_bytes =
      static_cast<size_t>(CeilOfRatio<int64_t>(num_elements, 2));
  std::vector<uint8_t> packed(packed_bytes, 0x78);
  Literal literal(host_shape);

  for (auto _ : state) {
    CHECK_OK(common_client->Delinearize(packed, device_shape, &literal,
                                        *memory_space));
    benchmark::DoNotOptimize(literal.untyped_data());
    benchmark::ClobberMemory();
  }
  state.SetBytesProcessed(state.iterations() * num_elements);
}

BENCHMARK(BM_DelinearizeInt4)->Arg(4096)->Arg(1 << 20)->Arg(64 << 20);
// NOLINTEND(misc-include-cleaner)

}  // namespace
}  // namespace xla
