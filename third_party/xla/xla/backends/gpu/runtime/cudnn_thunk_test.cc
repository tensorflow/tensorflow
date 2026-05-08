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

#include "xla/backends/gpu/runtime/cudnn_thunk.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend.h"  // IWYU pragma: keep - cudnn frontend headers are not hermetic
#include "third_party/cudnn_frontend/include/cudnn_frontend/graph_interface.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_utils.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_dnn.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using tsl::proto_testing::EqualsProto;

TEST(CuDnnThunkTest, TestSerializationDeserialization) {
  CudnnThunkProto cudnn_thunk_proto;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        fingerprint: "fingerprint"
        args {
          slice { offset: 123 size: 456 }
          shape { element_type: U8 }
        }
        args {
          slice { offset: 789 size: 1011 }
          shape { element_type: U8 }
        }
        output_args: false
        output_args: true
        sdpa_dropout_seed: 123456789
      )pb",
      &cudnn_thunk_proto));

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";

  ThunkProto thunk_proto;
  *thunk_proto.mutable_thunk_info() = thunk_info.ToProto();
  *thunk_proto.mutable_cudnn_thunk() = cudnn_thunk_proto;

  BufferAllocation alloc0(/*index=*/0, /*size=*/2048, /*color=*/0);
  std::array buffer_allocations = {alloc0};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CuDnnThunk> thunk,
      CuDnnThunk::FromProto(thunk_info, cudnn_thunk_proto, buffer_allocations));

  EXPECT_THAT(thunk->ToProto(),
              absl_testing::IsOkAndHolds(EqualsProto(thunk_proto)));
}

//===----------------------------------------------------------------------===//
// Command-buffer Record() tests for CuDnnThunk.
//
// Covers both Record() branches:
//   * explicit path  — DnnGraph supports CreateDnnGraphCommand /
//                      UpdateDnnGraphCommand; exercised with a real
//                      se::gpu::CudnnGraph built with
//                      require_command_buffer=true.
//   * implicit path  — DnnGraph falls back to RecordTracedCommand; exercised
//                      with a test-only FakeDnnGraph that forces
//                      SupportsExplicitCommandBufferConstruction() -> false and
//                      whose Execute() emits a verifiable device op on the
//                      trace stream.
//
// Each path is tested in both RecordCreate and RecordUpdate modes.
//===----------------------------------------------------------------------===//

absl::StatusOr<se::StreamExecutor*> GpuExecutor() {
  ASSIGN_OR_RETURN(std::string canonical_name,
                   PlatformUtil::CanonicalPlatformName("gpu"));
  std::string name = absl::AsciiStrToUpper(canonical_name);
  ASSIGN_OR_RETURN(auto* platform, se::PlatformManager::PlatformWithName(name));
  return platform->ExecutorForDevice(0);
}

bool SupportsCudaGraphTracing(const se::StreamExecutor* executor) {
  const auto& desc = executor->GetDeviceDescription();
  const auto* cuda_cc = desc.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc == nullptr) {
    return false;
  }
  return std::min(desc.driver_version(), desc.compile_time_toolkit_version()) >=
         stream_executor::SemanticVersion(12, 3, 0);
}

// DnnGraph that forces Record() into the traced (implicit) path.
//
// SupportsExplicitCommandBufferConstruction() returns false so CuDnnThunk falls
// through to RecordTracedCommand, which traces Execute() onto the trace stream.
// Execute() memsets a 32-bit sentinel across the last operand (the output
// buffer), giving the test a verifiable observable side effect without needing
// a real cuDNN arithmetic kernel.
class FakeDnnGraph : public se::dnn::DnnGraph {
 public:
  explicit FakeDnnGraph(uint32_t sentinel) : sentinel_(sentinel) {}

  absl::Status Prepare(se::dnn::DnnSupport&,
                       const se::EngineOptions&) override {
    return absl::OkStatus();
  }
  absl::Status Build(se::dnn::DnnSupport&,
                     std::optional<int64_t> plan_id) override {
    return absl::OkStatus();
  }
  absl::Status Execute(se::Stream& stream,
                       absl::Span<se::DeviceAddressBase> operands,
                       int64_t local_device_ordinal) const override {
    if (operands.empty()) {
      return absl::InternalError("FakeDnnGraph::Execute received no operands");
    }
    se::DeviceAddressBase& output = operands.back();
    return stream.Memset32(&output, sentinel_, output.size());
  }
  void InitDropoutState(int64_t local_device_count, int64_t seed,
                        int64_t increment) override {}
  absl::StatusOr<bool> SupportsExplicitCommandBufferConstruction()
      const override {
    return false;
  }
  absl::Status PopulateOrUpdateRawCommandBuffer(
      se::Stream&, absl::Span<se::DeviceAddressBase>, RawCommandBufferHandle,
      bool do_update) override {
    return absl::UnimplementedError(
        "FakeDnnGraph::PopulateOrUpdateRawCommandBuffer");
  }

 private:
  uint32_t sentinel_;
};

// Fixture shared by all Record() tests.
//
// Matmul layout mirrors CommandBufferThunkTest.CuDnnCmd
// (cuda_command_buffer_thunk_test.cc): A(1×32×32) INT8 * itself → D(1×32×32)
// INT32. Input is zero-filled, so the explicit-path real-matmul output is also
// zero. The implicit-path FakeDnnGraph ignores the math and memsets the output
// to a 32-bit sentinel instead.
class CuDnnThunkCmdBufTest : public ::testing::Test {
 protected:
  static constexpr int kDimSize = 32;
  static constexpr int kTotalElements = kDimSize * kDimSize;
  static constexpr uint32_t kInitialOutput = 0xdeadbeefu;
  static constexpr uint32_t kFakeSentinel = 0x12345678u;

  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(executor_, GpuExecutor());
    if (!SupportsCudaGraphTracing(executor_)) {
      GTEST_SKIP() << "CUDA graph tracing is not supported";
    }
    if (executor_->AsDnn() == nullptr) {
      GTEST_SKIP() << "DNN support is not available on this platform";
    }

    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
    TF_ASSERT_OK_AND_ASSIGN(trace_stream_, executor_->CreateStream());

    input_buf_ = executor_->AllocateArray<int8_t>(kTotalElements);
    output_buf_ = executor_->AllocateArray<int32_t>(kTotalElements);
    TF_ASSERT_OK(stream_->MemZero(&input_buf_, input_buf_.size()));
    TF_ASSERT_OK(
        stream_->Memset32(&output_buf_, kInitialOutput, output_buf_.size()));
    TF_ASSERT_OK(stream_->BlockHostUntilDone());

    // ShapedSlice args: {input, input, output} — matmul(A, A) → D.
    Shape int8_shape = ShapeUtil::MakeShape(S8, {kTotalElements});
    Shape int32_shape = ShapeUtil::MakeShape(S32, {kTotalElements});
    args_.push_back({BufferAllocation::Slice(&alloc_input_, 0, kTotalElements),
                     int8_shape});
    args_.push_back({BufferAllocation::Slice(&alloc_input_, 0, kTotalElements),
                     int8_shape});
    args_.push_back({BufferAllocation::Slice(&alloc_output_, 0,
                                             kTotalElements * sizeof(int32_t)),
                     int32_shape});

    run_options_.mutable_run_options()->set_stream(stream_.get());
    TF_ASSERT_OK_AND_ASSIGN(
        CollectiveParams cp,
        CollectiveParams::Create(run_options_, /*async_streams=*/{},
                                 LocalDeviceId(executor_->device_ordinal())));
    collective_params_.emplace(std::move(cp));

    allocator_ =
        std::make_unique<se::StreamExecutorAddressAllocator>(executor_);
    allocations_ = std::make_unique<BufferAllocations>(
        std::vector<se::DeviceAddressBase>{input_buf_, output_buf_}, 0,
        allocator_.get());

    params_.emplace(Thunk::ExecuteParams::Create(
        run_options_, *allocations_, stream_.get(), trace_stream_.get(),
        &*collective_params_, /*collective_cliques=*/nullptr,
        /*collective_memory=*/nullptr));
  }

  // Builds a real cuDNN matmul graph and wraps it in a CuDnnThunk. Skips the
  // enclosing test if cuDNN is too old or the GPU is below Ampere.
  void BuildRealGraphThunk() {
    se::dnn::DnnSupport& dnn_support = *executor_->AsDnn();
    if (dnn_support.GetVersion().value_or(se::dnn::VersionInfo{0, 0, 0}) <
        se::dnn::VersionInfo(9, 7, 0)) {
      GTEST_SKIP() << "Requires cuDNN 9.7.0 or later.";
    }
    if (!executor_->GetDeviceDescription()
             .cuda_compute_capability()
             .IsAtLeastAmpere()) {
      GTEST_SKIP() << "Requires at least an Ampere GPU.";
    }

    se::gpu::CudnnGraph graph([]() {
      cudnn_frontend::graph::Graph g;
      g.set_compute_data_type(cudnn_frontend::DataType_t::INT32);
      auto lhs = g.tensor(cudnn_frontend::graph::Tensor_attributes()
                              .set_dim({1, kDimSize, kDimSize})
                              .set_stride({kDimSize * kDimSize, kDimSize, 1})
                              .set_data_type(cudnn_frontend::DataType_t::INT8)
                              .set_uid(1));
      auto rhs = g.tensor_like(lhs);
      rhs->set_uid(2);
      g.matmul(lhs, rhs, cudnn_frontend::graph::Matmul_attributes())
          ->set_output(true)
          .set_data_type(cudnn_frontend::DataType_t::INT32)
          .set_uid(3);
      return g;
    }());
    TF_ASSERT_OK(graph.Prepare(
        dnn_support, se::EngineOptions{/*require_determinism=*/false,
                                       /*allow_tf32=*/true,
                                       /*require_command_buffer=*/true}));
    TF_ASSERT_OK(graph.Build(dnn_support, /*plan_id=*/std::nullopt));
    ASSERT_THAT(graph.SupportsExplicitCommandBufferConstruction(),
                absl_testing::IsOkAndHolds(true));

    // The matmul graph we build here never needs workspace for this shape; the
    // existing cuda_command_buffer_thunk_test handles workspace > 0, but here
    // we skip it to keep the fixture small. A workspace would just mean an
    // additional BufferAllocation + operand.
    ASSERT_EQ(graph.Graph().get_workspace_size(), 0);

    std::vector<bool> output_args(args_.size(), false);
    output_args.back() = true;
    thunk_ = std::make_unique<CuDnnThunk>(
        /*fingerprint=*/"", Thunk::ThunkInfo(), args_, std::move(output_args));
    se::dnn::LazyDnnGraph prebuilt(
        std::make_unique<se::gpu::CudnnGraph>(std::move(graph)));
    thunk_->graph()->swap(prebuilt);

    InitializeThunk();
  }

  // Builds a CuDnnThunk backed by the FakeDnnGraph (forces the traced path).
  void BuildFakeGraphThunk() {
    std::vector<bool> output_args(args_.size(), false);
    output_args.back() = true;
    thunk_ = std::make_unique<CuDnnThunk>(
        /*fingerprint=*/"", Thunk::ThunkInfo(), args_, std::move(output_args));
    se::dnn::LazyDnnGraph prebuilt(
        std::make_unique<FakeDnnGraph>(kFakeSentinel));
    thunk_->graph()->swap(prebuilt);

    InitializeThunk();
  }

  void InitializeThunk() {
    Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
    TF_ASSERT_OK(thunk_->Initialize({executor_, source, allocations_.get(),
                                     stream_.get(), trace_stream_.get()}));
  }

  std::vector<int32_t> ReadOutput(const se::DeviceAddress<int32_t>& buf) {
    std::vector<int32_t> dst(kTotalElements, 0);
    TF_CHECK_OK(
        stream_->Memcpy(dst.data(), buf, kTotalElements * sizeof(int32_t)));
    TF_CHECK_OK(stream_->BlockHostUntilDone());
    return dst;
  }

  se::StreamExecutor* executor_ = nullptr;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<se::Stream> trace_stream_;
  se::DeviceAddress<int8_t> input_buf_;
  se::DeviceAddress<int32_t> output_buf_;

  // alloc_*_ must outlive thunk_ because the ShapedSlices in args_ hold raw
  // pointers into them. See the analogous comment in
  // CublasLtMatmulThunkCmdBufTest.
  BufferAllocation alloc_input_{/*index=*/0, kTotalElements, /*color=*/0};
  BufferAllocation alloc_output_{/*index=*/1, kTotalElements * sizeof(int32_t),
                                 /*color=*/0};

  std::vector<ShapedSlice> args_;
  std::unique_ptr<CuDnnThunk> thunk_;

  // allocator_ must outlive allocations_.
  std::unique_ptr<se::StreamExecutorAddressAllocator> allocator_;
  std::unique_ptr<BufferAllocations> allocations_;
  ServiceExecutableRunOptions run_options_;
  std::optional<CollectiveParams> collective_params_;
  // params_ holds references to allocations_, stream_, trace_stream_, and
  // collective_params_, all of which must outlive it.
  std::optional<Thunk::ExecuteParams> params_;
};

TEST_F(CuDnnThunkCmdBufTest, RecordCreateExplicit) {
  BuildRealGraphThunk();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor_->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* cmd,
      thunk_->Record(*params_, record_params,
                     Command::RecordCreate{/*dependencies=*/{}},
                     command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream_.get()));

  // Input is all zeros → matmul(A, A) is all zeros.
  EXPECT_EQ(ReadOutput(output_buf_), std::vector<int32_t>(kTotalElements, 0));
}

TEST_F(CuDnnThunkCmdBufTest, RecordUpdateExplicit) {
  BuildRealGraphThunk();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  // Create.
  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor_->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* cmd,
      thunk_->Record(*params_, record_params,
                     Command::RecordCreate{/*dependencies=*/{}},
                     command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream_.get()));
  EXPECT_EQ(ReadOutput(output_buf_), std::vector<int32_t>(kTotalElements, 0));

  // Rebind the output to a freshly-allocated buffer and update. Using a new
  // address skips the "same-operands" cache-hit path and exercises
  // UpdateDnnGraphCommand with changed arguments. Pre-fill the new buffer with
  // a non-zero sentinel so the post-submit zero fill is observable.
  se::DeviceAddress<int32_t> output1 =
      executor_->AllocateArray<int32_t>(kTotalElements);
  TF_ASSERT_OK(stream_->Memset32(&output1, kInitialOutput, output1.size()));
  BufferAllocations updated_allocations(
      std::vector<se::DeviceAddressBase>{input_buf_, output1}, 0,
      allocator_.get());
  Thunk::ExecuteParams updated_params = Thunk::ExecuteParams::Create(
      run_options_, updated_allocations, stream_.get(), trace_stream_.get(),
      &*collective_params_, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr);

  TF_ASSERT_OK(command_buffer->Update());
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk_->Record(updated_params, record_params, Command::RecordUpdate{cmd},
                     command_buffer.get()));
  // UpdateDnnGraphCommand updates the same command node in place regardless of
  // operand addresses, so the returned pointer is the original cmd.
  EXPECT_EQ(updated_cmd, cmd);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream_.get()));
  EXPECT_EQ(ReadOutput(output1), std::vector<int32_t>(kTotalElements, 0));
}

TEST_F(CuDnnThunkCmdBufTest, RecordCreateImplicit) {
  BuildFakeGraphThunk();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor_->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* cmd,
      thunk_->Record(*params_, record_params,
                     Command::RecordCreate{/*dependencies=*/{}},
                     command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream_.get()));

  EXPECT_EQ(ReadOutput(output_buf_),
            std::vector<int32_t>(kTotalElements,
                                 static_cast<int32_t>(kFakeSentinel)));
}

TEST_F(CuDnnThunkCmdBufTest, RecordUpdateImplicit) {
  BuildFakeGraphThunk();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  // Create.
  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor_->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* cmd,
      thunk_->Record(*params_, record_params,
                     Command::RecordCreate{/*dependencies=*/{}},
                     command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream_.get()));
  EXPECT_EQ(ReadOutput(output_buf_),
            std::vector<int32_t>(kTotalElements,
                                 static_cast<int32_t>(kFakeSentinel)));

  // Rebind the output to a freshly-allocated buffer and update. Using a new
  // address forces a cache miss in the traced-command-buffer cache (keyed on
  // operand addresses), so the underlying nested command buffer is re-traced.
  // The outer command node returned from Record is not guaranteed to be
  // pointer-identical to the original across backends — only assert non-null
  // and that the side effect re-occurs on the new buffer.
  se::DeviceAddress<int32_t> output1 =
      executor_->AllocateArray<int32_t>(kTotalElements);
  TF_ASSERT_OK(stream_->Memset32(&output1, kInitialOutput, output1.size()));
  BufferAllocations updated_allocations(
      std::vector<se::DeviceAddressBase>{input_buf_, output1}, 0,
      allocator_.get());
  Thunk::ExecuteParams updated_params = Thunk::ExecuteParams::Create(
      run_options_, updated_allocations, stream_.get(), trace_stream_.get(),
      &*collective_params_, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr);

  TF_ASSERT_OK(command_buffer->Update());
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk_->Record(updated_params, record_params, Command::RecordUpdate{cmd},
                     command_buffer.get()));
  ASSERT_NE(updated_cmd, nullptr);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream_.get()));
  EXPECT_EQ(ReadOutput(output1),
            std::vector<int32_t>(kTotalElements,
                                 static_cast<int32_t>(kFakeSentinel)));
}

}  // namespace
}  // namespace xla::gpu
