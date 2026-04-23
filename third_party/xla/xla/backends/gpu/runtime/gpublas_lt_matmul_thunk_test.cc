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

#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/backends/gpu/transforms/gemm_rewriter.h"
#include "xla/error_spec.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tests/hlo_test_base_legacy.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {
using absl_testing::IsOkAndHolds;
using tsl::proto_testing::EqualsProto;

class GpuBlasLtMatmulThunkTest : public HloTestBaseLegacy {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    auto debug_options = HloTestBaseLegacy::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    debug_options.set_xla_gpu_autotune_level(0);
    return debug_options;
  }
  se::StreamExecutor* default_exec() {
    return backend().default_stream_executor();
  }
  const se::DeviceDescription& device_desc(se::StreamExecutor* exec = nullptr) {
    if (exec == nullptr) {
      exec = default_exec();
    }
    return exec->GetDeviceDescription();
  }
  const se::GpuComputeCapability& gpu_comp(se::StreamExecutor* exec = nullptr) {
    return device_desc(exec).gpu_compute_capability();
  }

  void SetUp() override {
    if (auto* rocm = gpu_comp().rocm_compute_capability();
        rocm != nullptr && !rocm->has_hipblaslt()) {
      GTEST_SKIP() << "No hipblas-lt support on this architecture!";
    }
  }

  void CreateExecuteThunksFromHLO(se::StreamExecutor* executor,
                                  absl::string_view hlo_string);
};

class GpuBlasLtThunkBuilder {
 public:
  GpuBlasLtThunkBuilder(se::StreamExecutor* exec,
                        const se::GpuComputeCapability& gpu_comp)
      : exec_(exec), allocator_(exec), gpu_comp_(gpu_comp) {}

  absl::StatusOr<std::unique_ptr<CublasLtMatmulThunk>> CreateThunk(
      HloInstruction* gemm) {
    TF_ASSIGN_OR_RETURN(const auto gpu_config,
                        gemm->backend_config<GpuBackendConfig>());
    const auto& backend_config = gpu_config.gemm_backend_config();

    TF_ASSIGN_OR_RETURN(
        bool has_vector_bias,
        gpublas_lt::EpilogueAddsVectorBias(backend_config.epilogue()));
    bool has_matrix_bias = backend_config.beta() != 0;
    TF_ASSIGN_OR_RETURN(
        auto epilogue, gpublas_lt::AsBlasLtEpilogue(backend_config.epilogue()));

    std::vector<Shape> buf_shapes;
    for (auto op : gemm->operands()) {
      buf_shapes.push_back(op->shape());
    }
    const auto& output_shape =
        gemm->shape().IsTuple() ? gemm->shape().tuple_shapes(0) : gemm->shape();
    buf_shapes.push_back(output_shape);

    size_t idx = allocs_.size();

    std::vector<ShapedSlice> slices;
    slices.reserve(buf_shapes.size());
    for (const Shape& shape : buf_shapes) {
      int64_t size = ShapeUtil::ByteSizeOf(shape);
      mem_buffers_.emplace_back();
      TF_ASSIGN_OR_RETURN(mem_buffers_.back(),
                          allocator_.Allocate(exec_->device_ordinal(), size));
      allocs_.emplace_back(/*index=*/idx++, size, /*color=*/0);
      slices.push_back(
          {BufferAllocation::Slice{&allocs_.back(), /*offset*/ 0, size},
           shape});
    }
    // we need at least 3 buffers: lhs, rhs and output
    EXPECT_EQ(slices.size(),
              3 + size_t{has_matrix_bias} + size_t{has_vector_bias});
    TF_ASSIGN_OR_RETURN(auto gemm_config, GemmConfig::For(gemm, gpu_comp_));

    std::optional<ShapedSlice> bias;
    if (has_vector_bias) {
      bias = slices[has_matrix_bias ? 3 : 2];
    }

    Thunk::ThunkInfo thunk_info =
        Thunk::ThunkInfo::WithProfileAnnotation(gemm, ThunkId(1));
    std::string canonical_hlo = gemm->ToString(
        HloPrintOptions::Fingerprint().set_print_backend_config(true));

    return std::make_unique<CublasLtMatmulThunk>(
        std::move(thunk_info), std::move(canonical_hlo), std::move(gemm_config),
        epilogue,
        /*algorithm_idx*/ 0, backend_config.autotune_workspace_size(),
        slices[0], slices[1], has_matrix_bias ? slices[2] : slices.back(),
        slices.back(), bias, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt /* workspace */);
  }

  std::unique_ptr<BufferAllocations> buffer_allocations() {
    std::vector<se::DeviceAddressBase> buffers(mem_buffers_.size());
    for (size_t i = 0; i < buffers.size(); i++) {
      buffers[i] = *mem_buffers_[i];
    }
    return std::make_unique<BufferAllocations>(buffers, exec_->device_ordinal(),
                                               &allocator_);
  }

 private:
  se::StreamExecutor* exec_;
  stream_executor::StreamExecutorAddressAllocator allocator_;
  se::GpuComputeCapability gpu_comp_;
  std::deque<BufferAllocation> allocs_;
  std::vector<se::ScopedDeviceAddress<uint8_t>> mem_buffers_;
};

void GpuBlasLtMatmulThunkTest::CreateExecuteThunksFromHLO(
    se::StreamExecutor* executor, absl::string_view hlo_string) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          this->ParseAndReturnVerifiedModule(hlo_string));

  GemmRewriterOptions options;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(GemmRewriter(gpu_comp(executor),
                              /*toolkit_version=*/se::SemanticVersion{12, 4, 0},
                              options),
                 module.get()));
  ASSERT_TRUE(changed);

  GpuBlasLtThunkBuilder builder(executor, gpu_comp(executor));
  std::vector<std::unique_ptr<CublasLtMatmulThunk>> gemm_thunks;

  for (auto* instr : module->entry_computation()->instructions()) {
    if (IsCublasLtMatmul(*instr)) {
      TF_ASSERT_OK_AND_ASSIGN(auto thunk, builder.CreateThunk(instr));
      gemm_thunks.push_back(std::move(thunk));
    }
  }
  auto allocs = builder.buffer_allocations();
  ServiceExecutableRunOptions run_options;

  auto thread_func = [&](se::Stream* stream) -> absl::Status {
    auto thunk_params = Thunk::ExecuteParams::Create(
        run_options, *allocs, stream, stream, nullptr, nullptr, nullptr);

    Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
    for (auto& thunk : gemm_thunks) {
      TF_RETURN_IF_ERROR(
          thunk->Initialize({executor, source, allocs.get(), stream, stream}));
      TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(thunk_params));
    }
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    return absl::OkStatus();
  };

  // Running BlasLt thunks across multiple streams with shared matmul plan
  int num_streams = 10;
  struct StreamInfo {
    std::unique_ptr<se::Stream> stream;
    absl::Status result;
  };
  std::vector<StreamInfo> threads(num_streams);
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_streams",
                                 num_streams);
    // use two different loops to make sure all threads start at the same time
    for (auto& [s, _] : threads) {
      TF_ASSERT_OK_AND_ASSIGN(s, executor->CreateStream());
    }
    // some compilers complain about lambda capture of structured bindings
    for (auto& info : threads) {
      pool.Schedule([&] { info.result = thread_func(info.stream.get()); });
    }
  }
  for (const auto& [_, res] : threads) {
    TF_ASSERT_OK(res);
  }
}

const absl::string_view hlo_single_plan = R"(
HloModule SharedMatmulPlan

ENTRY test {
  x1 = f32[101,407] parameter(0)
  x2 = f32[101,407] parameter(1)
  x3 = f32[101,407] parameter(2)
  y = f32[407,400] parameter(3)
  z = f32[407,400] parameter(4)
  w = f32[407,400] parameter(5)
  dot_a = f32[101,400] dot(x1, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_b = f32[101,400] dot(x2, z), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_c = f32[101,400] dot(x3, w), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  mul_ab = f32[101,400] multiply(dot_a, dot_b)
  ROOT abc = f32[101,400] subtract(mul_ab, dot_c)
})";

// same as above but now we have non-default epilogue for one dot operation
const absl::string_view hlo_two_plans =
    R"(
HloModule SharedMatmulPlan

ENTRY test {
  x1 = f32[101,407] parameter(0)
  x2 = f32[101,407] parameter(1)
  x3 = f32[101,407] parameter(2)
  y = f32[407,400] parameter(3)
  z = f32[407,400] parameter(4)
  w = f32[407,400] parameter(5)
  c = f32[] constant(0)
  c_bcast = f32[101,400] broadcast(c), dimensions={}
  dot_a = f32[101,400] dot(x1, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  out_a = f32[101,400] maximum(dot_a, c_bcast)
  dot_b = f32[101,400] dot(x2, z), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot_c = f32[101,400] dot(x3, w), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  mul_ab = f32[101,400] multiply(out_a, dot_b)
  ROOT abc = f32[101,400] subtract(mul_ab, dot_c)
})";

TEST_F(GpuBlasLtMatmulThunkTest, SharedMatmulPlansUnit) {
  auto* exec = default_exec();
  auto* blas_lt = exec->AsBlas()->GetBlasLt();
  EXPECT_NE(blas_lt, nullptr);
  blas_lt->ClearMatmulPlanCache();

  CreateExecuteThunksFromHLO(exec, hlo_single_plan);
  // Assert that only one matmul plan was created
  EXPECT_EQ(blas_lt->GetMatmulPlanCacheSize(), 1);

  CreateExecuteThunksFromHLO(exec, hlo_two_plans);
  // Assert that we have now 2 MatmulPlans (one more created for ReLu epilogue).
  EXPECT_EQ(blas_lt->GetMatmulPlanCacheSize(), 2);
}

// Same as above but instead of creating thunks manually, we use XLA runtime
TEST_F(GpuBlasLtMatmulThunkTest, SharedMatmulPlansFunctional) {
  auto* exec = default_exec();
  auto* blas_lt = exec->AsBlas()->GetBlasLt();
  EXPECT_NE(blas_lt, nullptr);
  blas_lt->ClearMatmulPlanCache();

  EXPECT_TRUE(RunAndCompare(hlo_single_plan, ErrorSpec{1e-3, 1e-3}));
  // Assert that only one MatmulPlan cache entry was created.
  EXPECT_EQ(blas_lt->GetMatmulPlanCacheSize(), 1);

  EXPECT_TRUE(RunAndCompare(hlo_two_plans, ErrorSpec{1e-3, 1e-3}));
  // Assert that we have now 2 MatmulPlans (one more created for ReLu epilogue).
  EXPECT_EQ(blas_lt->GetMatmulPlanCacheSize(), 2);
}

// Mock BlasLt interface to test only the cache function
struct MockBlasLt : public se::gpu::BlasLt {
  absl::Status Init() override { return absl::OkStatus(); }

  absl::StatusOr<MatmulPlanPtr> GetMatmulPlan(const se::gpu::GemmConfig&,
                                              Epilogue) const override {
    return MatmulPlanPtr{};
  }

  absl::StatusOr<MatmulPlanPtr> GetGroupedMatmulPlan(
      se::gpu::GroupedGemmConfig&,
      const std::vector<Epilogue>&) const override {
    return MatmulPlanPtr{};
  }

  ~MockBlasLt() override = default;
};

TEST_F(GpuBlasLtMatmulThunkTest, CacheUnitTest) {
  auto thread_func = [&](MockBlasLt* blas_lt, const std::string& key,
                         int sleep_ms) -> absl::Status {
    auto create_func = [&]() -> absl::StatusOr<se::gpu::BlasLt::MatmulPlanPtr> {
      // We don't care about creation of matmul plans -> emulate it with a sleep
      absl::SleepFor(absl::Milliseconds(sleep_ms));
      return se::gpu::BlasLt::MatmulPlanPtr{};
    };

    return blas_lt->GetOrCreateMatmulPlan(key, create_func).status();
  };  // thread_func

  const int num_blas_lts = 30, num_streams = 30,
            total = num_blas_lts * num_streams, mod = 11;

  std::vector<absl::Status> results(total);
  std::vector<MockBlasLt> blas_lts(num_blas_lts);

  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_streams", total);
    std::random_device rand_dev;
    std::default_random_engine engine(rand_dev());
    std::uniform_int_distribution<int> uniform_sleeps(1, 500);

    for (int j = 0, k = 0; j < num_blas_lts; j++) {
      for (int i = 0; i < num_streams; i++, k++) {
        int sleep_ms = uniform_sleeps(engine), x = i + j + 1;
        // we could have same keys for different executors
        auto key = std::to_string((x * x * x) % mod);
        VLOG(1) << j << "," << i << " :" << key;
        pool.Schedule([&, key, sleep_ms, k, j] {
          results[k] = thread_func(&blas_lts[j], key, sleep_ms);
        });
      }
    }  // for j
  }  // end block
  for (auto& res : results) {
    TF_ASSERT_OK(res);
  }

  // We assert that we have the same number of cache entries for each executor
  // and that this number is <= mod (based on our logic to create keys)
  std::optional<size_t> size;
  for (const auto& blas_lt : blas_lts) {
    if (!size) {
      size = blas_lt.GetMatmulPlanCacheSize();
    } else {
      EXPECT_EQ(*size, blas_lt.GetMatmulPlanCacheSize());
    }
  }
  EXPECT_TRUE(size.has_value() && static_cast<int>(*size <= mod));
}

TEST_F(GpuBlasLtMatmulThunkTest, ThunkProtoSerialization) {
  constexpr absl::string_view kCublasLtMatmulThunkProtoText = R"pb(
    gemm_config {
      lhs_layout {
        order: ORDER_ROW_MAJOR
        num_rows: 101
        num_cols: 407
        batch_size: 1
        leading_dim_stride: 407
        dtype: F32
      }
      rhs_layout {
        order: ORDER_ROW_MAJOR
        num_rows: 407
        num_cols: 400
        batch_size: 1
        leading_dim_stride: 400
        dtype: F32
      }
      c_layout {
        order: ORDER_ROW_MAJOR
        num_rows: 101
        num_cols: 400
        batch_size: 1
        leading_dim_stride: 400
        dtype: F32
      }
      output_layout {
        order: ORDER_ROW_MAJOR
        num_rows: 101
        num_cols: 400
        batch_size: 1
        leading_dim_stride: 400
        dtype: F32
      }
      alpha_real: 1
      algorithm: -1
    }
    epilogue: EPILOGUE_DEFAULT
    canonical_hlo: "(f32[101,400]{1,0}, s8[33554432]{0}) custom-call(f32[101,407]{1,0}, f32[407,400]{1,0}), custom_call_target=\"__cublas$lt$matmul\", backend_config={\"operation_queue_id\":\"0\",\"gemm_backend_config\":{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"],\"algorithm\":\"ALG_UNSET\"},\"epilogue\":\"DEFAULT\",\"lhs_stride\":\"41107\",\"rhs_stride\":\"162800\",\"grad_x\":false,\"grad_y\":false,\"damax_output\":false},\"force_earliest_schedule\":false,\"reification_cost\":[]}"
    a {
      slice { size: 164428 buffer_allocation_index: 3 }
      shape {}
    }
    b {
      slice { size: 651200 buffer_allocation_index: 4 }
      shape {}
    }
    c {
      slice { size: 161600 buffer_allocation_index: 5 }
      shape {}
    }
    d {
      slice { size: 161600 buffer_allocation_index: 5 }
      shape {}
    }
  )pb";

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "test";

  CublasLtMatmulThunkProto proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(kCublasLtMatmulThunkProtoText,
                                                  &proto));

  std::vector<BufferAllocation> allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),  // UNUSED
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),  // UNUSED
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),  // UNUSED
      BufferAllocation(/*index=*/3, /*size=*/164428, /*color=*/0),
      BufferAllocation(/*index=*/4, /*size=*/651200, /*color=*/0),
      BufferAllocation(/*index=*/5, /*size=*/161600, /*color=*/0),
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      CublasLtMatmulThunk::FromProto(thunk_info, proto, allocations));

  ThunkProto reference_thunk_proto;
  *reference_thunk_proto.mutable_thunk_info() = thunk_info.ToProto();
  *reference_thunk_proto.mutable_cublas_lt_matmul_thunk() = proto;
  EXPECT_THAT(thunk->ToProto(),
              IsOkAndHolds(EqualsProto(reference_thunk_proto)));
}

TEST_F(GpuBlasLtMatmulThunkTest, ThunkProtoSerializationGroupedMatmul) {
  constexpr absl::string_view kCublasLtGroupedMatmulThunkProtoText = R"pb(
    grouped_gemm_config {
      m: 101
      n: 400
      k: 407
      batch_count: 1
      group_count: 1
      lhs_leading_dim_stride: 407
      rhs_leading_dim_stride: 400
      c_leading_dim_stride: 400
      output_leading_dim_stride: 400
      trans_a: BLAS_TRANSPOSE
      trans_b: BLAS_TRANSPOSE
      must_swap_operands: True
      alpha_real: 1
      type_a: F32
      type_b: F32
      type_c: F32
      type_d: F32
      stride_ragged_dim: 407
      stride_group_dim: 400
      output_stride_ragged_dim: 400
      ragged_mode: RAGGED_NON_CONTRACTING
    }
    epilogue: EPILOGUE_DEFAULT
    canonical_hlo: "(f32[101,400]{1,0}, s8[33554432]{0}) custom-call(f32[101,407]{1,0}, f32[2, 407,400]{1,0}, s32[2]{0}), custom_call_target=\"__cublas$lt$groupedMatmul backend_config={\"operation_queue_id\":\"0\", \"force_earliest_schedule\":false,\"reification_cost\":[], \"device_type\":\"DEVICE_TYPE_INVALID\", \"grouped_gemm_backend_config\":{\"gemm_backend_config\":{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"epilogue\":\"DEFAULT\",\"grad_x\":false,\"grad_y\":false,\"damax_output\":false},\"ragged_dot_dimension_numbers\":{\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"1\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"lhs_ragged_dimensions\":[\"0\"],\"rhs_group_dimensions\":[\"0\"]}}})"
    a {
      slice { size: 164428 buffer_allocation_index: 3 }
      shape {}
    }
    b {
      slice { size: 1302400 buffer_allocation_index: 4 }
      shape {}
    }
    c {
      slice { size: 161600 buffer_allocation_index: 5 }
      shape {}
    }
    d {
      slice { size: 161600 buffer_allocation_index: 5 }
      shape {}
    }
    group_sizes {
      slice { size: 8 buffer_allocation_index: 6 }
      shape {}
    }
  )pb";

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "test";

  CublasLtMatmulThunkProto proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      kCublasLtGroupedMatmulThunkProtoText, &proto));

  std::vector<BufferAllocation> allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0),  // UNUSED
      BufferAllocation(/*index=*/1, /*size=*/4, /*color=*/0),  // UNUSED
      BufferAllocation(/*index=*/2, /*size=*/4, /*color=*/0),  // UNUSED
      BufferAllocation(/*index=*/3, /*size=*/164428, /*color=*/0),
      BufferAllocation(/*index=*/4, /*size=*/651200, /*color=*/0),
      BufferAllocation(/*index=*/5, /*size=*/161600, /*color=*/0),
      BufferAllocation(/*index=*/6, /*size=*/8, /*color=*/0),
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> thunk,
      CublasLtMatmulThunk::FromProto(thunk_info, proto, allocations));

  ThunkProto reference_thunk_proto;
  *reference_thunk_proto.mutable_thunk_info() = thunk_info.ToProto();
  *reference_thunk_proto.mutable_cublas_lt_matmul_thunk() = proto;
  EXPECT_THAT(thunk->ToProto(),
              IsOkAndHolds(EqualsProto(reference_thunk_proto)));
}

//===----------------------------------------------------------------------===//
// Command buffer tests (Record)
//===----------------------------------------------------------------------===//

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

// Returns true if the GPU supports CUDA graph tracing (requires CUDA 12.3+).
static bool SupportsCudaGraphTracing(const se::StreamExecutor* executor) {
  const auto& desc = executor->GetDeviceDescription();
  const auto* cuda_cc = desc.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc == nullptr) {
    return false;
  }
  return std::min(desc.driver_version(), desc.compile_time_toolkit_version()) >=
         stream_executor::SemanticVersion(12, 3, 0);
}

// Fixture shared by all command-buffer Record tests.
//
// Matrix layout: A(2×4) * B(4×3) → D(2×3), beta=1 accumulated into C(2×3).
// A = [[1,2,3,4],[5,6,7,8]], B = ones(4×3), C = ones(2×3).
// Expected output: D = [[11,11,11],[27,27,27]].
class CublasLtMatmulThunkCmdBufTest : public ::testing::Test {
 protected:
  static constexpr int64_t kALength = sizeof(float) * 2 * 4;
  static constexpr int64_t kBLength = sizeof(float) * 4 * 3;
  static constexpr int64_t kCLength = sizeof(float) * 2 * 3;
  static constexpr int64_t kDLength = sizeof(float) * 2 * 3;
  static constexpr int64_t kWorkspaceLength = 1024 * 1024;

  void SetUp() override {
    executor_ = GpuExecutor();
    if (!SupportsCudaGraphTracing(executor_)) {
      GTEST_SKIP() << "CUDA graph tracing is not supported";
    }

    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
    TF_ASSERT_OK_AND_ASSIGN(trace_stream_, executor_->CreateStream());

    // Allocate and initialize device buffers.
    a_buf_ = executor_->AllocateArray<float>(2 * 4);
    b_buf_ = executor_->AllocateArray<float>(4 * 3);
    c_buf_ = executor_->AllocateArray<float>(2 * 3);
    d_buf_ = executor_->AllocateArray<float>(2 * 3);
    workspace_buf_ =
        executor_->AllocateArray<float>(kWorkspaceLength / sizeof(float));

    std::vector<float> a_arr{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> b_arr(12, 1.0f);
    std::vector<float> c_arr(6, 1.0f);
    TF_ASSERT_OK(stream_->Memcpy(&a_buf_, a_arr.data(), kALength));
    TF_ASSERT_OK(stream_->Memcpy(&b_buf_, b_arr.data(), kBLength));
    TF_ASSERT_OK(stream_->Memcpy(&c_buf_, c_arr.data(), kCLength));
    TF_ASSERT_OK(stream_->MemZero(&d_buf_, kDLength));
    TF_ASSERT_OK(stream_->MemZero(&workspace_buf_, kWorkspaceLength));

    // Build shaped slices from the member BufferAllocation objects.
    ShapedSlice slice_a{BufferAllocation::Slice(&alloc_a_, 0, kALength),
                        ShapeUtil::MakeShape(F32, {2, 4})};
    ShapedSlice slice_b{BufferAllocation::Slice(&alloc_b_, 0, kBLength),
                        ShapeUtil::MakeShape(F32, {4, 3})};
    ShapedSlice slice_c{BufferAllocation::Slice(&alloc_c_, 0, kCLength),
                        ShapeUtil::MakeShape(F32, {2, 3})};
    ShapedSlice slice_d{BufferAllocation::Slice(&alloc_d_, 0, kDLength),
                        ShapeUtil::MakeShape(F32, {2, 3})};
    ShapedSlice slice_workspace{
        BufferAllocation::Slice(&alloc_workspace_, 0, kWorkspaceLength),
        ShapeUtil::MakeShape(U8, {1024, 1024})};

    TF_ASSERT_OK_AND_ASSIGN(
        GemmConfig config,
        GemmConfig::For(
            ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), {}, {1},
            ShapeUtil::MakeShape(PrimitiveType::F32, {4, 3}), {}, {0},
            ShapeUtil::MakeShape(PrimitiveType::F32, {2, 3}), nullptr,
            ShapeUtil::MakeShape(PrimitiveType::F32, {2, 3}),
            /*alpha_real=*/1.0, /*alpha_imag=*/0, /*beta=*/1.0,
            PrecisionConfig::ALG_UNSET, std::nullopt,
            se::blas::kDefaultComputePrecision, false, false,
            se::gpu::ScaleMode::kNone,
            executor_->GetDeviceDescription().gpu_compute_capability()));

    thunk_.emplace(Thunk::ThunkInfo(), /*canonical_hlo=*/"", config,
                   se::gpu::BlasLt::Epilogue::kDefault, /*algorithm_idx=*/0,
                   /*autotune_workspace_size=*/0, slice_a, slice_b, slice_c,
                   slice_d, /*bias=*/std::nullopt, /*aux=*/std::nullopt,
                   /*a_scale=*/std::nullopt, /*b_scale=*/std::nullopt,
                   /*c_scale=*/std::nullopt, /*d_scale=*/std::nullopt,
                   /*d_amax=*/std::nullopt, slice_workspace);

    // Use raw new so the brace-init for BufferAllocations matches the original
    // call site exactly, sidestepping template argument deduction issues with
    // optional::emplace and initializer lists.
    allocator_.reset(new se::StreamExecutorAddressAllocator(executor_));
    allocations_.reset(new BufferAllocations(
        {a_buf_, b_buf_, c_buf_, d_buf_, workspace_buf_}, 0, allocator_.get()));

    Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
    TF_ASSERT_OK(thunk_->Initialize(
        {executor_, source, allocations_.get(), stream_.get()}));

    params_.emplace(Thunk::ExecuteParams::Create(
        run_options_, *allocations_, stream_.get(), trace_stream_.get(),
        /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
        /*collective_memory=*/nullptr));
  }

  se::StreamExecutor* executor_ = nullptr;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<se::Stream> trace_stream_;
  se::DeviceAddress<float> a_buf_, b_buf_, c_buf_, d_buf_, workspace_buf_;

  // alloc_*_ must be declared before thunk_ so they outlive it: thunk_ stores
  // BufferAllocation::Slices that hold raw pointers into these allocations.
  BufferAllocation alloc_a_{/*index=*/0, kALength, /*color=*/0};
  BufferAllocation alloc_b_{/*index=*/1, kBLength, /*color=*/0};
  BufferAllocation alloc_c_{/*index=*/2, kCLength, /*color=*/0};
  BufferAllocation alloc_d_{/*index=*/3, kDLength, /*color=*/0};
  BufferAllocation alloc_workspace_{/*index=*/4, kWorkspaceLength,
                                    /*color=*/0};

  std::optional<CublasLtMatmulThunk> thunk_;
  // allocator_ must be declared before allocations_ so it outlives it.
  std::unique_ptr<se::StreamExecutorAddressAllocator> allocator_;
  std::unique_ptr<BufferAllocations> allocations_;
  ServiceExecutableRunOptions run_options_;
  // params_ must be declared last: it holds references to allocations_,
  // stream_, and trace_stream_, all of which must outlive it.
  std::optional<Thunk::ExecuteParams> params_;
};

TEST_F(CublasLtMatmulThunkCmdBufTest, RecordCommandBuffer) {
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
  TF_ASSERT_OK(stream_->BlockHostUntilDone());

  std::vector<float> dst(6, 0.0f);
  TF_ASSERT_OK(stream_->Memcpy(dst.data(), d_buf_, kDLength));
  ASSERT_EQ(dst, std::vector<float>({11, 11, 11, 27, 27, 27}));
}

TEST_F(CublasLtMatmulThunkCmdBufTest, RecordCommandBufferUpdate) {
  CommandStateManager state;
  Command::RecordParams record_params = {state};

  // First recording: RecordCreate.
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
  TF_ASSERT_OK(stream_->BlockHostUntilDone());

  std::vector<float> dst(6, 0.0f);
  TF_ASSERT_OK(stream_->Memcpy(dst.data(), d_buf_, kDLength));
  ASSERT_EQ(dst, std::vector<float>({11, 11, 11, 27, 27, 27}));

  // Transition to update state; zero output to confirm re-execution.
  TF_ASSERT_OK(command_buffer->Update());
  TF_ASSERT_OK(stream_->MemZero(&d_buf_, kDLength));

  // Second recording: RecordUpdate with same buffers → cache hit, same cmd.
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk_->Record(*params_, record_params, Command::RecordUpdate{cmd},
                     command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);  // same command node is reused (cache hit)
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream_.get()));
  TF_ASSERT_OK(stream_->BlockHostUntilDone());

  std::fill(dst.begin(), dst.end(), 0.0f);
  TF_ASSERT_OK(stream_->Memcpy(dst.data(), d_buf_, kDLength));
  ASSERT_EQ(dst, std::vector<float>({11, 11, 11, 27, 27, 27}));
}

}  // namespace
}  // namespace xla::gpu
