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
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/error_spec.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

namespace {
using absl_testing::IsOkAndHolds;
using tsl::proto_testing::EqualsProto;

class GpuBlasLtMatmulThunkTest : public HloTestBase {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cublaslt(true);
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
  options.enable_cublaslt = GetDebugOptionsForTest().xla_gpu_enable_cublaslt();
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
        run_options, *allocs, stream, stream, nullptr, nullptr);

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
    canonical_hlo: "(f32[101,400]{1,0}, s8[33554432]{0}) custom-call(f32[101,407]{1,0}, f32[407,400]{1,0}), custom_call_target=\"__cublas$lt$matmul\", backend_config={\"operation_queue_id\":\"0\",\"wait_on_operation_queues\":[],\"gemm_backend_config\":{\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"],\"algorithm\":\"ALG_UNSET\"},\"epilogue\":\"DEFAULT\",\"lhs_stride\":\"41107\",\"rhs_stride\":\"162800\",\"grad_x\":false,\"grad_y\":false,\"damax_output\":false},\"force_earliest_schedule\":false,\"reification_cost\":[]}"
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
  thunk_info.execution_stream_id = 0;

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

}  // namespace
}  // namespace xla::gpu
