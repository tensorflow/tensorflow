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
#include <deque>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/error_spec.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {

namespace {

class GpuBlasLtMatmulThunkTest : public HloTestBase {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cublaslt(true);
    debug_options.set_xla_gpu_enable_triton_gemm(false);
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
    if (auto* rocm = std::get_if<se::RocmComputeCapability>(&gpu_comp());
        rocm != nullptr && !rocm->has_hipblaslt()) {
      GTEST_SKIP() << "No hipblas-lt support on this architecture!";
    }
  }

  void CreateExecuteThunksFromHLO(se::StreamExecutor* executor,
                                  absl::string_view hlo_string);
};

struct GpuBlasLtThunkBuilder {
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

    std::vector<BufferAllocation::Slice> slices;
    std::vector<size_t> buf_sizes;
    for (auto op : gemm->operands()) {
      auto size = ShapeUtil::ByteSizeOf(op->shape());
      buf_sizes.push_back(size);
    }
    const auto& output_shape =
        gemm->shape().IsTuple() ? gemm->shape().tuple_shapes(0) : gemm->shape();
    buf_sizes.push_back(ShapeUtil::ByteSizeOf(output_shape));

    size_t idx = allocs_.size();
    slices.reserve(buf_sizes.size());
    for (auto size : buf_sizes) {
      mem_buffers_.emplace_back();
      TF_ASSIGN_OR_RETURN(mem_buffers_.back(),
                          allocator_.Allocate(exec_->device_ordinal(), size));
      allocs_.emplace_back(/*index=*/idx++, size, /*color=*/0);
      slices.emplace_back(&allocs_.back(), /*offset*/ 0, size);
    }
    // we need at least 3 buffers: lhs, rhs and output
    EXPECT_EQ(slices.size(),
              3 + size_t{has_matrix_bias} + size_t{has_vector_bias});
    TF_ASSIGN_OR_RETURN(auto gemm_config, GemmConfig::For(gemm, gpu_comp_));

    BufferAllocation::Slice bias;
    if (has_vector_bias) {
      bias = slices[has_matrix_bias ? 3 : 2];
    }

    return std::make_unique<CublasLtMatmulThunk>(
        gemm, std::move(gemm_config), epilogue,
        /*algorithm_idx*/ 0, slices[0], slices[1],
        has_matrix_bias ? slices[2] : slices.back(), slices.back(), bias,
        BufferAllocation::Slice{} /* aux */,
        BufferAllocation::Slice{} /* a_scale */,
        BufferAllocation::Slice{} /* b_scale */,
        BufferAllocation::Slice{} /* c_scale */,
        BufferAllocation::Slice{} /* d_scale */,
        BufferAllocation::Slice{} /* d_amax */, std::nullopt /* workspace */);
  }

  std::unique_ptr<BufferAllocations> buffer_allocations() {
    std::vector<se::DeviceMemoryBase> buffers(mem_buffers_.size());
    for (size_t i = 0; i < buffers.size(); i++) {
      buffers[i] = *mem_buffers_[i];
    }
    return std::make_unique<BufferAllocations>(buffers, exec_->device_ordinal(),
                                               &allocator_);
  }

 private:
  se::StreamExecutor* exec_;
  se::StreamExecutorMemoryAllocator allocator_;
  se::GpuComputeCapability gpu_comp_;
  std::deque<BufferAllocation> allocs_;
  std::vector<se::OwningDeviceMemory> mem_buffers_;
};

void GpuBlasLtMatmulThunkTest::CreateExecuteThunksFromHLO(
    se::StreamExecutor* executor, absl::string_view hlo_string) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          this->ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloPass(
          GemmRewriter(gpu_comp(executor),
                       /*toolkit_version=*/se::SemanticVersion{12, 4, 0}),
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

}  // namespace
}  // namespace xla::gpu
