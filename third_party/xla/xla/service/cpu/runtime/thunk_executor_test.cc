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

#include "xla/service/cpu/runtime/thunk_executor.h"

#define EIGEN_USE_THREADS

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/buffer_allocations.h"
#include "xla/service/cpu/runtime/task.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"
#include "tsl/platform/threadpool.h"

namespace xla::cpu {
namespace {

using ::testing::ElementsAre;

// A test-only thunk for verifying thunk executor implementation:
//
//   dst += src (for all srcs and dsts slices)
//
// We generate random thunk sequences reading and writing different slices of
// the same buffer, and check that at run time it does not lead to any data
// races and produces expected result.
class AddI32Thunk final : public Thunk {
 public:
  AddI32Thunk(std::string name, std::vector<BufferAllocation::Slice> srcs,
              std::vector<BufferAllocation::Slice> dsts, bool inject_error,
              std::vector<std::string>* trace);

  static std::unique_ptr<Thunk> Create(
      std::string name, std::vector<BufferAllocation::Slice> srcs,
      std::vector<BufferAllocation::Slice> dsts, bool inject_error = false,
      std::vector<std::string>* trace = nullptr);

  static std::vector<MaybeOwningDeviceMemory> AsDeviceMemory(
      absl::Span<std::vector<int32_t>* const> data);

  // Executes `dst += src` for a single src/dst pair.
  static absl::Status Execute(const BufferAllocations* allocations,
                              BufferAllocation::Slice src_slice,
                              BufferAllocation::Slice dst_slice);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams&) final;

  BufferUses buffer_uses() const final;

 private:
  std::vector<BufferAllocation::Slice> srcs_;
  std::vector<BufferAllocation::Slice> dsts_;
  bool inject_error_;
  std::vector<std::string>* trace_;
};

std::unique_ptr<Thunk> AddI32Thunk::Create(
    std::string name, std::vector<BufferAllocation::Slice> srcs,
    std::vector<BufferAllocation::Slice> dsts, bool inject_error,
    std::vector<std::string>* trace) {
  return std::make_unique<AddI32Thunk>(std::move(name), std::move(srcs),
                                       std::move(dsts), inject_error, trace);
}

std::vector<MaybeOwningDeviceMemory> AddI32Thunk::AsDeviceMemory(
    absl::Span<std::vector<int32_t>* const> data) {
  std::vector<MaybeOwningDeviceMemory> buffers;
  for (auto& vec : data) {
    buffers.emplace_back(
        se::DeviceMemoryBase(vec->data(), vec->size() * sizeof(int32_t)));
  }
  return buffers;
}

AddI32Thunk::AddI32Thunk(std::string name,
                         std::vector<BufferAllocation::Slice> srcs,
                         std::vector<BufferAllocation::Slice> dsts,
                         bool inject_error, std::vector<std::string>* trace)
    : Thunk(Kind::kKernel, Info{name}),
      srcs_(std::move(srcs)),
      dsts_(std::move(dsts)),
      inject_error_(inject_error),
      trace_(trace) {}

absl::Status AddI32Thunk::Execute(const BufferAllocations* allocations,
                                  BufferAllocation::Slice src_slice,
                                  BufferAllocation::Slice dst_slice) {
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase src,
                      allocations->GetDeviceAddress(src_slice));

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase dst,
                      allocations->GetDeviceAddress(dst_slice));

  CHECK_EQ(src.size() % sizeof(int32_t), 0);
  CHECK_EQ(dst.size() % sizeof(int32_t), 0);

  int32_t* src_ptr = static_cast<int32_t*>(src.opaque());
  int32_t* dst_ptr = static_cast<int32_t*>(dst.opaque());
  size_t len = std::min(src.size(), dst.size()) / sizeof(int32_t);

  for (int j = 0; j < len; ++j) dst_ptr[j] += src_ptr[j];

  return absl::OkStatus();
}

tsl::AsyncValueRef<Thunk::ExecuteEvent> AddI32Thunk::Execute(
    const ExecuteParams& params) {
  if (trace_) trace_->push_back(info().op_name);

  auto execute = [&]() -> absl::Status {
    CHECK_EQ(srcs_.size(), dsts_.size());
    for (int i = 0; i < srcs_.size(); ++i) {
      TF_RETURN_IF_ERROR(
          Execute(params.buffer_allocations, srcs_.at(i), dsts_.at(i)));
    }
    return absl::OkStatus();
  };

  // Offload the execution to the intra-op thread pool.
  if (params.intra_op_threadpool) {
    auto event = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();
    params.intra_op_threadpool->getPool()->Schedule([&, event, execute] {
      if (inject_error_) {
        event.SetError(absl::InternalError("Injected error"));
      } else {
        CHECK_OK(execute());
        event.SetStateConcrete();
      }
    });
    return event;
  }

  if (inject_error_) {
    return tsl::MakeErrorAsyncValueRef(absl::InternalError("Injected error"));
  }

  TF_RETURN_IF_ERROR(execute());
  return Thunk::OkExecuteEvent();
}

AddI32Thunk::BufferUses AddI32Thunk::buffer_uses() const {
  BufferUses buffer_uses;
  for (const auto& src : srcs_) buffer_uses.push_back(BufferUse::Read(src));
  for (const auto& dst : dsts_) buffer_uses.push_back(BufferUse::Write(dst));
  return buffer_uses;
}

TEST(ThunkExecutorTest, Ordering) {
  BufferAllocation alloc(/*index=*/0, /*size=*/80, /*color=*/0);

  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/40);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/40, /*size=*/40);
  BufferAllocation::Slice slice2(&alloc, /*offset=*/20, /*size=*/40);

  ThunkSequence sequence;
  sequence.push_back(AddI32Thunk::Create("a", {slice0}, {slice0}));
  sequence.push_back(AddI32Thunk::Create("b", {slice1}, {slice1}));
  sequence.push_back(AddI32Thunk::Create("c", {slice2}, {slice2}));

  TF_ASSERT_OK_AND_ASSIGN(ThunkExecutor executor,
                          ThunkExecutor::Create(std::move(sequence)));

  EXPECT_THAT(executor.source(), ElementsAre(0, 1));
  EXPECT_THAT(executor.sink(), ElementsAre(2));
}

TEST(ThunkExecutorTest, TransitiveReduction) {
  BufferAllocation alloc(/*index=*/0, /*size=*/80, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/40);

  ThunkSequence sequence;
  sequence.push_back(AddI32Thunk::Create("a", {slice}, {slice}));
  sequence.push_back(AddI32Thunk::Create("b", {slice}, {slice}));
  sequence.push_back(AddI32Thunk::Create("c", {slice}, {slice}));

  TF_ASSERT_OK_AND_ASSIGN(ThunkExecutor executor,
                          ThunkExecutor::Create(std::move(sequence)));

  EXPECT_THAT(executor.source(), ElementsAre(0));
  EXPECT_THAT(executor.sink(), ElementsAre(2));

  EXPECT_THAT(executor.node_def(0).out_edges, ElementsAre(1));
  EXPECT_THAT(executor.node_def(1).in_edges, ElementsAre(0));
  EXPECT_THAT(executor.node_def(1).out_edges, ElementsAre(2));
  EXPECT_THAT(executor.node_def(2).in_edges, ElementsAre(1));
}

TEST(ThunkExecutorTest, Execute) {
  BufferAllocation alloc(/*index=*/0, /*size=*/80, /*color=*/0);

  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/40);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/40, /*size=*/40);
  BufferAllocation::Slice slice2(&alloc, /*offset=*/20, /*size=*/40);

  std::vector<std::string> trace;

  ThunkSequence sequence;
  sequence.push_back(AddI32Thunk::Create("a", {slice0}, {slice0},
                                         /*inject_error=*/false, &trace));
  sequence.push_back(AddI32Thunk::Create("b", {slice1}, {slice1},
                                         /*inject_error=*/false, &trace));
  sequence.push_back(AddI32Thunk::Create("c", {slice2}, {slice2},
                                         /*inject_error=*/false, &trace));

  TF_ASSERT_OK_AND_ASSIGN(ThunkExecutor executor,
                          ThunkExecutor::Create(std::move(sequence)));

  std::vector<int32_t> data(20, 1);  // shared src and dst allocation

  auto buffers = AddI32Thunk::AsDeviceMemory({&data});
  BufferAllocations allocations(buffers);

  Thunk::ExecuteParams params = {nullptr, &allocations};
  auto execute_event = executor.Execute(params, [&](ThunkExecutor::Task task) {
    trace.push_back("<TaskRunner>");
    task();
  });

  tsl::BlockUntilReady(execute_event);
  ASSERT_TRUE(execute_event.IsConcrete());

  EXPECT_THAT(trace, ElementsAre("<TaskRunner>", "b", "a", "c"));
  EXPECT_THAT(data, ElementsAre(2, 2, 2, 2, 2,                 // slice0
                                4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  // slice2
                                2, 2, 2, 2, 2));               // slice1
}

//===----------------------------------------------------------------------===//
// ThunkExecutor stress testing
//===----------------------------------------------------------------------===//

struct GeneratedThunkSequence {
  BufferAllocation src_alloc;
  BufferAllocation dst_alloc;

  std::vector<int32_t> src;
  std::vector<int32_t> dst;
  std::vector<int32_t> expected;

  std::vector<MaybeOwningDeviceMemory> expected_buffers;
  std::vector<MaybeOwningDeviceMemory> buffers;

  ThunkSequence sequence;
};

static absl::StatusOr<std::unique_ptr<GeneratedThunkSequence>>
GenerateThunkSequence(size_t num_elements, size_t num_thunks,
                      bool inject_errors = false) {
  auto g = std::make_unique<GeneratedThunkSequence>(GeneratedThunkSequence{
      BufferAllocation(/*index=*/0, num_elements * sizeof(int32_t), 0),
      BufferAllocation(/*index=*/1, num_elements * sizeof(int32_t), 0),
      /*src=*/std::vector<int32_t>(num_elements, 1),
      /*dst=*/std::vector<int32_t>(num_elements, 0),
      /*expected=*/std::vector<int32_t>(num_elements, 0),
  });

  g->expected_buffers = AddI32Thunk::AsDeviceMemory({&g->src, &g->expected});
  g->buffers = AddI32Thunk::AsDeviceMemory({&g->src, &g->dst});

  std::minstd_rand0 engine;

  std::uniform_int_distribution<size_t> offset_dist(0, num_elements - 1);
  std::uniform_int_distribution<size_t> size_dist(32, 64);
  std::uniform_int_distribution<size_t> inject_error_dist(0, num_thunks / 10);

  // Returns a random slice of the allocation.
  auto random_slice = [&](BufferAllocation* alloc) {
    size_t start = offset_dist(engine);
    size_t size = std::min(num_elements - start, size_dist(engine));
    return BufferAllocation::Slice(alloc, start * sizeof(int32_t),
                                   size * sizeof(int32_t));
  };

  for (int i = 0; i < num_thunks; ++i) {
    BufferAllocation::Slice src = random_slice(&g->src_alloc);
    BufferAllocation::Slice dst = random_slice(&g->dst_alloc);

    // Pre-compute expected result while building the thunk sequence.
    BufferAllocations allocations(g->expected_buffers);
    TF_RETURN_IF_ERROR(AddI32Thunk::Execute(&allocations, src, dst));

    bool inject_error = inject_errors && inject_error_dist(engine) == 0;
    g->sequence.push_back(
        AddI32Thunk::Create(absl::StrCat(i), {src}, {dst}, inject_error));
  }

  return g;
}

// Parameterized thunk executor stress tests that builds a random thunk sequence
// and optionally uses a thread pool to execute thunk executor tasks.
class ThunkExecutorStressTest
    : public testing::TestWithParam<std::tuple<int32_t, bool, bool, bool>> {
 public:
  void SetUp() override {
    auto& [_, use_task_runner, use_device, inject_errors] = GetParam();

    use_task_runner_ = use_task_runner;
    use_device_ = use_device;

    // Both the task runner and the intra-op device share the same underlying
    // thread pool, and we test that they do not deadlock each other and
    // everything works via chaining together asynchronous events. It is a
    // common source of deadlocks to wait for the completion of tasks scheduled
    // into the same thread pool where awaiting thread is executing.
    if (use_task_runner_ || use_device_) {
      thread_pool_.emplace(tsl::Env::Default(), "thunk-executor", 8);
      device_.emplace(thread_pool_->AsEigenThreadPool(),
                      thread_pool_->NumThreads());
    }
  }

  ThunkExecutor::TaskRunner task_runner() {
    if (!use_task_runner_) return nullptr;
    return [&](ThunkExecutor::Task task) {
      thread_pool_->Schedule(ToCopyableTask(std::move(task)));
    };
  }

  Eigen::ThreadPoolDevice* device() {
    if (!use_device_) return nullptr;
    return &*device_;
  }

 private:
  bool use_task_runner_;
  bool use_device_;
  std::optional<tsl::thread::ThreadPool> thread_pool_;
  std::optional<Eigen::ThreadPoolDevice> device_;
};

TEST_P(ThunkExecutorStressTest, Execute) {
  auto [num_thunks, use_task_runner, use_device, inject_errors] = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GeneratedThunkSequence> g,
      GenerateThunkSequence(/*num_elements=*/1024, num_thunks, inject_errors));

  TF_ASSERT_OK_AND_ASSIGN(ThunkExecutor executor,
                          ThunkExecutor::Create(std::move(g->sequence)));

  BufferAllocations allocations(g->buffers);
  Thunk::ExecuteParams params = {nullptr, &allocations, nullptr, device()};

  auto execute_event = executor.Execute(params, task_runner());
  tsl::BlockUntilReady(execute_event);

  if (inject_errors) {
    ASSERT_TRUE(execute_event.IsError());
    EXPECT_EQ(execute_event.GetError(), absl::InternalError("Injected error"));
  } else {
    ASSERT_TRUE(execute_event.IsConcrete());
    EXPECT_EQ(g->dst, g->expected);
  }
}

INSTANTIATE_TEST_SUITE_P(ThunkExecutor, ThunkExecutorStressTest,
                         testing::Combine(testing::ValuesIn({10, 100, 1000}),
                                          testing::Bool(), testing::Bool(),
                                          testing::Bool()));

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

static void BM_SyncThunkExecutor(benchmark::State& state) {
  const size_t num_thunks = state.range(0);

  auto g = GenerateThunkSequence(/*num_elements=*/1024, num_thunks).value();
  auto e = ThunkExecutor::Create(std::move(g->sequence)).value();

  BufferAllocations allocations(g->buffers);
  Thunk::ExecuteParams params = {nullptr, &allocations};

  for (auto _ : state) {
    auto execute_event = e.Execute(params, nullptr);
    tsl::BlockUntilReady(execute_event);
    CHECK(execute_event.IsConcrete());
  }
}

static void BM_AsyncThunkExecutor(benchmark::State& state) {
  const size_t num_thunks = state.range(0);

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "thunk-executor", 8);
  Eigen::ThreadPoolDevice device(thread_pool.AsEigenThreadPool(),
                                 thread_pool.NumThreads());

  auto g = GenerateThunkSequence(/*num_elements=*/1024, num_thunks).value();
  auto e = ThunkExecutor::Create(std::move(g->sequence)).value();

  BufferAllocations allocations(g->buffers);
  Thunk::ExecuteParams params = {nullptr, &allocations, nullptr, &device};

  for (auto _ : state) {
    auto execute_event = e.Execute(params, [&](ThunkExecutor::Task task) {
      thread_pool.Schedule(ToCopyableTask(std::move(task)));
    });
    tsl::BlockUntilReady(execute_event);
    CHECK(execute_event.IsConcrete());
  }
}

BENCHMARK(BM_SyncThunkExecutor)
    ->MeasureProcessCPUTime()
    ->Arg(1)
    ->Arg(16)
    ->Arg(64)
    ->Arg(128)
    ->Arg(258)
    ->Arg(512);

BENCHMARK(BM_AsyncThunkExecutor)
    ->MeasureProcessCPUTime()
    ->Arg(1)
    ->Arg(16)
    ->Arg(64)
    ->Arg(128)
    ->Arg(258)
    ->Arg(512);

}  // namespace
}  // namespace xla::cpu
