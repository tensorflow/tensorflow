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

#include "xla/backends/cpu/runtime/thunk_executor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thread_pool_task_runner.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

using ::testing::ElementsAre;

// We use a global static variable to simulate a shared resource. We check that
// thunk executor correctly orders access to this resource by running the test
// with a thread sanitizer and checking that there are no data races.
static int64_t shared_resource;

// An adaptor from a lambda that runs tasks and a TaskRunner API.
template <typename Runner, typename WorkerId>
class TaskRunnerAdaptor : public Thunk::TaskRunner {
 public:
  TaskRunnerAdaptor(Runner runner, WorkerId worker_id)
      : runner_(std::move(runner)), worker_id_(std::move(worker_id)) {}

  void operator()(Thunk::Task task) final { runner_(std::move(task)); }

  std::optional<int64_t> current_worker_id() const final {
    return worker_id_();
  }

 private:
  Runner runner_;
  WorkerId worker_id_;
};

template <typename Runner>
auto MakeTaskRunnerFrom(Runner&& runner) {
  auto no_id = []() { return std::nullopt; };
  return TaskRunnerAdaptor<Runner, decltype(no_id)>(
      std::forward<Runner>(runner), no_id);
}

template <typename Runner, typename WorkerId>
auto MakeTaskRunnerFrom(Runner&& runner, WorkerId&& worker_id) {
  return TaskRunnerAdaptor<Runner, WorkerId>(std::forward<Runner>(runner),
                                             std::forward<WorkerId>(worker_id));
}

// A test-only thunk for verifying thunk executor implementation:
//
//   dst += src (for all srcs and dsts slices)
//
// We generate random thunk sequences reading and writing different slices of
// the same buffer, and check that at run time it does not lead to any data
// races and produces expected result.
//
// We also emulate shared resource access by writing to the global static
// `shared_resource` variable and detecting data races with thread sanitizer.
class AddI32Thunk final : public Thunk {
 public:
  AddI32Thunk(std::string name, std::vector<BufferAllocation::Slice> srcs,
              std::vector<BufferAllocation::Slice> dsts,
              std::vector<std::string>* trace,
              std::optional<Resource::Kind> shared_resource, bool inject_error);

  static std::unique_ptr<Thunk> Create(
      std::string name, std::vector<BufferAllocation::Slice> srcs,
      std::vector<BufferAllocation::Slice> dsts,
      std::vector<std::string>* trace = nullptr,
      std::optional<Resource::Kind> shared_resource = std::nullopt,
      bool inject_error = false);

  // Executes `dst += src` for a single src/dst pair.
  static absl::Status Execute(const BufferAllocations* allocations,
                              BufferAllocation::Slice src_slice,
                              BufferAllocation::Slice dst_slice);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams&) final;

  BufferUses buffer_uses() const final;
  ResourceUses resource_uses() const final;

 private:
  std::vector<BufferAllocation::Slice> srcs_;
  std::vector<BufferAllocation::Slice> dsts_;
  std::vector<std::string>* trace_;
  std::optional<Resource::Kind> shared_resource_;
  bool inject_error_;
};

std::unique_ptr<Thunk> AddI32Thunk::Create(
    std::string name, std::vector<BufferAllocation::Slice> srcs,
    std::vector<BufferAllocation::Slice> dsts, std::vector<std::string>* trace,
    std::optional<Resource::Kind> shared_resource, bool inject_error) {
  return std::make_unique<AddI32Thunk>(std::move(name), std::move(srcs),
                                       std::move(dsts), trace, shared_resource,
                                       inject_error);
}

AddI32Thunk::AddI32Thunk(std::string name,
                         std::vector<BufferAllocation::Slice> srcs,
                         std::vector<BufferAllocation::Slice> dsts,
                         std::vector<std::string>* trace,
                         std::optional<Resource::Kind> shared_resource,
                         bool inject_error)
    : Thunk(Kind::kKernel, Info{name}),
      srcs_(std::move(srcs)),
      dsts_(std::move(dsts)),
      trace_(trace),
      shared_resource_(shared_resource),
      inject_error_(inject_error) {}

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

  for (int j = 0; j < len; ++j) {
    dst_ptr[j] += src_ptr[j];
  }

  return absl::OkStatus();
}

tsl::AsyncValueRef<Thunk::ExecuteEvent> AddI32Thunk::Execute(
    const ExecuteParams& params) {
  if (trace_) {
    trace_->push_back(info().op_name);
  }

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

    // Collective communicator resource creates only a scheduling edge,
    // incrementing the shared resource value from the intra-op thread pool
    // might lead to data races. ThunkExecutor will only guarantee that calls
    // to Execute() are properly synchronized.
    if (shared_resource_ &&
        *shared_resource_ == Resource::kCollectiveCommunicator) {
      shared_resource++;
    }

    params.intra_op_threadpool->getPool()->Schedule([&, event, execute] {
      // Token creates an execution edge, and it means that dependent thunks
      // will wait for the completion of execution of all dependencies, and we
      // verify that we don't have data races by mutating shared resource from a
      // task that runs on a thread pool.
      if (shared_resource_ && *shared_resource_ == Resource::kToken) {
        shared_resource++;
      }

      if (inject_error_) {
        event.SetError(absl::InternalError("Injected error"));
      } else {
        CHECK_OK(execute());
        event.SetStateConcrete();
      }
    });
    return event;
  }

  if (shared_resource_) {
    shared_resource++;
  }

  if (inject_error_) {
    return tsl::MakeErrorAsyncValueRef(absl::InternalError("Injected error"));
  }

  TF_RETURN_IF_ERROR(execute());
  return Thunk::OkExecuteEvent();
}

AddI32Thunk::BufferUses AddI32Thunk::buffer_uses() const {
  BufferUses buffer_uses;
  for (const auto& src : srcs_) {
    buffer_uses.push_back(BufferUse::Read(src));
  }
  for (const auto& dst : dsts_) {
    buffer_uses.push_back(BufferUse::Write(dst));
  }
  return buffer_uses;
}

AddI32Thunk::ResourceUses AddI32Thunk::resource_uses() const {
  static std::shared_ptr<Resource>* token_resource =
      new std::shared_ptr<Resource>(Resource::Create(Resource::kToken));

  static std::shared_ptr<Resource>* comm_resource =
      new std::shared_ptr<Resource>(
          Resource::Create(Resource::kCollectiveCommunicator));

  if (!shared_resource_) {
    return ResourceUses{};
  }

  switch (*shared_resource_) {
    case Resource::kToken:
      return ResourceUses{ResourceUse::Write(*token_resource)};
    case Resource::kCollectiveCommunicator:
      return ResourceUses{ResourceUse::Write(*comm_resource)};
  }
}

static ThunkExecutor::Options OptionsForTest() {
  return ThunkExecutor::Options{/*execute_sequential_buffer_threshold=*/0,
                                /*execute_sequential_num_thunks_threshold=*/0};
}

TEST(ThunkExecutorTest, FifoReadyQueueTest) {
  ThunkExecutor::FifoReadyQueue queue({});

  // Check basic queue properties.
  EXPECT_TRUE(queue.Empty());
  EXPECT_EQ(queue.Size(), 0);

  queue.Push(1);
  queue.Push(2);
  queue.Push(3);

  ASSERT_EQ(queue.Size(), 3);

  EXPECT_EQ(queue.Pop(), 1);
  EXPECT_EQ(queue.Pop(), 2);
  EXPECT_EQ(queue.Pop(), 3);

  EXPECT_TRUE(queue.Empty());
  ASSERT_EQ(queue.Size(), 0);

  // Prepare queue for PopHalf test case.
  queue.Push(1);
  queue.Push(2);
  queue.Push(3);

  // Pop half of the queue.
  ThunkExecutor::FifoReadyQueue half0 = queue.PopHalf();
  ASSERT_EQ(half0.Size(), 2);
  EXPECT_EQ(half0.Pop(), 2);
  EXPECT_EQ(half0.Pop(), 3);

  // Check that the rest is still in the queue.
  ASSERT_EQ(queue.Size(), 1);

  // Pop the rest of the queue.
  ThunkExecutor::FifoReadyQueue half1 = queue.PopHalf();
  ASSERT_EQ(half1.Size(), 1);

  // Check that all nodes were returned from PopHalf.
  EXPECT_EQ(queue.Size(), 0);

  // Add 5 elements to test Pop followed by PopHalf.
  queue.Push(1);
  queue.Push(2);
  queue.Push(3);
  queue.Push(4);
  queue.Push(5);

  EXPECT_EQ(queue.Pop(), 1);

  // Check that PopHalf returns 2 last nodes.
  ThunkExecutor::FifoReadyQueue half2 = queue.PopHalf();
  ASSERT_EQ(half2.Size(), 2);
  EXPECT_EQ(half2.Pop(), 4);
  EXPECT_EQ(half2.Pop(), 5);
}

TEST(ThunkExecutorTest, LifoReadyQueueTest) {
  ThunkExecutor::LifoReadyQueue queue({});

  // Check basic queue properties.
  EXPECT_TRUE(queue.Empty());
  EXPECT_EQ(queue.Size(), 0);

  queue.Push(1);
  queue.Push(2);
  queue.Push(3);

  ASSERT_EQ(queue.Size(), 3);

  EXPECT_EQ(queue.Pop(), 3);
  EXPECT_EQ(queue.Pop(), 2);
  EXPECT_EQ(queue.Pop(), 1);

  EXPECT_TRUE(queue.Empty());
  EXPECT_EQ(queue.Size(), 0);

  // Prepare queue for PopHalf test case.
  queue.Push(1);
  queue.Push(2);
  queue.Push(3);

  // Pop half of the queue.
  ThunkExecutor::LifoReadyQueue half0 = queue.PopHalf();
  ASSERT_EQ(half0.Size(), 2);
  EXPECT_EQ(half0.Pop(), 2);
  EXPECT_EQ(half0.Pop(), 1);

  // Check that the rest is still in the queue.
  ASSERT_EQ(queue.Size(), 1);

  // Pop the rest of the queue.
  ThunkExecutor::LifoReadyQueue half1 = queue.PopHalf();
  ASSERT_EQ(half1.Size(), 1);

  // ASSERT_EQ that all nodes were returned from PopHalf.
  EXPECT_EQ(queue.Size(), 0);

  // Add 5 elements to test Pop followed by PopHalf.
  queue.Push(1);
  queue.Push(2);
  queue.Push(3);
  queue.Push(4);
  queue.Push(5);

  EXPECT_EQ(queue.Pop(), 5);

  // Check that PopHalf returns first 2 nodes.
  ThunkExecutor::LifoReadyQueue half2 = queue.PopHalf();
  ASSERT_EQ(half2.Size(), 3);
  EXPECT_EQ(half2.Pop(), 3);
  EXPECT_EQ(half2.Pop(), 2);
  EXPECT_EQ(half2.Pop(), 1);
}

TEST(ThunkExecutorTest, PriorityReadyQueueTest) {
  std::vector<ThunkExecutor::NodeDef> nodes_defs(16);
  for (size_t i = 0; i < nodes_defs.size(); ++i) {
    nodes_defs[i].priority = i;
  }

  ThunkExecutor::PriorityReadyQueue queue(nodes_defs, {});
  // Check basic queue properties.
  EXPECT_TRUE(queue.Empty());
  EXPECT_EQ(queue.Size(), 0);

  queue.Push(1);
  queue.Push(3);
  queue.Push(2);

  EXPECT_EQ(queue.Pop(), 3);
  EXPECT_EQ(queue.Pop(), 2);
  EXPECT_EQ(queue.Pop(), 1);

  EXPECT_TRUE(queue.Empty());
  EXPECT_EQ(queue.Size(), 0);

  // Prepare queue for PopHalf test case.
  queue.Push(2);
  queue.Push(1);
  queue.Push(3);

  // Pop half of the queue.
  ThunkExecutor::PriorityReadyQueue half0 = queue.PopHalf();
  ASSERT_EQ(half0.Size(), 2);
  EXPECT_EQ(half0.Pop(), 2);
  EXPECT_EQ(half0.Pop(), 1);

  // Check that the rest is still in the queue.
  ASSERT_EQ(queue.Size(), 1);

  // Pop the rest of the queue.
  ThunkExecutor::PriorityReadyQueue half1 = queue.PopHalf();
  ASSERT_EQ(half1.Size(), 1);
  EXPECT_EQ(half1.Pop(), 3);

  // Check that all nodes were returned from PopHalf.
  ASSERT_EQ(queue.Size(), 0);

  // Add 5 elements to test Pop followed by PopHalf.
  queue.Push(4);
  queue.Push(3);
  queue.Push(5);
  queue.Push(1);
  queue.Push(2);

  EXPECT_EQ(queue.Pop(), 5);

  // Check that PopHalf returns 2 last nodes.
  ThunkExecutor::PriorityReadyQueue half2 = queue.PopHalf();
  ASSERT_EQ(half2.Size(), 2);
  EXPECT_EQ(half2.Pop(), 2);
  EXPECT_EQ(half2.Pop(), 1);
}

TEST(ThunkExecutorTest, Execute) {
  BufferAllocation alloc(/*index=*/0, /*size=*/80, /*color=*/0);

  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/40);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/40, /*size=*/40);
  BufferAllocation::Slice slice2(&alloc, /*offset=*/20, /*size=*/40);

  std::vector<std::string> trace;

  ThunkSequence sequence;
  sequence.push_back(AddI32Thunk::Create("a", {slice0}, {slice0}, &trace));
  sequence.push_back(AddI32Thunk::Create("b", {slice1}, {slice1}, &trace));
  sequence.push_back(AddI32Thunk::Create("c", {slice2}, {slice2}, &trace));

  TF_ASSERT_OK_AND_ASSIGN(
      ThunkExecutor executor,
      ThunkExecutor::Create(std::move(sequence), OptionsForTest()));

  // Shared src and dst allocation.
  auto data = LiteralUtil::CreateFull({20}, int32_t{1});
  BufferAllocations allocations = CreateBufferAllocations(data);

  auto task_runner = MakeTaskRunnerFrom(
      [&](Thunk::Task task) {
        trace.push_back("<TaskRunner>");
        task();
      },
      // Always return current worker id as 0.
      [] { return 0; });

  Thunk::ExecuteParams params = {nullptr, &allocations};
  params.task_runner = &task_runner;
  params.session =
      Thunk::ExecuteSession(/*max_workers=*/8, /*split_threshold=*/0);

  auto execute_event = executor.Execute(params);

  tsl::BlockUntilReady(execute_event);
  ASSERT_TRUE(execute_event.IsConcrete());

  EXPECT_THAT(trace, ElementsAre("<TaskRunner>", "b", "a", "c"));
  EXPECT_EQ(data, LiteralUtil::CreateR1<int32_t>({2, 2, 2, 2, 2,     // slice0
                                                  4, 4, 4, 4, 4,     // slice2
                                                  4, 4, 4, 4, 4,     // ...
                                                  2, 2, 2, 2, 2}));  // slice1
}

//===----------------------------------------------------------------------===//
// ThunkExecutor resource isolation testing
//===----------------------------------------------------------------------===//

// No-op thunk that completes execution on a separate thread pool. We use this
// thunk to test that ThunkExecutor can jump out of a separate thread pool to
// continue execution in the intra-op thread pool. This is important for
// resource isolation as we don't want to accidentally continue with expensive
// execution on a non blocking IO callbacks thread pool.
class NoOpAsyncThunk : public Thunk {
 public:
  NoOpAsyncThunk(std::string name, BufferAllocation::Slice slice)
      : Thunk(Kind::kKernel, Info{std::move(name)}), slice_(slice) {}

  static std::unique_ptr<NoOpAsyncThunk> Create(std::string name,
                                                BufferAllocation::Slice slice) {
    return std::make_unique<NoOpAsyncThunk>(std::move(name), slice);
  }

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams&) final {
    auto ret = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();
    ThreadPool()->Schedule([ret] {
      tsl::Env::Default()->SleepForMicroseconds(10 * 1000);
      ret.SetStateConcrete();
    });
    return ret;
  }

  BufferUses buffer_uses() const override {
    return BufferUses{BufferUse::Write(slice_)};
  }

 private:
  static tsl::thread::ThreadPool* ThreadPool() {
    static auto* thread_pool =
        new tsl::thread::ThreadPool(tsl::Env::Default(), "no-op-thunk", 8);
    return thread_pool;
  }

  BufferAllocation::Slice slice_;
};

TEST(ThunkExecutorTest, ExecuteOnCorrectThreadPool) {
  BufferAllocation alloc(/*index=*/0, /*size=*/60, /*color=*/0);

  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/20);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/20, /*size=*/20);
  BufferAllocation::Slice slice2(&alloc, /*offset=*/40, /*size=*/20);

  std::array<BufferAllocation::Slice, 3> slices = {slice0, slice1, slice2};

  ThunkSequence sequence;
  for (int i = 0; i < 100; ++i) {
    sequence.push_back(NoOpAsyncThunk::Create(absl::StrCat(i), slices[i % 3]));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      ThunkExecutor executor,
      ThunkExecutor::Create(std::move(sequence), OptionsForTest()));

  auto data = LiteralUtil::CreateFull({60}, uint8_t{1});
  BufferAllocations allocations = CreateBufferAllocations(data);

  // Task runner must be used only when ThunkExecutor detects that it runs on a
  // wrong thread and has to jump into the task runner.
  std::atomic<int32_t> num_tasks = 0;
  auto task_runner = MakeTaskRunnerFrom([&](Thunk::Task task) {
    ++num_tasks;
    task();
  });

  Thunk::ExecuteParams params = {nullptr, &allocations};
  params.task_runner = &task_runner;
  params.session =
      Thunk::ExecuteSession(/*max_workers=*/1, /*split_threshold=*/1000);

  auto execute_event = executor.Execute(params);

  tsl::BlockUntilReady(execute_event);
  ASSERT_TRUE(execute_event.IsConcrete());

  // We compare using GE because thread scheduling introduces small
  // non-determinism and ThunkExecutor might resume after NoOpAsyncThunk already
  // completes its execution event.
  EXPECT_GE(num_tasks, 90);
}

//===----------------------------------------------------------------------===//
// ThunkExecutor stress testing
//===----------------------------------------------------------------------===//

// We generate random thunk sequences that may or may not use a shared resource.
struct SharedResourceUse {
  enum class Kind { kNo, kAll, kRandom };

  Kind kind;
  std::optional<Resource::Kind> resource_kind;
};

struct GeneratedThunkSequence {
  explicit GeneratedThunkSequence(int64_t num_elements)
      : src(LiteralUtil::CreateFull({num_elements}, int32_t{1})),
        dst(LiteralUtil::CreateFull({num_elements}, int32_t{0})),
        expected(LiteralUtil::CreateFull({num_elements}, int32_t{0})),
        src_alloc(CreateBufferAllocation(0, src)),
        dst_alloc(CreateBufferAllocation(1, dst)),
        expected_shared_resource_value(0),
        expected_literals({&src, &expected}),
        literals({&src, &dst}) {}

  Literal src;
  Literal dst;
  Literal expected;

  BufferAllocation src_alloc;
  BufferAllocation dst_alloc;

  int32_t expected_shared_resource_value;

  std::vector<Literal*> expected_literals;
  std::vector<Literal*> literals;

  ThunkSequence sequence;
};

static absl::StatusOr<std::unique_ptr<GeneratedThunkSequence>>
GenerateThunkSequence(size_t num_elements, size_t num_thunks,
                      SharedResourceUse shared_resource_use,
                      bool inject_errors) {
  auto g = std::make_unique<GeneratedThunkSequence>(num_elements);
  g->sequence.reserve(num_thunks);

  std::minstd_rand0 engine;

  std::uniform_int_distribution<size_t> offset_dist(0, num_elements - 1);
  std::uniform_int_distribution<size_t> size_dist(32, 64);
  std::uniform_int_distribution<size_t> use_resource_dist(0, num_thunks / 10);
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
    BufferAllocations allocations =
        CreateBufferAllocations(absl::MakeSpan(g->expected_literals));
    TF_RETURN_IF_ERROR(AddI32Thunk::Execute(&allocations, src, dst));

    auto use_resource = [&]() -> std::optional<Resource::Kind> {
      switch (shared_resource_use.kind) {
        case SharedResourceUse::Kind::kNo:
          return std::nullopt;
        case SharedResourceUse::Kind::kAll:
          return shared_resource_use.resource_kind;
        case SharedResourceUse::Kind::kRandom:
          if (use_resource_dist(engine) == 0) {
            return shared_resource_use.resource_kind;
          }
          return std::nullopt;
      }
    }();

    if (use_resource) {
      g->expected_shared_resource_value++;
    }

    bool inject_error = inject_errors && inject_error_dist(engine) == 0;
    g->sequence.push_back(AddI32Thunk::Create(absl::StrCat(i), {src}, {dst},
                                              /*trace=*/nullptr, use_resource,
                                              inject_error));
  }

  return g;
}

// Parameterized thunk executor stress tests that builds a random thunk sequence
// and optionally uses a thread pool to execute thunk executor tasks.
class ThunkExecutorStressTest
    : public testing::TestWithParam<
          std::tuple<int32_t, bool, bool, SharedResourceUse, bool, bool>> {
 public:
  void SetUp() override {
    auto& [num_thunks, use_task_runner, use_device, shared_resource_use,
           inject_errors, use_priority_ready_queue] = GetParam();

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
      task_runner_.emplace(thread_pool_->AsEigenThreadPool());
    }
  }

  Thunk::TaskRunner* task_runner() {
    return use_task_runner_ ? &*task_runner_ : nullptr;
  }

  Eigen::ThreadPoolDevice* device() {
    return use_device_ ? &*device_ : nullptr;
  }

 private:
  bool use_task_runner_;
  bool use_device_;
  std::optional<tsl::thread::ThreadPool> thread_pool_;
  std::optional<Eigen::ThreadPoolDevice> device_;
  std::optional<ThreadPoolTaskRunner> task_runner_;
};

TEST_P(ThunkExecutorStressTest, Execute) {
  auto [num_thunks, use_task_runner, use_device, shared_resource_use,
        inject_errors, use_priority_ready_queue] = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GeneratedThunkSequence> g,
      GenerateThunkSequence(/*num_elements=*/1024, num_thunks,
                            shared_resource_use, inject_errors));

  ThunkExecutor::Options executor_options = {
      /*execute_sequential_buffer_threshold=*/0,
      /*use_priority_ready_queue=*/use_priority_ready_queue,
  };

  TF_ASSERT_OK_AND_ASSIGN(
      ThunkExecutor executor,
      ThunkExecutor::Create(std::move(g->sequence), executor_options));

  BufferAllocations allocations =
      CreateBufferAllocations(absl::MakeSpan(g->literals));
  Thunk::ExecuteParams params = {nullptr, &allocations, nullptr, device(),
                                 task_runner()};

  shared_resource = 0;

  auto execute_event = executor.Execute(params);
  tsl::BlockUntilReady(execute_event);

  if (inject_errors) {
    ASSERT_TRUE(execute_event.IsError());
    EXPECT_EQ(execute_event.GetError(), absl::InternalError("Injected error"));
  } else {
    ASSERT_TRUE(execute_event.IsConcrete());
    EXPECT_EQ(shared_resource, g->expected_shared_resource_value);
    EXPECT_EQ(g->dst, g->expected);
  }
}

// We keep the number of thunks smaller in debug builds as otherwise it takes
// too long to run the tests. In optimized builds we can afford to run longer
// thunk sequences to get more coverage.
auto NumTestThunks() {
#ifdef NDEBUG
  return testing::ValuesIn({10, 50, 100});
#else
  return testing::ValuesIn({10, 100, 500});
#endif
}

// Create aliases for all possible combinations of shared resource use.
static constexpr auto kToken = Resource::Kind::kToken;
static constexpr auto kComm = Resource::Kind::kCollectiveCommunicator;
static constexpr auto kNoResource = SharedResourceUse::Kind::kNo;
static constexpr auto kAllResource = SharedResourceUse::Kind::kAll;
static constexpr auto kRandomResource = SharedResourceUse::Kind::kRandom;

INSTANTIATE_TEST_SUITE_P(
    ThunkExecutor, ThunkExecutorStressTest,
    testing::Combine(
        /*num_thunks=*/NumTestThunks(),
        /*use_task_runner=*/testing::Bool(),
        /*use_device=*/testing::Bool(),
        /*shared_resource_use=*/
        testing::Values(SharedResourceUse{kNoResource, kToken},
                        SharedResourceUse{kAllResource, kToken},
                        SharedResourceUse{kRandomResource, kToken},
                        SharedResourceUse{kNoResource, kComm},
                        SharedResourceUse{kAllResource, kComm},
                        SharedResourceUse{kRandomResource, kComm}),
        /*inject_errors=*/testing::Bool(),
        /*use_priority_ready_queue=*/testing::Bool()));

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

static void BM_FifoReadyQueuePushPop(benchmark::State& state) {
  ThunkExecutor::FifoReadyQueue queue({});
  const size_t num_push_pop = state.range(0);

  for (auto _ : state) {
    for (int i = 0; i < num_push_pop; ++i) {
      queue.Push(i);
    }
    for (int i = 0; i < num_push_pop; ++i) {
      benchmark::DoNotOptimize(queue.Pop());
    }
  }
}

static void BM_FifoReadyQueuePushPopHalf(benchmark::State& state) {
  ThunkExecutor::FifoReadyQueue queue({});
  const size_t num_push_pop = state.range(0);

  for (auto _ : state) {
    for (int i = 0; i < num_push_pop; ++i) {
      queue.Push(i);
    }
    benchmark::DoNotOptimize(queue.PopHalf());
  }
}

static void BM_PriorityReadyQueuePushPop(benchmark::State& state) {
  std::vector<ThunkExecutor::NodeDef> nodes_defs(16);
  for (size_t i = 0; i < nodes_defs.size(); ++i) {
    nodes_defs[i].priority = i;
  }

  std::default_random_engine rng;
  absl::c_shuffle(nodes_defs, rng);

  ThunkExecutor::PriorityReadyQueue queue(nodes_defs, {});
  const size_t num_push_pop = state.range(0);

  for (auto _ : state) {
    for (int i = 0; i < num_push_pop; ++i) {
      queue.Push(i);
    }
    for (int i = 0; i < num_push_pop; ++i) {
      benchmark::DoNotOptimize(queue.Pop());
    }
  }
}

static void BM_PriorityReadyQueuePushPopHalf(benchmark::State& state) {
  std::vector<ThunkExecutor::NodeDef> nodes_defs(16);
  for (size_t i = 0; i < nodes_defs.size(); ++i) {
    nodes_defs[i].priority = i;
  }

  std::default_random_engine rng;
  absl::c_shuffle(nodes_defs, rng);

  ThunkExecutor::PriorityReadyQueue queue(nodes_defs, {});
  const size_t num_push_pop = state.range(0);

  for (auto _ : state) {
    for (int i = 0; i < num_push_pop; ++i) {
      queue.Push(i);
    }
    benchmark::DoNotOptimize(queue.PopHalf());
  }
}

#define BENCHMARK_READY_QUEUE(name) \
  BENCHMARK(name)                   \
      ->MeasureProcessCPUTime()     \
      ->Arg(1)                      \
      ->Arg(2)                      \
      ->Arg(4)                      \
      ->Arg(8)                      \
      ->Arg(16)

BENCHMARK_READY_QUEUE(BM_FifoReadyQueuePushPop);
BENCHMARK_READY_QUEUE(BM_FifoReadyQueuePushPopHalf);
BENCHMARK_READY_QUEUE(BM_PriorityReadyQueuePushPop);
BENCHMARK_READY_QUEUE(BM_PriorityReadyQueuePushPopHalf);

static void BM_CreateThunkExecutor(benchmark::State& state) {
  const size_t num_thunks = state.range(0);

  for (auto _ : state) {
    auto g = GenerateThunkSequence(/*num_elements=*/1024, num_thunks,
                                   {kNoResource}, false);
    CHECK_OK(ThunkExecutor::Create(std::move((*g)->sequence), OptionsForTest())
                 .status());
  }
}

static void BM_SequentialThunkExecutor(benchmark::State& state) {
  const size_t num_thunks = state.range(0);

  auto g = GenerateThunkSequence(/*num_elements=*/1024, num_thunks,
                                 /*shared_resource_use=*/{kAllResource, kToken},
                                 /*inject_errors=*/false)
               .value();
  auto e =
      ThunkExecutor::Create(std::move(g->sequence), OptionsForTest()).value();

  BufferAllocations allocations =
      CreateBufferAllocations(absl::MakeSpan(g->literals));
  Thunk::ExecuteParams params = {nullptr, &allocations};

  for (auto _ : state) {
    auto execute_event = e.Execute(params);
    tsl::BlockUntilReady(execute_event);
    CHECK(execute_event.IsConcrete());
  }
}

static void BM_SyncThunkExecutor(benchmark::State& state) {
  const size_t num_thunks = state.range(0);

  auto g = GenerateThunkSequence(/*num_elements=*/1024, num_thunks,
                                 /*shared_resource_use=*/{kNoResource},
                                 /*inject_errors=*/false);
  auto e = ThunkExecutor::Create(std::move((*g)->sequence), OptionsForTest());

  BufferAllocations allocations =
      CreateBufferAllocations(absl::MakeSpan((*g)->literals));
  Thunk::ExecuteParams params = {nullptr, &allocations};

  for (auto _ : state) {
    auto execute_event = e->Execute(params);
    tsl::BlockUntilReady(execute_event);
    CHECK(execute_event.IsConcrete());
  }
}

static void BM_AsyncThunkExecutor(benchmark::State& state) {
  const size_t num_thunks = state.range(0);

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "thunk-executor", 8);
  Eigen::ThreadPoolDevice device(thread_pool.AsEigenThreadPool(),
                                 thread_pool.NumThreads());

  auto g = GenerateThunkSequence(/*num_elements=*/1024, num_thunks,
                                 /*shared_resource_use=*/{kNoResource},
                                 /*inject_errors=*/false);
  auto e = ThunkExecutor::Create(std::move((*g)->sequence), OptionsForTest());

  BufferAllocations allocations =
      CreateBufferAllocations(absl::MakeSpan((*g)->literals));
  ThreadPoolTaskRunner task_runner(thread_pool.AsEigenThreadPool());

  Thunk::ExecuteParams params = {nullptr, &allocations, nullptr, &device,
                                 &task_runner};

  for (auto _ : state) {
    auto execute_event = e->Execute(params);
    tsl::BlockUntilReady(execute_event);
    CHECK(execute_event.IsConcrete());
  }
}

#define BENCHMARK_THUNK_EXECUTOR(name) \
  BENCHMARK(name)                      \
      ->MeasureProcessCPUTime()        \
      ->Arg(1)                         \
      ->Arg(2)                         \
      ->Arg(4)                         \
      ->Arg(8)                         \
      ->Arg(16)                        \
      ->Arg(32)                        \
      ->Arg(64)                        \
      ->Arg(128)                       \
      ->Arg(256)                       \
      ->Arg(512)

BENCHMARK_THUNK_EXECUTOR(BM_CreateThunkExecutor);
BENCHMARK_THUNK_EXECUTOR(BM_SequentialThunkExecutor);
BENCHMARK_THUNK_EXECUTOR(BM_SyncThunkExecutor);
BENCHMARK_THUNK_EXECUTOR(BM_AsyncThunkExecutor);

}  // namespace
}  // namespace xla::cpu
