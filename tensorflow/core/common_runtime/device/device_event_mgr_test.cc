/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/common_runtime/device/device_event_mgr.h"

#include <atomic>

#include "absl/synchronization/notification.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/tsl/framework/device_id.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

// Subclass EventMgr to access its private constructor.
class TEST_EventMgr : public EventMgr {
 public:
  TEST_EventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options)
      : EventMgr(se, gpu_options) {}
};

class TEST_EventMgrHelper {
 public:
  explicit TEST_EventMgrHelper(EventMgr* em) : em_(em) {
    // The polling loop can interfere with the measurements made here, and
    // isn't needed since the member PollEvents() always clears the queue.
    // The tested behavior is slightly different from what may occur in
    // ordinary execution.
    StopPollingLoop();
  }

  size_t queue_size() {
    mutex_lock l(em_->mu_);
    size_t n = 0;
    for (const auto& [stream, events_and_callbacks] : em_->callbacks_) {
      n += events_and_callbacks.size();
    }
    return n;
  }

  size_t free_size() {
    mutex_lock l(em_->mu_);
    return em_->free_events_.size();
  }

  void PollEvents() {
    while (queue_size() > 0) {
      EventMgr::ToFreeVector to_free;
      {
        mutex_lock l(em_->mu_);
        em_->PollEvents(nullptr, &to_free);
      }
      em_->FreeMemory(to_free);
    }
  }

  void StopPollingLoop() { return em_->StopPollingLoop(); }

  void StartPollingLoop() { return em_->StartPollingLoop(); }

 private:
  EventMgr* em_;
};

static std::atomic_int_fast64_t live_tensor_bytes(0);

// A TensorBuffer that counts live memory usage for testing
class TestTensorBuffer : public TensorBuffer {
 public:
  explicit TestTensorBuffer(size_t bytes)
      : TensorBuffer(nullptr), bytes_(bytes) {
    live_tensor_bytes += bytes_;
  }
  ~TestTensorBuffer() override { live_tensor_bytes -= bytes_; }

  size_t size() const override { return bytes_; }

  // Not used in this test
  TensorBuffer* root_buffer() override { return nullptr; }
  void FillAllocationDescription(AllocationDescription* arg) const override {}

 private:
  size_t bytes_;
};

namespace {

TEST(EventMgr, Empty) {
  auto stream_exec = se::GPUMachineManager()->ExecutorForDevice(0).value();
  TEST_EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, th.queue_size());
  EXPECT_EQ(0, th.free_size());
}

// Tests that WarnIfInCallback() triggers correctly.
TEST(EventMgr, WarnIfInCallback) {
  auto stream_exec = se::GPUMachineManager()->ExecutorForDevice(0).value();
  TEST_EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_exec->CreateStream());
  bool hit = false;
  th.StartPollingLoop();
  device_event_mgr::WarnIfInCallback([&hit] { hit = true; });
  EXPECT_FALSE(hit);
  absl::Notification note;
  em.ThenExecute(stream.get(), [&hit, &note]() {
    device_event_mgr::WarnIfInCallback([&hit, &note] {
      hit = true;
      note.Notify();
    });
  });
  note.WaitForNotification();
  EXPECT_TRUE(hit);
}
}  // namespace

// Provides access to private resources of BaseGPUDevice.
class GPUDeviceTestHelper {
 public:
  GPUDeviceTestHelper(size_t memory_limit, int pending_cap) {
    SessionOptions sops;
    device_ =
        DeviceFactory::NewDevice(DEVICE_GPU, sops, "/job:a/replica:0/task:0");
    gpu_.reset(reinterpret_cast<BaseGPUDevice*>(device_.release()));
    gpu_allocator_ = GPUProcessState::singleton()->GetGPUAllocator(
        GPUOptions(), tsl::TfDeviceId(0), memory_limit, /*peer_gpu_ids=*/{});
    host_allocator_ = GPUProcessState::singleton()->GetGpuHostAllocator(
        /*options=*/{}, /*numa_node=*/0);
  }

  BaseGPUDevice* gpu() { return gpu_.get(); }
  Allocator* gpu_allocator() { return gpu_allocator_; }
  Allocator* host_allocator() { return host_allocator_; }
  se::Stream* compute_stream() { return gpu_->stream_->compute; }
  se::Stream* h2d_stream() { return gpu_->stream_->host_to_device; }
  se::Stream* d2h_stream() { return gpu_->stream_->device_to_host; }
  se::Stream* d2d_stream() { return gpu_->stream_->device_to_device[0]; }
  EventMgr* event_mgr() { return gpu_->em_; }
  int pending_cap() { return gpu_->pending_cap_; }

 private:
  std::unique_ptr<Device> device_;
  std::unique_ptr<BaseGPUDevice> gpu_;
  Allocator* gpu_allocator_;
  Allocator* host_allocator_;
};

namespace {

// Class that can queue some GPU data transfers and simple kernels.
class EMBenchmarkHelper {
  GPUDeviceTestHelper* gpu_helper_;
  // We need one of these for each Add op in the chain.
  std::vector<std::unique_ptr<OpKernel>> add_kernels_;
  std::vector<OpKernelContext::Params*> add_params_;
  std::vector<std::unique_ptr<OpKernelContext>> add_contexts_;
  // The rest of these are one per chain.
  NodeDef add_node_def_;
  NodeDef id_node_def_;
  gtl::InlinedVector<TensorValue, 4> add_inputs_;
  std::vector<AllocatorAttributes> allocator_attrs_;
  gtl::InlinedVector<Tensor, 4> gpu_inputs_;
  gtl::InlinedVector<Tensor, 4> gpu_outputs_;
  gtl::InlinedVector<Tensor, 4> host_inputs_;
  gtl::InlinedVector<Tensor, 4> host_outputs_;

 public:
  // Length of tensors.  TODO(tucker): make this a variable parameter.
  static constexpr int kTDim = 1024;

  int num_ops() const { return add_kernels_.size(); }
  size_t tensor_size() const {
    return add_inputs_.empty() ? 0 : add_inputs_[0]->NumElements();
  }

  Tensor& host_outputs(int i) { return host_outputs_[i]; }
  Tensor& host_inputs(int i) { return host_inputs_[i]; }

  EMBenchmarkHelper(GPUDeviceTestHelper* h) : gpu_helper_(h) {}

  void ReInit(int num_ops, int tensor_size) {
    gpu_inputs_.clear();
    while (gpu_inputs_.size() < 2) {
      gpu_inputs_.push_back(Tensor(gpu_helper_->gpu_allocator(), DT_FLOAT,
                                   {tensor_size}, AllocationAttributes()));
    }
    gpu_outputs_.clear();
    while (gpu_outputs_.empty()) {
      gpu_outputs_.push_back(Tensor(gpu_helper_->gpu_allocator(), DT_FLOAT,
                                    {tensor_size}, AllocationAttributes()));
    }
    host_inputs_.clear();
    while (host_inputs_.size() < 2) {
      int instance_index = host_inputs_.size();
      host_inputs_.push_back(Tensor(gpu_helper_->host_allocator(), DT_FLOAT,
                                    {tensor_size}, AllocationAttributes()));
      for (int i = 0; i < tensor_size; ++i) {
        host_inputs_.back().flat<float>()(i) =
            i * (1.0 + (0.5 * instance_index));
      }
    }
    host_outputs_.clear();
    while (host_outputs_.empty()) {
      host_outputs_.push_back(Tensor(gpu_helper_->host_allocator(), DT_FLOAT,
                                     {tensor_size}, AllocationAttributes()));
      for (int i = 0; i < tensor_size; ++i) {
        host_outputs_.back().flat<float>()(i) = -1;
      }
    }
    add_kernels_.clear();
    add_params_.clear();
    while (add_kernels_.size() < num_ops) {
      MakeAddOp();
    }
  }

  std::unique_ptr<OpKernel> GetOpKernel(const NodeDef& node_def,
                                        Status* status) {
    return CreateOpKernel("GPU", gpu_helper_->gpu(),
                          gpu_helper_->gpu_allocator(), node_def,
                          TF_GRAPH_DEF_VERSION, status);
  }

  void MakeAddOp() {
    if (add_kernels_.empty()) {
      TF_ASSERT_OK(NodeDefBuilder("add_op", "Add")
                       .Input(FakeInput(DT_FLOAT))
                       .Input(FakeInput(DT_FLOAT))
                       .Device("/job:a/replica:0/task:0/GPU:0")
                       .Finalize(&add_node_def_));
    }
    Status status;
    add_kernels_.emplace_back(GetOpKernel(add_node_def_, &status));
    TF_ASSERT_OK(status);
    add_params_.push_back(new OpKernelContext::Params);
    PrepOpKernel(add_params_.back(), add_kernels_.back().get());
  }

  void SetOutputAttrs(OpKernelContext::Params* params,
                      std::vector<AllocatorAttributes>* attrs) {
    attrs->clear();
    for (int index = 0; index < params->op_kernel->num_outputs(); index++) {
      AllocatorAttributes attr;
      const bool on_host =
          (params->op_kernel->output_memory_types()[index] == HOST_MEMORY);
      attr.set_on_host(on_host);
      attrs->push_back(attr);
    }
    params->output_attr_array = attrs->data();
    params->forward_from_array = {};
  }

  void PrepOpKernel(OpKernelContext::Params* params, OpKernel* kernel) {
    // This mimics what happens in ExecutorState::Process to run
    // a single graph node.
    params->step_id = 1;
    params->device = gpu_helper_->gpu();
    params->log_memory = false;
    params->rendezvous = nullptr;
    params->collective_executor = nullptr;
    params->session_state = nullptr;  // ???
    params->session_handle = "session_handle";
    params->tensor_store = nullptr;
    params->cancellation_manager = nullptr;

    params->call_frame = nullptr;
    params->function_library = nullptr;
    params->runner = nullptr;
    params->graph_collector = nullptr;

    params->step_container = nullptr;
    params->slice_reader_cache = nullptr;
    params->resource_manager = gpu_helper_->gpu()->resource_manager();

    params->stats_collector = nullptr;
    params->inc_num_deferred_ops_function = nullptr;
    params->dec_num_deferred_ops_function = nullptr;

    params->op_device_context = nullptr;
    params->track_allocations = false;
    params->op_kernel = kernel;
    params->frame_iter = FrameAndIter(0, 0);
    params->is_input_dead = false;

    if (add_inputs_.empty()) {
      add_inputs_.resize(2);
      add_inputs_[0] = TensorValue(&gpu_inputs_[0]);
      add_inputs_[1] = TensorValue(&gpu_inputs_[1]);
    }
    params->inputs = add_inputs_;
    SetOutputAttrs(params, &allocator_attrs_);
  }

  struct TimeSet {
    int iter = 0;
    int64_t start = 0;
    int64_t copy_done = 0;
    int64_t compute_done = 0;
    int64_t final_copy = 0;
    int64_t all_done = 0;
  };

  // Display sampled iteration times giving the approximate breakdown
  // within iterations and overall curve.
  void DisplayTimes(std::vector<TimeSet>* times) {
    LOG(INFO) << "Summarize set of " << times->size() << " iters";
    for (auto& ts : *times) {
      ts.final_copy = ts.all_done - ts.compute_done;
      ts.compute_done = ts.compute_done - ts.copy_done;
      ts.copy_done = ts.copy_done - ts.start;
      ts.all_done = ts.all_done - ts.start;
    }
    struct TSSort {
      bool operator()(const TimeSet& a, const TimeSet& b) {
        return a.all_done < b.all_done;
      }
    };
    std::sort(times->begin(), times->end(), TSSort());
    int64_t last_time = 0;
    // Display first, last and every > 5% change.
    for (int i = 0; i < times->size(); ++i) {
      if (i == (times->size() - 1) ||
          (times->at(i).all_done >= (1.05 * last_time))) {
        LOG(INFO) << "rank " << i << " iter: " << times->at(i).iter
                  << " copy: " << times->at(i).copy_done
                  << " compute: " << times->at(i).compute_done
                  << " copy back: " << times->at(i).final_copy
                  << " sum: " << times->at(i).all_done;
        last_time = times->at(i).all_done;
      }
    }
  }

  // Queue one work unit on the GPU as follows:
  // 1. Copy 2 input tensors from CPU to GPU using h2d stream.
  // 2. Instruct compute stream to wait on h2d stream.
  // 3. Queue a sequence of Add ops on the compute stream, all using
  //    the same input tensors, allocating their own output tensors.
  // 4. Instruct d2h stream to wait on the compute stream.
  // 5. Copy final output tensor back to the CPU.
  // 6. Instruct the EventMgr to execute callback when the final tensor
  //    copy completes.
  // If event_after_add == true then additionally instruct the EventMgr
  //    to execute the callback after each Add completes.
  // The optional times parameter is used for gathering detailed timing
  // data.
  void DoAddChain(int adds_per_copy, int rounds, bool event_after_add,
                  std::function<void()> callback, std::vector<TimeSet>* times) {
    // Take an extra ref on the inputs so that the add doesn't compute in place.
    Tensor alias0(gpu_inputs_[0]);
    Tensor alias1(gpu_inputs_[1]);
    for (int r = 0; r < rounds; ++r) {
      if (times) {
        times->at(r).iter = r;
        times->at(r).start = Env::Default()->NowMicros();
      }
      TF_ASSERT_OK(
          gpu_helper_->h2d_stream()->WaitFor(gpu_helper_->compute_stream()));
      // Begin by copying the input values from CPU to GPU.
      const int64_t src_bytes = host_inputs_[0].TotalBytes();
      se::DeviceMemoryBase gpu_dst_ptr0(DMAHelper::base(&gpu_inputs_[0]),
                                        src_bytes);
      TF_ASSERT_OK(gpu_helper_->h2d_stream()->Memcpy(
          &gpu_dst_ptr0, DMAHelper::base(&host_inputs_[0]), src_bytes));
      se::DeviceMemoryBase gpu_dst_ptr1(DMAHelper::base(&gpu_inputs_[1]),
                                        src_bytes);
      TF_ASSERT_OK(gpu_helper_->h2d_stream()->Memcpy(
          &gpu_dst_ptr1, DMAHelper::base(&host_inputs_[1]), src_bytes));
      TF_ASSERT_OK(
          gpu_helper_->compute_stream()->WaitFor(gpu_helper_->h2d_stream()));
      if (times) {
        gpu_helper_->event_mgr()->ThenExecute(
            gpu_helper_->compute_stream(), [times, r]() {
              times->at(r).copy_done = Env::Default()->NowMicros();
            });
      }
      std::unique_ptr<OpKernelContext> ctx;
      for (int apc = 0; apc < adds_per_copy; ++apc) {
        ctx.reset(new OpKernelContext(add_params_[apc], 1));
        gpu_helper_->gpu()->Compute(add_kernels_[apc].get(), ctx.get());
        TF_ASSERT_OK(ctx->status());
        if (event_after_add) {
          gpu_helper_->event_mgr()->ThenExecute(gpu_helper_->compute_stream(),
                                                callback);
        }
      }
      // Finish by copying output back to CPU.
      if (times) {
        gpu_helper_->event_mgr()->ThenExecute(
            gpu_helper_->compute_stream(), [times, r]() {
              times->at(r).compute_done = Env::Default()->NowMicros();
            });
      }
      TF_ASSERT_OK(
          gpu_helper_->d2h_stream()->WaitFor(gpu_helper_->compute_stream()));
      const int64_t return_bytes = ctx->mutable_output(0)->TotalBytes();
      se::DeviceMemoryBase gpu_src_ptr(DMAHelper::base(ctx->mutable_output(0)),
                                       return_bytes);
      TF_ASSERT_OK(gpu_helper_->d2h_stream()->Memcpy(
          DMAHelper::base(&host_outputs_[0]), gpu_src_ptr, return_bytes));
      gpu_helper_->event_mgr()->ThenExecute(gpu_helper_->d2h_stream(),
                                            callback);
      if (times) {
        gpu_helper_->event_mgr()->ThenExecute(
            gpu_helper_->d2h_stream(), [times, r]() {
              times->at(r).all_done = Env::Default()->NowMicros();
            });
      }
    }
  }
};

static void BM_no_ops(::testing::benchmark::State& state) {
  const int threads = state.range(0);
  const int iters = state.max_iterations;

  auto stream_exec = se::GPUMachineManager()->ExecutorForDevice(0).value();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_exec->CreateStream());
  TEST_EventMgr em(stream_exec, GPUOptions());

  auto benchmark_exec = [&]() {
    std::atomic<int> counter;
    counter.store(0, std::memory_order_seq_cst);
    se::Stream* stream_ptr = stream.get();
    auto runner = [&em, &counter, stream_ptr, iters]() {
      auto callback = [&counter]() { counter.fetch_add(1); };
      for (int i = 0; i < iters; ++i) {
        em.ThenExecute(stream_ptr, callback);
      }
    };
    for (int t = 0; t < threads; ++t) {
      Env::Default()->SchedClosure(runner);
    }
    int expected = iters * threads;
    while (counter < expected) {
      Env::Default()->SleepForMicroseconds(1);
    }
  };

#ifdef PLATFORM_GOOGLE

  // The timer starts automatically
  while (state.KeepRunningBatch(state.max_iterations)) {
    benchmark_exec();
  }
#else
  // The tensorflow's own implementation of the benchmark does not support
  // running-batch (yet), therefore we had to use the Stop/StartTimer.
  // FIXME: Remove this if-def once we switched all tensorflow's benchmarks to
  // using the OSS benchmark library.

  state.ResumeTiming();
  benchmark_exec();
  state.PauseTiming();
#endif
}
BENCHMARK(BM_no_ops)->UseRealTime()->Arg(4)->Arg(8)->Arg(32);

// Benchmark functions are defined at top level.  In order to provide a real,
// persistent GPUDevice to the following function it also needs to be at top
// level.  But then we can't clean it up without a cuda runtime error, so we
// just leak it.
GPUDeviceTestHelper* gpu_helper = nullptr;
EMBenchmarkHelper* bm_helper = nullptr;
mutex helper_mu;

#ifdef PLATFORM_GOOGLE
static void BM_chain_ops(::testing::benchmark::State& state, int tensor_size,
                         int adds_per_round, bool event_after_add,
                         int pending_cap) {
#else
static void BM_chain_ops(::testing::benchmark::State& state, int tensor_size,
                         int adds_per_round, bool event_after_add,
                         int pending_cap, int threads) {
#endif
  const int iters = state.max_iterations;
  {
    mutex_lock l(helper_mu);
    if (gpu_helper && gpu_helper->pending_cap() != pending_cap) {
      delete bm_helper;
      bm_helper = nullptr;
      delete gpu_helper;
      gpu_helper = nullptr;
    }
    if (!gpu_helper) {
      gpu_helper = new GPUDeviceTestHelper(1 << 24, pending_cap);
      bm_helper = new EMBenchmarkHelper(gpu_helper);
    }
    if (bm_helper->num_ops() != adds_per_round ||
        bm_helper->tensor_size() != tensor_size) {
      bm_helper->ReInit(adds_per_round, tensor_size);
    }
  }
  std::vector<EMBenchmarkHelper::TimeSet> times;
  std::vector<EMBenchmarkHelper::TimeSet>* time_ptr = nullptr;
  if (VLOG_IS_ON(1)) {
    times.resize(iters);
    time_ptr = &times;
  }
  std::atomic<int> counter;
  counter.store(0, std::memory_order_seq_cst);
  auto callback = [&counter]() { counter.fetch_add(1); };
  // First iter is always slow, so do one prior to the timed loop.
  int expected = 1 + (event_after_add ? adds_per_round : 0);
  bm_helper->DoAddChain(adds_per_round, 1, event_after_add, callback, nullptr);
  while (counter < expected) {
    Env::Default()->SleepForMicroseconds(1);
  }
  counter = 0;

#ifdef PLATFORM_GOOGLE
  while (state.KeepRunningBatch(state.max_iterations)) {
    expected = iters * (1 + (event_after_add ? adds_per_round : 0));
    bm_helper->DoAddChain(adds_per_round, iters, event_after_add, callback,
                          time_ptr);
    while (counter < expected) {
      Env::Default()->SleepForMicroseconds(1);
    }
  }
#else
  state.ResumeTiming();
  expected = threads * iters * (1 + (event_after_add ? adds_per_round : 0));
  for (int i = 0; i < threads; ++i) {
    Env::Default()->SchedClosure(
        [callback, iters, adds_per_round, event_after_add, time_ptr]() {
          bm_helper->DoAddChain(adds_per_round, iters, event_after_add,
                                callback, time_ptr);
        });
  }
  while (counter < expected) {
    Env::Default()->SleepForMicroseconds(1);
  }
  state.PauseTiming();
#endif
  VLOG(1) << "counter = " << counter << " post_execute Output: "
          << bm_helper->host_outputs(0).SummarizeValue(64);
  if (time_ptr) bm_helper->DisplayTimes(time_ptr);
}

#ifdef PLATFORM_GOOGLE
static void BM_chain_1024_1_false(::testing::benchmark::State& state) {
  BM_chain_ops(state, 1024, 1, false, 0);
}

static void BM_chain_1024_1_true(::testing::benchmark::State& state) {
  BM_chain_ops(state, 1024, 1, true, 0);
}

static void BM_chain_1024_10_false(::testing::benchmark::State& state) {
  BM_chain_ops(state, 1024, 10, false, 0);
}

static void BM_chain_1024_10_true(::testing::benchmark::State& state) {
  BM_chain_ops(state, 1024, 10, true, 0);
}

static void BM_chain_1024_100_false(::testing::benchmark::State& state) {
  BM_chain_ops(state, 1024, 100, false, 0);
}

static void BM_chain_1024_100_true(::testing::benchmark::State& state) {
  BM_chain_ops(state, 1024, 100, true, 0);
}

static void BM_chain_1M_1_false(::testing::benchmark::State& state) {
  BM_chain_ops(state, 1 << 20, 1, false, 0);
}

static void BM_chain_1M_1_true(::testing::benchmark::State& state) {
  BM_chain_ops(state, 1 << 20, 1, true, 0);
}

static void BM_chain_1M_10_false(::testing::benchmark::State& state) {
  BM_chain_ops(state, 1 << 20, 10, false, 0);
}

static void BM_chain_1M_10_true(::testing::benchmark::State& state) {
  BM_chain_ops(state, 1 << 20, 10, true, 0);
}

static void BM_chain_1M_100_false(::testing::benchmark::State& state) {
  BM_chain_ops(state, 1 << 20, 100, false, 0);
}

static void BM_chain_1M_100_true(::testing::benchmark::State& state) {
  BM_chain_ops(state, 1 << 20, 100, true, 0);
}

BENCHMARK(BM_chain_1024_1_false)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1024_1_true)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1024_1_false)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1024_1_true)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1024_1_false)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1024_1_true)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1024_10_false)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1024_10_true)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1024_10_false)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1024_10_true)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1024_100_false)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1024_100_true)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1024_100_false)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1024_100_true)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1024_100_false)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1024_100_true)->UseRealTime()->Threads(8);

BENCHMARK(BM_chain_1M_1_false)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1M_1_true)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1M_1_false)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1M_1_true)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1M_1_false)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1M_1_true)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1M_10_false)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1M_10_true)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1M_10_false)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1M_10_true)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1M_100_false)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1M_100_true)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1M_100_false)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1M_100_true)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1M_100_false)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1M_100_true)->UseRealTime()->Threads(8);
#else
static void BM_chain_1024_1_false(::testing::benchmark::State& state) {
  const int threads = state.range(0);
  BM_chain_ops(state, 1024, 1, false, 0, threads);
}

static void BM_chain_1024_1_true(::testing::benchmark::State& state) {
  const int threads = state.range(0);
  BM_chain_ops(state, 1024, 1, true, 0, threads);
}

static void BM_chain_1024_10_false(::testing::benchmark::State& state) {
  const int threads = state.range(0);
  BM_chain_ops(state, 1024, 10, false, 0, threads);
}

static void BM_chain_1024_10_true(::testing::benchmark::State& state) {
  const int threads = state.range(0);
  BM_chain_ops(state, 1024, 10, true, 0, threads);
}

static void BM_chain_1024_100_false(::testing::benchmark::State& state) {
  const int threads = state.range(0);
  BM_chain_ops(state, 1024, 100, false, 0, threads);
}

static void BM_chain_1024_100_true(::testing::benchmark::State& state) {
  const int threads = state.range(0);
  BM_chain_ops(state, 1024, 100, true, 0, threads);
}

static void BM_chain_1M_1_false(::testing::benchmark::State& state) {
  const int threads = state.range(0);
  BM_chain_ops(state, 1 << 20, 1, false, 0, threads);
}

static void BM_chain_1M_1_true(::testing::benchmark::State& state) {
  const int threads = state.range(0);
  BM_chain_ops(state, 1 << 20, 1, true, 0, threads);
}

static void BM_chain_1M_10_false(::testing::benchmark::State& state) {
  const int threads = state.range(0);
  BM_chain_ops(state, 1 << 20, 10, false, 0, threads);
}

static void BM_chain_1M_10_true(::testing::benchmark::State& state) {
  const int threads = state.range(0);
  BM_chain_ops(state, 1 << 20, 10, true, 0, threads);
}

static void BM_chain_1M_100_false(::testing::benchmark::State& state) {
  const int threads = state.range(0);
  BM_chain_ops(state, 1 << 20, 100, false, 0, threads);
}

static void BM_chain_1M_100_true(::testing::benchmark::State& state) {
  const int threads = state.range(0);
  BM_chain_ops(state, 1 << 20, 100, true, 0, threads);
}

BENCHMARK(BM_chain_1024_1_false)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1024_1_true)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1024_1_false)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1024_1_true)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1024_1_false)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1024_1_true)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1024_10_false)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1024_10_true)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1024_10_false)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1024_10_true)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1024_100_false)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1024_100_true)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1024_100_false)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1024_100_true)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1024_100_false)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1024_100_true)->UseRealTime()->Arg(8);

BENCHMARK(BM_chain_1M_1_false)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1M_1_true)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1M_1_false)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1M_1_true)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1M_1_false)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1M_1_true)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1M_10_false)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1M_10_true)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1M_10_false)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1M_10_true)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1M_100_false)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1M_100_true)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1M_100_false)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1M_100_true)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1M_100_false)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1M_100_true)->UseRealTime()->Arg(8);
#endif
}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
