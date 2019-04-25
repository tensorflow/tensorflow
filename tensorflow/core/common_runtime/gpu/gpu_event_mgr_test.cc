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

#if GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"

#include <atomic>
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

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
    return em_->used_events_.size();
  }

  size_t free_size() {
    mutex_lock l(em_->mu_);
    return em_->free_events_.size();
  }

  void QueueTensors(se::Stream* stream, TensorReferenceVector* tensors) {
    mutex_lock l(em_->mu_);
    em_->QueueTensors(stream, tensors);
  }

  void PollEvents(bool is_dedicated_poller) {
    while (queue_size() > 0) {
      // For ordinary tensor frees, this function
      // should synchronously harvest all complete
      // events and execute the corresponding memory frees.
      EventMgr::ToFreeVector to_free;
      {
        mutex_lock l(em_->mu_);
        em_->PollEvents(is_dedicated_poller, &to_free);
      }
      em_->FreeMemory(to_free);
    }
  }

  void StopPollingLoop() { em_->StopPollingLoop(); }

  void StartPollingLoop() { em_->StartPollingLoop(); }

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
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, th.queue_size());
  EXPECT_EQ(0, th.free_size());
}

static void AddTensorReference(TensorReferenceVector* v, int64 size) {
  TestTensorBuffer* buf = new TestTensorBuffer(size);
  v->push_back(TensorReference(buf));
  buf->Unref();
}

// Delaying polling until after several enqueings should grow the
// total number of allocated events.  Once we have enough events for
// the max simultaneously pending, we should not allocate any more.
TEST(EventMgr, DelayedPolling) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, th.queue_size());
  TensorReferenceVector* v = nullptr;
  std::unique_ptr<se::Stream> stream(new se::Stream(stream_exec));
  CHECK(stream);
  stream->Init();
  for (int i = 0; i < 5; ++i) {
    v = new TensorReferenceVector;
    AddTensorReference(v, 100 * 1048576);
    th.QueueTensors(stream.get(), v);
    EXPECT_EQ(i + 1, th.queue_size());
    EXPECT_EQ(0, th.free_size());
  }
  th.PollEvents(false);
  EXPECT_EQ(0, th.queue_size());
  EXPECT_EQ(5, th.free_size());
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 5; ++i) {
      v = new TensorReferenceVector;
      AddTensorReference(v, 100 * 1048576);
      th.QueueTensors(stream.get(), v);
      EXPECT_EQ(i + 1, th.queue_size());
      EXPECT_EQ(4 - i, th.free_size());
    }
    th.PollEvents(false);
    EXPECT_EQ(0, th.queue_size());
    EXPECT_EQ(5, th.free_size());
  }
}

TEST(EventMgr, FlushLargeTensorImmediately) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, live_tensor_bytes);
  std::unique_ptr<se::Stream> stream(new se::Stream(stream_exec));
  CHECK(stream);
  stream->Init();
  for (int i = 0; i < 5; ++i) {
    TensorReferenceVector v;
    AddTensorReference(&v, 100 * 1048576);
    em.ThenDeleteTensors(stream.get(), v);
    th.PollEvents(false);  // Ensure things get registered to be freed by Poll
    EXPECT_EQ(0, live_tensor_bytes);
  }
}

TEST(EventMgr, ManySmallTensorsFlushedImmediately) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, live_tensor_bytes);
  std::unique_ptr<se::Stream> stream(new se::Stream(stream_exec));
  CHECK(stream);
  stream->Init();
  for (int i = 0; i < 5; ++i) {
    TensorReferenceVector v;
    for (int i = 0; i < 1000; i++) {
      AddTensorReference(&v, 100 * 1024);
    }
    em.ThenDeleteTensors(stream.get(), v);
    th.PollEvents(false);  // Harvest the tensors ready to be freed.
    EXPECT_EQ(0, live_tensor_bytes);
  }
}

TEST(EventMgr, StreamSwitchingFlushesImmediately) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, live_tensor_bytes);
  std::unique_ptr<se::Stream> stream1(new se::Stream(stream_exec));
  std::unique_ptr<se::Stream> stream2(new se::Stream(stream_exec));
  stream1->Init();
  stream2->Init();
  TensorReferenceVector v1;
  AddTensorReference(&v1, 1024);
  em.ThenDeleteTensors(stream1.get(), v1);

  TensorReferenceVector v2;
  AddTensorReference(&v2, 1024);
  int64 initial_live_bytes = live_tensor_bytes;
  em.ThenDeleteTensors(stream2.get(), v2);
  th.PollEvents(false);  // Ensure things get registered to be freed by Poll
  // Different stream should cause first tensor to get deleted
  EXPECT_GT(initial_live_bytes, live_tensor_bytes);
}

TEST(EventMgr, ManySmallTensorsSeparateCallsFlushed) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, live_tensor_bytes);
  std::unique_ptr<se::Stream> stream(new se::Stream(stream_exec));
  CHECK(stream);
  stream->Init();
  for (int i = 0; i < 5; ++i) {
    for (int i = 0; i < 1000; i++) {
      TensorReferenceVector v;
      AddTensorReference(&v, 100 * 1024);
      em.ThenDeleteTensors(stream.get(), v);
    }
    th.PollEvents(false);  // Ensure things get registered to be freed by Poll
    // Some of the tensors at least should be flushed
    EXPECT_GT(1000 * 100 * 1024, live_tensor_bytes);
  }
}

// Deleting the EventMgr when events are still pending should shut
// down gracefully.
TEST(EventMgr, NonEmptyShutdown) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, th.queue_size());
  EXPECT_EQ(0, th.free_size());
  std::unique_ptr<se::Stream> stream(new se::Stream(stream_exec));
  CHECK(stream);
  stream->Init();
  for (int i = 0; i < 5; ++i) {
    TensorReferenceVector* v = new TensorReferenceVector;
    AddTensorReference(v, 100 * 1048576);
    th.QueueTensors(stream.get(), v);
    EXPECT_EQ(1 + i, th.queue_size());
    EXPECT_EQ(0, th.free_size());
  }
}

// Tests that WarnIfInCallback() triggers correctly.
TEST(EventMgr, WarnIfInCallback) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  std::unique_ptr<se::Stream> stream(new se::Stream(stream_exec));
  CHECK(stream);
  stream->Init();
  bool hit = false;
  gpu_event_mgr::WarnIfInCallback([&hit] { hit = true; });
  EXPECT_FALSE(hit);
  Notification note;
  em.ThenExecute(stream.get(), [&hit, &note]() {
    gpu_event_mgr::WarnIfInCallback([&hit, &note] {
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
  GPUDeviceTestHelper(size_t memory_limit) {
    SessionOptions sops;
    device_ =
        DeviceFactory::NewDevice(DEVICE_GPU, sops, "/job:a/replica:0/task:0");
    gpu_.reset(reinterpret_cast<BaseGPUDevice*>(device_.release()));
    gpu_allocator_ = GPUProcessState::singleton()->GetGPUAllocator(
        GPUOptions(), TfGpuId(0), memory_limit);
    host_allocator_ = GPUProcessState::singleton()->GetGpuHostAllocator(0);
  }

  BaseGPUDevice* gpu() { return gpu_.get(); }
  Allocator* gpu_allocator() { return gpu_allocator_; }
  Allocator* host_allocator() { return host_allocator_; }
  se::Stream* compute_stream() { return gpu_->streams_[0]->compute; }
  se::Stream* h2d_stream() { return gpu_->streams_[0]->host_to_device; }
  se::Stream* d2h_stream() { return gpu_->streams_[0]->device_to_host; }
  se::Stream* d2d_stream() { return gpu_->streams_[0]->device_to_device[0]; }
  EventMgr* event_mgr() { return gpu_->em_.get(); }

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
  static const int kTDim = 1024;

  int num_ops() const { return add_kernels_.size(); }
  size_t tensor_size() const {
    return add_inputs_.empty() ? 0 : add_inputs_[0]->NumElements();
  }

  Tensor& host_outputs(int i) { return host_outputs_[i]; }
  Tensor& host_inputs(int i) { return host_inputs_[i]; }

  EMBenchmarkHelper(GPUDeviceTestHelper* h) : gpu_helper_(h) {}

  void ReInit(int num_ops) {
    gpu_inputs_.clear();
    while (gpu_inputs_.size() < 2) {
      gpu_inputs_.push_back(Tensor(gpu_helper_->gpu_allocator(), DT_FLOAT,
                                   {kTDim}, AllocationAttributes()));
    }
    gpu_outputs_.clear();
    while (gpu_outputs_.size() < 1) {
      gpu_outputs_.push_back(Tensor(gpu_helper_->gpu_allocator(), DT_FLOAT,
                                    {kTDim}, AllocationAttributes()));
    }
    host_inputs_.clear();
    while (host_inputs_.size() < 2) {
      int instance_index = host_inputs_.size();
      host_inputs_.push_back(Tensor(gpu_helper_->host_allocator(), DT_FLOAT,
                                    {kTDim}, AllocationAttributes()));
      for (int i = 0; i < kTDim; ++i) {
        host_inputs_.back().flat<float>()(i) =
            i * (1.0 + (0.5 * instance_index));
      }
    }
    host_outputs_.clear();
    while (host_outputs_.size() < 1) {
      host_outputs_.push_back(Tensor(gpu_helper_->host_allocator(), DT_FLOAT,
                                     {kTDim}, AllocationAttributes()));
      for (int i = 0; i < kTDim; ++i) {
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
    params->output_attr_array = gtl::vector_as_array(attrs);
    params->forward_from_array = {};
  }

  void PrepOpKernel(OpKernelContext::Params* params, OpKernel* kernel) {
    // This mimics what happens in ExecutorState::Process to run
    // a single graph node.
    params->step_id = 1;
    params->device = gpu_helper_->gpu();
    params->log_memory = false;
    params->record_tensor_accesses = false;
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
    params->input_device_contexts = nullptr;
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
    params->inputs = &add_inputs_;
    params->input_alloc_attrs = nullptr;
    SetOutputAttrs(params, &allocator_attrs_);
  }

  struct TimeSet {
    int iter = 0;
    int64 start = 0;
    int64 copy_done = 0;
    int64 compute_done = 0;
    int64 final_copy = 0;
    int64 all_done = 0;
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
    int64 last_time = 0;
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
      gpu_helper_->h2d_stream()->ThenWaitFor(gpu_helper_->compute_stream());
      // Begin by copying the input values from CPU to GPU.
      const int64 src_bytes = host_inputs_[0].TotalBytes();
      se::DeviceMemoryBase gpu_dst_ptr0(DMAHelper::base(&gpu_inputs_[0]),
                                        src_bytes);
      gpu_helper_->h2d_stream()->ThenMemcpy(
          &gpu_dst_ptr0, DMAHelper::base(&host_inputs_[0]), src_bytes);
      se::DeviceMemoryBase gpu_dst_ptr1(DMAHelper::base(&gpu_inputs_[1]),
                                        src_bytes);
      gpu_helper_->h2d_stream()->ThenMemcpy(
          &gpu_dst_ptr1, DMAHelper::base(&host_inputs_[1]), src_bytes);
      gpu_helper_->compute_stream()->ThenWaitFor(gpu_helper_->h2d_stream());
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
      gpu_helper_->d2h_stream()->ThenWaitFor(gpu_helper_->compute_stream());
      const int64 return_bytes = ctx->mutable_output(0)->TotalBytes();
      se::DeviceMemoryBase gpu_src_ptr(DMAHelper::base(ctx->mutable_output(0)),
                                       return_bytes);
      gpu_helper_->d2h_stream()->ThenMemcpy(DMAHelper::base(&host_outputs_[0]),
                                            gpu_src_ptr, return_bytes);
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

static void BM_no_ops(int iters, int threads) {
  testing::StopTiming();
#ifdef PLATFORM_GOOGLE
  BenchmarkUseRealTime();
#else
  testing::UseRealTime();
#endif  // PLATFORM_GOOGLE
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  std::unique_ptr<se::Stream> stream(new se::Stream(stream_exec));
  CHECK(stream);
  stream->Init();
  EventMgr em(stream_exec, GPUOptions());  //, stream.get());
  testing::StartTiming();
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
}
BENCHMARK(BM_no_ops)->Arg(4);
BENCHMARK(BM_no_ops)->Arg(8);
BENCHMARK(BM_no_ops)->Arg(32);

// Benchmark functions are defined at top level.  In order to provide a real,
// persistent GPUDevice to the following function it also needs to be at top
// level.  But then we can't clean it up without a cuda runtime error, so we
// just leak it.
GPUDeviceTestHelper* gpu_helper = nullptr;
EMBenchmarkHelper* bm_helper = nullptr;
mutex helper_mu;

#ifdef PLATFORM_GOOGLE
static void BM_chain_ops(int iters, int adds_per_round, bool event_after_add) {
#else
static void BM_chain_ops(int iters, int adds_per_round, bool event_after_add,
                         int threads) {
#endif
  testing::StopTiming();
#ifdef PLATFORM_GOOGLE
  BenchmarkUseRealTime();
#else
  testing::UseRealTime();
#endif  // PLATFORM_GOOGLE
  {
    mutex_lock l(helper_mu);
    if (!gpu_helper) {
      gpu_helper = new GPUDeviceTestHelper(1 << 20);
      bm_helper = new EMBenchmarkHelper(gpu_helper);
    }
    if (bm_helper->num_ops() != adds_per_round) {
      bm_helper->ReInit(adds_per_round);
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
  while (counter < 1) {
    Env::Default()->SleepForMicroseconds(1);
  }
  counter = 0;
  testing::StartTiming();
#ifdef PLATFORM_GOOGLE
  expected = iters * (1 + (event_after_add ? adds_per_round : 0));
  bm_helper->DoAddChain(adds_per_round, iters, event_after_add, callback,
                        time_ptr);
#else
  expected = threads * iters * (1 + (event_after_add ? adds_per_round : 0));
  for (int i = 0; i < threads; ++i) {
    Env::Default()->SchedClosure(
        [callback, iters, adds_per_round, event_after_add, time_ptr]() {
          bm_helper->DoAddChain(adds_per_round, iters, event_after_add,
                                callback, time_ptr);
        });
  }
#endif
  while (counter < expected) {
    Env::Default()->SleepForMicroseconds(1);
  }
  testing::StopTiming();
  VLOG(1) << "counter = " << counter << " post_execute Output: "
          << bm_helper->host_outputs(0).SummarizeValue(64);
  if (time_ptr) bm_helper->DisplayTimes(time_ptr);
}

#ifdef PLATFORM_GOOGLE
static void BM_chain_1_false(int iters) { BM_chain_ops(iters, 1, false); }

static void BM_chain_1_true(int iters) { BM_chain_ops(iters, 1, true); }

static void BM_chain_10_false(int iters) { BM_chain_ops(iters, 10, false); }

static void BM_chain_10_true(int iters) { BM_chain_ops(iters, 10, true); }

static void BM_chain_100_false(int iters) { BM_chain_ops(iters, 100, false); }

static void BM_chain_100_true(int iters) { BM_chain_ops(iters, 100, true); }

BENCHMARK(BM_chain_1_false)->Threads(1);
BENCHMARK(BM_chain_1_true)->Threads(1);
BENCHMARK(BM_chain_1_false)->Threads(2);
BENCHMARK(BM_chain_1_true)->Threads(2);
BENCHMARK(BM_chain_1_false)->Threads(8);
BENCHMARK(BM_chain_1_true)->Threads(8);
BENCHMARK(BM_chain_10_false)->Threads(1);
BENCHMARK(BM_chain_10_true)->Threads(1);
BENCHMARK(BM_chain_10_false)->Threads(8);
BENCHMARK(BM_chain_10_true)->Threads(8);
BENCHMARK(BM_chain_100_false)->Threads(1);
BENCHMARK(BM_chain_100_true)->Threads(1);
BENCHMARK(BM_chain_100_false)->Threads(8);
BENCHMARK(BM_chain_100_true)->Threads(8);
#else
static void BM_chain_1_false(int iters, int threads) {
  BM_chain_ops(iters, 1, false, threads);
}

static void BM_chain_1_true(int iters, int threads) {
  BM_chain_ops(iters, 1, true, threads);
}

static void BM_chain_10_false(int iters, int threads) {
  BM_chain_ops(iters, 10, false, threads);
}

static void BM_chain_10_true(int iters, int threads) {
  BM_chain_ops(iters, 10, true, threads);
}

static void BM_chain_100_false(int iters, int threads) {
  BM_chain_ops(iters, 100, false, threads);
}

static void BM_chain_100_true(int iters, int threads) {
  BM_chain_ops(iters, 100, true, threads);
}

BENCHMARK(BM_chain_1_false)->Arg(1);
BENCHMARK(BM_chain_1_true)->Arg(1);
BENCHMARK(BM_chain_1_false)->Arg(2);
BENCHMARK(BM_chain_1_true)->Arg(2);
BENCHMARK(BM_chain_1_false)->Arg(8);
BENCHMARK(BM_chain_1_true)->Arg(8);
BENCHMARK(BM_chain_10_false)->Arg(1);
BENCHMARK(BM_chain_10_true)->Arg(1);
BENCHMARK(BM_chain_10_false)->Arg(8);
BENCHMARK(BM_chain_10_true)->Arg(8);
BENCHMARK(BM_chain_100_false)->Arg(1);
BENCHMARK(BM_chain_100_true)->Arg(1);
BENCHMARK(BM_chain_100_false)->Arg(8);
BENCHMARK(BM_chain_100_true)->Arg(8);
#endif
}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
