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

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/threadpool_interface.h"

namespace tensorflow {
namespace {

// Simulates ThreadPoolInterface implementations (such as FiberPool and
// UnboundedThreadPool) that return -1 from CurrentThreadId() for every worker.
class MissingWorkerIdThreadPool : public thread::ThreadPoolInterface {
 public:
  MissingWorkerIdThreadPool(int backing_threads, int reported_num_threads)
      : backing_pool_(Env::Default(), "missing_worker_id_test",
                      backing_threads),
        reported_num_threads_(reported_num_threads) {}

  void Schedule(std::function<void()> fn) override {
    schedule_count_.fetch_add(1, std::memory_order_relaxed);
    backing_pool_.Schedule(std::move(fn));
  }

  void ScheduleWithHint(std::function<void()> fn, int start,
                        int limit) override {
    schedule_count_.fetch_add(1, std::memory_order_relaxed);
    backing_pool_.ScheduleWithHint(std::move(fn), start, limit);
  }

  void Cancel() override { backing_pool_.AsEigenThreadPool()->Cancel(); }

  int NumThreads() const override { return reported_num_threads_; }

  int CurrentThreadId() const override { return -1; }

  int64_t schedule_count() const {
    return schedule_count_.load(std::memory_order_relaxed);
  }

 private:
  thread::ThreadPool backing_pool_;
  int reported_num_threads_;
  std::atomic<int64_t> schedule_count_{0};
};

class BincountOpTest : public OpsTestBase {
 protected:
  void SetCustomWorkerThreads(int backing_threads, int reported_num_threads) {
    if (orig_worker_threads_ == nullptr) {
      orig_worker_threads_ = device_->tensorflow_cpu_worker_threads();
    }
    custom_raw_pool_ = std::make_unique<MissingWorkerIdThreadPool>(
        backing_threads, reported_num_threads);
    custom_thread_pool_ =
        std::make_unique<thread::ThreadPool>(custom_raw_pool_.get());
    custom_worker_threads_.num_threads = reported_num_threads;
    custom_worker_threads_.workers = custom_thread_pool_.get();
    device_->set_tensorflow_cpu_worker_threads(&custom_worker_threads_);
  }

  int64_t custom_schedule_count() const {
    return custom_raw_pool_->schedule_count();
  }

  void TearDown() override {
    if (orig_worker_threads_ != nullptr) {
      device_->set_tensorflow_cpu_worker_threads(
          const_cast<DeviceBase::CpuWorkerThreads*>(orig_worker_threads_));
    }
    OpsTestBase::TearDown();
  }

  template <typename T>
  void Verify1DBincount(const char* op_name, bool binary_output, bool weighted,
                        int32_t num_bins, int32_t count_per_bin) {
    const int64_t arr_size = static_cast<int64_t>(num_bins) * count_per_bin;
    const DataType val_dtype = DataTypeToEnum<T>::v();

    NodeDefBuilder builder(op_name, op_name);
    builder.Input(FakeInput(DT_INT32))
        .Input(FakeInput(DT_INT32))
        .Input(FakeInput(val_dtype))
        .Attr("T", val_dtype);
    if (std::string(op_name) == "DenseBincount") {
      builder.Attr("Tidx", DT_INT32).Attr("binary_output", binary_output);
    }
    TF_ASSERT_OK(builder.Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    AddInput<int32_t>(TensorShape({arr_size}),
                      [num_bins](int i) { return i % num_bins; });
    AddInputFromArray<int32_t>(TensorShape({}), {num_bins});
    if (weighted) {
      AddInput<T>(TensorShape({arr_size}), [](int) { return T(1); });
    } else {
      AddInputFromArray<T>(TensorShape({0}), {});
    }

    TF_ASSERT_OK(RunOpKernel());

    const Tensor* output = GetOutput(0);
    ASSERT_EQ(output->NumElements(), num_bins);
    const auto out_flat = output->flat<T>();
    const T expected = binary_output ? T(1) : static_cast<T>(count_per_bin);
    for (int32_t b = 0; b < num_bins; ++b) {
      ASSERT_EQ(out_flat(b), expected) << "at bin " << b;
    }
  }

 private:
  const DeviceBase::CpuWorkerThreads* orig_worker_threads_ = nullptr;
  std::unique_ptr<MissingWorkerIdThreadPool> custom_raw_pool_;
  std::unique_ptr<thread::ThreadPool> custom_thread_pool_;
  DeviceBase::CpuWorkerThreads custom_worker_threads_;
};

TEST_F(BincountOpTest, DenseBincountUnweightedWithMissingWorkerId) {
  SetCustomWorkerThreads(/*backing_threads=*/8, /*reported_num_threads=*/8);
  Verify1DBincount<int32_t>("DenseBincount", /*binary_output=*/false,
                            /*weighted=*/false, /*num_bins=*/4,
                            /*count_per_bin=*/50000);
  EXPECT_GT(custom_schedule_count(), 0);
}

TEST_F(BincountOpTest, DenseBincountWeightedWithMissingWorkerId) {
  SetCustomWorkerThreads(/*backing_threads=*/8, /*reported_num_threads=*/8);
  Verify1DBincount<float>("DenseBincount", /*binary_output=*/false,
                          /*weighted=*/true, /*num_bins=*/4,
                          /*count_per_bin=*/50000);
}

TEST_F(BincountOpTest, DenseBincountBinaryWithMissingWorkerId) {
  SetCustomWorkerThreads(/*backing_threads=*/8, /*reported_num_threads=*/8);
  Verify1DBincount<int32_t>("DenseBincount", /*binary_output=*/true,
                            /*weighted=*/false, /*num_bins=*/4,
                            /*count_per_bin=*/10000);
}

TEST_F(BincountOpTest, DenseBincountBinaryParallelIgnoresOutOfRangeValues) {
  SetCustomWorkerThreads(/*backing_threads=*/8, /*reported_num_threads=*/8);
  constexpr int64_t kArrSize = 20000;
  constexpr int32_t kNumBins = 4;

  TF_ASSERT_OK(NodeDefBuilder("dense_bincount_bin_oob", "DenseBincount")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tidx", DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("binary_output", true)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInput<int32_t>(TensorShape({kArrSize}), [](int i) {
    return (i % 4 < 2) ? (i % 4) : (kNumBins + 1000000);
  });
  AddInputFromArray<int32_t>(TensorShape({}), {kNumBins});
  AddInputFromArray<int32_t>(TensorShape({0}), {});

  TF_ASSERT_OK(RunOpKernel());

  const Tensor* output = GetOutput(0);
  ASSERT_EQ(output->NumElements(), kNumBins);
  const auto out_flat = output->flat<int32_t>();
  EXPECT_EQ(out_flat(0), 1);
  EXPECT_EQ(out_flat(1), 1);
  EXPECT_EQ(out_flat(2), 0);
  EXPECT_EQ(out_flat(3), 0);
}

TEST_F(BincountOpTest, BincountWithMissingWorkerId) {
  SetCustomWorkerThreads(/*backing_threads=*/8, /*reported_num_threads=*/8);
  Verify1DBincount<int32_t>("Bincount", /*binary_output=*/false,
                            /*weighted=*/false, /*num_bins=*/4,
                            /*count_per_bin=*/50000);
}

TEST_F(BincountOpTest, DenseBincountWithNegativeReportedNumThreads) {
  // UnboundedThreadPool returns -1 for both NumThreads() and CurrentThreadId().
  SetCustomWorkerThreads(/*backing_threads=*/4, /*reported_num_threads=*/-1);
  Verify1DBincount<int32_t>("DenseBincount", /*binary_output=*/false,
                            /*weighted=*/false, /*num_bins=*/4,
                            /*count_per_bin=*/10000);
}

TEST_F(BincountOpTest, DenseBincountBinaryWithNegativeReportedNumThreads) {
  SetCustomWorkerThreads(/*backing_threads=*/4, /*reported_num_threads=*/-1);
  Verify1DBincount<int32_t>("DenseBincount", /*binary_output=*/true,
                            /*weighted=*/false, /*num_bins=*/4,
                            /*count_per_bin=*/10000);
  EXPECT_EQ(custom_schedule_count(), 0);
}

TEST_F(BincountOpTest, DenseBincountBinarySmallInputStaysSingleThreaded) {
  SetCustomWorkerThreads(/*backing_threads=*/8, /*reported_num_threads=*/8);
  Verify1DBincount<int32_t>("DenseBincount", /*binary_output=*/true,
                            /*weighted=*/false, /*num_bins=*/4,
                            /*count_per_bin=*/100);
  EXPECT_EQ(custom_schedule_count(), 0);
}

TEST_F(BincountOpTest, DenseBincount2DWithMissingWorkerId) {
  SetCustomWorkerThreads(/*backing_threads=*/8, /*reported_num_threads=*/8);

  constexpr int64_t kNumRows = 16384;
  constexpr int64_t kNumCols = 16;
  constexpr int32_t kNumBins = 8;

  TF_ASSERT_OK(NodeDefBuilder("dense_bincount_2d", "DenseBincount")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tidx", DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("binary_output", false)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInput<int32_t>(TensorShape({kNumRows, kNumCols}),
                    [](int i) { return i % kNumBins; });
  AddInputFromArray<int32_t>(TensorShape({}), {kNumBins});
  AddInputFromArray<int32_t>(TensorShape({0}), {});

  TF_ASSERT_OK(RunOpKernel());

  const Tensor* output = GetOutput(0);
  ASSERT_EQ(output->dims(), 2);
  ASSERT_EQ(output->dim_size(0), kNumRows);
  ASSERT_EQ(output->dim_size(1), kNumBins);
  const auto out_mat = output->matrix<int32_t>();
  for (int64_t r = 0; r < kNumRows; ++r) {
    for (int32_t b = 0; b < kNumBins; ++b) {
      ASSERT_EQ(out_mat(r, b), kNumCols / kNumBins)
          << "at (" << r << ", " << b << ")";
    }
  }
}

TEST_F(BincountOpTest, DenseBincount2DInt64NegativeValue) {
  TF_ASSERT_OK(NodeDefBuilder("dense_bincount_2d_int64_neg", "DenseBincount")
                   .Input(FakeInput(DT_INT64))
                   .Input(FakeInput(DT_INT64))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tidx", DT_INT64)
                   .Attr("T", DT_INT32)
                   .Attr("binary_output", false)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  constexpr int64_t kNegVal = -(int64_t{1} << 32);
  AddInputFromArray<int64_t>(TensorShape({1, 2}), {0, kNegVal});
  AddInputFromArray<int64_t>(TensorShape({}), {4});
  AddInputFromArray<int32_t>(TensorShape({0}), {});

  absl::Status status = RunOpKernel();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(status.message().find("-4294967296"), std::string::npos);
}

}  // namespace

static Graph* Bincount(int arr_size, int nbins) {
  Graph* g = new Graph(OpRegistry::Global());

  Tensor arr(DT_INT32, TensorShape({arr_size}));
  arr.flat<int32_t>() = arr.flat<int32_t>().setRandom().abs();

  Tensor size(DT_INT32, TensorShape({static_cast<int32_t>(1)}));
  size.flat<int32_t>()(0) = static_cast<int32_t>(nbins);

  Tensor weights(DT_INT32, TensorShape({0}));

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Bincount")
                  .Input(test::graph::Constant(g, arr))
                  .Input(test::graph::Constant(g, size))
                  .Input(test::graph::Constant(g, weights))
                  .Attr("T", DT_INT32)
                  .Finalize(g, &node));
  return g;
}

#define BM_BincountDev(K, NBINS, type)                                     \
  static void BM_Bincount##_##type##_##K##_##NBINS(                        \
      ::testing::benchmark::State& state) {                                \
    test::Benchmark(#type, Bincount(K * 1024, NBINS),                      \
                    /*old_benchmark_api=*/false)                           \
        .Run(state);                                                       \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * K * \
                            1024);                                         \
  }                                                                        \
  BENCHMARK(BM_Bincount##_##type##_##K##_##NBINS);

BM_BincountDev(32, 1000, cpu);
BM_BincountDev(32, 2000, cpu);
BM_BincountDev(32, 5000, cpu);
BM_BincountDev(64, 1000, cpu);
BM_BincountDev(64, 2000, cpu);
BM_BincountDev(64, 5000, cpu);
BM_BincountDev(128, 1000, cpu);
BM_BincountDev(128, 2000, cpu);
BM_BincountDev(128, 5000, cpu);

BM_BincountDev(32, 1000, gpu);
BM_BincountDev(32, 2000, gpu);
BM_BincountDev(32, 5000, gpu);
BM_BincountDev(64, 1000, gpu);
BM_BincountDev(64, 2000, gpu);
BM_BincountDev(64, 5000, gpu);
BM_BincountDev(128, 1000, gpu);
BM_BincountDev(128, 2000, gpu);
BM_BincountDev(128, 5000, gpu);

}  // end namespace tensorflow
