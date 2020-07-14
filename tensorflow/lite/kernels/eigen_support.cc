/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/eigen_support.h"

#include <functional>
#include <memory>
#include <utility>

#include "tensorflow/lite/arena_planner.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/eigen_spatial_convolutions.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace eigen_support {
namespace {

// For legacy reasons, we use 4 threads by default unless the thread count is
// explicitly specified by the context.
const int kDefaultNumThreadpoolThreads = 4;

bool IsValidNumThreads(int num_threads) { return num_threads >= -1; }
int GetNumThreads(int num_threads) {
  return num_threads > -1 ? num_threads : kDefaultNumThreadpoolThreads;
}

#ifndef EIGEN_DONT_ALIGN
// Eigen may require buffers to be aligned to 16, 32 or 64 bytes depending on
// hardware architecture and build configurations.
// If the static assertion fails, try to increase `kDefaultTensorAlignment` to
// in `arena_planner.h` to 32 or 64.
static_assert(
    kDefaultTensorAlignment % EIGEN_MAX_ALIGN_BYTES == 0,
    "kDefaultArenaAlignment doesn't comply with Eigen alignment requirement.");
#endif  // EIGEN_DONT_ALIGN

// Helper routine for updating the global Eigen thread count used for OpenMP.
void SetEigenNbThreads(int threads) {
#if defined(EIGEN_HAS_OPENMP)
  // The global Eigen thread count is only used when OpenMP is enabled. As this
  // call causes problems with tsan, make it only when OpenMP is available.
  Eigen::setNbThreads(threads);
#endif  // defined(EIGEN_HAS_OPENMP)
}

// We have a single global threadpool for all convolution operations. This means
// that inferences started from different threads may block each other, but
// since the underlying resource of CPU cores should be consumed by the
// operations anyway, it shouldn't affect overall performance. Note that we
// also avoid ThreadPool creation if the target thread count is 1, avoiding
// unnecessary overhead, and more closely mimicking Gemmlowp threadpool
// behavior.
class EigenThreadPoolWrapper : public Eigen::ThreadPoolInterface {
 public:
  // Takes ownership of 'pool'
  explicit EigenThreadPoolWrapper(int num_threads) {
    // Avoid creating any threads for the single-threaded case.
    if (num_threads > 1) {
      pool_.reset(new Eigen::ThreadPool(num_threads));
    }
  }
  ~EigenThreadPoolWrapper() override {}

  void Schedule(std::function<void()> fn) override {
    if (pool_) {
      pool_->Schedule(std::move(fn));
    } else {
      fn();
    }
  }
  int NumThreads() const override { return pool_ ? pool_->NumThreads() : 1; }
  int CurrentThreadId() const override {
    return pool_ ? pool_->CurrentThreadId() : 0;
  }

 private:
  // May be null if num_threads <= 1.
  std::unique_ptr<Eigen::ThreadPool> pool_;
};

// Utility class for lazily creating an Eigen thread pool/device only when used.
class LazyEigenThreadPoolHolder {
 public:
  explicit LazyEigenThreadPoolHolder(int num_threads) {
    SetNumThreads(num_threads);
  }

  // Gets the ThreadPoolDevice, creating if necessary.
  const Eigen::ThreadPoolDevice* GetThreadPoolDevice() {
    if (!device_) {
      thread_pool_wrapper_.reset(
          new EigenThreadPoolWrapper(target_num_threads_));
      device_.reset(new Eigen::ThreadPoolDevice(thread_pool_wrapper_.get(),
                                                target_num_threads_));
    }
    return device_.get();
  }

  // Updates the thread count, invalidating the ThreadPoolDevice if necessary.
  void SetNumThreads(int num_threads) {
    const int target_num_threads = GetNumThreads(num_threads);
    if (target_num_threads_ != target_num_threads) {
      target_num_threads_ = target_num_threads;
      // As the device references the thread pool wrapper, destroy it first.
      device_.reset();
      thread_pool_wrapper_.reset();
    }
  }

 private:
  int target_num_threads_ = kDefaultNumThreadpoolThreads;
  // Both device_ and thread_pool_wrapper_ are lazily created.
  std::unique_ptr<Eigen::ThreadPoolDevice> device_;
  std::unique_ptr<Eigen::ThreadPoolInterface> thread_pool_wrapper_;
};

struct RefCountedEigenContext : public TfLiteExternalContext {
  std::unique_ptr<LazyEigenThreadPoolHolder> thread_pool_holder;
  int num_references = 0;
};

RefCountedEigenContext* GetEigenContext(TfLiteContext* context) {
  return reinterpret_cast<RefCountedEigenContext*>(
      context->GetExternalContext(context, kTfLiteEigenContext));
}

TfLiteStatus Refresh(TfLiteContext* context) {
  if (IsValidNumThreads(context->recommended_num_threads)) {
    SetEigenNbThreads(GetNumThreads(context->recommended_num_threads));
  }

  auto* ptr = GetEigenContext(context);
  if (ptr != nullptr) {
    ptr->thread_pool_holder->SetNumThreads(context->recommended_num_threads);
  }

  return kTfLiteOk;
}

}  // namespace

void IncrementUsageCounter(TfLiteContext* context) {
  auto* ptr = GetEigenContext(context);
  if (ptr == nullptr) {
    if (IsValidNumThreads(context->recommended_num_threads)) {
      SetEigenNbThreads(context->recommended_num_threads);
    }
    ptr = new RefCountedEigenContext;
    ptr->type = kTfLiteEigenContext;
    ptr->Refresh = Refresh;
    ptr->thread_pool_holder.reset(
        new LazyEigenThreadPoolHolder(context->recommended_num_threads));
    ptr->num_references = 0;
    context->SetExternalContext(context, kTfLiteEigenContext, ptr);
  }
  ptr->num_references++;
}

void DecrementUsageCounter(TfLiteContext* context) {
  auto* ptr = GetEigenContext(context);
  if (ptr == nullptr) {
    TF_LITE_FATAL(
        "Call to DecrementUsageCounter() not preceded by "
        "IncrementUsageCounter()");
  }
  if (--ptr->num_references == 0) {
    delete ptr;
    context->SetExternalContext(context, kTfLiteEigenContext, nullptr);
  }
}

const Eigen::ThreadPoolDevice* GetThreadPoolDevice(TfLiteContext* context) {
  auto* ptr = GetEigenContext(context);
  if (ptr == nullptr) {
    TF_LITE_FATAL(
        "Call to GetFromContext() not preceded by IncrementUsageCounter()");
  }
  return ptr->thread_pool_holder->GetThreadPoolDevice();
}

}  // namespace eigen_support
}  // namespace tflite
