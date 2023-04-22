/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"

#include <sstream>

#include "tensorflow/compiler/tf2tensorrt/utils/trt_allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/mutex.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

string CalibrationContext::TerminateCalibration() {
  mutex_lock l(mu_);
  if (terminated_) return calibration_table_;

  TRTInt8Calibrator* raw_calibrator = calibrator_.get();
  raw_calibrator->waitAndSetDone();
  terminated_ = true;

  // At this point the calibration thread `thr_` is woken up and can
  // transfer the ownership of `calibrator_` and `engine_` at any time, so
  // it's not safe to use `calibrator_` below, but we can still access it
  // using raw pointer.
  // TODO(laigd): make TRTEngineOp::AllocateCalibrationResources() a member
  // function of this class instead.

  thr_->join();
  calibration_table_ = raw_calibrator->getCalibrationTableAsString();
  return calibration_table_;
}

const absl::string_view kTfTrtContainerName = "TF-TRT";

Logger& TRTEngineCacheResource::GetLogger() {
  static Logger* logger = new Logger();
  return *logger;
}

TRTEngineCacheResource::TRTEngineCacheResource(OpKernelContext* ctx,
                                               size_t capacity)
    : cache_(capacity) {
  auto device = ctx->device();
  auto alloc = device->GetAllocator(AllocatorAttributes());
  if (!alloc) {
    LOG(ERROR) << "Can't find device allocator for gpu device "
               << device->name();
    allocator_ = nullptr;
  } else {
    allocator_.reset(new TRTDeviceAllocator(alloc));
  }
}

TRTEngineCacheResource::~TRTEngineCacheResource() {
  VLOG(1) << "Destroying TRTEngineCacheResource...";
}

string TRTEngineCacheResource::DebugString() const {
  std::stringstream oss;
  using std::dec;
  using std::endl;
  using std::hex;
  oss << "TRTEngineCacheResource: ";
  oss << "TRTBaseAllocator = " << hex << allocator_.get() << dec << ", ";
  oss << "LRUCache = " << hex << &cache_ << dec << endl;
  oss << "Containing " << cache_.size() << " entries: " << endl;
  for (const auto& item : cache_) {
    mutex_lock lock(item.second->mu);
    oss << TensorShapeUtils::ShapeListString(item.first) << ": " << hex
        << "ICudaEngine: " << item.second->cuda_engine.get() << ", "
        << "IExecutionContext: ";
    absl::c_for_each(
        item.second->execution_contexts,
        [&](const ExecutionContext& ctx) { oss << ctx.get() << ","; });
    oss << dec << endl;
  }
  return oss.str();
}

EngineContext* TRTEngineCacheResource::GetEngineContext(
    const std::vector<TensorShape>& input_shapes) {
  EngineContext* engine_context = nullptr;
  int64 min_matched_batch_size = kint64max;
  for (const auto& pair : cache_) {
    const std::vector<TensorShape>& cached_input_shapes = pair.first;
    // This should not happen, but just for safety.
    if (input_shapes.size() != cached_input_shapes.size()) {
      LOG(ERROR) << "Input shape list size mismatch"
                 << ", cached size: " << cached_input_shapes.size()
                 << " vs. input size: " << input_shapes.size();
    }
    if (AreShapesCompatible(input_shapes, cached_input_shapes)) {
      const int cached_batch_size = cached_input_shapes[0].dim_size(0);
      if (min_matched_batch_size > cached_batch_size) {
        min_matched_batch_size = cached_batch_size;
        engine_context = pair.second.get();
      }
    }
  }
  return engine_context;
}

EngineContext* TRTEngineCacheResource::GetEngineContext(const int profile_id) {
  if (profiles_.NeedProfiles() && profile_id >= profiles_.GetNumProfiles()) {
    LOG(ERROR) << "Out of range: profile_id " << profile_id
               << " is larger than number of profiles "
               << profiles_.GetNumProfiles();
    return nullptr;
  }
  if (cache_.size() > 1) {
    LOG(ERROR) << "Cache is expected to have at most "
               << "1 engine in explicit batch mode where profiles are used.";
    return nullptr;
  }
  if (cache_.size() == 0) {
    return nullptr;
  }
  return cache_.begin()->second.get();
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
