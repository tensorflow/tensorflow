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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_KERNELS_TRT_ENGINE_OP_H_
#define TENSORFLOW_CONTRIB_TENSORRT_KERNELS_TRT_ENGINE_OP_H_

#include <memory>
#include <vector>

#include "tensorflow/contrib/tensorrt/resources/trt_allocator.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/mutex.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "cuda/include/cuda_runtime_api.h"
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
class Logger;
class TRTInt8Calibrator;
class TRTCalibrationResource;
class AsyncHelper;
//  TODO(Sami): Remove this file?
class TRTEngineOp : public AsyncOpKernel {
 public:
  explicit TRTEngineOp(OpKernelConstruction* context);

  void ComputeAsync(OpKernelContext* context,
                    tensorflow::AsyncOpKernel::DoneCallback done) override;
  ~TRTEngineOp();

 private:
  template <typename T>
  struct Destroyer {
    void operator()(T* d) {
      if (d) d->destroy();
    }
  };

  // Execute calibration
  void ExecuteCalibration(tensorflow::OpKernelContext* ctx,
                          AsyncHelper* helper);

  // Construct a function handle for executing native funcdef graph
  tensorflow::Status ConstructFunctionHandle(tensorflow::OpKernelContext* ctx);

  // Execute replaced native segment as function Op.
  void ExecuteNativeSegment(tensorflow::OpKernelContext* ctx,
                            AsyncHelper* helper);

  // Allocate necessary resources for calibration
  tensorflow::Status AllocateCalibrationResources(
      tensorflow::OpKernelContext* ctx,
      tensorflow::tensorrt::TRTCalibrationResource** cr);

  // TODO(samikama): context should go to a resource manager!
  typedef std::pair<std::shared_ptr<nvinfer1::ICudaEngine>,
                    std::shared_ptr<nvinfer1::IExecutionContext>>
      EngineCtxPair;
  EngineCtxPair GetEngine(int batch_size, OpKernelContext* ctx,
                          bool ignore_dim_change = true);

  // Return engine batch closest to input batch.
  int GetEngineBatch(OpKernelContext* ctx);

  // map to keep engines and their execution context.
  std::unordered_map<int, EngineCtxPair> engine_map_;
  std::vector<string> input_nodes_;
  std::vector<string> output_nodes_;
  // keep device allocator for TRT
  std::unordered_map<string, std::shared_ptr<nvinfer1::IGpuAllocator>>
      allocators_;
  string serialized_segment_;
  string funcdef_name_;
  string calibration_data_;
  tensorflow::GraphDef segment_graph_;
  std::unordered_map<string, std::pair<void*, size_t>> device_buffers_;
  std::vector<tensorflow::PersistentTensor> dev_tensors_;
  int precision_mode_;
  bool static_engine_;
  bool calibration_mode_;
  bool fixed_input_size_;
  std::vector<int> cached_engine_batches_;
  int max_cached_engines_;
  tensorflow::int64 workspace_size_;
  tensorflow::mutex engine_mutex_;
  tensorflow::FunctionLibraryRuntime::Handle native_func_;
  std::unique_ptr<TRTInt8Calibrator> calibrator_;
};

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CONTRIB_TENSORRT_KERNELS_TRT_ENGINE_OP_H_
