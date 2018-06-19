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

#include "tensorflow/contrib/tensorrt/convert/utils.h"
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
class TRTInt8Calibrator;
class TRTCalibrationResource;
class AsyncHelper;
//  TODO(Sami): Remove this file?

//  This OP can construct TRTEngine on the fly and if construction of engine
//  fails, executes equivalent subgraph as a TensorFlow function.
class TRTEngineOp : public AsyncOpKernel {
 public:
  explicit TRTEngineOp(OpKernelConstruction* context);

  void ComputeAsync(OpKernelContext* context,
                    tensorflow::AsyncOpKernel::DoneCallback done) override;
  ~TRTEngineOp();

 private:
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
  typedef std::pair<TrtUniquePtrType<nvinfer1::ICudaEngine>,
                    TrtUniquePtrType<nvinfer1::IExecutionContext>>
      EngineCtxPair;
  EngineCtxPair& GetEngine(int batch_size, OpKernelContext* ctx);

  // Return engine batch closest to input batch.
  int GetEngineBatch(OpKernelContext* ctx);

  nvinfer1::IGpuAllocator* GetAllocator(OpKernelContext* ctx);

  // map to keep engines and their execution context for given batch size.
  std::unordered_map<int, EngineCtxPair> engine_map_;
  std::vector<string> input_nodes_;
  std::vector<string> output_nodes_;

  // keep device allocator for TRT.
  std::unique_ptr<TRTDeviceAllocator> allocator_;

  // serialized protobuf segment or trt engine depending on static_engine_ flag.
  string serialized_segment_;

  // Name of the function for TF native execution of the segment.
  string funcdef_name_;

  // GraphDef representation of the segment.
  tensorflow::GraphDef segment_graph_;

  // Lookup table for temporary staging areas of input tensors for calibration.
  std::unordered_map<string, std::pair<void*, size_t>> device_buffers_;

  // Temporary staging areas for calibration inputs.
  std::vector<tensorflow::PersistentTensor> dev_tensors_;

  // Engine Precision mode.
  int precision_mode_;

  // Whether engine is constructed during the conversion or needs to be
  // constructed from protobuf segment.
  bool static_engine_;

  // Whether to calibrate INT8 engine.
  bool calibration_mode_;

  // Whether non-batch ranks of the inputs are assumed to be fixed or not for
  // engine construction.
  bool fixed_input_size_;

  // Batches of the cached engines
  std::vector<int> cached_engine_batches_;

  // Maximum number of cached engines
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
