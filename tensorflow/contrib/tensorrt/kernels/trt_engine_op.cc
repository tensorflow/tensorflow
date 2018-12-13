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
#include "tensorflow/contrib/tensorrt/kernels/trt_engine_op.h"

#include <algorithm>

#include "tensorflow/contrib/tensorrt/convert/convert_nodes.h"
#include "tensorflow/contrib/tensorrt/convert/utils.h"
#include "tensorflow/contrib/tensorrt/log/trt_logger.h"
#include "tensorflow/contrib/tensorrt/plugin/trt_plugin_factory.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resource_manager.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resources.h"
#include "tensorflow/contrib/tensorrt/test/utils.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "cuda/include/cuda_runtime_api.h"

namespace tensorflow {
namespace tensorrt {
static Logger logger;
using ::nvinfer1::IRuntime;
using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;

// A helper class to call done() when destructed for asynchronous execution.
// Helps simultaneous execution of native and TRT engines.
class AsyncHelper : public tensorflow::core::RefCounted {
 public:
  AsyncHelper(AsyncOpKernel::DoneCallback done) { done_ = done; }
  ~AsyncHelper() override { done_(); }

 private:
  AsyncOpKernel::DoneCallback done_;
};

#define TYPECASE(dt, X, Y)                                                \
  case dt: {                                                              \
    return (void*)X->flat<tensorflow::EnumToDataType<dt>::Type>().data(); \
  }

void* GetTensorAddress(const Tensor* tensor_ptr) {
  auto tensor_type = tensor_ptr->dtype();
  switch (tensor_type) {
    TYPECASE(tensorflow::DT_FLOAT, tensor_ptr, dest_ptr);
    TYPECASE(tensorflow::DT_HALF, tensor_ptr, dest_ptr);
    TYPECASE(tensorflow::DT_INT8, tensor_ptr, dest_ptr);
    default: {
      LOG(ERROR) << "Unsupported Data type "
                 << tensorflow::DataTypeString(tensor_type);
      return nullptr;
    }
  }
}

tensorflow::Status TRTEngineOp::ConstructFunctionHandle(OpKernelContext* ctx) {
  VLOG(1) << "Constructing function handle";
  auto lib = ctx->function_library();
  if (lib == nullptr) {
    return tensorflow::errors::Internal("Context function library is null");
  }
  auto fdef = lib->GetFunctionLibraryDefinition()->Find(funcdef_name_);
  if (fdef == nullptr) {
    return tensorflow::errors::Internal("Native FunctionDef ", funcdef_name_,
                                        " can't be found in function library");
  }
  tensorflow::FunctionLibraryRuntime::InstantiateOptions inst_ops;
  inst_ops.overlay_lib = nullptr;
  inst_ops.state_handle = "";
  inst_ops.target = ctx->device()->name();
  native_func_ = 0;
  auto status = lib->Instantiate(funcdef_name_, AttrSlice(&fdef->attr()),
                                 inst_ops, &native_func_);
  if (!status.ok()) {
    LOG(ERROR) << " Instantiating native function " << funcdef_name_
               << " failed!";
  }
  return status;
}

TRTEngineOp::TRTEngineOp(OpKernelConstruction* context)
    : AsyncOpKernel(context) {
  // read serialized_engine
  OP_REQUIRES_OK(context,
                 context->GetAttr("serialized_segment", &serialized_segment_));
  OP_REQUIRES_OK(context,
                 context->GetAttr("workspace_size_bytes", &workspace_size_));
  OP_REQUIRES_OK(context, context->GetAttr("static_engine", &static_engine_));
  if (!static_engine_) {
    if (!segment_graph_.ParseFromString(serialized_segment_)) {
      LOG(ERROR) << "Parsing segment graph failed!";
      context->SetStatus(tensorflow::errors::InvalidArgument(
          "Failed to parse segment graphdef!"));
      return;
    }
    serialized_segment_.resize(0);
  }
  VLOG(1) << "Constructing " << name();
  string precision_string;
  OP_REQUIRES_OK(context,
                 context->GetAttr("precision_mode", &precision_string));
  string calibration_data;
  OP_REQUIRES_OK(context,
                 context->GetAttr("calibration_data", &calibration_data));
  OP_REQUIRES_OK(context,
                 context->GetAttr("segment_funcdef_name", &funcdef_name_));
  OP_REQUIRES_OK(context, GetPrecisionMode(precision_string, &precision_mode_));
  OP_REQUIRES_OK(context,
                 context->GetAttr("use_calibration", &use_calibration_));
  calibration_mode_ = (use_calibration_ && precision_mode_ == INT8MODE &&
                       calibration_data.size() == 0);
  if (calibration_data.size()) {
    calibrator_.reset(new TRTInt8Calibrator(calibration_data));
    calibration_data.resize(0);
  }
  native_func_ = tensorflow::kInvalidHandle;
  OP_REQUIRES_OK(context, context->GetAttr("max_cached_engines_count",
                                           &max_cached_engines_));
  OP_REQUIRES_OK(context,
                 context->GetAttr("fixed_input_size", &fixed_input_size_));
  OP_REQUIRES_OK(context, context->GetAttr("cached_engine_batches",
                                           &cached_engine_batches_));
  std::sort(cached_engine_batches_.begin(), cached_engine_batches_.end());
  if (VLOG_IS_ON(1)) {
    string s("Engine Batches= ");
    for (auto i : cached_engine_batches_) {
      StrAppend(&s, i, " ");
    }
    VLOG(1) << s;
  }
}

void TRTEngineOp::ExecuteNativeSegment(OpKernelContext* ctx,
                                       AsyncHelper* helper) {
  std::vector<Tensor> inputs;
  std::vector<Tensor>* outputs = new std::vector<Tensor>();
  if (native_func_ == tensorflow::kInvalidHandle) {
    auto status = ConstructFunctionHandle(ctx);
    if (!status.ok()) {
      LOG(ERROR) << "Couldn't construct function handle " << funcdef_name_;
      ctx->SetStatus(status);
      return;
    }
  }
  auto lib = ctx->function_library();
  tensorflow::FunctionLibraryRuntime::Options opts;
  opts.step_id = ctx->step_id();
  opts.rendezvous = ctx->rendezvous();
  opts.cancellation_manager = ctx->cancellation_manager();
  opts.runner = ctx->runner();
  for (int i = 0; i < ctx->num_inputs(); i++) {
    inputs.push_back(ctx->input(i));
  }
  helper->Ref();  // Increment count for calculating native graph
  VLOG(1) << "Executing native segment: " << name();
  lib->Run(opts, native_func_, inputs, outputs,
           [this, ctx, outputs, helper](const tensorflow::Status& s) {
             tensorflow::core::ScopedUnref sc(helper);
             VLOG(1) << "Native Segment completed";
             if (!s.ok()) {
               ctx->SetStatus(s);
               return;
             }
             for (size_t t = 0; t < outputs->size(); ++t) {
               ctx->set_output(t, outputs->at(t));
             }
             test::AddTestValue(StrCat(this->name(), ":ExecuteNativeSegment"),
                                "done");
             delete outputs;
           });
}

void TRTEngineOp::ExecuteCalibration(OpKernelContext* ctx,
                                     AsyncHelper* helper) {
  VLOG(1) << "Executing TRT calibration: " << name();
  helper->Ref();
  tensorflow::core::ScopedUnref sc(helper);
  // TODO(aaroey): remove the ResourceMgr singleton.
  auto trt_rm = TRTResourceManager::instance();
  auto res_mgr = trt_rm->getManager("TRTCalibration");
  TRTCalibrationResource* calib_res = nullptr;
  auto status = res_mgr->LookupOrCreate(
      funcdef_name_, "Calibrator", &calib_res,
      {[ctx, this](TRTCalibrationResource** cr) -> tensorflow::Status {
        return this->AllocateCalibrationResources(ctx, cr);
      }});
  if (!status.ok()) {
    ctx->SetStatus(status);
    return;
  }
  int num_inputs = ctx->num_inputs();
  // Pass input data to calibrator
  std::unordered_map<string, void*> input_data;
  for (int i = 0; i < num_inputs; i++) {
    const Tensor& t = ctx->input(i);
    void* data_address = GetTensorAddress(&t);
    if (data_address == nullptr) {
      ctx->SetStatus(tensorflow::errors::InvalidArgument(
          "Unsupported data type encountered in input ", i));
      return;
    }
    // Check the allocated buffer is sufficient for input
    const auto device_tensor = dev_tensors_.at(i).AccessTensor(ctx);
    CHECK_EQ(t.TotalBytes(), device_tensor->TotalBytes());
    input_data.emplace(StrCat(kInputPHName, i), data_address);
  }
  VLOG(2) << "Filled map for sending";
  // copied from cuda_kernel_helper since it seems only valid in *.cu.cc files
  const cudaStream_t* stream = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(ctx->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GpuStreamMemberHack()));
  calib_res->calibrator_->setBatch(input_data, *stream);
  test::AddTestValue(StrCat(name(), ":ExecuteCalibration"), "done");
  VLOG(2) << "Passed calibration data";
  ExecuteNativeSegment(ctx, helper);
}

int TRTEngineOp::GetEngineBatch(OpKernelContext* ctx) {
  int num_batch = ctx->input(0).shape().dim_size(0);
  int smallest_engine = 0;
  for (const auto i : cached_engine_batches_) {
    if (i >= num_batch) {
      smallest_engine = i;
      break;
    }
  }
  // TODO(sami): Need an LRU here
  if (smallest_engine == 0) {
    if (max_cached_engines_ > cached_engine_batches_.size()) {
      smallest_engine = num_batch;
      cached_engine_batches_.push_back(num_batch);
      VLOG(1) << "Running with batch size " << num_batch;
    } else {
      string msg =
          StrCat("Engine buffer is full. buffer limit=", max_cached_engines_,
                 ", current entries=");
      for (auto i : cached_engine_batches_) StrAppend(&msg, i, ",");
      StrAppend(&msg, " requested batch=", num_batch);
      LOG(WARNING) << msg;
      return -1;
    }
  }
  return smallest_engine;
}

void TRTEngineOp::ComputeAsync(OpKernelContext* ctx,
                               AsyncOpKernel::DoneCallback done) {
  auto helper = new AsyncHelper(done);
  tensorflow::core::ScopedUnref sc(helper);
  if (calibration_mode_) {
    ExecuteCalibration(ctx, helper);
    return;
  }
  const int smallest_engine = GetEngineBatch(ctx);
  if (smallest_engine < 0) {
    LOG(WARNING) << "Failed to get engine batch, running native segment for "
                 << name();
    ExecuteNativeSegment(ctx, helper);
    return;
  }

  const int num_batch = ctx->input(0).shape().dim_size(0);
  auto& engine_ctx_pair = GetEngine(smallest_engine, ctx);
  auto& trt_engine_ptr = engine_ctx_pair.first;
  if (!trt_engine_ptr) {
    LOG(WARNING) << "Engine retrieval for batch size " << num_batch
                 << " failed. Running native segment for " << name();
    ExecuteNativeSegment(ctx, helper);
    return;
  }
  const bool retry = ExecuteTrtEngine(ctx, num_batch, trt_engine_ptr.get(),
                                      engine_ctx_pair.second.get());
  if (retry) {
    LOG(WARNING) << "Failed to execute engine, "
                 << "retrying with native segment for " << name();
    ExecuteNativeSegment(ctx, helper);
    return;
  }
}

bool TRTEngineOp::ExecuteTrtEngine(
    OpKernelContext* ctx, const int num_batch,
    nvinfer1::ICudaEngine* trt_engine_ptr,
    nvinfer1::IExecutionContext* trt_execution_context_ptr) {
  VLOG(1) << "Executing TRT engine: " << name();
  const bool kRetry = true;
  const int num_binding = ctx->num_inputs() + ctx->num_outputs();
  std::vector<void*> buffers(num_binding);
  for (int i = 0; i < ctx->num_inputs(); i++) {
    const string input_name = StrCat(kInputPHName, i);
    const int binding_index =
        trt_engine_ptr->getBindingIndex(input_name.c_str());
    if (binding_index == -1) {
      LOG(ERROR) << "Input node not found, at " << input_name;
      return kRetry;
    }

    const Tensor& input_tensor = ctx->input(i);
    const TensorShape& input_shape = input_tensor.shape();
    if (num_batch != input_shape.dim_size(0)) {
      LOG(ERROR) << "Input data has inconsistent batch size: " << num_batch
                 << " vs " << input_shape.dim_size(0);
      return kRetry;
    }
    auto dtype = trt_engine_ptr->getBindingDataType(binding_index);
    switch (dtype) {
      case nvinfer1::DataType::kFLOAT:
        buffers[binding_index] = (void*)(input_tensor.flat<float>().data());
        break;
      case nvinfer1::DataType::kHALF:
        LOG(ERROR) << "FP16 inputs are not supported yet!";
        return kRetry;
      case nvinfer1::DataType::kINT8:
        LOG(ERROR) << "INT8 inputs are not supported yet!";
        return kRetry;
      case nvinfer1::DataType::kINT32:
        buffers[binding_index] = (void*)(input_tensor.flat<int32>().data());
        break;
      default:
        LOG(ERROR) << "Unknown TRT data type: " << int(dtype);
        return kRetry;
    }
  }

  for (int i = 0; i < ctx->num_outputs(); i++) {
    // Create an output tensor
    const string output_name = StrCat(kOutputPHName, i);
    const int binding_index =
        trt_engine_ptr->getBindingIndex(output_name.c_str());
    Tensor* output_tensor = nullptr;

    TensorShape output_shape;
    if (binding_index != -1) {
      auto dims = trt_engine_ptr->getBindingDimensions(binding_index);
      std::vector<int> trt_shape(dims.nbDims + 1);
      trt_shape[0] = num_batch;
      for (int j = 0; j < dims.nbDims; j++) trt_shape[j + 1] = dims.d[j];
      auto status = TensorShapeUtils::MakeShape(
          trt_shape.data(), trt_shape.size(), &output_shape);
      if (!status.ok()) {
        LOG(ERROR) << "Failed to get output shape: " << status;
        return kRetry;
      }
    } else {
      LOG(ERROR) << "Output node not found, at " << output_name;
      return kRetry;
    }
    auto status = ctx->allocate_output(i, output_shape, &output_tensor);
    if (!status.ok()) {
      LOG(ERROR) << "Allocating output failed with " << status;
      ctx->SetStatus(status);
      // Do not retry since we cannot allocate the same output twice.
      // TODO(aaroey): ideally we should retry, fix this.
      return !kRetry;
    }
    auto dtype = trt_engine_ptr->getBindingDataType(binding_index);
    switch (dtype) {
      case nvinfer1::DataType::kFLOAT:
        buffers[binding_index] =
            reinterpret_cast<void*>(output_tensor->flat<float>().data());
        break;
      case nvinfer1::DataType::kHALF:
        LOG(WARNING) << "half size is not supported yet!";
        return kRetry;
      case nvinfer1::DataType::kINT8:
        LOG(WARNING) << "int8 is not supported yet!";
        return kRetry;
      case nvinfer1::DataType::kINT32:
        buffers[binding_index] =
            reinterpret_cast<void*>(output_tensor->flat<int32>().data());
        break;
      default:
        LOG(WARNING) << "Unknown TRT data type: " << static_cast<int>(dtype);
        return kRetry;
    }
  }
  // Copied from cuda_kernel_helper since it seems only valid in *.cu.cc files
  const cudaStream_t* stream = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(ctx->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GpuStreamMemberHack()));

  // TODO(jie): trt enqueue does not return error
  auto ret = trt_execution_context_ptr->enqueue(num_batch, &buffers[0], *stream,
                                                nullptr);
  if (!ret) {
    LOG(WARNING) << "Failed to enqueue batch for TRT engine: " << name();
    return kRetry;
  }
  test::AddTestValue(StrCat(name(), ":ExecuteTrtEngine"), "done");
  // Synchronization will be done by TF.
  return !kRetry;
}

TRTEngineOp::~TRTEngineOp() {
  // We need to manually destroy the engine and execution context before
  // the allocator is destructed.
  for (auto& eng : engine_map_) {
    eng.second.first.reset();
    eng.second.second.reset();
  }
  allocator_.reset();
}

nvinfer1::IGpuAllocator* TRTEngineOp::GetAllocator(OpKernelContext* ctx) {
  if (allocator_) return allocator_.get();
  auto device = ctx->device();
  auto alloc = device->GetAllocator(tensorflow::AllocatorAttributes());
  if (!alloc) {
    LOG(ERROR) << "Can't find device allocator for gpu device "
               << device->name();
    return nullptr;
  }
  allocator_.reset(new TRTDeviceAllocator(alloc));
  return allocator_.get();
}

TRTEngineOp::EngineCtxPair& TRTEngineOp::GetEngine(int batch_size,
                                                   OpKernelContext* ctx) {
  static EngineCtxPair null_pair = {
      TrtUniquePtrType<nvinfer1::ICudaEngine>(nullptr),
      TrtUniquePtrType<nvinfer1::IExecutionContext>(nullptr)};
  // TODO(sami): This method needs to be re-written to use resource manager and
  // with LRU mechanism option.
  tensorflow::mutex_lock lock(engine_mutex_);

  if (static_engine_) {
    if (engine_map_.size()) {
      if (engine_map_.begin()->first >= batch_size) {
        return engine_map_.begin()->second;
      }
      return null_pair;
    }
    TrtUniquePtrType<IRuntime> infer(nvinfer1::createInferRuntime(logger));
    auto allocator = GetAllocator(ctx);
    if (allocator == nullptr) {
      return null_pair;
    }
    infer->setGpuAllocator(allocator);
    TrtUniquePtrType<nvinfer1::ICudaEngine> static_engine(
        infer->deserializeCudaEngine(serialized_segment_.c_str(),
                                     serialized_segment_.size(),
                                     PluginFactoryTensorRT::GetInstance()));
    auto raw_static_engine = static_engine.get();
    const auto max_batch_size = raw_static_engine->getMaxBatchSize();
    engine_map_[max_batch_size] = {
        std::move(static_engine),
        TrtUniquePtrType<nvinfer1::IExecutionContext>(
            raw_static_engine->createExecutionContext())};
    // Runtime is safe to delete after engine creation
    serialized_segment_.clear();
    if (max_batch_size < batch_size) {
      return null_pair;
    }
    return engine_map_.at(max_batch_size);
  }  // static_engine_

  // Handle the dynamic engine case.
  auto engine_it = engine_map_.find(batch_size);
  if (engine_it == engine_map_.end() &&
      engine_map_.size() < (size_t)max_cached_engines_) {
    nvinfer1::IGpuAllocator* allocator = nullptr;
    allocator = GetAllocator(ctx);
    if (allocator == nullptr) {
      return null_pair;
    }
    std::vector<tensorflow::PartialTensorShape> shapes;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      shapes.emplace_back(ctx->input(i).shape());
    }
    TrtUniquePtrType<nvinfer1::ICudaEngine> engine;
    bool convert_successfully = false;
    LOG(INFO) << "Building a new TensorRT engine for " << name()
              << " with batch size " << batch_size;
    // Up to this point, calibrator_ can never be empty, since otherwise it
    // means calibration_mode_ is true and this path won't get executed.
    auto status = convert::ConvertGraphDefToEngine(
        segment_graph_, precision_mode_, batch_size, workspace_size_, shapes,
        &logger, allocator, calibrator_.get(), &engine, use_calibration_,
        &convert_successfully);
    if (!status.ok()) {
      if (convert_successfully) {
        // This means it fail to build the engine even when the network is built
        // successfully, probably due to internal issues. In this case we don't
        // retry in the future.
        engine_map_[batch_size] = {nullptr, nullptr};
      }
      LOG(WARNING) << "Engine creation for batch size " << batch_size
                   << " failed " << status;
      return null_pair;
    }
    VLOG(1) << "Conversion is done";
    TrtUniquePtrType<nvinfer1::IExecutionContext> exec_context(
        engine->createExecutionContext());
    engine_map_[batch_size] = {std::move(engine), std::move(exec_context)};
  }
  return engine_map_.at(batch_size);
}

tensorflow::Status TRTEngineOp::AllocateCalibrationResources(
    OpKernelContext* ctx, TRTCalibrationResource** cr) {
  auto cres = new TRTCalibrationResource();
  *cr = cres;
  // Get the allocator.
  auto alloc = ctx->device()->GetAllocator(tensorflow::AllocatorAttributes());
  if (!alloc) {
    LOG(WARNING) << "Can't get device allocator will not be able to "
                    "allocate memory from TensorFlow memory pool";
    cres->allocator_.reset(new TRTCudaAllocator);
  } else {
    cres->allocator_.reset(new TRTDeviceAllocator(alloc));
  }
  // Get the input shapes.
  const int batch_size = ctx->input(0).dim_size(0);
  const int num_inputs = ctx->num_inputs();
  std::vector<tensorflow::PartialTensorShape> shapes;
  dev_tensors_.resize(num_inputs);
  VLOG(1) << " Constructing calibrator";
  for (int i = 0; i < num_inputs; i++) {
    // allocate workspace on device for inputs
    const tensorflow::Tensor& t = ctx->input(i);
    shapes.emplace_back(t.shape());
    Tensor* device_tensor;
    TF_RETURN_IF_ERROR(ctx->allocate_persistent(
        t.dtype(), t.shape(), &dev_tensors_.at(i), &device_tensor));
    CHECK_EQ(t.TotalBytes(), device_tensor->TotalBytes());
    void* device_address = GetTensorAddress(device_tensor);
    if (device_address == nullptr) {
      return tensorflow::errors::InvalidArgument(
          "Unsupported data type encountered in input ", i);
    }
    device_buffers_.emplace(
        StrCat(kInputPHName, i),
        std::pair<void*, size_t>(device_address, device_tensor->TotalBytes()));
  }
  cres->calibrator_.reset(
      new TRTInt8Calibrator(device_buffers_, batch_size, name()));
  const string label(name());
  auto segment_graph = &segment_graph_;
  const int platform_gpu_id =
      ctx->device()->tensorflow_gpu_device_info()->gpu_id;
  if (platform_gpu_id < 0) {
    LOG(ERROR) << "Can't get gpu_device_info from context->device()";
    return tensorflow::errors::InvalidArgument(
        "Context->device doesn't contain device info!");
  }
  const int64 workspace_size_bytes = workspace_size_;
  cres->thr_.reset(new std::thread([cres, label, segment_graph, shapes,
                                    platform_gpu_id, workspace_size_bytes]() {
    LOG(INFO) << "Starting calibration thread on device " << platform_gpu_id
              << ", Calibration Resource @ " << cres;
    auto err = cudaSetDevice(platform_gpu_id);
    if (err != cudaSuccess) {
      // TODO(aaroey): should return error here.
      LOG(ERROR) << "Couldn't set cuda device to " << platform_gpu_id
                 << " in calibration thread";
    }
    // ConvertGraphDefToEngine() will try to build the engine. This thread
    // will loop inside buildCudaEngine() consuming the calibration data
    // that is set by the TF op, and drive the builder until calibrator returns
    // false. Engine is discarded after calibration table is generated
    //
    // TODO(aaroey): maybe setting the max batch size using the python
    // calibration wrapper class.
    auto s = convert::ConvertGraphDefToEngine(
        *segment_graph, INT8MODE, cres->calibrator_->getBatchSize(),
        workspace_size_bytes, shapes, &cres->logger_, cres->allocator_.get(),
        cres->calibrator_.get(), &cres->engine_,
        /*use_calibration=*/true,
        /*convert_successfully=*/nullptr);
    if (!s.ok()) {
      LOG(ERROR) << "Calibration failed: " << s;
      cres->calibrator_->setDone();  // Ignore further pushes
    }
    VLOG(1) << "Calibration loop terminated " << label;
  }));
  VLOG(1) << "initialized calibrator resource";
  return tensorflow::Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("TRTEngineOp").Device(DEVICE_GPU), TRTEngineOp);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
