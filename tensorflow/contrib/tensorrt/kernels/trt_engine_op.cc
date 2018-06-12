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
#include "tensorflow/contrib/tensorrt/log/trt_logger.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resource_manager.h"
#include "tensorflow/contrib/tensorrt/resources/trt_resources.h"
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
static ::tensorflow::tensorrt::Logger logger;
using IRuntime = nvinfer1::IRuntime;
using Dims = nvinfer1::Dims;

namespace tensorrt {
using tensorflow::strings::StrAppend;
using tensorflow::strings::StrCat;
// A helper class to call done() for asynchronous execution.
// Helps simultaneous execution of native and TRT engines.
class AsyncHelper : public tensorflow::core::RefCounted {
 public:
  AsyncHelper(tensorflow::AsyncOpKernel::DoneCallback done) { done_ = done; }
  ~AsyncHelper() override { done_(); }

 private:
  tensorflow::AsyncOpKernel::DoneCallback done_;
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
      LOG(FATAL) << "Unsupported Data type "
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
    return tensorflow::errors::Internal(
        StrCat("Native FunctionDef ", funcdef_name_,
               " can't be found in function library"));
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

  string precision_string;
  OP_REQUIRES_OK(context,
                 context->GetAttr("precision_mode", &precision_string));
  OP_REQUIRES_OK(context,
                 context->GetAttr("calibration_data", &calibration_data_));
  OP_REQUIRES_OK(context,
                 context->GetAttr("segment_funcdef_name", &funcdef_name_));
  if (precision_string == "FP32") {
    precision_mode_ = tensorflow::tensorrt::convert::FP32MODE;
  } else if (precision_string == "FP16") {
    precision_mode_ = tensorflow::tensorrt::convert::FP16MODE;
  } else if (precision_string == "INT8") {
    precision_mode_ = tensorflow::tensorrt::convert::INT8MODE;
  }
  calibration_mode_ =
      precision_mode_ == tensorflow::tensorrt::convert::INT8MODE &&
      calibration_data_.size() == 0;
  if (calibration_data_.size()) {
    calibrator_.reset(new TRTInt8Calibrator(calibration_data_));
    calibration_data_.resize(0);
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

void TRTEngineOp::ExecuteNativeSegment(tensorflow::OpKernelContext* ctx,
                                       AsyncHelper* helper) {
  if (!calibration_mode_) {
    VLOG(1) << "Executing native engine";
  }
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
  VLOG(1) << "Executing native segment " << name();
  lib->Run(opts, native_func_, inputs, outputs,
           [ctx, outputs, helper](const tensorflow::Status& s) {
             tensorflow::core::ScopedUnref SC(helper);
             VLOG(1) << "Native Segment completed";
             if (!s.ok()) {
               ctx->SetStatus(s);
               return;
             }
             for (size_t t = 0; t < outputs->size(); ++t) {
               ctx->set_output(t, outputs->at(t));
             }
             delete outputs;
             return;
           });
  return;
}

void TRTEngineOp::ExecuteCalibration(tensorflow::OpKernelContext* ctx,
                                     AsyncHelper* helper) {
  tensorflow::core::ScopedUnref SC(helper);
  auto TRT_RM = tensorflow::tensorrt::TRTResourceManager::instance();
  auto res_mgr = TRT_RM->getManager("TRTCalibration");
  tensorflow::tensorrt::TRTCalibrationResource* calib_res = nullptr;
  auto status = res_mgr->LookupOrCreate(
      funcdef_name_, "Calibrator", &calib_res,
      {[ctx, this](tensorflow::tensorrt::TRTCalibrationResource** cr)
           -> tensorflow::Status {
        return this->AllocateCalibrationResources(ctx, cr);
      }});
  if (!status.ok()) {
    ctx->SetStatus(status);
    return;
  }
  ExecuteNativeSegment(ctx, helper);
  int num_inputs = ctx->num_inputs();
  // Pass input data to calibrator
  std::unordered_map<string, void*> input_data;
  for (int i = 0; i < num_inputs; i++) {
    const Tensor& t = ctx->input(i);
    void* data_address = GetTensorAddress(&t);
    const auto device_tensor = dev_tensors_.at(i).AccessTensor(ctx);
    CHECK_EQ(t.TotalBytes(),
             device_tensor->TotalBytes());  // use the tensor so FW keeps it
    input_data.emplace(StrCat("InputPH_", i), data_address);
  }
  VLOG(2) << "Filled map for sending";
  // copied from cuda_kernel_helper since it seems only valid in *.cu.cc files
  const cudaStream_t* stream = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(ctx->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->CudaStreamMemberHack()));
  calib_res->calibrator_->setBatch(input_data, *stream);
  VLOG(2) << "Passed calibration data";
  return;
}

int TRTEngineOp::GetEngineBatch(tensorflow::OpKernelContext *ctx){
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
      string s("Engine buffer is full. buffer limit= ");
      StrAppend(&s, max_cached_engines_, ", current entries= ");
      for (auto i : cached_engine_batches_) StrAppend(&s, i, ", ");
      StrAppend(&s, "Requested batch= ", num_batch);
      LOG(ERROR) << s;
      ctx->SetStatus(tensorflow::errors::ResourceExhausted(
          "Requested batch size is not available and engine cache is full"));
      return -1;
    }
  }
  return smallest_engine;
}

void TRTEngineOp::ComputeAsync(tensorflow::OpKernelContext* ctx,
                               tensorflow::AsyncOpKernel::DoneCallback done) {
  auto ah = new AsyncHelper(done);
  tensorflow::core::ScopedUnref SC(ah);
  if (calibration_mode_) {
    ah->Ref();
    ExecuteCalibration(ctx, ah);
    return;
  }
  int num_binding = ctx->num_inputs() + ctx->num_outputs();
  std::vector<void*> buffers(num_binding);
  int smallest_engine=GetEngineBatch(ctx);
  if(smallest_engine<0)return;
  int num_batch=ctx->input(0).shape().dim_size(0);
  size_t binding_index;
  auto engine_ctx_pair = GetEngine(smallest_engine, ctx, fixed_input_size_);
  auto trt_engine_ptr_ = engine_ctx_pair.first;
  if (!trt_engine_ptr_) {
    LOG(WARNING) << "Engine retrieval for batch size " << num_batch
                 << " failed Running native segment";
    ExecuteNativeSegment(ctx, ah);
    return;
  }
  for (int i = 0; i < ctx->num_inputs(); i++) {
    string inp_name = "InputPH_";
    // Grab the input tensor
    tensorflow::strings::StrAppend(&inp_name, i);
    binding_index = trt_engine_ptr_->getBindingIndex(inp_name.c_str());

    const Tensor& input_tensor = ctx->input(i);
    const TensorShape& input_shape = input_tensor.shape();
    if (num_batch != input_shape.dim_size(0)) {
      LOG(ERROR) << "input data inconsistent batch size";
      ctx->SetStatus(tensorflow::errors::FailedPrecondition(
          "Different batch sizes between input tensors"));
      return;
    }
    auto dtype = trt_engine_ptr_->getBindingDataType(binding_index);
    switch (dtype) {
      case nvinfer1::DataType::kFLOAT:
        buffers[binding_index] = (void*)(input_tensor.flat<float>().data());
        break;
      case nvinfer1::DataType::kHALF:
        LOG(ERROR) << "FP16 inputs are not supported yet!";
        ctx->SetStatus(tensorflow::errors::InvalidArgument(
            "FP16 inputs are not supported!"));
        return;
        break;
      case nvinfer1::DataType::kINT8:
        LOG(ERROR) << "INT8 inputs are not supported yet!";
        ctx->SetStatus(tensorflow::errors::InvalidArgument(
            "INT8 inputs are not supported!"));
        return;
        break;
      default:
        LOG(ERROR) << "Unknown TRT data type: " << int(dtype);
        ctx->SetStatus(tensorflow::errors::InvalidArgument(
            "Unknown ouput TRT data type! " + int(dtype)));
        return;
        break;
    }
  }

  for (int i = 0; i < ctx->num_outputs(); i++) {
    // This is bad that we have to reallocate output buffer every run.
    // Create an output tensor
    string output_name = "OutputPH_";
    tensorflow::strings::StrAppend(&output_name, i);
    binding_index = trt_engine_ptr_->getBindingIndex(output_name.c_str());
    Tensor* output_tensor = nullptr;

    TensorShape output_shape;
    if (binding_index != -1) {
      auto dims = trt_engine_ptr_->getBindingDimensions(binding_index);
      std::vector<int> trt_shape(dims.nbDims + 1);
      trt_shape[0] = num_batch;
      for (int j = 0; j < dims.nbDims; j++) trt_shape[j + 1] = dims.d[j];
      OP_REQUIRES_OK(
          ctx, TensorShapeUtils::MakeShape(trt_shape.data(), trt_shape.size(),
                                           &output_shape));
    } else {
      LOG(ERROR) << "output node not found, at " << output_name;
      ctx->SetStatus(tensorflow::errors::Internal("output " + output_name +
                                                  " but couldn't be found!"));
      return;
    }
    auto status = ctx->allocate_output(i, output_shape, &output_tensor);
    if (!status.ok()) {
      LOG(ERROR) << "Allocating output failed with " << status;
      ctx->SetStatus(status);
      return;
    }
    auto dtype = trt_engine_ptr_->getBindingDataType(binding_index);
    switch (dtype) {
      case nvinfer1::DataType::kFLOAT:
        buffers[binding_index] =
            reinterpret_cast<void*>(output_tensor->flat<float>().data());
        break;
      case nvinfer1::DataType::kHALF:
        LOG(ERROR) << "half size is not supported yet!";
        ctx->SetStatus(tensorflow::errors::InvalidArgument(
            "Half outputs are not supported!"));
        return;
        break;
      case nvinfer1::DataType::kINT8:
        LOG(ERROR) << "int8 is not supported yet!";
        ctx->SetStatus(tensorflow::errors::InvalidArgument(
            "INT8 outputs are not supported!"));
        return;
        break;
      default:
        LOG(ERROR) << "Unknown TRT data type: " << int(dtype);
        ctx->SetStatus(tensorflow::errors::InvalidArgument(
            "Unsupported output data type! " + int(dtype)));
        return;
        break;
    }
  }
  // copied from cuda_kernel_helper since it seems only valid in *.cu.cc files
  const cudaStream_t* stream = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(ctx->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->CudaStreamMemberHack()));

  // TODO(jie): trt enqueue does not return error
  auto trt_execution_context_ptr = engine_ctx_pair.second;
  auto ret = trt_execution_context_ptr->enqueue(num_batch, &buffers[0], *stream,
                                                nullptr);
  VLOG(2) << "enqueue returns: " << ret;
  // sync should be done by TF.
}

TRTEngineOp::~TRTEngineOp() {
  // Order matters!
  for (auto eng : engine_map_) {
    eng.second.first.reset();
    eng.second.second.reset();
  }
  for (auto alloc : allocators_) alloc.second.reset();
}

TRTEngineOp::EngineCtxPair TRTEngineOp::GetEngine(int batch_size,
                                                   OpKernelContext* ctx,
                                                   bool ignore_dim_change) {
  tensorflow::mutex_lock lock(engine_mutex_);
  if (static_engine_) {
    if (engine_map_.size()) {
      if (engine_map_.begin()->first >= batch_size) {
        return engine_map_.begin()->second;
      } else {
        return {nullptr, nullptr};
      }
    } else {
      IRuntime* infer = nvinfer1::createInferRuntime(logger);
#if NV_TENSORRT_MAJOR > 3
      auto device = ctx->device();
      auto dev_allocator =
          device->GetAllocator(tensorflow::AllocatorAttributes());
      if (!dev_allocator) {
        LOG(FATAL) << "Can't find device allocator for gpu device "
                   << device->name();
      }
      allocator_ = std::make_shared<TRTDeviceAllocator>(dev_allocator);
      infer->setGpuAllocator(allocator_.get());
#endif
      std::shared_ptr<nvinfer1::ICudaEngine> static_engine(
          infer->deserializeCudaEngine(serialized_segment_.c_str(),
                                       serialized_segment_.size(), nullptr),
          Destroyer<nvinfer1::ICudaEngine>());
      engine_map_.insert({static_engine->getMaxBatchSize(),
                          {static_engine,
                           {static_engine->createExecutionContext(),
                            Destroyer<nvinfer1::IExecutionContext>()}}});
      // Runtime is safe to delete after engine creation
      infer->destroy();
      serialized_segment_.clear();
      if (static_engine->getMaxBatchSize() < batch_size) {
        return {nullptr, nullptr};
      }
      return engine_map_.at(static_engine->getMaxBatchSize());
    }
  } else {
    auto engine_it = engine_map_.find(batch_size);
    if (engine_it == engine_map_.end() &&
        engine_map_.size() < (size_t)max_cached_engines_) {
      auto builder_ = std::shared_ptr<nvinfer1::IBuilder>(
          nvinfer1::createInferBuilder(logger),
          Destroyer<nvinfer1::IBuilder>());  // reset the builder to ensure
                                             // device is correct
#if NV_TENSORRT_MAJOR > 3
      auto device = context->device();
      auto device_name = device->name();
      if (allocators_.count(device_name)) {
        builder_->setGpuAllocator(allocators_.at(device_name).get());
      } else {
        std::make_shared<TRTDeviceAllocator> auto dev_allocator =
            device->GetAllocator(tensorflow::AllocatorAttributes());
        if (!dev_allocator) {
          LOG(ERROR) << "Can't find device allocator for gpu device "
                     << device->name();
          ctx->SetStatus(
              tensorflow::errors::Internal("Can't get device allocator"));
          return nullptr;
        }
        auto allocator_ = std::make_shared<TRTDeviceAllocator>(dev_allocator);
        builder_->setGpuAllocator(allocator_.get());
        allocators_.insert({device_name, allocator});
      }
#endif
      VLOG(1) << name() << " Constructing a new engine with batch size "
              << batch_size;
      builder_->setMaxBatchSize(batch_size);
      if (precision_mode_ == tensorflow::tensorrt::convert::FP16MODE) {
        builder_->setHalf2Mode(true);
      } else if (precision_mode_ == tensorflow::tensorrt::convert::INT8MODE) {
        builder_->setInt8Mode(true);
        builder_->setInt8Calibrator(calibrator_.get());
      }
      builder_->setMaxWorkspaceSize(workspace_size_);
      nvinfer1::ICudaEngine* engine = nullptr;
      std::vector<tensorflow::PartialTensorShape> shapes;
      for (int i = 0; i < ctx->num_inputs(); ++i) {
        shapes.emplace_back(ctx->input(i).shape());
      }
      auto status = tensorflow::tensorrt::convert::ConvertSubgraphToEngine(
          segment_graph_, builder_.get(), shapes, &engine, precision_mode_);
      if (engine) {
        engine_map_[batch_size] = {
            std::shared_ptr<nvinfer1::ICudaEngine>(
                engine, Destroyer<nvinfer1::ICudaEngine>()),
            std::shared_ptr<nvinfer1::IExecutionContext>(
                engine->createExecutionContext(),
                Destroyer<nvinfer1::IExecutionContext>())};
      } else {
        LOG(ERROR) << "Engine creation for batch size " << batch_size
                   << " failed";
        ctx->SetStatus(tensorflow::errors::Internal("Engine creation failed!"));
        engine_map_[batch_size] = {nullptr, nullptr};
        return {nullptr, nullptr};
      }
    }
    return engine_map_.at(batch_size);
  }
}

tensorflow::Status TRTEngineOp::AllocateCalibrationResources(
    tensorflow::OpKernelContext* ctx,
    tensorflow::tensorrt::TRTCalibrationResource** cr) {
  auto cres = new TRTCalibrationResource();
  *cr = cres;
  cres->logger_ = new tensorflow::tensorrt::Logger();
  cres->builder_ = nvinfer1::createInferBuilder(*(cres->logger_));
#if NV_TENSORRT_MAJOR > 3
  auto dev = ctx->device();
  auto dev_allocator = dev->GetAllocator(tensorflow::AllocatorAttributes());
  if (!dev_allocator) {
    LOG(WARNING) << "Can't get device allocator will not be able to "
                    "allocate memory from TensorFlow memory pool";
    cres->allocator_ =
        std::make_shared<tensorflow::tensorrt::TRTCudaAllocator>();
  } else {
    cres->allocator_ =
        std::make_shared<tensorflow::tensorrt::TRTDeviceAllocator>(
            dev_allocator);
  }
  cres->builder_->setGpuAllocator(cres->allocator_.get());
#endif
  int batch_size = ctx->input(0).dim_size(0);
  cres->builder_->setMaxBatchSize(batch_size);
  cres->builder_->setInt8Mode(true);
  cres->builder_->setMaxWorkspaceSize(workspace_size_);
  cres->engine_ = nullptr;
  std::vector<tensorflow::PartialTensorShape> shapes;
  int num_inputs = ctx->num_inputs();
  // first run instantiate calibrator
  dev_tensors_.resize(num_inputs);
  VLOG(1) << " Constructing calibrator";
  for (int i = 0; i < num_inputs; i++) {
    // allocate workspace on device for inputs
    const tensorflow::Tensor& t = ctx->input(i);
    shapes.emplace_back(t.shape());
    TF_RETURN_IF_ERROR(ctx->allocate_persistent(t.dtype(), t.shape(),
                                                &dev_tensors_.at(i), nullptr));
    const auto device_tensor = dev_tensors_.at(i).AccessTensor(ctx);
    CHECK_EQ(t.TotalBytes(), device_tensor->TotalBytes());
    void* device_address = GetTensorAddress(device_tensor);
    device_buffers_.emplace(
        StrCat("InputPH_", i),
        std::pair<void*, size_t>(device_address, device_tensor->TotalBytes()));
  }
  cres->calibrator_ =
      new TRTInt8Calibrator(device_buffers_, batch_size, name());
  cres->builder_->setInt8Calibrator(cres->calibrator_);
  string label(name());
  auto segment_graph = &segment_graph_;
  cres->thr_ = new std::thread([cres, label, segment_graph, shapes]() {
    VLOG(1) << "Starting calibration thread, Calibration Resource @ " << cres;
    auto s = tensorflow::tensorrt::convert::ConvertSubgraphToEngine(
        *segment_graph, cres->builder_, shapes, &cres->engine_,
        tensorflow::tensorrt::convert::INT8MODE);  // will loop until we
                                                   // terminate calibration
    if (!s.ok()) {
      LOG(ERROR) << "Calibration thread failed with " << s;
    }
    VLOG(1) << "Calibration loop terminated " << label;
  });
  VLOG(1) << "initialized calibrator resource";
  return tensorflow::Status::OK();
}
REGISTER_KERNEL_BUILDER(Name("TRTEngineOp").Device(DEVICE_GPU), TRTEngineOp);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
