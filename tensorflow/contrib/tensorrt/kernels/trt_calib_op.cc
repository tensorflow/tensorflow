//
// Created by skama on 1/25/18.
//

#include "tensorflow/contrib/tensorrt/kernels/trt_calib_op.h"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "tensorflow/contrib/tensorrt/resources/TRTInt8Calibrator.h"
#include "tensorflow/contrib/tensorrt/resources/TRTResourceManager.h"
#include "tensorflow/contrib/tensorrt/resources/TRTResources.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace trt {
TRTCalibOp::TRTCalibOp(OpKernelConstruction* context) : OpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("segment_nodes", &segment_nodes_));
  OP_REQUIRES_OK(context, context->GetAttr("input_names", &input_names_));
  OP_REQUIRES_OK(context, context->GetAttr("resource_name", &repo_name));
};

#define TYPECASE(dt, X, Y)                                             \
  case dt: {                                                           \
    Y = (void*)X->flat<tensorflow::EnumToDataType<dt>::Type>().data(); \
    break;                                                             \
  }
#define GET_TENSOR_ADDRESS(tensor_ptr, dest_ptr)               \
  {                                                            \
    auto TENSOR_TYPE = tensor_ptr->dtype();                    \
    switch (TENSOR_TYPE) {                                     \
      TYPECASE(tensorflow::DT_FLOAT, tensor_ptr, dest_ptr);    \
      TYPECASE(tensorflow::DT_HALF, tensor_ptr, dest_ptr);     \
      TYPECASE(tensorflow::DT_INT8, tensor_ptr, dest_ptr);     \
      default: {                                               \
        LOG(FATAL) << "Unsupported Data type "                 \
                   << tensorflow::DataTypeString(TENSOR_TYPE); \
        break;                                                 \
      }                                                        \
    }                                                          \
  }
void TRTCalibOp::Compute(tensorflow::OpKernelContext* ctx) {
  auto trt_rm = tensorflow::trt::TRTResourceManager::instance();
  VLOG(0) << "Op Name= " << name() << " nodedef name= " << repo_name;
  auto resmgr = trt_rm->getManager("TRTCalibOps");
  tensorflow::trt::TRTCalibrationResource* calibRes = nullptr;
  auto status = resmgr->Lookup(repo_name, repo_name, &calibRes);
  VLOG(0) << "SAMI status " << status.ToString();
  if (status.ok()) {
    int batchSize = ctx->input(0).dim_size(0);
    VLOG(0) << "SAMI Batchsize= " << batchSize;
    int numInputs = ctx->num_inputs();
    VLOG(0) << "SAMI numInputs= " << numInputs;
    dev_tensors_.resize(numInputs);
    if (calibRes->calibrator == nullptr) {
      VLOG(0) << " Constructing calibrator";
      // first run
      for (int i = 0; i < numInputs; i++) {
        const tensorflow::Tensor& t = ctx->input(i);
        VLOG(0) << "Tensor " << i << " " << t.shape().DebugString();
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_persistent(t.dtype(), t.shape(),
                                                &dev_tensors_.at(i), nullptr));
        const auto dTensor = dev_tensors_.at(i).AccessTensor(ctx);
        CHECK_EQ(t.TotalBytes(), dTensor->TotalBytes());
        void* devAddr = nullptr;
        GET_TENSOR_ADDRESS(dTensor, devAddr)
        device_buffers_.emplace(
            input_names_.at(i),
            std::pair<void*, size_t>(devAddr, dTensor->TotalBytes()));
      }
      calibRes->calibrator = new TRTInt8Calibrator(device_buffers_, batchSize);
      calibRes->thr = new std::thread([calibRes]() {
        calibRes->engine = calibRes->builder->buildCudaEngine(
            *calibRes->network);  // will loop until we terminate calibrator
        VLOG(1) << "Calibration loop terminated";
      });
      VLOG(0) << "SAMI intialized calibrator resource";
    }

    std::unordered_map<std::string, void*> input_data;
    for (int i = 0; i < numInputs; i++) {
      const Tensor& t = ctx->input(i);
      void* data_address = nullptr;
      const Tensor* t_ptr = &t;
      GET_TENSOR_ADDRESS(t_ptr, data_address);
      const auto dTensor = dev_tensors_.at(i).AccessTensor(ctx);
      CHECK_EQ(t.TotalBytes(),
               dTensor->TotalBytes());  // use the tensor so FW keeps it
      input_data.emplace(input_names_.at(i), data_address);
      ctx->set_output(i, t);
    }
    VLOG(0) << "Filled map";
    calibRes->calibrator->setBatch(input_data);
    VLOG(0) << "Passed calibration data";
  } else {
    ctx->SetStatus(status);
    return;
  }
};

#undef TYPECASE

REGISTER_KERNEL_BUILDER(Name("TRTCalibOp").Device(DEVICE_GPU), TRTCalibOp);

}  // namespace trt
}  // namespace tensorflow