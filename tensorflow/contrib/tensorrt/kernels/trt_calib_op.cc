//
// Created by skama on 1/25/18.
//

#include "tensorflow/contrib/tensorrt/kernels/trt_calib_op.h"
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include "tensorflow/contrib/tensorrt/resources/TRTInt8Calibrator.h"
#include "tensorflow/contrib/tensorrt/resources/TRTResourceManager.h"
#include "tensorflow/contrib/tensorrt/resources/TRTResources.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
namespace tensorflow{
namespace trt{
TRTCalibOp::TRTCalibOp(OpKernelConstruction* context) : OpKernel(context){
  OP_REQUIRES_OK(context,
                 context->GetAttr("segment_nodes", &segment_nodes_));
  OP_REQUIRES_OK(context, context->GetAttr("input_names", &input_names_));
  dev_tensors_.resize(segment_nodes_.size());

};

void TRTCalibOp::Compute(OpKernelContext *ctx) {
  auto trt_rm = tensorflow::trt::TRTResourceManager::instance();
  auto resmgr = trt_rm->getManager(name());
  TRTCalibrationResource *calibRes= nullptr;
  auto status=resmgr->Lookup(name(), name(), &calibRes);
  if (status.ok()){
    int batchSize=ctx->input(0).dim_size(0);
    int numInputs=ctx->num_inputs();
    if ( calibRes->calibrator == nullptr){// first run
      for(int i = 0 ; i < numInputs; i++){
        const Tensor& t=ctx->input(i);
        OP_REQUIRES_OK(ctx, ctx->allocate_persistent(t.dtype(), t.shape(),&dev_tensors_.at(i), nullptr));
        const auto dTensor=dev_tensors_.at(i).AccessTensor(ctx);
        CHECK_EQ(t.TotalBytes(),dTensor->TotalBytes());
        auto dType=t.dtype();
        void* devAddr=(void*)dTensor->flat<tensorflow::EnumToDataType<dType>::Type>().data();
        device_buffers_.emplace({input_names_.at(i),std::make_pair(devAddr,dTensor->TotalBytes())});
      }
      calibRes->calibrator=new TRTInt8Calibrator(device_buffers_,batchSize);
      auto builder=calibRes->builder;
      calibRes->thr=new std::thread([calibRes](){
        calibRes->engine=calibRes->builder->buildCudaEngine(*calibRes->network);  // will loop until we terminate calibrator
      });
    }
    std::unordered_map<std::string, void*> input_data;
    for(int i = 0; i < numInputs; i++){
      const Tensor& t = ctx->input(i);
      auto dType = t.dtype();
      void* data_address = (void*)t.flat<tensorflow::EnumToDataType<dType>::Type>().data();
      const auto dTensor = dev_tensors_.at(i).AccessTensor(ctx);
      CHECK_EQ(t.TotalBytes(), dTensor->TotalBytes()); // use the tensor so FW keeps it
      input_data.emplace(input_names_.at(i), data_address);
      ctx->set_output(i,t);
    }
    calibRes->calibrator->setBatch(input_data);
  }else{
    ctx->SetStatus(status);
    return;
  }

};

}
}