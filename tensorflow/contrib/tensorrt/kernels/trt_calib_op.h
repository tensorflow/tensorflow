//
// Created by skama on 1/25/18.
//

#ifndef TFGITHUB_TRT_CALIB_OP_H
#define TFGITHUB_TRT_CALIB_OP_H

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace trt {
class TRTCalibOp: public OpKernel {
public:
  explicit TRTCalibOp(OpKernelConstruction* context);

  void Compute(OpKernelContext* context) override;

 private:
  std::vector<std::string> segment_nodes_;
  std::vector<std::string> input_names_;
  std::vector<tensorflow::TensorShape> shapes_;
  std::unordered_map<std::string, std::pair<void*, size_t>> device_buffers_;
  std::vector<tensorflow::PersistentTensor> dev_tensors_;

};
}
}
#endif //TFGITHUB_TRT_CALIB_OP_H
