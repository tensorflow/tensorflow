#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_KERNELS_XCORE_OPS_RESOLVER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_KERNELS_XCORE_OPS_RESOLVER_H_

#include "tensorflow/lite/experimental/micro/compatibility.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace micro {

class XcoreOpsResolver : public MicroMutableOpResolver {
 public:
  XcoreOpsResolver();

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_KERNELS_XCORE_OPS_RESOLVER_H_
