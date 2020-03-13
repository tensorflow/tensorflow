#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XCORE_OPS_RESOLVER_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XCORE_OPS_RESOLVER_H_

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

class XcoreOpsResolver : public AllOpsResolver {
 public:
  XcoreOpsResolver();

 private:

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XCORE_OPS_RESOLVER_H_
