#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XCORE_OPS_RESOLVER_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XCORE_OPS_RESOLVER_H_

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

constexpr int num_xcore_ops = 13;

void add_custom_ops(MicroMutableOpResolver<num_xcore_ops> *resolver);

class XcoreOpsResolver : public MicroMutableOpResolver<num_xcore_ops> {
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
