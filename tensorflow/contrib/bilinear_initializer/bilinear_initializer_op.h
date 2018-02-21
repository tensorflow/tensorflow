#ifndef TENSORFLOW_CONTRIB_BILINEAR_INITIALIZER_BILINEAR_INITIALIZER_OPS_H_
#define TENSORFLOW_CONTRIB_BILINEAR_INITIALIZER_BILINEAR_INITIALIZER_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace bilinear_initializer {

/* Bilinear Initializer Operation
 *
 * Computes batch filters with bilinear weights. Result is a
 * 4D tensor of dimension [H, W, C, N] where:
 * (1) H = height of kernel
 * (2) W = width of kernel
 * (3) C = number of input channels
 * (4) N = number of output channels (number of filters)
 * Typical use case is weight initialization for deconvolution layer.
 */
template<typename T>
class BilinearInitializerOp : public OpKernel {
  public:
    explicit BilinearInitializerOp(OpKernelConstruction* context);

    void Compute(OpKernelContext* context) override;
};

} // namespace bilinear_initializer
} // namespace tensorflow

#endif
