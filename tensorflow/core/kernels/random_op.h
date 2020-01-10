#ifndef TENSORFLOW_KERNELS_RANDOM_OP_H_
#define TENSORFLOW_KERNELS_RANDOM_OP_H_

namespace tensorflow {

class OpKernelContext;

namespace functor {

template <typename Device, class Distribution>
struct FillPhiloxRandom;

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_RANDOM_OP_H_
