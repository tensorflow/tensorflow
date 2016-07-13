
#ifndef TENSORFLOW_KERNELS_NORM_OPS_H_
#define TENSORFLOW_KERNELS_NORM_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct Norm1D {
   void operator()(const Device& d, typename TTypes<T>::ConstFlat x,
		   typename TTypes<T>::ConstFlat mean,
		   typename TTypes<T>::ConstFlat stdd,
		   typename TTypes<T>::Flat out);
};

} // end namespace functor
} // end namespace tensorflow
#endif // TENSORFLOW_KERNELS_NORM_OPS_H_
