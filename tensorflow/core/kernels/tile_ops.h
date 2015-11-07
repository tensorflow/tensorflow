#ifndef TENSORFLOW_KERNELS_TILE_OPS_H_
#define TENSORFLOW_KERNELS_TILE_OPS_H_

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace functor {

template <typename Device, typename T, int NDIM>
struct Tile {
  void operator()(const Device& d, typename TTypes<T, NDIM>::Tensor out,
                  typename TTypes<T, NDIM>::ConstTensor in,
                  const Eigen::array<int32, NDIM>& broadcast_array) const {
    out.device(d) = in.broadcast(broadcast_array);
  }
};

template <typename Device, typename T, int NDIM>
struct TileGrad {
  void operator()(const Device& d, typename TTypes<T, NDIM>::Tensor out,
                  typename TTypes<T, NDIM>::ConstTensor in,
                  const Eigen::DSizes<ptrdiff_t, NDIM>& indices,
                  const Eigen::DSizes<ptrdiff_t, NDIM>& sizes,
                  bool first) const {
    if (first) {
      out.device(d) = in.slice(indices, sizes);
    } else {
      out.device(d) += in.slice(indices, sizes);
    }
  }
};

template <typename Device, typename T, int NDIM, int REDUCEDNDIM>
struct ReduceAndReshape {
  void operator()(const Device& d, typename TTypes<T, NDIM>::Tensor out,
                  typename TTypes<T, NDIM>::ConstTensor in,
                  const Eigen::DSizes<ptrdiff_t, REDUCEDNDIM>& reduce_dim,
                  const Eigen::DSizes<ptrdiff_t, NDIM>& reshape_dim) const {
    out.device(d) = in.sum(reduce_dim).reshape(reshape_dim);
  }
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_TILE_OPS_H_
