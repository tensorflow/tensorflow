#include "tensorflow/compiler/plugin/poplar/driver/ops/graph_caching_util.h"

#include <poplar/Tensor.hpp>

namespace xla {
namespace poplarplugin {
namespace graph_caching_util {

PoplarTensorSignature GetPoplarTensorSignature(const poplar::Tensor& tensor) {
  return {tensor.elementType(), tensor.shape()};
}

}  // namespace graph_caching_util
}  // namespace poplarplugin
}  // namespace xla
