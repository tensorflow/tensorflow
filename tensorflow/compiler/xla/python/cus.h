
#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_CUS_TYPE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_CUS_TYPE_H_

#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

xla::StatusOr<pybind11::object> CusDtype();

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_CUS_TYPE_H_
