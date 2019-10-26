/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/python/types.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

namespace py = pybind11;

xla::StatusOr<PrimitiveType> DtypeToPrimitiveType(const py::dtype& np_type) {
  static auto* types =
      new absl::flat_hash_map<std::pair<char, int>, PrimitiveType>({
          {{'b', 1}, PRED},
          {{'i', 1}, S8},
          {{'i', 2}, S16},
          {{'i', 4}, S32},
          {{'i', 8}, S64},
          {{'u', 1}, U8},
          {{'u', 2}, U16},
          {{'u', 4}, U32},
          {{'u', 8}, U64},
          {{'f', 2}, F16},
          {{'f', 4}, F32},
          {{'f', 8}, F64},
          {{'c', 8}, C64},
          {{'c', 16}, C128},
      });
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    return InvalidArgument("Unknown NumPy type %c size %d", np_type.kind(),
                           np_type.itemsize());
  }
  return it->second;
}

xla::StatusOr<py::dtype> PrimitiveTypeToDtype(PrimitiveType type) {
  switch (type) {
    case PRED:
      return py::dtype::of<bool>();
    case S8:
      return py::dtype::of<int8>();
    case S16:
      return py::dtype::of<int16>();
    case S32:
      return py::dtype::of<int32>();
    case S64:
      return py::dtype::of<int64>();
    case U8:
      return py::dtype::of<uint8>();
    case U16:
      return py::dtype::of<uint16>();
    case U32:
      return py::dtype::of<uint32>();
    case U64:
      return py::dtype::of<uint64>();
    case F16:
      return py::dtype("e");
    case F32:
      return py::dtype::of<float>();
    case F64:
      return py::dtype::of<double>();
    case C64:
      return py::dtype::of<std::complex<float>>();
    case C128:
      return py::dtype::of<std::complex<double>>();
    default:
      return Unimplemented("Unimplemented primitive type %s",
                           PrimitiveType_Name(type));
  }
}

// Returns a numpy-style format descriptor string for `type`.
StatusOr<std::string> FormatDescriptorForPrimitiveType(PrimitiveType type) {
  switch (type) {
    case PRED:
      return py::format_descriptor<bool>::format();
    case S8:
      return py::format_descriptor<int8>::format();
    case S16:
      return py::format_descriptor<int16>::format();
    case S32:
      return py::format_descriptor<int32>::format();
    case S64:
      return py::format_descriptor<int64>::format();
    case U8:
      return py::format_descriptor<uint8>::format();
    case U16:
      return py::format_descriptor<uint16>::format();
    case U32:
      return py::format_descriptor<uint32>::format();
    case U64:
      return py::format_descriptor<uint64>::format();
    case F16:
      return std::string("e");
    case F32:
      return py::format_descriptor<float>::format();
    case F64:
      return py::format_descriptor<double>::format();
    case C64:
      return py::format_descriptor<std::complex<float>>::format();
    case C128:
      return py::format_descriptor<std::complex<double>>::format();
    default:
      return Unimplemented("Unimplemented primitive type %s",
                           PrimitiveType_Name(type));
  }
}

// Returns the strides for `shape`.
std::vector<ssize_t> StridesForShape(const Shape& shape) {
  std::vector<ssize_t> strides;
  CHECK(shape.IsArray());
  CHECK(shape.has_layout());

  strides.resize(shape.dimensions_size());
  ssize_t stride = ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  for (int i : shape.layout().minor_to_major()) {
    strides.at(i) = stride;
    stride *= shape.dimensions(i);
  }
  return strides;
}

StatusOr<py::object> LiteralToPython(std::shared_ptr<xla::Literal> literal) {
  xla::Literal& m = *literal;
  if (m.shape().IsTuple()) {
    std::vector<Literal> elems = m.DecomposeTuple();
    std::vector<py::object> arrays(elems.size());
    for (int i = 0; i < elems.size(); ++i) {
      TF_ASSIGN_OR_RETURN(
          arrays[i],
          LiteralToPython(absl::make_unique<Literal>(std::move(elems[i]))));
    }
    py::tuple result(elems.size());
    for (int i = 0; i < elems.size(); ++i) {
      PyTuple_SET_ITEM(result.ptr(), i, arrays[i].release().ptr());
    }
    return result;
  }
  TF_RET_CHECK(m.shape().IsArray());

  py::object literal_object = py::cast(literal);
  TF_ASSIGN_OR_RETURN(std::string format, FormatDescriptorForPrimitiveType(
                                              m.shape().element_type()));
  py::buffer_info info(
      m.untyped_data(),  // Pointer to buffer
      xla::ShapeUtil::ByteSizeOfPrimitiveType(
          m.shape().element_type()),  // Size of one scalar
      format,                         // Python struct-style format descriptor
      m.shape().dimensions_size(),    // Number of dimensions
      m.shape().dimensions(),         // Buffer dimensions
      StridesForShape(m.shape())      // Strides (in bytes) for each index
  );
  return py::array(pybind11::dtype(info), info.shape, info.strides, info.ptr,
                   literal_object);
}

StatusOr<PythonBufferTree> GetPythonBufferTree(const py::object& argument) {
  PythonBufferTree tree;
  if (py::isinstance<py::tuple>(argument)) {
    py::tuple tuple = py::reinterpret_borrow<py::tuple>(argument);
    std::vector<Shape> host_shapes(tuple.size());
    for (int i = 0; i < host_shapes.size(); ++i) {
      TF_ASSIGN_OR_RETURN(PythonBufferTree subtree,
                          GetPythonBufferTree(tuple[i]));
      tree.leaves.reserve(tree.leaves.size() + subtree.leaves.size());
      std::move(subtree.leaves.begin(), subtree.leaves.end(),
                std::back_inserter(tree.leaves));
      tree.arrays.reserve(tree.arrays.size() + subtree.arrays.size());
      std::move(subtree.arrays.begin(), subtree.arrays.end(),
                std::back_inserter(tree.arrays));
      host_shapes[i] = std::move(subtree.shape);
    }
    tree.shape = ShapeUtil::MakeTupleShape(host_shapes);
  } else {
    pybind11::detail::type_caster<BorrowingLiteral> caster;
    if (!caster.load(argument, /*convert=*/true)) {
      return InvalidArgument("Invalid array value.");
    }
    DCHECK_EQ(caster.arrays.size(), 1);
    tree.arrays.push_back(std::move(caster.arrays.front()));
    tree.leaves.push_back(std::move(*caster));
    tree.shape = tree.leaves.front().shape();
  }
  return tree;
}

py::tuple IntSpanToTuple(absl::Span<int64 const> xs) {
  py::tuple out(xs.size());
  for (int i = 0; i < xs.size(); ++i) {
    out[i] = py::int_(xs[i]);
  }
  return out;
}

std::vector<int64> IntSequenceToVector(const py::object& sequence) {
  std::vector<int64> output;
  for (auto item : sequence) {
    output.push_back(item.cast<int64>());
  }
  return output;
}

}  // namespace xla
