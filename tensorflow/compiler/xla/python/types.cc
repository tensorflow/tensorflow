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

#include <algorithm>
#include <complex>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/python/exceptions.h"
#include "tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

namespace py = pybind11;

namespace {

struct CustomDtypes {
  py::dtype bfloat16;
  py::dtype float8_e4m3fn;
  py::dtype float8_e4m3b11fnuz;
  py::dtype float8_e4m3fnuz;
  py::dtype float8_e5m2;
  py::dtype float8_e5m2fnuz;
  py::dtype int4;
  py::dtype uint4;
};

const CustomDtypes& GetCustomDtypes() {
  static const CustomDtypes& custom_dtypes = *[]() {
    py::module ml_dtypes = py::module::import("ml_dtypes");
    auto* dtypes = new CustomDtypes;
    dtypes->bfloat16 = py::dtype::from_args(ml_dtypes.attr("bfloat16"));
    dtypes->float8_e4m3fn =
        py::dtype::from_args(ml_dtypes.attr("float8_e4m3fn"));
    dtypes->float8_e5m2 = py::dtype::from_args(ml_dtypes.attr("float8_e5m2"));
    dtypes->float8_e4m3b11fnuz =
        py::dtype::from_args(ml_dtypes.attr("float8_e4m3b11fnuz"));
    dtypes->float8_e4m3fnuz =
        py::dtype::from_args(ml_dtypes.attr("float8_e4m3fnuz"));
    dtypes->float8_e5m2fnuz =
        py::dtype::from_args(ml_dtypes.attr("float8_e5m2fnuz"));
    dtypes->int4 = py::dtype::from_args(ml_dtypes.attr("int4"));
    dtypes->uint4 = py::dtype::from_args(ml_dtypes.attr("uint4"));
    return dtypes;
  }();
  return custom_dtypes;
}

}  // namespace

xla::StatusOr<PrimitiveType> DtypeToPrimitiveType(const py::dtype& np_type) {
  static auto& builtin_dtypes =
      *new absl::flat_hash_map<std::tuple<char, char, int>, PrimitiveType>({
          {{'?', 'b', 1}, PRED},
          {{'b', 'i', 1}, S8},
          {{'h', 'i', 2}, S16},
          {{'i', 'i', 4}, S32},
          {{'l', 'i', 4}, S32},
          {{'q', 'i', 8}, S64},
          {{'l', 'i', 8}, S64},
          {{'B', 'u', 1}, U8},
          {{'H', 'u', 2}, U16},
          {{'I', 'u', 4}, U32},
          {{'L', 'u', 4}, U32},
          {{'Q', 'u', 8}, U64},
          {{'L', 'u', 8}, U64},
          {{'e', 'f', 2}, F16},
          {{'f', 'f', 4}, F32},
          {{'d', 'f', 8}, F64},
          {{'F', 'c', 8}, C64},
          {{'D', 'c', 16}, C128},
      });
  auto builtin_it = builtin_dtypes.find(
      {np_type.char_(), np_type.kind(), np_type.itemsize()});
  if (builtin_it != builtin_dtypes.end()) {
    return builtin_it->second;
  }

  struct DtypeEq {
    bool operator()(const py::dtype& a, const py::dtype& b) const {
      return a.equal(b);
    }
  };
  struct DtypeHash {
    ssize_t operator()(const py::dtype& key) const { return py::hash(key); }
  };
  static auto* custom_dtype_map = []() {
    const CustomDtypes& custom_dtypes = GetCustomDtypes();
    auto* map =
        new absl::flat_hash_map<py::dtype, PrimitiveType, DtypeHash, DtypeEq>();
    map->emplace(custom_dtypes.bfloat16, BF16);
    map->emplace(custom_dtypes.float8_e4m3fn, F8E4M3FN);
    map->emplace(custom_dtypes.float8_e4m3b11fnuz, F8E4M3B11FNUZ);
    map->emplace(custom_dtypes.float8_e4m3fnuz, F8E4M3FNUZ);
    map->emplace(custom_dtypes.float8_e5m2, F8E5M2);
    map->emplace(custom_dtypes.float8_e5m2fnuz, F8E5M2FNUZ);
    map->emplace(custom_dtypes.int4, S4);
    map->emplace(custom_dtypes.uint4, U4);
    return map;
  }();

  auto custom_it = custom_dtype_map->find(np_type);
  if (custom_it != custom_dtype_map->end()) {
    return custom_it->second;
  }
  return InvalidArgument("Unknown NumPy dtype %s char %c kind %c itemsize %d",
                         static_cast<std::string>(py::repr(np_type)),
                         np_type.char_(), np_type.kind(), np_type.itemsize());
}

xla::StatusOr<py::dtype> PrimitiveTypeToDtype(PrimitiveType type) {
  const CustomDtypes& custom_dtypes = GetCustomDtypes();
  switch (type) {
    case PRED:
      return py::dtype::of<bool>();
    case S4:
      return custom_dtypes.int4;
    case S8:
      return py::dtype::of<int8_t>();
    case S16:
      return py::dtype::of<int16_t>();
    case S32:
      return py::dtype::of<int32_t>();
    case S64:
      return py::dtype::of<int64_t>();
    case U4:
      return custom_dtypes.uint4;
    case U8:
      return py::dtype::of<uint8_t>();
    case U16:
      return py::dtype::of<uint16_t>();
    case U32:
      return py::dtype::of<uint32_t>();
    case U64:
      return py::dtype::of<uint64_t>();
    case F8E4M3FN:
      return custom_dtypes.float8_e4m3fn;
    case F8E4M3B11FNUZ:
      return custom_dtypes.float8_e4m3b11fnuz;
    case F8E4M3FNUZ:
      return custom_dtypes.float8_e4m3fnuz;
    case F8E5M2:
      return custom_dtypes.float8_e5m2;
    case F8E5M2FNUZ:
      return custom_dtypes.float8_e5m2fnuz;
    case BF16:
      return custom_dtypes.bfloat16;
    case F16:
      return py::dtype("e");  // PEP 3118 code for "float16
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

StatusOr<pybind11::dtype> IfrtDtypeToDtype(ifrt::DType dtype) {
  const CustomDtypes& custom_dtypes = GetCustomDtypes();
  switch (dtype.kind()) {
    case ifrt::DType::kPred:
      return py::dtype::of<bool>();
    case ifrt::DType::kS4:
      return custom_dtypes.int4;
    case ifrt::DType::kS8:
      return py::dtype::of<int8_t>();
    case ifrt::DType::kS16:
      return py::dtype::of<int16_t>();
    case ifrt::DType::kS32:
      return py::dtype::of<int32_t>();
    case ifrt::DType::kS64:
      return py::dtype::of<int64_t>();
    case ifrt::DType::kU4:
      return custom_dtypes.uint4;
    case ifrt::DType::kU8:
      return py::dtype::of<uint8_t>();
    case ifrt::DType::kU16:
      return py::dtype::of<uint16_t>();
    case ifrt::DType::kU32:
      return py::dtype::of<uint32_t>();
    case ifrt::DType::kU64:
      return py::dtype::of<uint64_t>();
    case ifrt::DType::kF16:
      return py::dtype("e");  // PEP 3118 code for "float16"
    case ifrt::DType::kF32:
      return py::dtype::of<float>();
    case ifrt::DType::kF64:
      return py::dtype::of<double>();
    case ifrt::DType::kBF16:
      return custom_dtypes.bfloat16;
    case ifrt::DType::kC64:
      return py::dtype::of<std::complex<float>>();
    case ifrt::DType::kC128:
      return py::dtype::of<std::complex<double>>();
    case ifrt::DType::kF8E4M3FN:
      return custom_dtypes.float8_e4m3fn;
    case ifrt::DType::kF8E4M3B11FNUZ:
      return custom_dtypes.float8_e4m3b11fnuz;
    case ifrt::DType::kF8E4M3FNUZ:
      return custom_dtypes.float8_e4m3fnuz;
    case ifrt::DType::kF8E5M2:
      return custom_dtypes.float8_e5m2;
    case ifrt::DType::kF8E5M2FNUZ:
      return custom_dtypes.float8_e5m2fnuz;
    case ifrt::DType::kString:
      // PEP 3118 code for "pointer to Python Object". We use Python objects
      // instead of 'U' (Unicode string) or 'V' (raw data) because the latter
      // two are fixed length, and thus, require encoding the maximum length as
      // part of dtype. Using 'O' allows us to represent variable-length bytes
      // and is also consistent with TensorFlow's tensor -> ndarray conversion
      // logic (see `TF_DataType_to_PyArray_TYPE`).
      return py::dtype("O");
    default:
      return Unimplemented("Unimplemented primitive type %s",
                           dtype.DebugString());
  }
}

const NumpyScalarTypes& GetNumpyScalarTypes() {
  static const NumpyScalarTypes* singleton = []() {
    NumpyScalarTypes* dtypes = new NumpyScalarTypes();
    py::module numpy = py::module::import("numpy");
    py::module ml_dtypes = py::module::import("ml_dtypes");
    dtypes->np_bool = py::object(numpy.attr("bool_"));
    dtypes->np_int4 = py::object(ml_dtypes.attr("int4"));
    dtypes->np_int8 = py::object(numpy.attr("int8"));
    dtypes->np_int16 = py::object(numpy.attr("int16"));
    dtypes->np_int32 = py::object(numpy.attr("int32"));
    dtypes->np_int64 = py::object(numpy.attr("int64"));
    dtypes->np_uint4 = py::object(ml_dtypes.attr("uint4"));
    dtypes->np_uint8 = py::object(numpy.attr("uint8"));
    dtypes->np_uint16 = py::object(numpy.attr("uint16"));
    dtypes->np_uint32 = py::object(numpy.attr("uint32"));
    dtypes->np_uint64 = py::object(numpy.attr("uint64"));
    dtypes->np_bfloat16 = py::object(ml_dtypes.attr("bfloat16"));
    dtypes->np_float8_e4m3fn = py::object(ml_dtypes.attr("float8_e4m3fn"));
    dtypes->np_float8_e4m3b11fnuz =
        py::object(ml_dtypes.attr("float8_e4m3b11fnuz"));
    dtypes->np_float8_e5m2 = py::object(ml_dtypes.attr("float8_e5m2"));
    dtypes->np_float8_e4m3fnuz = py::object(ml_dtypes.attr("float8_e4m3fnuz"));
    dtypes->np_float8_e5m2fnuz = py::object(ml_dtypes.attr("float8_e5m2fnuz"));
    dtypes->np_float16 = py::object(numpy.attr("float16"));
    dtypes->np_float32 = py::object(numpy.attr("float32"));
    dtypes->np_float64 = py::object(numpy.attr("float64"));
    dtypes->np_complex64 = py::object(numpy.attr("complex64"));
    dtypes->np_complex128 = py::object(numpy.attr("complex128"));
    dtypes->np_longlong = py::object(numpy.attr("longlong"));
    dtypes->np_intc = py::object(numpy.attr("intc"));
    return dtypes;
  }();
  return *singleton;
}

const char* PEP3118FormatDescriptorForPrimitiveType(PrimitiveType type) {
  // We use an "=" prefix to indicate that we prefer "standard" types like
  // np.int32 rather than "native" types like np.cint. pybind11 does not qualify
  // its format descriptors.
  switch (type) {
    case PRED:
      return "?";
    case S8:
      return "=b";
    case S16:
      return "=h";
    case S32:
      return "=i";
    case S64:
      return "=q";
    case U8:
      return "=B";
    case U16:
      return "=H";
    case U32:
      return "=I";
    case U64:
      return "=Q";
    case F16:
      return "=e";
    case F32:
      return "=f";
    case F64:
      return "=d";
    case C64:
      return "=Zf";
    case C128:
      return "=Zd";
    default:
      return nullptr;
  }
}

StatusOr<py::str> TypeDescriptorForPrimitiveType(PrimitiveType type) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define ENDIAN_PREFIX "<"
#else
#define ENDIAN_PREFIX ">"
#endif
  switch (type) {
    case PRED:
      return py::str("|b1");
    case S8:
      return py::str("|i1");
    case S16:
      return py::str(ENDIAN_PREFIX "i2");
    case S32:
      return py::str(ENDIAN_PREFIX "i4");
    case S64:
      return py::str(ENDIAN_PREFIX "i8");
    case U8:
      return py::str("|u1");
    case U16:
      return py::str(ENDIAN_PREFIX "u2");
    case U32:
      return py::str(ENDIAN_PREFIX "u4");
    case U64:
      return py::str(ENDIAN_PREFIX "u8");
    case BF16:
      return py::str(ENDIAN_PREFIX "V2");
    case F16:
      return py::str(ENDIAN_PREFIX "f2");
    case F32:
      return py::str(ENDIAN_PREFIX "f4");
    case F64:
      return py::str(ENDIAN_PREFIX "f8");
    case C64:
      return py::str(ENDIAN_PREFIX "c8");
    case C128:
      return py::str(ENDIAN_PREFIX "c16");
    default:
      return Unimplemented("Unimplemented primitive type %s",
                           PrimitiveType_Name(type));
  }
}

PrimitiveType Squash64BitTypes(PrimitiveType type) {
  switch (type) {
    case S64:
      return S32;
    case U64:
      return U32;
    case F64:
      return F32;
    case C128:
      return C64;
    default:
      return type;
  }
}

// Returns the strides for `shape`.
std::vector<int64_t> ByteStridesForShape(const Shape& shape) {
  std::vector<int64_t> strides;
  CHECK(shape.IsArray());
  CHECK(shape.has_layout());
  return ByteStridesForShape(shape.element_type(), shape.dimensions(),
                             shape.layout());
}

static std::vector<int64_t> StridesForShapeHelper(
    PrimitiveType element_type, absl::Span<const int64_t> dimensions,
    const xla::Layout& layout, int64_t innermost_stride_size) {
  CHECK_EQ(dimensions.size(), layout.minor_to_major().size());
  std::vector<int64_t> strides;
  strides.resize(dimensions.size());
  int64_t stride = innermost_stride_size;
  for (int i : layout.minor_to_major()) {
    strides[i] = stride;
    stride *= dimensions[i];
  }
  return strides;
}

std::vector<int64_t> ByteStridesForShape(PrimitiveType element_type,
                                         absl::Span<const int64_t> dimensions,
                                         const xla::Layout& layout) {
  return StridesForShapeHelper(
      element_type, dimensions, layout,
      ShapeUtil::ByteSizeOfPrimitiveType(element_type));
}

std::vector<int64_t> StridesForShape(PrimitiveType element_type,
                                     absl::Span<const int64_t> dimensions,
                                     const xla::Layout& layout) {
  return StridesForShapeHelper(element_type, dimensions, layout,
                               /*innermost_stride_size=*/1);
}

StatusOr<py::object> LiteralToPython(std::shared_ptr<xla::Literal> literal) {
  xla::Literal& m = *literal;
  if (m.shape().IsTuple()) {
    std::vector<Literal> elems = m.DecomposeTuple();
    std::vector<py::object> arrays(elems.size());
    for (int i = 0; i < elems.size(); ++i) {
      TF_ASSIGN_OR_RETURN(
          arrays[i],
          LiteralToPython(std::make_unique<Literal>(std::move(elems[i]))));
    }
    py::tuple result(elems.size());
    for (int i = 0; i < elems.size(); ++i) {
      PyTuple_SET_ITEM(result.ptr(), i, arrays[i].release().ptr());
    }
    return result;
  }
  TF_RET_CHECK(m.shape().IsArray());

  py::object literal_object = py::cast(literal);
  TF_ASSIGN_OR_RETURN(py::dtype dtype,
                      PrimitiveTypeToDtype(m.shape().element_type()));
  return py::array(dtype, m.shape().dimensions(),
                   ByteStridesForShape(m.shape()), m.untyped_data(),
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

template <typename IntType>
static py::tuple IntSpanToTupleHelper(absl::Span<IntType const> xs) {
  py::tuple out(xs.size());
  for (int i = 0; i < xs.size(); ++i) {
    out[i] = py::int_(xs[i]);
  }
  return out;
}

template <>
pybind11::tuple SpanToTuple(absl::Span<int const> xs) {
  return IntSpanToTupleHelper(xs);
}
template <>
pybind11::tuple SpanToTuple(absl::Span<int64_t const> xs) {
  return IntSpanToTupleHelper(xs);
}

std::optional<CastToArrayResult> CastToArray(py::handle h) {
  py::array array = py::array::ensure(
      h, py::array::c_style | py::detail::npy_api::NPY_ARRAY_ALIGNED_);
  if (!array) {
    return std::nullopt;
  }
  auto type_or_status = DtypeToPrimitiveType(array.dtype());
  if (!type_or_status.ok()) {
    throw xla::XlaRuntimeError(type_or_status.status());
  }
  PrimitiveType type = type_or_status.value();

  absl::InlinedVector<int64_t, 4> dims(array.ndim());
  for (int i = 0; i < array.ndim(); ++i) {
    dims[i] = array.shape(i);
  }
  Shape shape = ShapeUtil::MakeShape(type, dims);
  if (array.size() * array.itemsize() != ShapeUtil::ByteSizeOf(shape)) {
    throw xla::XlaRuntimeError(absl::StrCat(
        "Size mismatch for buffer: ", array.size() * array.itemsize(), " vs. ",
        ShapeUtil::ByteSizeOf(shape)));
  }
  return CastToArrayResult{array, static_cast<const char*>(array.data()),
                           shape};
}

}  // namespace xla
