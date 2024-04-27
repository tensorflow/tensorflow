/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/python/types.h"

#include <Python.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/ndarray.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/exceptions.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/nb_helpers.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace nb = nanobind;

namespace {

struct CustomDtypes {
  nb_dtype bfloat16;
  nb_dtype float8_e4m3fn;
  nb_dtype float8_e4m3b11fnuz;
  nb_dtype float8_e4m3fnuz;
  nb_dtype float8_e5m2;
  nb_dtype float8_e5m2fnuz;
  nb_dtype int4;
  nb_dtype uint4;
};

const CustomDtypes& GetCustomDtypes() {
  static const CustomDtypes& custom_dtypes = *[]() {
    nb::module_ ml_dtypes = nb::module_::import_("ml_dtypes");
    auto* dtypes = new CustomDtypes;
    dtypes->bfloat16 = nb_dtype::from_args(ml_dtypes.attr("bfloat16"));
    dtypes->float8_e4m3fn =
        nb_dtype::from_args(ml_dtypes.attr("float8_e4m3fn"));
    dtypes->float8_e5m2 = nb_dtype::from_args(ml_dtypes.attr("float8_e5m2"));
    dtypes->float8_e4m3b11fnuz =
        nb_dtype::from_args(ml_dtypes.attr("float8_e4m3b11fnuz"));
    dtypes->float8_e4m3fnuz =
        nb_dtype::from_args(ml_dtypes.attr("float8_e4m3fnuz"));
    dtypes->float8_e5m2fnuz =
        nb_dtype::from_args(ml_dtypes.attr("float8_e5m2fnuz"));
    dtypes->int4 = nb_dtype::from_args(ml_dtypes.attr("int4"));
    dtypes->uint4 = nb_dtype::from_args(ml_dtypes.attr("uint4"));
    return dtypes;
  }();
  return custom_dtypes;
}

}  // namespace

absl::StatusOr<PrimitiveType> DtypeToPrimitiveType(const nb_dtype& np_type) {
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
    bool operator()(const nb_dtype& a, const nb_dtype& b) const {
      return a.equal(b);
    }
  };
  struct DtypeHash {
    ssize_t operator()(const nb_dtype& key) const { return nb_hash(key); }
  };
  static auto* custom_dtype_map = []() {
    const CustomDtypes& custom_dtypes = GetCustomDtypes();
    auto* map =
        new absl::flat_hash_map<nb_dtype, PrimitiveType, DtypeHash, DtypeEq>();
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
                         nb::cast<std::string_view>(nb::repr(np_type)),
                         np_type.char_(), np_type.kind(), np_type.itemsize());
}

absl::StatusOr<nb_dtype> PrimitiveTypeToNbDtype(PrimitiveType type) {
  const CustomDtypes& custom_dtypes = GetCustomDtypes();
  auto to_nb_dtype = [](int typenum) -> nb_dtype {
    return nb::steal<nb_dtype>(
        reinterpret_cast<PyObject*>(PyArray_DescrFromType(typenum)));
  };
  switch (type) {
    case PRED:
      return to_nb_dtype(NPY_BOOL);
    case S4:
      return custom_dtypes.int4;
    case S8:
      return to_nb_dtype(NPY_INT8);
    case S16:
      return to_nb_dtype(NPY_INT16);
    case S32:
      return to_nb_dtype(NPY_INT32);
    case S64:
      return to_nb_dtype(NPY_INT64);
    case U4:
      return custom_dtypes.uint4;
    case U8:
      return to_nb_dtype(NPY_UINT8);
    case U16:
      return to_nb_dtype(NPY_UINT16);
    case U32:
      return to_nb_dtype(NPY_UINT32);
    case U64:
      return to_nb_dtype(NPY_UINT64);
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
      return to_nb_dtype(NPY_HALF);
    case F32:
      return to_nb_dtype(NPY_FLOAT);
    case F64:
      return to_nb_dtype(NPY_DOUBLE);
    case C64:
      return to_nb_dtype(NPY_COMPLEX64);
    case C128:
      return to_nb_dtype(NPY_COMPLEX128);
    default:
      return Unimplemented("Unimplemented primitive type %s",
                           PrimitiveType_Name(type));
  }
}

absl::StatusOr<nb_dtype> IfrtDtypeToNbDtype(ifrt::DType dtype) {
  const CustomDtypes& custom_dtypes = GetCustomDtypes();
  auto to_nb_dtype = [](int typenum) -> nb_dtype {
    return nb::steal<nb_dtype>(
        reinterpret_cast<PyObject*>(PyArray_DescrFromType(typenum)));
  };
  switch (dtype.kind()) {
    case ifrt::DType::kPred:
      return to_nb_dtype(NPY_BOOL);
    case ifrt::DType::kS4:
      return custom_dtypes.int4;
    case ifrt::DType::kS8:
      return to_nb_dtype(NPY_INT8);
    case ifrt::DType::kS16:
      return to_nb_dtype(NPY_INT16);
    case ifrt::DType::kS32:
      return to_nb_dtype(NPY_INT32);
    case ifrt::DType::kS64:
      return to_nb_dtype(NPY_INT64);
    case ifrt::DType::kU4:
      return custom_dtypes.uint4;
    case ifrt::DType::kU8:
      return to_nb_dtype(NPY_UINT8);
    case ifrt::DType::kU16:
      return to_nb_dtype(NPY_UINT16);
    case ifrt::DType::kU32:
      return to_nb_dtype(NPY_UINT32);
    case ifrt::DType::kU64:
      return to_nb_dtype(NPY_UINT64);
    case ifrt::DType::kF16:
      return to_nb_dtype(NPY_HALF);
    case ifrt::DType::kF32:
      return to_nb_dtype(NPY_FLOAT);
    case ifrt::DType::kF64:
      return to_nb_dtype(NPY_DOUBLE);
    case ifrt::DType::kBF16:
      return custom_dtypes.bfloat16;
    case ifrt::DType::kC64:
      return to_nb_dtype(NPY_COMPLEX64);
    case ifrt::DType::kC128:
      return to_nb_dtype(NPY_COMPLEX128);
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
      return to_nb_dtype(NPY_OBJECT);
    default:
      return Unimplemented("Unimplemented primitive type %s",
                           dtype.DebugString());
  }
}

absl::StatusOr<ifrt::DType> DtypeToIfRtDType(nb_dtype dtype) {
  TF_ASSIGN_OR_RETURN(auto primitive_type, DtypeToPrimitiveType(dtype));
  return ifrt::ToDType(primitive_type);
}

absl::StatusOr<nb_dtype> IfrtDtypeToDtypeWithTokenCanonicalization(
    ifrt::DType dtype) {
  if (dtype.kind() == ifrt::DType::kToken) {
    // Treat token as bool.
    return nb::steal<nb_dtype>(
        reinterpret_cast<PyObject*>(PyArray_DescrFromType(NPY_BOOL)));
  }

  return IfrtDtypeToNbDtype(dtype);
}

const NumpyScalarTypes& GetNumpyScalarTypes() {
  static const NumpyScalarTypes* singleton = []() {
    NumpyScalarTypes* dtypes = new NumpyScalarTypes();
    nb::module_ numpy = nb::module_::import_("numpy");
    nb::module_ ml_dtypes = nb::module_::import_("ml_dtypes");
    dtypes->np_bool = nb::object(numpy.attr("bool_"));
    dtypes->np_int4 = nb::object(ml_dtypes.attr("int4"));
    dtypes->np_int8 = nb::object(numpy.attr("int8"));
    dtypes->np_int16 = nb::object(numpy.attr("int16"));
    dtypes->np_int32 = nb::object(numpy.attr("int32"));
    dtypes->np_int64 = nb::object(numpy.attr("int64"));
    dtypes->np_uint4 = nb::object(ml_dtypes.attr("uint4"));
    dtypes->np_uint8 = nb::object(numpy.attr("uint8"));
    dtypes->np_uint16 = nb::object(numpy.attr("uint16"));
    dtypes->np_uint32 = nb::object(numpy.attr("uint32"));
    dtypes->np_uint64 = nb::object(numpy.attr("uint64"));
    dtypes->np_bfloat16 = nb::object(ml_dtypes.attr("bfloat16"));
    dtypes->np_float8_e4m3fn = nb::object(ml_dtypes.attr("float8_e4m3fn"));
    dtypes->np_float8_e4m3b11fnuz =
        nb::object(ml_dtypes.attr("float8_e4m3b11fnuz"));
    dtypes->np_float8_e5m2 = nb::object(ml_dtypes.attr("float8_e5m2"));
    dtypes->np_float8_e4m3fnuz = nb::object(ml_dtypes.attr("float8_e4m3fnuz"));
    dtypes->np_float8_e5m2fnuz = nb::object(ml_dtypes.attr("float8_e5m2fnuz"));
    dtypes->np_float16 = nb::object(numpy.attr("float16"));
    dtypes->np_float32 = nb::object(numpy.attr("float32"));
    dtypes->np_float64 = nb::object(numpy.attr("float64"));
    dtypes->np_complex64 = nb::object(numpy.attr("complex64"));
    dtypes->np_complex128 = nb::object(numpy.attr("complex128"));
    dtypes->np_longlong = nb::object(numpy.attr("longlong"));
    dtypes->np_intc = nb::object(numpy.attr("intc"));
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

absl::StatusOr<nb::str> TypeDescriptorForPrimitiveType(PrimitiveType type) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define ENDIAN_PREFIX "<"
#else
#define ENDIAN_PREFIX ">"
#endif
  switch (type) {
    case PRED:
      return nb::str("|b1");
    case S8:
      return nb::str("|i1");
    case S16:
      return nb::str(ENDIAN_PREFIX "i2");
    case S32:
      return nb::str(ENDIAN_PREFIX "i4");
    case S64:
      return nb::str(ENDIAN_PREFIX "i8");
    case U8:
      return nb::str("|u1");
    case U16:
      return nb::str(ENDIAN_PREFIX "u2");
    case U32:
      return nb::str(ENDIAN_PREFIX "u4");
    case U64:
      return nb::str(ENDIAN_PREFIX "u8");
    case BF16:
      return nb::str(ENDIAN_PREFIX "V2");
    case F16:
      return nb::str(ENDIAN_PREFIX "f2");
    case F32:
      return nb::str(ENDIAN_PREFIX "f4");
    case F64:
      return nb::str(ENDIAN_PREFIX "f8");
    case C64:
      return nb::str(ENDIAN_PREFIX "c8");
    case C128:
      return nb::str(ENDIAN_PREFIX "c16");
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

absl::StatusOr<nb::object> LiteralToPython(
    std::shared_ptr<xla::Literal> literal) {
  xla::Literal& m = *literal;
  if (m.shape().IsTuple()) {
    std::vector<Literal> elems = m.DecomposeTuple();
    std::vector<nb::object> arrays(elems.size());
    for (int i = 0; i < elems.size(); ++i) {
      TF_ASSIGN_OR_RETURN(
          arrays[i],
          LiteralToPython(std::make_unique<Literal>(std::move(elems[i]))));
    }
    nb::tuple result = nb::steal<nb::tuple>(PyTuple_New(elems.size()));
    for (int i = 0; i < elems.size(); ++i) {
      PyTuple_SET_ITEM(result.ptr(), i, arrays[i].release().ptr());
    }
    return result;
  }
  TF_RET_CHECK(m.shape().IsArray());

  nb::object literal_object = nb::cast(literal);
  TF_ASSIGN_OR_RETURN(nb_dtype dtype,
                      PrimitiveTypeToNbDtype(m.shape().element_type()));
  return nb_numpy_ndarray(dtype, m.shape().dimensions(),
                          ByteStridesForShape(m.shape()), m.untyped_data(),
                          literal_object);
}

nb::tuple MutableSpanToNbTuple(absl::Span<nb::object> xs) {
  nb::tuple out = nb::steal<nb::tuple>(PyTuple_New(xs.size()));
  for (int i = 0; i < xs.size(); ++i) {
    PyTuple_SET_ITEM(out.ptr(), i, xs[i].release().ptr());
  }
  return out;
}

std::optional<CastToArrayResult> CastToArray(nb::handle h) {
  auto array =
      nb_numpy_ndarray::ensure(h, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
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
