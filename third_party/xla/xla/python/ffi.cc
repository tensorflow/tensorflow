/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/ffi.h"

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "include/dlpack/dlpack.h"
#include "llvm/Support/Casting.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/callback.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/nb_absl_span.h"  // IWYU pragma: keep
#include "xla/python/nb_numpy.h"
#include "xla/python/py_host_callback.h"
#include "xla/python/types.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

namespace nb = nanobind;

namespace {
const char* const kDlTensorCapsuleName = "dltensor";

struct DLPackTensor {
  std::vector<int64_t> shape;
  DLManagedTensor tensor;
};

void DLPackTensorDeleter(DLManagedTensor* t) {
  if (t) {
    delete static_cast<DLPackTensor*>(t->manager_ctx);
  }
}

absl::StatusOr<DLDeviceType> DLDeviceTypeForDeviceType(int32_t device_type) {
  switch (device_type) {
    case nb::device::cpu::value:
      return kDLCPU;
    case nb::device::cuda::value:
      return kDLCUDA;
    case nb::device::rocm::value:
      return kDLROCM;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported device type: %d", device_type));
  }
}

absl::StatusOr<DLDevice> DLDeviceForDevice(int32_t device_type,
                                           int32_t device_ordinal) {
  DLDevice context;
  TF_ASSIGN_OR_RETURN(context.device_type,
                      DLDeviceTypeForDeviceType(device_type));
  context.device_id = device_ordinal;
  return context;
}

absl::StatusOr<DLDataType> PrimitiveTypeToDLDataType(PrimitiveType type) {
  switch (type) {
    case S8:
      return DLDataType{kDLInt, 8, 1};
    case S16:
      return DLDataType{kDLInt, 16, 1};
    case S32:
      return DLDataType{kDLInt, 32, 1};
    case S64:
      return DLDataType{kDLInt, 64, 1};
    case U8:
      return DLDataType{kDLUInt, 8, 1};
    case U16:
      return DLDataType{kDLUInt, 16, 1};
    case U32:
      return DLDataType{kDLUInt, 32, 1};
    case U64:
      return DLDataType{kDLUInt, 64, 1};
    case F16:
      return DLDataType{kDLFloat, 16, 1};
    case F32:
      return DLDataType{kDLFloat, 32, 1};
    case F64:
      return DLDataType{kDLFloat, 64, 1};
    case BF16:
      return DLDataType{kDLBfloat, 16, 1};
    case PRED:
      return DLDataType{kDLBool, 8, 1};
    case C64:
      return DLDataType{kDLComplex, 64, 1};
    case C128:
      return DLDataType{kDLComplex, 128, 1};
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "XLA type %s has no DLPack equivalent", PrimitiveType_Name(type)));
  }
}
}  // namespace

class PyContext {
 public:
  enum Stage {
    kInstantiate = XLA_FFI_ExecutionStage_INSTANTIATE,
    kPrepare = XLA_FFI_ExecutionStage_PREPARE,
    kInitialize = XLA_FFI_ExecutionStage_INITIALIZE,
    kExecute = XLA_FFI_ExecutionStage_EXECUTE,
  };

  PyContext(const XLA_FFI_Api* api, XLA_FFI_ExecutionContext* ctx,
            XLA_FFI_ExecutionStage stage)
      : api_(api), ctx_(ctx), stage_(stage) {}

  Stage stage() const { return static_cast<Stage>(stage_); }

  absl::StatusOr<uintptr_t> stream() const {
    XLA_FFI_Stream_Get_Args args;
    args.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
    args.extension_start = nullptr;
    args.ctx = ctx_;
    args.stream = nullptr;
    if (XLA_FFI_Error* error = api_->XLA_FFI_Stream_Get(&args)) {
      return ffi::TakeStatus(error);
    }
    return absl::bit_cast<uintptr_t>(args.stream);
  }

  int32_t device_ordinal() const {
    return api_->internal_api->XLA_FFI_INTERNAL_DeviceOrdinal_Get(ctx_);
  }

 private:
  const XLA_FFI_Api* api_;
  XLA_FFI_ExecutionContext* ctx_;
  XLA_FFI_ExecutionStage stage_;
};

namespace {
template <bool ReadOnly, typename Framework>
struct ArrayType {
  typedef nb::ndarray<Framework> type;
};

template <typename Framework>
struct ArrayType<true, Framework> {
  typedef nb::ndarray<nb::ro, Framework> type;
};
}  // namespace

class PyAnyBuffer {
 public:
  PyAnyBuffer(int32_t device_type, int32_t device_ordinal, void* data,
              absl::Span<int64_t const> dimensions, PrimitiveType element_type)
      : device_type_(device_type),
        device_ordinal_(device_ordinal),
        data_(data),
        dimensions_(dimensions),
        element_type_(element_type) {}
  explicit PyAnyBuffer(int32_t device_type, int32_t device_ordinal,
                       ffi::AnyBuffer buf)
      : PyAnyBuffer(device_type, device_ordinal, buf.untyped_data(),
                    buf.dimensions(), buf.element_type()) {}
  explicit PyAnyBuffer(int32_t device_type, int32_t device_ordinal,
                       ffi::Result<ffi::AnyBuffer> buf)
      : PyAnyBuffer(device_type, device_ordinal, buf->untyped_data(),
                    buf->dimensions(), buf->element_type()) {}

  absl::StatusOr<nb_dtype> dtype() const {
    return PrimitiveTypeToNbDtype(element_type_);
  }
  absl::Span<int64_t const>::size_type ndim() const {
    return dimensions_.size();
  }
  nb::tuple shape() const { return SpanToNbTuple(dimensions_); }

  absl::StatusOr<nb_numpy_ndarray> NumpyArray() const {
    if (device_type_ != nb::device::cpu::value) {
      return absl::UnimplementedError(
          "Buffer.__array__ is only supported on CPU.");
    }

    TF_ASSIGN_OR_RETURN(auto dtype, this->dtype());
    nb_numpy_ndarray array(dtype, dimensions_, /* strides= */ std::nullopt,
                           data_, nb::cast(this));

    // TODO(danfm): We don't seem to be allowed to set this flag like this
    // because the array doesn't own its data.
    // array.attr("flags").attr("writeable") = nb::bool_(writeable_);

    return array;
  }

  absl::StatusOr<nb::dict> CudaArrayInterface() const {
    if (device_type_ != nb::device::cuda::value) {
      return absl::UnimplementedError(
          "Buffer.__cuda_array_interface__ is only supported on CUDA.");
    }

    switch (element_type_) {
      case PrimitiveType::PRED:
      case PrimitiveType::S8:
      case PrimitiveType::S16:
      case PrimitiveType::S32:
      case PrimitiveType::S64:
      case PrimitiveType::U8:
      case PrimitiveType::U16:
      case PrimitiveType::U32:
      case PrimitiveType::U64:
      case PrimitiveType::BF16:
      case PrimitiveType::F16:
      case PrimitiveType::F32:
      case PrimitiveType::F64:
      case PrimitiveType::C64:
      case PrimitiveType::C128:
        break;

      default:
        return absl::UnimplementedError(absl::StrFormat(
            "Buffer.__cuda_array_interface__ is not supported for %s buffers.",
            PrimitiveType_Name(element_type_)));
    }

    nb::dict result;
    result["shape"] = SpanToNbTuple(dimensions_);
    TF_ASSIGN_OR_RETURN(result["typestr"],
                        TypeDescriptorForPrimitiveType(element_type_));
    nb::tuple data =
        nb::make_tuple(nb::int_(absl::bit_cast<std::uintptr_t>(data_)),
                       /* !writeable_ */ false);
    result["data"] = data;
    result["version"] = nb::int_(2);
    return result;
  }

  absl::StatusOr<nb::capsule> DLPack() const {
    auto pack = std::make_unique<DLPackTensor>();
    pack->tensor.manager_ctx = pack.get();
    pack->tensor.deleter = DLPackTensorDeleter;

    // Using legacy DLPack API, because that's all PyTorch supports, but we
    // should eventually update to the versioned API.
    // pack->tensor.version =
    //     DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    // pack->tensor.flags = writeable_ ? 0 : DLPACK_FLAG_BITMASK_READ_ONLY;

    DLTensor& dt = pack->tensor.dl_tensor;
    dt.data = data_;
    TF_ASSIGN_OR_RETURN(dt.device,
                        DLDeviceForDevice(device_type_, device_ordinal_));
    dt.ndim = dimensions_.size();
    TF_ASSIGN_OR_RETURN(dt.dtype, PrimitiveTypeToDLDataType(element_type_));
    pack->shape = std::vector<int64_t>(dimensions_.begin(), dimensions_.end());
    dt.shape = reinterpret_cast<std::int64_t*>(pack->shape.data());
    dt.strides = nullptr;
    dt.byte_offset = 0;

    // We cannot use nanobind's capsule object constructor because we need to
    // detect if the capsule name has been changed in the deleter, but nanobind
    // hides the underlying Python object from the deleter.
    nb::capsule capsule = nb::steal<nb::capsule>(
        PyCapsule_New(&pack.release()->tensor, kDlTensorCapsuleName,
                      [](PyObject* obj) noexcept {
                        DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(
                            PyCapsule_GetPointer(obj, kDlTensorCapsuleName));
                        if (dlmt) {
                          DLPackTensorDeleter(dlmt);
                        } else {
                          // The tensor has been deleted. Clear any error from
                          // PyCapsule_GetPointer.
                          PyErr_Clear();
                        }
                      }));
    if (!capsule.ptr()) {
      throw nb::python_error();
    }

    return capsule;
  }

  nb::tuple DLPackDevice() const {
    return nb::make_tuple(device_type_, device_ordinal_);
  }

 private:
  int32_t device_type_;
  int32_t device_ordinal_;
  void* data_;
  absl::Span<int64_t const> dimensions_;
  PrimitiveType element_type_;
};

template <XLA_FFI_ExecutionStage Stage, typename Device>
absl::Status FfiCallbackImpl(
    int32_t device_ordinal, const XLA_FFI_Api* api,
    XLA_FFI_ExecutionContext* ctx,
    std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>* callbacks,
    uint64_t index, ffi::RemainingArgs args, ffi::RemainingRets rets) {
  if (index >= callbacks->size()) {
    return absl::InvalidArgumentError("Callback index out of range.");
  }
  auto loaded_callback = llvm::dyn_cast_or_null<PyCpuLoadedHostCallback>(
      callbacks->at(index).get());
  if (loaded_callback == nullptr) {
    return absl::InternalError(
        "Expected a PyCpuLoadedHostCallback, got something else.");
  }
  CpuCallback* callback = loaded_callback->cpu_callback();

  nb::gil_scoped_acquire gil;
  auto nb_args =
      nb::steal<nb::tuple>(PyTuple_New(1 + args.size() + rets.size()));

  PyContext py_ctx(api, ctx, Stage);
  PyTuple_SET_ITEM(nb_args.ptr(), 0, nb::cast(py_ctx).release().ptr());

  size_t offset = 1;
  for (size_t i = 0; i < args.size(); ++i, ++offset) {
    TF_ASSIGN_OR_RETURN(auto arg, args.get<ffi::AnyBuffer>(i));
    PyAnyBuffer py_buffer(Device::value, device_ordinal, arg);
    PyTuple_SET_ITEM(nb_args.ptr(), offset,
                     nb::cast(py_buffer).release().ptr());
  }

  for (size_t i = 0; i < rets.size(); ++i, ++offset) {
    TF_ASSIGN_OR_RETURN(auto ret, rets.get<ffi::AnyBuffer>(i));
    PyAnyBuffer py_buffer(Device::value, device_ordinal, ret);
    PyTuple_SET_ITEM(nb_args.ptr(), offset,
                     nb::cast(py_buffer).release().ptr());
  }

  EnterHostCallback();
  absl::StatusOr<nb::tuple> maybe_result_tuple = callback->FfiCall(nb_args);
  LeaveHostCallback();
  TF_RETURN_IF_ERROR(maybe_result_tuple.status());

  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(
    kFfiCallback,
    (FfiCallbackImpl<XLA_FFI_ExecutionStage_EXECUTE, nb::device::cpu>),
    ffi::Ffi::Bind()
        .Ctx<ffi::DeviceOrdinal>()
        .Ctx<ffi::FfiApi>()
        .Ctx<ffi::FfiExecutionContext>()
        .Ctx<ffi::UserData<
            std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>>>()
        .Attr<uint64_t>("index")
        .RemainingArgs()
        .RemainingRets());
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_python_buffer_callback",
                         "Host", kFfiCallback);

XLA_FFI_DEFINE_HANDLER(
    kFfiCallbackCuda,
    (FfiCallbackImpl<XLA_FFI_ExecutionStage_EXECUTE, nb::device::cuda>),
    ffi::Ffi::Bind()
        .Ctx<ffi::DeviceOrdinal>()
        .Ctx<ffi::FfiApi>()
        .Ctx<ffi::FfiExecutionContext>()
        .Ctx<ffi::UserData<
            std::vector<tsl::RCReference<ifrt::LoadedHostCallback>>>>()
        .Attr<uint64_t>("index")
        .RemainingArgs()
        .RemainingRets());
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_python_buffer_callback",
                         "CUDA", kFfiCallbackCuda);

void BuildFfiSubmodule(nb::module_& m) {
  tsl::ImportNumpy();

  nb::module_ ffi_module =
      m.def_submodule("ffi", "Python bindings for the XLA FFI.");

  nb::class_<PyAnyBuffer> buffer(ffi_module, "Buffer");
  buffer.def_prop_ro("dtype", ValueOrThrowWrapper(&PyAnyBuffer::dtype));
  buffer.def_prop_ro("ndim", &PyAnyBuffer::ndim);
  buffer.def_prop_ro("shape", &PyAnyBuffer::shape);
  buffer.def(
      "__array__",
      [](PyAnyBuffer self, nb::object dtype, nb::object copy) {
        if (!dtype.is_none()) {
          throw nb::value_error(
              "dtype parameter is not supported by Buffer.__array__.");
        }
        if (!copy.is_none() && nb::cast<bool>(copy)) {
          throw nb::value_error(
              "Buffer.__array__ with copy=True is not supported.");
        }
        return ValueOrThrow(self.NumpyArray());
      },
      nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none());
  buffer.def_prop_ro("__cuda_array_interface__",
                     ValueOrThrowWrapper(&PyAnyBuffer::CudaArrayInterface));
  buffer.def(
      "__dlpack__",
      [](PyAnyBuffer self, nb::object stream, nb::object max_version,
         nb::object dl_device, nb::object copy) {
        if (!copy.is_none() && nb::cast<bool>(copy)) {
          throw nb::value_error(
              "Buffer.__dlpack__ with copy=True is not supported.");
        }
        // TODO(danfm): Handle other optional inputs.
        return ValueOrThrow(self.DLPack());
      },
      nb::arg("stream") = nb::none(), nb::arg("max_version") = nb::none(),
      nb::arg("dl_device") = nb::none(), nb::arg("copy") = nb::none());
  buffer.def("__dlpack_device__", &PyAnyBuffer::DLPackDevice);

  nb::enum_<PyContext::Stage>(ffi_module, "ExecutionStage")
      .value("INSTANTIATE", PyContext::Stage::kInstantiate)
      .value("PREPARE", PyContext::Stage::kPrepare)
      .value("INITIALIZE", PyContext::Stage::kInitialize)
      .value("EXECUTE", PyContext::Stage::kExecute)
      .export_values();

  nb::class_<PyContext> context(ffi_module, "ExecutionContext");
  context.def_prop_ro("stage", &PyContext::stage);
  context.def_prop_ro("device_ordinal", &PyContext::device_ordinal);
  context.def_prop_ro("stream", ValueOrThrowWrapper(&PyContext::stream));
}

}  // namespace xla
