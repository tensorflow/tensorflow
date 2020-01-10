/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/register_types.h"

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

bool DeviceType::operator<(const DeviceType& other) const {
  return type_ < other.type_;
}

bool DeviceType::operator==(const DeviceType& other) const {
  return type_ == other.type_;
}

std::ostream& operator<<(std::ostream& os, const DeviceType& d) {
  os << d.type();
  return os;
}

const char* const DEVICE_DEFAULT = "DEFAULT";
const char* const DEVICE_CPU = "CPU";
const char* const DEVICE_GPU = "GPU";
const char* const DEVICE_SYCL = "SYCL";

const std::string DeviceName<Eigen::ThreadPoolDevice>::value = DEVICE_CPU;
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
const std::string DeviceName<Eigen::GpuDevice>::value = DEVICE_GPU;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#ifdef TENSORFLOW_USE_SYCL
const std::string DeviceName<Eigen::SyclDevice>::value = DEVICE_SYCL;
#endif  // TENSORFLOW_USE_SYCL

namespace {
string DataTypeStringInternal(DataType dtype) {
  switch (dtype) {
    case DT_INVALID:
      return "INVALID";
    case DT_FLOAT:
      return "float";
    case DT_DOUBLE:
      return "double";
    case DT_INT32:
      return "int32";
    case DT_UINT32:
      return "uint32";
    case DT_UINT8:
      return "uint8";
    case DT_UINT16:
      return "uint16";
    case DT_INT16:
      return "int16";
    case DT_INT8:
      return "int8";
    case DT_STRING:
      return "string";
    case DT_COMPLEX64:
      return "complex64";
    case DT_COMPLEX128:
      return "complex128";
    case DT_INT64:
      return "int64";
    case DT_UINT64:
      return "uint64";
    case DT_BOOL:
      return "bool";
    case DT_QINT8:
      return "qint8";
    case DT_QUINT8:
      return "quint8";
    case DT_QUINT16:
      return "quint16";
    case DT_QINT16:
      return "qint16";
    case DT_QINT32:
      return "qint32";
    case DT_BFLOAT16:
      return "bfloat16";
    case DT_HALF:
      return "half";
    case DT_RESOURCE:
      return "resource";
    case DT_VARIANT:
      return "variant";
    default:
      LOG(ERROR) << "Unrecognized DataType enum value " << dtype;
      return strings::StrCat("unknown dtype enum (", dtype, ")");
  }
}
}  // end namespace

string DataTypeString(DataType dtype) {
  if (IsRefType(dtype)) {
    DataType non_ref = static_cast<DataType>(dtype - kDataTypeRefOffset);
    return strings::StrCat(DataTypeStringInternal(non_ref), "_ref");
  }
  return DataTypeStringInternal(dtype);
}

bool DataTypeFromString(StringPiece sp, DataType* dt) {
  if (str_util::EndsWith(sp, "_ref")) {
    sp.remove_suffix(4);
    DataType non_ref;
    if (DataTypeFromString(sp, &non_ref) && !IsRefType(non_ref)) {
      *dt = static_cast<DataType>(non_ref + kDataTypeRefOffset);
      return true;
    } else {
      return false;
    }
  }

  if (sp == "float" || sp == "float32") {
    *dt = DT_FLOAT;
    return true;
  } else if (sp == "double" || sp == "float64") {
    *dt = DT_DOUBLE;
    return true;
  } else if (sp == "int32") {
    *dt = DT_INT32;
    return true;
  } else if (sp == "uint32") {
    *dt = DT_UINT32;
    return true;
  } else if (sp == "uint8") {
    *dt = DT_UINT8;
    return true;
  } else if (sp == "uint16") {
    *dt = DT_UINT16;
    return true;
  } else if (sp == "int16") {
    *dt = DT_INT16;
    return true;
  } else if (sp == "int8") {
    *dt = DT_INT8;
    return true;
  } else if (sp == "string") {
    *dt = DT_STRING;
    return true;
  } else if (sp == "complex64") {
    *dt = DT_COMPLEX64;
    return true;
  } else if (sp == "complex128") {
    *dt = DT_COMPLEX128;
    return true;
  } else if (sp == "int64") {
    *dt = DT_INT64;
    return true;
  } else if (sp == "uint64") {
    *dt = DT_UINT64;
    return true;
  } else if (sp == "bool") {
    *dt = DT_BOOL;
    return true;
  } else if (sp == "qint8") {
    *dt = DT_QINT8;
    return true;
  } else if (sp == "quint8") {
    *dt = DT_QUINT8;
    return true;
  } else if (sp == "qint16") {
    *dt = DT_QINT16;
    return true;
  } else if (sp == "quint16") {
    *dt = DT_QUINT16;
    return true;
  } else if (sp == "qint32") {
    *dt = DT_QINT32;
    return true;
  } else if (sp == "bfloat16") {
    *dt = DT_BFLOAT16;
    return true;
  } else if (sp == "half" || sp == "float16") {
    *dt = DT_HALF;
    return true;
  } else if (sp == "resource") {
    *dt = DT_RESOURCE;
    return true;
  } else if (sp == "variant") {
    *dt = DT_VARIANT;
    return true;
  }
  return false;
}

string DeviceTypeString(const DeviceType& device_type) {
  return device_type.type();
}

string DataTypeSliceString(const DataTypeSlice types) {
  string out;
  for (auto it = types.begin(); it != types.end(); ++it) {
    strings::StrAppend(&out, ((it == types.begin()) ? "" : ", "),
                       DataTypeString(*it));
  }
  return out;
}

bool DataTypeAlwaysOnHost(DataType dt) {
  // Includes DT_STRING and DT_RESOURCE.
  switch (dt) {
    case DT_STRING:
    case DT_STRING_REF:
    case DT_RESOURCE:
      return true;
    default:
      return false;
  }
}

int DataTypeSize(DataType dt) {
#define CASE(T)                  \
  case DataTypeToEnum<T>::value: \
    return sizeof(T);
  switch (dt) {
    TF_CALL_POD_TYPES(CASE);
    TF_CALL_QUANTIZED_TYPES(CASE);
    // TF_CALL_QUANTIZED_TYPES() macro does no cover quint16 and qint16, since
    // they are not supported widely, but are explicitly listed here for
    // bitcast.
    TF_CALL_qint16(CASE);
    TF_CALL_quint16(CASE);

    // uint32 and uint64 aren't included in TF_CALL_POD_TYPES because we
    // don't want to define kernels for them at this stage to avoid binary
    // bloat.
    TF_CALL_uint32(CASE);
    TF_CALL_uint64(CASE);
    default:
      return 0;
  }
#undef CASE
}

}  // namespace tensorflow
