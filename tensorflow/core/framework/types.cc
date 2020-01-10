#include "tensorflow/core/framework/types.h"

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

const char* const DEVICE_CPU = "CPU";
const char* const DEVICE_GPU = "GPU";

string DataTypeString(DataType dtype) {
  if (IsRefType(dtype)) {
    DataType non_ref = static_cast<DataType>(dtype - kDataTypeRefOffset);
    return strings::StrCat(DataTypeString(non_ref), "_ref");
  }
  switch (dtype) {
    case DT_INVALID:
      return "INVALID";
    case DT_FLOAT:
      return "float";
    case DT_DOUBLE:
      return "double";
    case DT_INT32:
      return "int32";
    case DT_UINT8:
      return "uint8";
    case DT_INT16:
      return "int16";
    case DT_INT8:
      return "int8";
    case DT_STRING:
      return "string";
    case DT_COMPLEX64:
      return "complex64";
    case DT_INT64:
      return "int64";
    case DT_BOOL:
      return "bool";
    case DT_QINT8:
      return "qint8";
    case DT_QUINT8:
      return "quint8";
    case DT_QINT32:
      return "qint32";
    case DT_BFLOAT16:
      return "bfloat16";
    default:
      LOG(FATAL) << "Unrecognized DataType enum value " << dtype;
      return "";
  }
}

bool DataTypeFromString(StringPiece sp, DataType* dt) {
  if (sp.ends_with("_ref")) {
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
  } else if (sp == "uint8") {
    *dt = DT_UINT8;
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
  } else if (sp == "int64") {
    *dt = DT_INT64;
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
  } else if (sp == "qint32") {
    *dt = DT_QINT32;
    return true;
  } else if (sp == "bfloat16") {
    *dt = DT_BFLOAT16;
    return true;
  }
  return false;
}

string DeviceTypeString(DeviceType device_type) { return device_type.type(); }

string DataTypeSliceString(const DataTypeSlice types) {
  string out;
  for (auto it = types.begin(); it != types.end(); ++it) {
    strings::StrAppend(&out, ((it == types.begin()) ? "" : ", "),
                       DataTypeString(*it));
  }
  return out;
}

DataTypeVector AllTypes() {
  return {DT_FLOAT, DT_DOUBLE, DT_INT32,     DT_UINT8, DT_INT16,
          DT_INT8,  DT_STRING, DT_COMPLEX64, DT_INT64, DT_BOOL,
          DT_QINT8, DT_QUINT8, DT_QINT32};
}

#ifndef __ANDROID__

DataTypeVector RealNumberTypes() {
  return {DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8};
}

DataTypeVector QuantizedTypes() { return {DT_QINT8, DT_QUINT8, DT_QINT32}; }

DataTypeVector RealAndQuantizedTypes() {
  return {DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64,  DT_UINT8,
          DT_INT16, DT_INT8,   DT_QINT8, DT_QUINT8, DT_QINT32};
}

DataTypeVector NumberTypes() {
  return {DT_FLOAT, DT_DOUBLE,    DT_INT64, DT_INT32,  DT_UINT8, DT_INT16,
          DT_INT8,  DT_COMPLEX64, DT_QINT8, DT_QUINT8, DT_QINT32};
}

#else  // __ANDROID__

DataTypeVector RealNumberTypes() { return {DT_FLOAT, DT_INT32}; }

DataTypeVector NumberTypes() {
  return {DT_FLOAT, DT_INT32, DT_QINT8, DT_QUINT8, DT_QINT32};
}

DataTypeVector QuantizedTypes() { return {DT_QINT8, DT_QUINT8, DT_QINT32}; }

DataTypeVector RealAndQuantizedTypes() {
  return {DT_FLOAT, DT_INT32, DT_QINT8, DT_QUINT8, DT_QINT32};
}

#endif  // __ANDROID__

// TODO(jeff): Maybe unify this with Tensor::CanUseDMA, or the underlying
// is_simple<T> in tensor.cc (and possible choose a more general name?)
bool DataTypeCanUseMemcpy(DataType dt) {
  switch (dt) {
    case DT_FLOAT:
    case DT_DOUBLE:
    case DT_INT32:
    case DT_UINT8:
    case DT_INT16:
    case DT_INT8:
    case DT_COMPLEX64:
    case DT_INT64:
    case DT_BOOL:
    case DT_QINT8:
    case DT_QUINT8:
    case DT_QINT32:
    case DT_BFLOAT16:
      return true;
    default:
      return false;
  }
}

bool DataTypeIsQuantized(DataType dt) {
  switch (dt) {
    case DT_QINT8:
    case DT_QUINT8:
    case DT_QINT32:
      return true;
    default:
      return false;
  }
}

}  // namespace tensorflow
