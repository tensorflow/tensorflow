/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"
#include "third_party/tensorrt/NvInferPlugin.h"

namespace tensorflow {
namespace tensorrt {
using absl::StrAppend;
using absl::StrCat;
void InitializeTrtPlugins() {
  static mutex plugin_mutex(LINKER_INITIALIZED);
  static bool plugin_initialized = false;
  static Logger trt_logger;
  mutex_lock lock(plugin_mutex);
  if (plugin_initialized) return;

  plugin_initialized = initLibNvInferPlugins(&trt_logger, "");
  if (!plugin_initialized) {
    LOG(ERROR) << "Failed to initialize TensorRT plugins, and conversion may "
                  "fail later.";
  }

  int num_trt_plugins = 0;
  nvinfer1::IPluginCreator* const* trt_plugin_creator_list =
      getPluginRegistry()->getPluginCreatorList(&num_trt_plugins);
  if (!trt_plugin_creator_list) {
    LOG(WARNING) << "Can not find any TensorRT plugins in registry.";
  } else {
    VLOG(1) << "Found the following " << num_trt_plugins
            << " TensorRT plugins in registry:";
    for (int i = 0; i < num_trt_plugins; ++i) {
      if (!trt_plugin_creator_list[i]) {
        LOG(WARNING) << "TensorRT plugin at index " << i
                     << " is not accessible (null pointer returned by "
                        "getPluginCreatorList for this plugin)";
      } else {
        VLOG(1) << "  " << trt_plugin_creator_list[i]->getPluginName();
      }
    }
  }
}
#define TYPESTR(x) TypeToString<x>()
template <typename T>
string TypeToString();
#define TYPESTRING(x)             \
  template <>                     \
  string TypeToString<x>() {      \
    return std::move(string(#x)); \
  }
TYPESTRING(float);
TYPESTRING(double);
TYPESTRING(int8);
TYPESTRING(short);
TYPESTRING(int);
TYPESTRING(Eigen::half);
#undef TYPESTRING

template <typename T>
void TypeDump(string* msg, T val) {
  StrAppend(msg, val, " ");
}
template <>
void TypeDump<Eigen::half>(string* msg, Eigen::half val) {
  StrAppend(msg, Eigen::half_impl::half_to_float(val), " ");
}
template <typename T>
void DumpField(const char* name, const Tensor& vals, int max_len) {
  int to_dump = std::min((int)vals.NumElements(), max_len);
  const T* val_vec = vals.flat<T>().data();
  string msg =
      StrCat(name, " , ", TYPESTR(T), "[", vals.NumElements(), "]", " v= ");
  for (int f = 0; f < to_dump; f++) {
    TypeDump(&msg, *(val_vec + f));
  }
  VLOG(0) << "Field " << msg;
}
#undef TYPESTR

Status ConstructPlugin(const AttrSlice& attrs, const string& name,
                       nvinfer1::IPluginV2*& plugin, bool validation_only) {
  plugin = nullptr;

  string plugin_name, plugin_version;
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "plugin_name", &plugin_name));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "plugin_version", &plugin_version));
  auto registry = getPluginRegistry();
  if (!registry) {
    return errors::Unavailable("Plugin registry is missing!");
  }
  auto creator =
      registry->getPluginCreator(plugin_name.c_str(), plugin_version.c_str());
  if (!creator) {
    return errors::Unavailable("No plugin creator is registered for plugin '",
                               plugin_name, "', version", plugin_version,
                               ". Did you loaded your plugin library?");
  }
  const auto field_names = creator->getFieldNames();
  std::vector<nvinfer1::PluginField> fields;
  std::vector<Tensor> store;
  // need special treatment of dims;
  std::vector<nvinfer1::Dims> dim_store;

  if (field_names) {
    int num_fields = field_names->nbFields;
    auto field = field_names->fields;
    for (int i = 0; i < num_fields; ++i) {
      if (!field->name) {
        LOG(WARNING) << "Field " << i << " in plugin field map for " << name
                     << " is nullptr";
        field++;
        continue;
      }
      VLOG(2) << "Trying to parse plugin requested attribute " << field->name;
      string hidden_attr = StrCat("_", field->name);
      const AttrValue* attr_value = nullptr;
      Status attr_found=attrs.Find(hidden_attr, &attr_value);
      if (!attr_found.ok()) {
        LOG(WARNING) << "Field " << field->name
                     << " is not defined in plugin node_def for " << name
                     << " skipping field.";
        field++;
        continue;
      }

      const auto& attr_type = attr_value->value_case();
      if (attr_type == AttrValue::kList) {
        LOG(WARNING) << "Can't parse lists yet!";
        field++;
        continue;
      }
      if (attr_type != AttrValue::kTensor) {
        LOG(WARNING) << field->name << " for op " << name
                     << " is not a tensor. Need args to be encoded in tensors!";
        field++;
        continue;
      }
      Tensor t;
      if (!t.FromProto(attr_value->tensor())) {
        return errors::InvalidArgument("Can't parse tensor for op ", name,
                                       " from attribute ", field->name);
      }

      switch (field->type) {
        case nvinfer1::PluginFieldType::kFLOAT16: {
          store.emplace_back(std::move(t));
          if (VLOG_IS_ON(2)) {
            DumpField<Eigen::half>(field->name, store.back(), 5);
          }
          fields.emplace_back(field->name,
                              store.back().flat<Eigen::half>().data(),
                              field->type, store.back().NumElements());
          break;
        }
        case nvinfer1::PluginFieldType::kFLOAT32: {
          store.emplace_back(std::move(t));
          if (VLOG_IS_ON(2)) {
            DumpField<float>(field->name, store.back(), 5);
          }
          fields.emplace_back(field->name, store.back().flat<float>().data(),
                              field->type, store.back().NumElements());
          break;
        }
        case nvinfer1::PluginFieldType::kFLOAT64: {
          store.emplace_back(std::move(t));
          if (VLOG_IS_ON(2)) {
            DumpField<double>(field->name, store.back(), 5);
          }
          fields.emplace_back(field->name, store.back().flat<double>().data(),
                              field->type, store.back().NumElements());
          break;
        }
        case nvinfer1::PluginFieldType::kINT8: {
          store.emplace_back(std::move(t));
          if (VLOG_IS_ON(2)) {
            DumpField<int8>(field->name, store.back(), 5);
          }
          fields.emplace_back(field->name, store.back().flat<int8>().data(),
                              field->type, store.back().NumElements());
          break;
        }
        case nvinfer1::PluginFieldType::kINT16: {
          store.emplace_back(std::move(t));
          if (VLOG_IS_ON(2)) {
            DumpField<short>(field->name, store.back(), 5);
          }
          fields.emplace_back(field->name, store.back().flat<short>().data(),
                              field->type, store.back().NumElements());
          break;
        }
        case nvinfer1::PluginFieldType::kINT32: {
          store.emplace_back(std::move(t));
          if (VLOG_IS_ON(2)) {
            DumpField<int>(field->name, store.back(), 5);
          }
          fields.emplace_back(field->name, store.back().flat<int>().data(),
                              field->type, store.back().NumElements());
          break;
        }
        case nvinfer1::PluginFieldType::kCHAR: {
          store.emplace_back(std::move(t));
          if (VLOG_IS_ON(2)) {
            // treat chars as int for dumping
            DumpField<int8>(field->name, store.back(), 5);
          }
          fields.emplace_back(field->name, store.back().flat<int8>().data(),
                              field->type, store.back().NumElements());
          break;
        }
        case nvinfer1::PluginFieldType::kDIMS: {
          if (t.dims() != 2) {
            return errors::InvalidArgument(
                "Dims in ", field->name, " for op ", name,
                " should be encoded in a tensor of shape [2,nbDims]");
          }
          int n_dims = t.dim_size(1);
          if (n_dims > nvinfer1::Dims::MAX_DIMS) {
            return errors::InvalidArgument(
                "Dims in ", field->name, " for op ", name, " has size ", n_dims,
                " bigger than max dims ", nvinfer1::Dims::MAX_DIMS);
          }
          const auto& m = t.matrix<int>();
          nvinfer1::Dims d;
          d.nbDims = n_dims;
          // find a better way!
          for (int k = 0; k < n_dims; ++k) {
            d.d[k] = m(0, k);
            switch ((nvinfer1::DimensionType)m(1, k)) {
              case nvinfer1::DimensionType::kSPATIAL: {
                d.type[k] = nvinfer1::DimensionType::kSPATIAL;
                break;
              }
              case nvinfer1::DimensionType::kCHANNEL: {
                d.type[k] = nvinfer1::DimensionType::kCHANNEL;
                break;
              }
              case nvinfer1::DimensionType::kINDEX: {
                d.type[k] = nvinfer1::DimensionType::kINDEX;
                break;
              }
              case nvinfer1::DimensionType::kSEQUENCE: {
                d.type[k] = nvinfer1::DimensionType::kSEQUENCE;
                break;
              }
              default: {
                return errors::InvalidArgument("Dim ", k, " in ", field->name,
                                               " for op ", name,
                                               " has unknown dim type");
                break;
              }
            }
          }
          dim_store.emplace_back(std::move(d));
          fields.emplace_back(field->name, &dim_store.back(), field->type, 1);
          break;
        }
        case nvinfer1::PluginFieldType::kUNKNOWN:
          return errors::InvalidArgument("kUNKNOWN type at attribute ",
                                         field->name, " for op ", name,
                                         ", need all attibutes defined!");
        default: {
          LOG(FATAL) << "UNKNOWN field type!";
          break;
        }
      }
      field++;
    }
    if (fields.size() != num_fields) {
      LOG(WARNING) << "Plugin " << plugin_name << " for op " << name
                   << " asked for " << num_fields << " attributes but got "
                   << fields.size();
    }
  }
  if (validation_only) {
    return Status::OK();
  }
  nvinfer1::PluginField* pfs = nullptr;
  if (fields.size()) {
    pfs = &fields[0];
  }
  nvinfer1::PluginFieldCollection pfc{(int)fields.size(), pfs};
  plugin = creator->createPlugin(name.c_str(), &pfc);
  if (!plugin) {
    return errors::Internal("Creator failed to construct a plugin object for ",
                            name);
  }
  return Status::OK();
}
}  // namespace tensorrt
}  // namespace tensorflow
#else
namespace tensorflow {
namespace tensorrt {
void InitializeTrtPlugins() {
  LOG(ERROR) << "=============================================================="
                "======================";
  LOG(ERROR) << "Tensorflow is compiled without TensorRT. This functions "
                "should not have been called!";
  LOG(ERROR) << "=============================================================="
                "======================";
}
Status ConstructPlugin(const NodeDef& node_def, nvinfer1::IPluginV2*& plugin,
                       bool validation_only) {
  LOG(ERROR) << "=============================================================="
                "======================";
  LOG(ERROR) << "Tensorflow is compiled without TensorRT. This functions "
                "should not have been called!";
  LOG(ERROR) << "=============================================================="
                "======================";
  return errors::Unavailable("Tensorflow is compiled without TensorRT support");
}
}  // namespace tensorrt
}  // namespace tensorflow
#endif
#endif
namespace tensorflow {
namespace tensorrt {

Status TrtPrecisionModeToName(TrtPrecisionMode mode, string* name) {
  switch (mode) {
    case TrtPrecisionMode::FP32:
      *name = "FP32";
      break;
    case TrtPrecisionMode::FP16:
      *name = "FP16";
      break;
    case TrtPrecisionMode::INT8:
      *name = "INT8";
      break;
    default:
      return errors::OutOfRange("Unknown precision mode");
  }
  return Status::OK();
}

Status TrtPrecisionModeFromName(const string& name, TrtPrecisionMode* mode) {
  if (name == "FP32") {
    *mode = TrtPrecisionMode::FP32;
  } else if (name == "FP16") {
    *mode = TrtPrecisionMode::FP16;
  } else if (name == "INT8") {
    *mode = TrtPrecisionMode::INT8;
  } else {
    return errors::InvalidArgument("Invalid precision mode name: ", name);
  }
  return Status::OK();
}

}  // namespace tensorrt
}  // namespace tensorflow
