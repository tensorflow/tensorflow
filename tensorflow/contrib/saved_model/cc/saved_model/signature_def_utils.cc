/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/saved_model/cc/saved_model/signature_def_utils.h"

#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

namespace {
template <class T>
Status FindInProtobufMap(StringPiece description,
                         const protobuf::Map<string, T>& map, const string& key,
                         const T** value) {
  const auto it = map.find(key);
  if (it == map.end()) {
    return errors::NotFound("Could not find ", description, " for key: ", key);
  }
  *value = &it->second;
  return Status::OK();
}

// Looks up the TensorInfo for the given key in the given map and verifies that
// its datatype matches the given correct datatype.
bool VerifyTensorInfoForKeyInMap(const protobuf::Map<string, TensorInfo>& map,
                                 const string& key, DataType correct_dtype) {
  const TensorInfo* tensor_info;
  const Status& status = FindInProtobufMap("", map, key, &tensor_info);
  if (!status.ok()) {
    return false;
  }
  if (tensor_info->dtype() != correct_dtype) {
    return false;
  }
  return true;
}

bool IsValidPredictSignature(const SignatureDef& signature_def) {
  if (signature_def.method_name() != kPredictMethodName) {
    return false;
  }
  if (signature_def.inputs().empty()) {
    return false;
  }
  if (signature_def.outputs().empty()) {
    return false;
  }
  return true;
}

bool IsValidRegressionSignature(const SignatureDef& signature_def) {
  if (signature_def.method_name() != kRegressMethodName) {
    return false;
  }
  if (!VerifyTensorInfoForKeyInMap(signature_def.inputs(), kRegressInputs,
                                   DT_STRING)) {
    return false;
  }
  if (!VerifyTensorInfoForKeyInMap(signature_def.outputs(), kRegressOutputs,
                                   DT_FLOAT)) {
    return false;
  }
  return true;
}

bool IsValidClassificationSignature(const SignatureDef& signature_def) {
  if (signature_def.method_name() != kClassifyMethodName) {
    return false;
  }
  if (!VerifyTensorInfoForKeyInMap(signature_def.inputs(), kClassifyInputs,
                                   DT_STRING)) {
    return false;
  }
  if (signature_def.outputs().empty()) {
    return false;
  }
  for (auto const& output : signature_def.outputs()) {
    const string& key = output.first;
    const TensorInfo& tensor_info = output.second;
    if (key == kClassifyOutputClasses) {
      if (tensor_info.dtype() != DT_STRING) {
        return false;
      }
    } else if (key == kClassifyOutputScores) {
      if (tensor_info.dtype() != DT_FLOAT) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace

Status FindSignatureDefByKey(const MetaGraphDef& meta_graph_def,
                             const string& signature_def_key,
                             const SignatureDef** signature_def) {
  return FindInProtobufMap("SignatureDef", meta_graph_def.signature_def(),
                           signature_def_key, signature_def);
}

Status FindInputTensorInfoByKey(const SignatureDef& signature_def,
                                const string& tensor_info_key,
                                const TensorInfo** tensor_info) {
  return FindInProtobufMap("input TensorInfo", signature_def.inputs(),
                           tensor_info_key, tensor_info);
}

Status FindOutputTensorInfoByKey(const SignatureDef& signature_def,
                                 const string& tensor_info_key,
                                 const TensorInfo** tensor_info) {
  return FindInProtobufMap("output TensorInfo", signature_def.outputs(),
                           tensor_info_key, tensor_info);
}

Status FindInputTensorNameByKey(const SignatureDef& signature_def,
                                const string& tensor_info_key, string* name) {
  const TensorInfo* tensor_info;
  TF_RETURN_IF_ERROR(
      FindInputTensorInfoByKey(signature_def, tensor_info_key, &tensor_info));
  *name = tensor_info->name();
  return Status::OK();
}

Status FindOutputTensorNameByKey(const SignatureDef& signature_def,
                                 const string& tensor_info_key, string* name) {
  const TensorInfo* tensor_info;
  TF_RETURN_IF_ERROR(
      FindOutputTensorInfoByKey(signature_def, tensor_info_key, &tensor_info));
  *name = tensor_info->name();
  return Status::OK();
}

bool IsValidSignature(const SignatureDef& signature_def) {
  return IsValidClassificationSignature(signature_def) ||
         IsValidRegressionSignature(signature_def) ||
         IsValidPredictSignature(signature_def);
}

}  // namespace tensorflow
