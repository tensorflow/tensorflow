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

#include "tensorflow/lite/delegates/gpu/gl/compiler/rename.h"

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/object_accessor.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/parameter_accessor.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
#include "tensorflow/lite/delegates/gpu/gl/uniform_parameter.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

// Rewrites names of all parameters according to returned values from the
// given NameFunctor.
class ParameterRewriter : public InlineRewrite {
 public:
  ParameterRewriter(const std::string& inline_delimiter,
                    const NameFunctor& name_func)
      : inline_delimiter_(inline_delimiter), name_func_(name_func) {}

  RewriteStatus Rewrite(absl::string_view input, std::string* output) final {
    auto ref = parameter_accessor_internal::Parse(input);
    if (ref.name.empty()) {
      absl::StrAppend(output, "INVALID_SYNTAX");
      return RewriteStatus::ERROR;
    }

    auto it =
        name_to_param_.find(std::string(ref.name.data(), ref.name.size()));
    if (it == name_to_param_.end()) {
      return RewriteStatus::NOT_RECOGNIZED;
    }

    // reconstruct access using the new name.
    absl::StrAppend(output, inline_delimiter_, it->second.name);
    if (!ref.index.empty()) {
      absl::StrAppend(output, "[", ref.index, "]");
    }
    absl::StrAppend(output, ref.field, inline_delimiter_);
    return RewriteStatus::SUCCESS;
  }

  // Return true if parameter was successfully added.
  bool AddParameter(UniformParameter param) {
    std::string old_name = param.name;
    param.name = name_func_(old_name);
    return name_to_param_.insert({old_name, std::move(param)}).second;
  }

  // Returns a collection of uniform parameters with updated names.
  std::vector<UniformParameter> GetUniformParameters() const {
    std::vector<UniformParameter> params;
    params.reserve(name_to_param_.size());
    for (auto& param : name_to_param_) {
      params.push_back(param.second);
    }
    return params;
  }

 private:
  const std::string inline_delimiter_;
  const NameFunctor name_func_;

  std::unordered_map<std::string, UniformParameter> name_to_param_;
};

// Rewrites names of all objects according to returned values from the
// given NameFunctor.
class ObjectRewriter : public InlineRewrite {
 public:
  ObjectRewriter(const std::string& inline_delimiter,
                 const NameFunctor& name_func)
      : inline_delimiter_(inline_delimiter), name_func_(name_func) {}

  RewriteStatus Rewrite(absl::string_view input, std::string* output) final {
    // Splits 'a = b' into {'a','b'}.
    std::pair<absl::string_view, absl::string_view> n =
        absl::StrSplit(input, absl::MaxSplits('=', 1), absl::SkipWhitespace());
    if (n.first.empty()) {
      return RewriteStatus::NOT_RECOGNIZED;
    }

    if (n.second.empty()) {
      return RewriteRead(absl::StripAsciiWhitespace(n.first), output);
    }
    return RewriteWrite(absl::StripAsciiWhitespace(n.first),
                        absl::StripAsciiWhitespace(n.second), output);
  }

  // Return true if an object was successfully added.
  bool AddObject(const std::string& name, Object object) {
    std::string new_name = name_func_(name);
    return name_to_object_.insert({name, {new_name, std::move(object)}}).second;
  }

  // Returns a collection of registered objects with updated names.
  std::vector<std::pair<std::string, Object>> GetObjects() const {
    std::vector<std::pair<std::string, Object>> objects;
    objects.reserve(name_to_object_.size());
    for (auto& o : name_to_object_) {
      objects.push_back(o.second);
    }
    return objects;
  }

 private:
  RewriteStatus RewriteRead(absl::string_view location, std::string* output) {
    auto element = object_accessor_internal::ParseElement(location);
    if (element.object_name.empty()) {
      absl::StrAppend(output, "UNABLE_TO_PARSE_INDEXED_ELEMENT");
      return RewriteStatus::ERROR;
    }
    auto it = name_to_object_.find(
        std::string(element.object_name.data(), element.object_name.size()));
    if (it == name_to_object_.end()) {
      return RewriteStatus::NOT_RECOGNIZED;
    }
    absl::StrAppend(output, inline_delimiter_, it->second.first, "[",
                    absl::StrJoin(element.indices, ","), "]",
                    inline_delimiter_);
    return RewriteStatus::SUCCESS;
  }

  RewriteStatus RewriteWrite(absl::string_view location,
                             absl::string_view value, std::string* output) {
    // name[index1, index2...] = value
    auto element = object_accessor_internal::ParseElement(location);
    if (element.object_name.empty()) {
      absl::StrAppend(output, "UNABLE_TO_PARSE_INDEXED_ELEMENT");
      return RewriteStatus::ERROR;
    }
    auto it = name_to_object_.find(
        std::string(element.object_name.data(), element.object_name.size()));
    if (it == name_to_object_.end()) {
      return RewriteStatus::NOT_RECOGNIZED;
    }
    absl::StrAppend(output, inline_delimiter_, it->second.first, "[",
                    absl::StrJoin(element.indices, ","), "] = ", value,
                    inline_delimiter_);
    return RewriteStatus::SUCCESS;
  }

  const std::string inline_delimiter_;
  const NameFunctor name_func_;

  std::unordered_map<std::string, std::pair<std::string, Object>>
      name_to_object_;
};

}  // namespace

Status Rename(const NameFunctor& name_func, GeneratedCode* code) {
  ParameterRewriter param_rewriter("$", name_func);
  ObjectRewriter object_rewriter("$", name_func);
  for (auto&& param : code->parameters) {
    if (!param_rewriter.AddParameter(std::move(param))) {
      return InternalError("Parameter name already exists");
    }
  }
  for (auto&& object : code->objects) {
    if (!object_rewriter.AddObject(object.first, std::move(object.second))) {
      return InternalError("Object name already exists");
    }
  }
  TextPreprocessor preprocessor('$', /* keep_unknown_rewrites = */ true);
  preprocessor.AddRewrite(&param_rewriter);
  preprocessor.AddRewrite(&object_rewriter);
  std::string source_code;
  RETURN_IF_ERROR(preprocessor.Rewrite(code->source_code, &source_code));
  code->source_code = source_code;
  code->parameters = param_rewriter.GetUniformParameters();
  code->objects = object_rewriter.GetObjects();
  return OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
