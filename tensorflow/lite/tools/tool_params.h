/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_TOOL_PARAMS_H_
#define TENSORFLOW_LITE_TOOLS_TOOL_PARAMS_H_
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tflite {
namespace tools {

template <typename T>
class TypedToolParam;

class ToolParam {
 protected:
  enum class ParamType { TYPE_INT32, TYPE_FLOAT, TYPE_BOOL, TYPE_STRING };
  template <typename T>
  static ParamType GetValueType();

 public:
  template <typename T>
  static std::unique_ptr<ToolParam> Create(const T& default_value,
                                           int position = 0) {
    auto* param = new TypedToolParam<T>(default_value);
    param->SetPosition(position);
    return std::unique_ptr<ToolParam>(param);
  }

  template <typename T>
  TypedToolParam<T>* AsTyped() {
    AssertHasSameType(GetValueType<T>(), type_);
    return static_cast<TypedToolParam<T>*>(this);
  }

  template <typename T>
  const TypedToolParam<T>* AsConstTyped() const {
    AssertHasSameType(GetValueType<T>(), type_);
    return static_cast<const TypedToolParam<T>*>(this);
  }

  virtual ~ToolParam() {}
  explicit ToolParam(ParamType type)
      : has_value_set_(false), position_(0), type_(type) {}

  bool HasValueSet() const { return has_value_set_; }

  int GetPosition() const { return position_; }
  void SetPosition(int position) { position_ = position; }

  virtual void Set(const ToolParam&) {}

  virtual std::unique_ptr<ToolParam> Clone() const = 0;

 protected:
  bool has_value_set_;

  // Represents the relative ordering among a set of params.
  // Note: in our code, a ToolParam is generally used together with a
  // tflite::Flag so that its value could be set when parsing commandline flags.
  // In this case, the `position_` is simply the index of the particular flag
  // into the list of commandline flags (i.e. named 'argv' in general).
  int position_;

 private:
  static void AssertHasSameType(ParamType a, ParamType b);

  const ParamType type_;
};

template <typename T>
class TypedToolParam : public ToolParam {
 public:
  explicit TypedToolParam(const T& value)
      : ToolParam(GetValueType<T>()), value_(value) {}

  void Set(const T& value) {
    value_ = value;
    has_value_set_ = true;
  }

  const T& Get() const { return value_; }

  void Set(const ToolParam& other) override {
    Set(other.AsConstTyped<T>()->Get());
    SetPosition(other.AsConstTyped<T>()->GetPosition());
  }

  std::unique_ptr<ToolParam> Clone() const override {
    return ToolParam::Create<T>(value_, position_);
  }

 private:
  T value_;
};

// A map-like container for holding values of different types.
class ToolParams {
 public:
  // Add a ToolParam instance `value` w/ `name` to this container.
  void AddParam(const std::string& name, std::unique_ptr<ToolParam> value) {
    params_[name] = std::move(value);
  }

  void RemoveParam(const std::string& name) { params_.erase(name); }

  bool HasParam(const std::string& name) const {
    return params_.find(name) != params_.end();
  }

  bool Empty() const { return params_.empty(); }

  const ToolParam* GetParam(const std::string& name) const {
    const auto& entry = params_.find(name);
    if (entry == params_.end()) return nullptr;
    return entry->second.get();
  }

  template <typename T>
  void Set(const std::string& name, const T& value, int position = 0) {
    AssertParamExists(name);
    params_.at(name)->AsTyped<T>()->Set(value);
    params_.at(name)->AsTyped<T>()->SetPosition(position);
  }

  template <typename T>
  bool HasValueSet(const std::string& name) const {
    AssertParamExists(name);
    return params_.at(name)->AsConstTyped<T>()->HasValueSet();
  }

  template <typename T>
  int GetPosition(const std::string& name) const {
    AssertParamExists(name);
    return params_.at(name)->AsConstTyped<T>()->GetPosition();
  }

  template <typename T>
  T Get(const std::string& name) const {
    AssertParamExists(name);
    return params_.at(name)->AsConstTyped<T>()->Get();
  }

  // Set the value of all same parameters from 'other'.
  void Set(const ToolParams& other);

  // Merge the value of all parameters from 'other'. 'overwrite' indicates
  // whether the value of the same paratmeter is overwritten or not.
  void Merge(const ToolParams& other, bool overwrite = false);

 private:
  void AssertParamExists(const std::string& name) const;
  std::unordered_map<std::string, std::unique_ptr<ToolParam>> params_;
};

#define LOG_TOOL_PARAM(params, type, name, description, verbose)      \
  do {                                                                \
    TFLITE_MAY_LOG(INFO, (verbose) || params.HasValueSet<type>(name)) \
        << description << ": [" << params.Get<type>(name) << "]";     \
  } while (0)

}  // namespace tools
}  // namespace tflite
#endif  // TENSORFLOW_LITE_TOOLS_TOOL_PARAMS_H_
