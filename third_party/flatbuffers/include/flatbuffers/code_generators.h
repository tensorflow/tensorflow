/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLATBUFFERS_CODE_GENERATORS_H_
#define FLATBUFFERS_CODE_GENERATORS_H_

#include <map>
#include <sstream>
#include "flatbuffers/idl.h"

namespace flatbuffers {

// Utility class to assist in generating code through use of text templates.
//
// Example code:
//   CodeWriter code("\t");
//   code.SetValue("NAME", "Foo");
//   code += "void {{NAME}}() { printf("%s", "{{NAME}}"); }";
//   code.SetValue("NAME", "Bar");
//   code += "void {{NAME}}() { printf("%s", "{{NAME}}"); }";
//   std::cout << code.ToString() << std::endl;
//
// Output:
//  void Foo() { printf("%s", "Foo"); }
//  void Bar() { printf("%s", "Bar"); }
class CodeWriter {
 public:
  CodeWriter(std::string pad = std::string())
      : pad_(pad), cur_ident_lvl_(0), ignore_ident_(false) {}

  // Clears the current "written" code.
  void Clear() {
    stream_.str("");
    stream_.clear();
  }

  // Associates a key with a value.  All subsequent calls to operator+=, where
  // the specified key is contained in {{ and }} delimiters will be replaced by
  // the given value.
  void SetValue(const std::string &key, const std::string &value) {
    value_map_[key] = value;
  }

  std::string GetValue(const std::string &key) const {
    const auto it = value_map_.find(key);
    return it == value_map_.end() ? "" : it->second;
  }

  // Appends the given text to the generated code as well as a newline
  // character.  Any text within {{ and }} delimeters is replaced by values
  // previously stored in the CodeWriter by calling SetValue above.  The newline
  // will be suppressed if the text ends with the \\ character.
  void operator+=(std::string text);

  // Returns the current contents of the CodeWriter as a std::string.
  std::string ToString() const { return stream_.str(); }

  // Increase ident level for writing code
  void IncrementIdentLevel() { cur_ident_lvl_++; }
  // Decrease ident level for writing code
  void DecrementIdentLevel() {
    if (cur_ident_lvl_) cur_ident_lvl_--;
  }

 private:
  std::map<std::string, std::string> value_map_;
  std::stringstream stream_;
  std::string pad_;
  int cur_ident_lvl_;
  bool ignore_ident_;

  // Add ident padding (tab or space) based on ident level
  void AppendIdent(std::stringstream &stream);
};

class BaseGenerator {
 public:
  virtual bool generate() = 0;

  static std::string NamespaceDir(const Parser &parser, const std::string &path,
                                  const Namespace &ns);

 protected:
  BaseGenerator(const Parser &parser, const std::string &path,
                const std::string &file_name,
                std::string qualifying_start,
                std::string qualifying_separator)
      : parser_(parser),
        path_(path),
        file_name_(file_name),
        qualifying_start_(qualifying_start),
        qualifying_separator_(qualifying_separator) {}
  virtual ~BaseGenerator() {}

  // No copy/assign.
  BaseGenerator &operator=(const BaseGenerator &);
  BaseGenerator(const BaseGenerator &);

  std::string NamespaceDir(const Namespace &ns) const;

  static const char *FlatBuffersGeneratedWarning();

  static std::string FullNamespace(const char *separator, const Namespace &ns);

  static std::string LastNamespacePart(const Namespace &ns);

  // tracks the current namespace for early exit in WrapInNameSpace
  // c++, java and csharp returns a different namespace from
  // the following default (no early exit, always fully qualify),
  // which works for js and php
  virtual const Namespace *CurrentNameSpace() const { return nullptr; }

  // Ensure that a type is prefixed with its namespace even within
  // its own namespace to avoid conflict between generated method
  // names and similarly named classes or structs
  std::string WrapInNameSpace(const Namespace *ns,
                              const std::string &name) const;

  std::string WrapInNameSpace(const Definition &def) const;

  std::string GetNameSpace(const Definition &def) const;

  const Parser &parser_;
  const std::string &path_;
  const std::string &file_name_;
  const std::string qualifying_start_;
  const std::string qualifying_separator_;
};

struct CommentConfig {
  const char *first_line;
  const char *content_line_prefix;
  const char *last_line;
};

extern void GenComment(const std::vector<std::string> &dc,
                       std::string *code_ptr, const CommentConfig *config,
                       const char *prefix = "");

class FloatConstantGenerator {
 public:
  virtual ~FloatConstantGenerator() {}
  std::string GenFloatConstant(const FieldDef &field) const;

 private:
  virtual std::string Value(double v, const std::string &src) const = 0;
  virtual std::string Inf(double v) const = 0;
  virtual std::string NaN(double v) const = 0;

  virtual std::string Value(float v, const std::string &src) const = 0;
  virtual std::string Inf(float v) const = 0;
  virtual std::string NaN(float v) const = 0;

  template<typename T>
  std::string GenFloatConstantImpl(const FieldDef &field) const;
};

class SimpleFloatConstantGenerator : public FloatConstantGenerator {
 public:
  SimpleFloatConstantGenerator(const char *nan_number,
                               const char *pos_inf_number,
                               const char *neg_inf_number);

 private:
  std::string Value(double v,
                    const std::string &src) const FLATBUFFERS_OVERRIDE;
  std::string Inf(double v) const FLATBUFFERS_OVERRIDE;
  std::string NaN(double v) const FLATBUFFERS_OVERRIDE;

  std::string Value(float v, const std::string &src) const FLATBUFFERS_OVERRIDE;
  std::string Inf(float v) const FLATBUFFERS_OVERRIDE;
  std::string NaN(float v) const FLATBUFFERS_OVERRIDE;

  const std::string nan_number_;
  const std::string pos_inf_number_;
  const std::string neg_inf_number_;
};

// C++, C#, Java like generator.
class TypedFloatConstantGenerator : public FloatConstantGenerator {
 public:
  TypedFloatConstantGenerator(const char *double_prefix,
                              const char *single_prefix, const char *nan_number,
                              const char *pos_inf_number,
                              const char *neg_inf_number = "");

 private:
  std::string Value(double v,
                    const std::string &src) const FLATBUFFERS_OVERRIDE;
  std::string Inf(double v) const FLATBUFFERS_OVERRIDE;

  std::string NaN(double v) const FLATBUFFERS_OVERRIDE;

  std::string Value(float v, const std::string &src) const FLATBUFFERS_OVERRIDE;
  std::string Inf(float v) const FLATBUFFERS_OVERRIDE;
  std::string NaN(float v) const FLATBUFFERS_OVERRIDE;

  std::string MakeNaN(const std::string &prefix) const;
  std::string MakeInf(bool neg, const std::string &prefix) const;

  const std::string double_prefix_;
  const std::string single_prefix_;
  const std::string nan_number_;
  const std::string pos_inf_number_;
  const std::string neg_inf_number_;
};

}  // namespace flatbuffers

#endif  // FLATBUFFERS_CODE_GENERATORS_H_
