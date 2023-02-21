/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PRINTER_H_
#define TENSORFLOW_COMPILER_XLA_PRINTER_H_

#include <string>

#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"

namespace xla {

// Abstract "printer" interface.
//
// This interface is used in XLA to print data structures into a human-readaable
// form. Most users can use either `StringPrinter` or `CordPrinter` to retrieve
// the result in a string format; power users may implement their own printer to
// implement "streamed printing" if needed.
class Printer {
 public:
  virtual ~Printer() = default;

  // Appends the given string to the printer.
  virtual void Append(const absl::AlphaNum& a) = 0;
};

// A printer implementation that accumulates printed strings into `std::string`.
class StringPrinter : public Printer {
 public:
  void Append(const absl::AlphaNum& a) override;

  std::string ToString() &&;

 private:
  std::string result_;
};

// A printer implementation that accumulates printed strings into `absl::Cord`.
class CordPrinter : public Printer {
 public:
  void Append(const absl::AlphaNum& a) override;

  absl::Cord ToCord() &&;

 private:
  void AppendImpl(const absl::AlphaNum& a);
  void AppendBuffer();

  absl::CordBuffer buffer_;
  absl::Cord result_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PRINTER_H_
