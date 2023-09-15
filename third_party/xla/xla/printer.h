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

#ifndef XLA_PRINTER_H_
#define XLA_PRINTER_H_

#include <iterator>
#include <string>

#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
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

// Utility functions that appends a list of elements to a Printer as if by
// calling printer->Append(absl::StrJoin(...)), but does it in-place.
template <typename Range, typename PrintFunc>
void AppendJoin(Printer* printer, const Range& range,
                absl::string_view separator, PrintFunc&& print) {
  AppendJoin(printer, range.begin(), range.end(), separator,
             std::forward<PrintFunc>(print));
}

template <typename Iterator, typename PrintFunc>
void AppendJoin(Printer* printer, Iterator start, Iterator end,
                absl::string_view separator, PrintFunc&& print) {
  if (ABSL_PREDICT_FALSE(start == end)) return;
  print(printer, *start);
  std::advance(start, 1);
  while (start != end) {
    printer->Append(separator);
    print(printer, *start);
    std::advance(start, 1);
  }
}

template <typename Range>
void AppendJoin(Printer* printer, const Range& range,
                absl::string_view separator) {
  AppendJoin(printer, range, separator,
             [](Printer* printer, auto& element) { printer->Append(element); });
}

// Utility function that appends multiple elements to a Printer as if by calling
// printer->Append(absl::StrCat(...)), but does it in-place.
inline void AppendCat(Printer* printer, const absl::AlphaNum& a,
                      const absl::AlphaNum& b) {
  printer->Append(a);
  printer->Append(b);
}

inline void AppendCat(Printer* printer, const absl::AlphaNum& a,
                      const absl::AlphaNum& b, const absl::AlphaNum& c) {
  printer->Append(a);
  printer->Append(b);
  printer->Append(c);
}

inline void AppendCat(Printer* printer, const absl::AlphaNum& a,
                      const absl::AlphaNum& b, const absl::AlphaNum& c,
                      const absl::AlphaNum& d) {
  printer->Append(a);
  printer->Append(b);
  printer->Append(c);
  printer->Append(d);
}

}  // namespace xla

#endif  // XLA_PRINTER_H_
