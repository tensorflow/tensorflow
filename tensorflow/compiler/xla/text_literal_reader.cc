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

#include "tensorflow/compiler/xla/text_literal_reader.h"

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

StatusOr<std::unique_ptr<Literal>> TextLiteralReader::ReadPath(
    tensorflow::StringPiece path) {
  CHECK(!tensorflow::str_util::EndsWith(path, ".gz"))
      << "TextLiteralReader no longer supports reading .gz files";
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  Status s =
      tensorflow::Env::Default()->NewRandomAccessFile(std::string(path), &file);
  if (!s.ok()) {
    return s;
  }

  TextLiteralReader reader(file.release());
  return reader.ReadAllLines();
}

TextLiteralReader::TextLiteralReader(tensorflow::RandomAccessFile* file)
    : file_(file) {}

namespace {
// This is an optimized version of tensorflow::str_util::Split which uses
// StringPiece for the delimited strings and uses an out parameter for the
// result to avoid vector creation/destruction.
void SplitByDelimToStringPieces(tensorflow::StringPiece text, char delim,
                                std::vector<tensorflow::StringPiece>* result) {
  result->clear();

  if (text.empty()) {
    return;
  }

  // The following loop is a little strange: its bound is text.size() + 1
  // instead of the more typical text.size().
  // The final iteration of the loop (when i is equal to text.size()) handles
  // the trailing token.
  size_t token_start = 0;
  for (size_t i = 0; i < text.size() + 1; i++) {
    if (i == text.size() || text[i] == delim) {
      tensorflow::StringPiece token(text.data() + token_start, i - token_start);
      result->push_back(token);
      token_start = i + 1;
    }
  }
}
}  // namespace

StatusOr<std::unique_ptr<Literal>> TextLiteralReader::ReadAllLines() {
  tensorflow::io::RandomAccessInputStream stream(file_.get());
  tensorflow::io::BufferedInputStream buf(&stream, 65536);
  string shape_string;
  Status s = buf.ReadLine(&shape_string);
  if (!s.ok()) {
    return s;
  }

  tensorflow::StringPiece sp(shape_string);
  if (tensorflow::str_util::RemoveWhitespaceContext(&sp) > 0) {
    string tmp = std::string(sp);
    shape_string = tmp;
  }
  TF_ASSIGN_OR_RETURN(Shape shape, ShapeUtil::ParseShapeString(shape_string));
  if (shape.element_type() != F32) {
    return Unimplemented(
        "unsupported element type for text literal reading: %s",
        ShapeUtil::HumanString(shape).c_str());
  }

  auto result = MakeUnique<Literal>(shape);
  const float fill = std::numeric_limits<float>::quiet_NaN();
  result->PopulateWithValue<float>(fill);
  std::vector<tensorflow::StringPiece> pieces;
  std::vector<tensorflow::StringPiece> coordinates;
  std::vector<int64> coordinate_values;
  string line;
  while (buf.ReadLine(&line).ok()) {
    SplitByDelimToStringPieces(line, ':', &pieces);
    tensorflow::StringPiece coordinates_string = pieces[0];
    tensorflow::StringPiece value_string = pieces[1];
    tensorflow::str_util::RemoveWhitespaceContext(&coordinates_string);
    tensorflow::str_util::RemoveWhitespaceContext(&value_string);
    if (!tensorflow::str_util::ConsumePrefix(&coordinates_string, "(")) {
      return InvalidArgument(
          "expected '(' at the beginning of coordinates: \"%s\"", line.c_str());
    }
    if (!tensorflow::str_util::ConsumeSuffix(&coordinates_string, ")")) {
      return InvalidArgument("expected ')' at the end of coordinates: \"%s\"",
                             line.c_str());
    }
    float value;
    if (!tensorflow::strings::safe_strtof(std::string(value_string).c_str(),
                                          &value)) {
      return InvalidArgument("could not parse value as float: \"%s\"",
                             std::string(value_string).c_str());
    }
    SplitByDelimToStringPieces(coordinates_string, ',', &coordinates);
    coordinate_values.clear();
    for (tensorflow::StringPiece piece : coordinates) {
      int64 coordinate_value;
      if (!tensorflow::strings::safe_strto64(piece, &coordinate_value)) {
        return InvalidArgument(
            "could not parse coordinate member as int64: \"%s\"",
            std::string(piece).c_str());
      }
      coordinate_values.push_back(coordinate_value);
    }
    if (coordinate_values.size() != shape.dimensions_size()) {
      return InvalidArgument(
          "line did not have expected number of coordinates; want %d got %zu: "
          "\"%s\"",
          shape.dimensions_size(), coordinate_values.size(), line.c_str());
    }
    result->Set<float>(coordinate_values, value);
  }
  return std::move(result);
}

}  // namespace xla
