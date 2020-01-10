#include "tensorflow/core/framework/op_gen_lib.h"

#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

string WordWrap(StringPiece prefix, StringPiece str, int width) {
  const string indent_next_line = "\n" + Spaces(prefix.size());
  width -= prefix.size();
  string result;
  strings::StrAppend(&result, prefix);

  while (!str.empty()) {
    if (static_cast<int>(str.size()) <= width) {
      // Remaining text fits on one line.
      strings::StrAppend(&result, str);
      break;
    }
    auto space = str.rfind(' ', width);
    if (space == StringPiece::npos) {
      // Rather make a too-long line and break at a space.
      space = str.find(' ');
      if (space == StringPiece::npos) {
        strings::StrAppend(&result, str);
        break;
      }
    }
    // Breaking at character at position <space>.
    StringPiece to_append = str.substr(0, space);
    str.remove_prefix(space + 1);
    // Remove spaces at break.
    while (to_append.ends_with(" ")) {
      to_append.remove_suffix(1);
    }
    while (str.Consume(" ")) {
    }

    // Go on to the next line.
    strings::StrAppend(&result, to_append);
    if (!str.empty()) strings::StrAppend(&result, indent_next_line);
  }

  return result;
}

bool ConsumeEquals(StringPiece* description) {
  if (description->Consume("=")) {
    while (description->Consume(" ")) {  // Also remove spaces after "=".
    }
    return true;
  }
  return false;
}

}  // namespace tensorflow
