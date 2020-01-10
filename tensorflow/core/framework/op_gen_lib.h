#ifndef TENSORFLOW_FRAMEWORK_OP_GEN_LIB_H_
#define TENSORFLOW_FRAMEWORK_OP_GEN_LIB_H_

#include <string>
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

inline string Spaces(int n) { return string(n, ' '); }

// Wrap prefix + str to be at most width characters, indenting every line
// after the first by prefix.size() spaces.  Intended use case is something
// like prefix = "  Foo(" and str is a list of arguments (terminated by a ")").
// TODO(josh11b): Option to wrap on ", " instead of " " when possible.
string WordWrap(StringPiece prefix, StringPiece str, int width);

// Looks for an "=" at the beginning of *description.  If found, strips it off
// (and any following spaces) from *description and return true.  Otherwise
// returns false.
bool ConsumeEquals(StringPiece* description);

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_OP_GEN_LIB_H_
