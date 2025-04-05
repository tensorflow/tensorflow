#include <string>
#include <type_traits>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton {

// Better version of llvm::join.  This one works when T is an integer or any
// other type which defines operator<<(raw_ostream).
template <typename C>
std::string join(C &&container, llvm::StringRef sep = ", ") {
  std::string ret;
  llvm::raw_string_ostream s(ret);
  for (const auto &elem : container) {
    if (!ret.empty())
      s << sep;
    s << elem;
  }
  return ret;
}

// Joins a container of elements into a string, using `sep` as a separator.
//
// fn is called to transform each element of the container before it's added to
// the string.  fn must have one of the following two signatures.
//
//   - void fn(llvm::raw_ostream&, E), where E is the element type of the
//     container, or
//   - T fn(E), where T is a type which can be passed to
//     raw_ostream::operator<<.
//
template <typename C, typename Fn>
std::string join(C &&container, llvm::StringRef sep, Fn &&fn) {
  std::string ret;
  llvm::raw_string_ostream s(ret);
  for (const auto &elem : container) {
    if (!ret.empty())
      s << sep;

    if constexpr (std::is_invocable_v<Fn, llvm::raw_ostream &,
                                      decltype(elem)>) {
      static_assert(
          std::is_void_v<
              std::invoke_result_t<Fn, llvm::raw_ostream &, decltype(elem)>>);
      fn(s, elem);
    } else {
      s << fn(elem);
    }
  }
  return ret;
}

} // namespace mlir::triton
