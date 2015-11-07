#include "tensorflow/stream_executor/blas.h"

#include "tensorflow/stream_executor/lib/strcat.h"

namespace perftools {
namespace gputools {
namespace blas {

string TransposeString(Transpose t) {
  switch (t) {
    case Transpose::kNoTranspose:
      return "NoTranspose";
    case Transpose::kTranspose:
      return "Transpose";
    case Transpose::kConjugateTranspose:
      return "ConjugateTranspose";
    default:
      LOG(FATAL) << "Unknown transpose " << static_cast<int32>(t);
  }
}

string UpperLowerString(UpperLower ul) {
  switch (ul) {
    case UpperLower::kUpper:
      return "Upper";
    case UpperLower::kLower:
      return "Lower";
    default:
      LOG(FATAL) << "Unknown upperlower " << static_cast<int32>(ul);
  }
}

string DiagonalString(Diagonal d) {
  switch (d) {
    case Diagonal::kUnit:
      return "Unit";
    case Diagonal::kNonUnit:
      return "NonUnit";
    default:
      LOG(FATAL) << "Unknown diagonal " << static_cast<int32>(d);
  }
}

string SideString(Side s) {
  switch (s) {
    case Side::kLeft:
      return "Left";
    case Side::kRight:
      return "Right";
    default:
      LOG(FATAL) << "Unknown side " << static_cast<int32>(s);
  }
}

}  // namespace blas
}  // namespace gputools
}  // namespace perftools
