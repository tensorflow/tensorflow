#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

namespace mlir {
// This target is a convenient dependency for users to auto-initialize MLIR
// internals.
static bool auto_init = []() {
  registerAllDialects();
  registerAllPasses();

  return true;
}();

} // namespace mlir
