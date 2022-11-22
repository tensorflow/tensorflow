// TODO: Add appropriate licenes ....

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_TENSOR_SPLITTER_V2_
#define TENSORFLOW_COMPILER_XLA_SERVICE_TENSOR_SPLITTER_V2_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which splits tensor values
// The difference between V1 and V2 is:
// V1 will create a while loop for every split path while V2 will merge mergable
// paths(whose paths have a common subpath longer than a threshold) into a
// single while-loop Example:
//         N1
//          |
//         N2
//        /   \
//      N3    N4
//      |     |
//      N5    N6
// There are two paths: one starts from N5 and another from N6 and they have a
// common subpath(N1->N2). In V1, two while loops will be created and the common
// part will be computed twice. In V2 such common paths will be detected and if
// they can be merged in to a single while-loop, they would be merged and in
// this case only one while-loop will be created and the common part will be
// calculated once but reused twice
class TensorSplitter : public HloModulePass {
 public:
  absl::string_view name() const override { return "tensor-splitter"; }

  StatusOr<bool> Run(HloModule* module) override;

  // Use this to retreive the configured split size in bytes.
  static int64_t TensorBytes(const std::string& option);
  static std::tuple<int64_t, int64_t> SplitSettings();
  static bool endsWith(const std::string& str, std::string pattern);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_TENSOR_SPLITTER_
