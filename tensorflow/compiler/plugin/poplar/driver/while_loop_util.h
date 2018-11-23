#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_WHILE_LOOP_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_WHILE_LOOP_UTIL_H_

/*
 * These functions are independent of poplar, and are included in the
 * optimizers target within the BUILD file.
 */

#include <vector>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace poplarplugin {

class WhileLoopUtil {
 public:
  static bool IsGTEFromParamIndex(const HloInstruction* inst,
                                  int64 param_index);
  static bool IsIntegralConstant(const HloInstruction* inst);
  static StatusOr<bool> IsIntegralConstantOfValue(const HloInstruction* inst,
                                                  const int32 value);

  // Find instructions which are incremented by 1 and for which the
  // resulting increment is *only* used in the output tuple of the while body in
  // the same index as the inst
  static StatusOr<std::vector<HloInstruction*>>
  FindMatchingGTEIncrementsInsideBody(const HloInstruction* inst,
                                      const HloComputation* while_body,
                                      HloOpcode opcode);
  static StatusOr<int32> CanConvertWhileToRepeat(HloInstruction* while_inst);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(WhileLoopUtil);
};

}  // namespace poplarplugin
}  // namespace xla

#endif
