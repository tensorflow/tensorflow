#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_FIND_ALL_USERS_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_FIND_ALL_USERS_H_

#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {

namespace poplarplugin {

using InstructionList = std::vector<HloInstruction*>;

/**
 * A class which finds all of the 'real' users of an instruction, looking
 * through kCall, kWhile, kTuple and kGetTupleElement operations.
 */
class FindAllUsers {
 public:
  FindAllUsers() {}

  void Find(HloInstruction* inst);

  std::set<HloInstruction*> Users() const;
  const std::set<InstructionList>& Paths() const;
  const InstructionList& PathFor(HloInstruction* target) const;

 private:
  void FindUsers(HloInstruction* tgt, const InstructionList& stack,
                 int64 index);

  InstructionList path;
  std::set<InstructionList> paths;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
