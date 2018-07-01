#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_CLASSIFICATION_PREDICATES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_CLASSIFICATION_PREDICATES_H_

namespace xla {

class HloInstruction;

namespace poplarplugin {

struct CompilerAnnotations;

bool IsForward(const HloInstruction*, const CompilerAnnotations&);
bool IsBackpropInput(const HloInstruction*, const CompilerAnnotations&);
bool IsBackpropFilter(const HloInstruction*, const CompilerAnnotations&);

}  // namespace poplarplugin
}  // namespace xla

#endif
