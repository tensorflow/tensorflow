#include "tensorflow/compiler/plugin/poplar/driver/tools/classification_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace poplarplugin {

bool IsForward(const HloInstruction* inst,
               const CompilerAnnotations& annotations) {
  if (annotations.classification_map.count(inst) == 0) {
    return false;
  }
  auto type = annotations.classification_map.at(inst);
  return type == ConvClassificationType::FORWARD;
}

bool IsBackpropInput(const HloInstruction* inst,
                     const CompilerAnnotations& annotations) {
  if (annotations.classification_map.count(inst) == 0) {
    return false;
  }
  auto type = annotations.classification_map.at(inst);
  return type == ConvClassificationType::BACKPROP_INPUT;
}

bool IsBackpropFilter(const HloInstruction* inst,
                      const CompilerAnnotations& annotations) {
  if (annotations.classification_map.count(inst) == 0) {
    return false;
  }
  auto type = annotations.classification_map.at(inst);
  return type == ConvClassificationType::BACKPROP_FILTER;
}

}  // namespace poplarplugin
}  // namespace xla
