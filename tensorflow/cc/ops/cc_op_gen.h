#ifndef TENSORFLOW_CC_OPS_CC_OP_GEN_H_
#define TENSORFLOW_CC_OPS_CC_OP_GEN_H_

#include "tensorflow/core/framework/op_def.pb.h"

namespace tensorflow {

// Result is written to files dot_h and dot_cc.
void WriteCCOps(const OpList& ops, const std::string& dot_h_fname,
                const std::string& dot_cc_fname);

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_CC_OP_GEN_H_
