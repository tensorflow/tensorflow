#ifndef TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_OP_GEN_H_
#define TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_OP_GEN_H_

#include <string>
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

// Result is printed to stdout.  hidden_ops should be a comma-separated
// list of Op names that should get a leading _ in the output.
void PrintPythonOps(const OpList& ops, const string& hidden_ops,
                    bool require_shapes);

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_OP_GEN_H_
