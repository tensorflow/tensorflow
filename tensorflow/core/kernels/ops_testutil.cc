#include "tensorflow/core/kernels/ops_testutil.h"

namespace tensorflow {
namespace test {

NodeDef Node(const string& name, const string& op,
             const std::vector<string>& inputs) {
  NodeDef def;
  def.set_name(name);
  def.set_op(op);
  for (const string& s : inputs) {
    def.add_input(s);
  }
  return def;
}

}  // namespace test
}  // namespace tensorflow
