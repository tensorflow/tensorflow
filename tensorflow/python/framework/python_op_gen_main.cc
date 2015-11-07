#include "tensorflow/python/framework/python_op_gen.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

void PrintAllPythonOps(const char* hidden, bool require_shapes) {
  OpList ops;
  OpRegistry::Global()->Export(false, &ops);
  PrintPythonOps(ops, hidden, require_shapes);
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc == 2) {
    tensorflow::PrintAllPythonOps("", std::string(argv[1]) == "1");
  } else if (argc == 3) {
    tensorflow::PrintAllPythonOps(argv[1], std::string(argv[2]) == "1");
  } else {
    return -1;
  }
  return 0;
}
