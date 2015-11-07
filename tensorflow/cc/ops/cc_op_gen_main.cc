#include "tensorflow/cc/ops/cc_op_gen.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {
namespace {

void PrintAllCCOps(const std::string& dot_h, const std::string& dot_cc,
                   bool include_internal) {
  OpList ops;
  OpRegistry::Global()->Export(include_internal, &ops);
  WriteCCOps(ops, dot_h, dot_cc);
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc != 4) {
    fprintf(stderr,
            "Usage: %s out.h out.cc include_internal\n"
            "  include_internal: 1 means include internal ops\n",
            argv[0]);
    exit(1);
  }

  bool include_internal = tensorflow::StringPiece("1") == argv[3];
  tensorflow::PrintAllCCOps(argv[1], argv[2], include_internal);
  return 0;
}
