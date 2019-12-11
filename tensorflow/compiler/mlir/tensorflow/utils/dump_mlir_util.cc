/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"

#include <cstdint>
#include <cstring>
#include <string>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {
struct NameCounts {
  mutex counts_mutex;
  llvm::StringMap<int64_t> counts;
};

std::string MakeUniqueFilename(string name) {
  static NameCounts& instance = *new NameCounts;

  // Remove illegal characters from `name`.
  for (int i = 0; i < name.size(); ++i) {
    char ch = name[i];
    if (ch == '/' || ch == '[' || ch == ']' || ch == '*' || ch == '?') {
      name[i] = '_';
    }
  }

  int count;
  {
    mutex_lock lock(instance.counts_mutex);
    count = instance.counts[name]++;
  }

  std::string filename = name;
  if (count > 0) {
    filename = llvm::formatv("{0}_{1}", filename, count).str();
  }
  filename = llvm::Twine(filename).concat(".mlir").str();
  return filename;
}
}  // namespace

std::string DumpMlirOpToFile(llvm::StringRef name, mlir::Operation* op,
                             llvm::StringRef dirname) {
  const char* dir = nullptr;
  if (!dirname.empty())
    dir = dirname.data();
  else
    dir = getenv("TF_DUMP_GRAPH_PREFIX");

  if (!dir) {
    LOG(WARNING)
        << "Failed to dump MLIR operation '"
        << op->getName().getStringRef().str() << "' to '" << name.str()
        << "' because dump location is not specified through either "
           "TF_DUMP_GRAPH_PREFIX environment variable or function argument.";
    return "(TF_DUMP_GRAPH_PREFIX not specified)";
  }

  std::string txt_op;
  {
    llvm::raw_string_ostream os(txt_op);
    op->print(os, mlir::OpPrintingFlags().useLocalScope());
    os.flush();
  }

  Env* env = Env::Default();
  std::string filepath;
  if (std::strncmp(dir, "-", 2) == 0) {
    LOG(INFO) << txt_op;
    filepath = "LOG(INFO)";
  } else {
    Status status = env->RecursivelyCreateDir(dir);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to create '" << dir
                   << "' directory for dumping MLIR operation '"
                   << op->getName().getStringRef().str() << "': " << status;
      return "(unavailable)";
    }
    filepath =
        llvm::Twine(dir).concat("/").concat(MakeUniqueFilename(name)).str();
    status = WriteStringToFile(env, filepath, txt_op);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to dump MLIR operation '"
                   << op->getName().getStringRef().str() << "' to file '"
                   << filepath << "': " << status;
      return "(unavailable)";
    }
  }

  LOG(INFO) << "Dumped MLIR operation '" << op->getName().getStringRef().str()
            << "' to '" << filepath << "'";
  return filepath;
}

}  // namespace tensorflow
