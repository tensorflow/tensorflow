#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/protobuf/config.pb.h"

int main() {
  tensorflow::ConfigProto config_proto;
  tensorflow::grappler::MetaOptimizer optimizer(nullptr, config_proto);
  return 0;
}