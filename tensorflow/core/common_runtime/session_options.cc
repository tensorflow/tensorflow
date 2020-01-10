#include "tensorflow/core/public/session_options.h"

#include "tensorflow/core/public/env.h"

namespace tensorflow {

SessionOptions::SessionOptions() : env(Env::Default()) {}

}  // namespace tensorflow
