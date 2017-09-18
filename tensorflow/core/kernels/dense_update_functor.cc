/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/dense_update_functor.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <>
struct DenseUpdate<CPUDevice, string, ASSIGN> {
  void operator()(const CPUDevice& d, typename TTypes<string>::Flat params,
                  typename TTypes<string>::ConstFlat update) {
    if (params.dimension(0) == 1) {
      params.data()->resize(update.data()->size());
      auto work = [&params, &update](int64 start, int64 end) {
        memmove(const_cast<char*>(params.data()->data()) + start,
                update.data()->data() + start, end - start);
      };
      d.parallelFor(update.data()->size(),
                    Eigen::TensorOpCost(.1,  // chosen to force large chunks
                                        .1, 0),
                    work);
    } else {
      auto work = [&params, &update](int64 start, int64 end) {
        for (int i = start; i < end; ++i) {
          params.data()[i].resize(update.data()[i].size());
          memmove(const_cast<char*>(params.data()[i].data()),
                  update.data()[i].data(), update.data()[i].size());
        }
      };
      int64 estimated_string_size;
      if (update.size() > 0) {
        // first element of the tensor seems as good a guess as any of the sizes
        // of the strings contained within...
        estimated_string_size =
            std::max(update.data()[0].size(), sizeof(string));
      } else {
        estimated_string_size = sizeof(string);
      }
      d.parallelFor(
          params.dimension(0),
          Eigen::TensorOpCost(estimated_string_size, estimated_string_size, 0),
          work);
    }
  }
};

}  // namespace functor

}  // namespace tensorflow
