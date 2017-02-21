/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/legacy_flags/parallel_check_op_flags.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace {

REGISTER_OP("ParallelCheck")
    .Attr("T: list(type) >= 0")
    .Input("expected: T")
    .Input("actual: T")
    .Output("result: T")
    .Doc(R"doc(
Op that compares two sets of inputs for near-identity, and propagates the first.
Inequality is logged to ERROR log.
)doc");

// Inputs 2*N tensors, outputs the first N inputs.
// Logs errors if input tensor i and i + N are not (near) identical
// in any position.
class ParallelCheckOp : public OpKernel {
 public:
  explicit ParallelCheckOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  template <typename T>
  int CompareTensors(DataType dtype, const char* v0, const char* v1,
                     int64 num_elts, int input_idx) {
    int failed = 0;
    const T* p0 = reinterpret_cast<const T*>(v0);
    const T* p1 = reinterpret_cast<const T*>(v1);
    double rtol;
    legacy_flags::ParallelCheckOpFlags* flags =
        legacy_flags::GetParallelCheckOpFlags();
    if (!tensorflow::strings::safe_strtod(flags->parallel_check_rtol.c_str(),
                                          &rtol)) {
      LOG(ERROR) << "can't convert parallel_check_rtol "
                 << flags->parallel_check_rtol << " to double";
    }
    double atol;
    if (!tensorflow::strings::safe_strtod(flags->parallel_check_atol.c_str(),
                                          &atol)) {
      LOG(ERROR) << "can't convert parallel_check_atol "
                 << flags->parallel_check_atol << " to double";
    }
    for (int i = 0; i < num_elts; ++i) {
      bool ok = (p0[i] == p1[i]);
      VLOG(2) << "output " << input_idx << " element " << i << ": " << p0[i];
      if (!ok) {
        if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
          float tolerance =
              std::max(atol, std::max(fabs(rtol * p0[i]), fabs(rtol * p1[i])));
          T diff = p0[i] - p1[i];
          if (diff < 0) diff = 0 - diff;
          ok = (diff <= tolerance);
        }
        if (ok) continue;
        LOG(ERROR) << "Op " << def().name() << " fails equality at output "
                   << input_idx << " type " << DataTypeString(dtype)
                   << " element " << i << ": std_val=" << p0[i]
                   << " test_val=" << p1[i] << " diff=" << (p0[i] - p1[i]);
        if (++failed > 10) break;
      }
    }
    return failed;
  }

  void Compute(OpKernelContext* ctx) override {
    VLOG(1) << "Compute " << def().name();
    const int num_pairs = ctx->num_inputs() / 2;
    for (int i = 0; i < num_pairs; ++i) {
      CHECK_EQ(ctx->input_dtype(i), ctx->input_dtype(i + num_pairs));
      Tensor t0 = ctx->input(i);
      Tensor t1 = ctx->input(i + num_pairs);
      int64 num_elts = t0.NumElements();
      CHECK_EQ(num_elts, t1.NumElements());

      // Compare inputs elementwise for near-exact equality.
      const char* v0 = t0.tensor_data().data();
      const char* v1 = t1.tensor_data().data();
      int failed = 0;
      switch (ctx->input_dtype(i)) {
        case DT_INT32:
          failed =
              CompareTensors<int32>(ctx->input_dtype(i), v0, v1, num_elts, i);
          break;
        case DT_INT64:
          failed =
              CompareTensors<int64>(ctx->input_dtype(i), v0, v1, num_elts, i);
          break;
        case DT_FLOAT:
          failed =
              CompareTensors<float>(ctx->input_dtype(i), v0, v1, num_elts, i);
          break;
        case DT_DOUBLE:
          failed =
              CompareTensors<double>(ctx->input_dtype(i), v0, v1, num_elts, i);
          break;
        case DT_BOOL:
          failed =
              CompareTensors<bool>(ctx->input_dtype(i), v0, v1, num_elts, i);
          break;
        default:
          LOG(FATAL) << "unimpl: " << ctx->input_dtype(i);
      }
      if (failed > 0) {
        LOG(ERROR) << "check failed for " << def().name() << " output " << i
                   << " num_elts: " << num_elts;
        legacy_flags::ParallelCheckOpFlags* flags =
            legacy_flags::GetParallelCheckOpFlags();
        if (flags->parallel_check_failfast) {
          LOG(QFATAL) << "failfast on first parallel-check failure";
        }
      } else {
        VLOG(1) << "check passed for " << def().name() << " output " << i
                << " num_elts: " << num_elts;
      }

      // Propagate the std value.
      if (IsRefType(ctx->input_dtype(i))) {
        ctx->forward_ref_input_to_ref_output(i, i);
      } else {
        ctx->set_output(i, ctx->input(i));
      }
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ParallelCheckOp);
};

REGISTER_KERNEL_BUILDER(Name("ParallelCheck").Device(DEVICE_CPU),
                        ParallelCheckOp);

}  // namespace
}  // namespace tensorflow
