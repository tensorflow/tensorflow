/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Operators that deal with SummaryProtos (encoded as DT_STRING tensors) as
// inputs or outputs in various ways.

// See docs in ../ops/summary_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/wav/wav_io.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class SummaryAudioOp : public OpKernel {
 public:
  explicit SummaryAudioOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("max_outputs", &max_outputs_));
    OP_REQUIRES(context, max_outputs_ > 0,
                errors::InvalidArgument("max_outputs must be > 0"));
    has_sample_rate_attr_ =
        context->GetAttr("sample_rate", &sample_rate_attr_).ok();
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& tag = c->input(0);
    const Tensor& tensor = c->input(1);
    OP_REQUIRES(c, TensorShapeUtils::IsScalar(tag.shape()),
                errors::InvalidArgument("Tag must be a scalar"));
    OP_REQUIRES(c, tensor.dims() >= 2 && tensor.dims() <= 3,
                errors::InvalidArgument("Tensor must be 3-D or 2-D, got: ",
                                        tensor.shape().DebugString()));
    const string& base_tag = tag.scalar<tstring>()();

    float sample_rate = sample_rate_attr_;
    if (!has_sample_rate_attr_) {
      const Tensor& sample_rate_tensor = c->input(2);
      OP_REQUIRES(c,
                  sample_rate_tensor.IsAligned() &&
                      sample_rate_tensor.NumElements() == 1,
                  errors::InvalidArgument(
                      "sample_rate must be rank-0 or contain a single value"));
      sample_rate = sample_rate_tensor.scalar<float>()();
    }
    OP_REQUIRES(c, sample_rate > 0.0f,
                errors::InvalidArgument("sample_rate must be > 0"));

    const int batch_size = tensor.dim_size(0);
    const int64_t length_frames = tensor.dim_size(1);
    const int64_t num_channels =
        tensor.dims() == 2 ? 1 : tensor.dim_size(tensor.dims() - 1);

    Summary s;
    const int N = std::min<int>(max_outputs_, batch_size);
    for (int i = 0; i < N; ++i) {
      Summary::Value* v = s.add_value();
      if (max_outputs_ > 1) {
        v->set_tag(absl::StrCat(base_tag, "/audio/", i));
      } else {
        v->set_tag(absl::StrCat(base_tag, "/audio"));
      }

      Summary::Audio* sa = v->mutable_audio();
      sa->set_sample_rate(sample_rate);
      sa->set_num_channels(num_channels);
      sa->set_length_frames(length_frames);
      sa->set_content_type("audio/wav");

      auto values =
          tensor.shaped<float, 3>({batch_size, length_frames, num_channels});
      const float* data =
          tensor.NumElements() == 0 ? nullptr : &values(i, 0, 0);

      size_t sample_rate_truncated = lrintf(sample_rate);
      if (sample_rate_truncated == 0) {
        sample_rate_truncated = 1;
      }
      OP_REQUIRES_OK(c, wav::EncodeAudioAsS16LEWav(
                            data, sample_rate_truncated, num_channels,
                            length_frames, sa->mutable_encoded_audio_string()));
    }

    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(SerializeToTString(s, &summary_tensor->scalar<tstring>()()));
  }

 private:
  int max_outputs_;
  bool has_sample_rate_attr_;
  float sample_rate_attr_;
};

REGISTER_KERNEL_BUILDER(Name("AudioSummaryV2").Device(DEVICE_CPU),
                        SummaryAudioOp);

// Deprecated -- this op is registered with sample_rate as an attribute for
// backwards compatibility.
REGISTER_KERNEL_BUILDER(Name("AudioSummary").Device(DEVICE_CPU),
                        SummaryAudioOp);

}  // namespace tensorflow
