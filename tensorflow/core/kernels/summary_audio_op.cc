/* Copyright 2016 Google Inc. All Rights Reserved.

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
    OP_REQUIRES_OK(context, context->GetAttr("sample_rate", &sample_rate_));
    OP_REQUIRES(context, sample_rate_ > 0.0f,
                errors::InvalidArgument("sample_rate must be > 0"));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& tag = c->input(0);
    const Tensor& tensor = c->input(1);
    OP_REQUIRES(c, IsLegacyScalar(tag.shape()),
                errors::InvalidArgument("Tag must be a scalar"));
    OP_REQUIRES(c, tensor.dims() >= 2 && tensor.dims() <= 3,
                errors::InvalidArgument("Tensor must be 3-D or 2-D, got: ",
                                        tensor.shape().DebugString()));
    const string& base_tag = tag.scalar<string>()();
    const int batch_size = tensor.dim_size(0);
    const int64 length_frames = tensor.dim_size(1);
    const int64 num_channels =
        tensor.dims() == 2 ? 1 : tensor.dim_size(tensor.dims() - 1);

    Summary s;
    const int N = std::min<int>(max_outputs_, batch_size);
    for (int i = 0; i < N; ++i) {
      Summary::Value* v = s.add_value();
      if (max_outputs_ > 1) {
        v->set_tag(strings::StrCat(base_tag, "/audio/", i));
      } else {
        v->set_tag(strings::StrCat(base_tag, "/audio"));
      }

      Summary::Audio* sa = v->mutable_audio();
      sa->set_sample_rate(sample_rate_);
      sa->set_num_channels(num_channels);
      sa->set_length_frames(length_frames);
      sa->set_content_type("audio/wav");

      auto values =
          tensor.shaped<float, 3>({batch_size, length_frames, num_channels});
      auto channels_by_frames = typename TTypes<float>::ConstMatrix(
          &values(i, 0, 0),
          Eigen::DSizes<Eigen::DenseIndex, 2>(length_frames, num_channels));
      size_t sample_rate_truncated = lrintf(sample_rate_);
      if (sample_rate_truncated == 0) {
        sample_rate_truncated = 1;
      }
      OP_REQUIRES_OK(
          c, wav::EncodeAudioAsS16LEWav(
                 channels_by_frames.data(), sample_rate_truncated, num_channels,
                 length_frames, sa->mutable_encoded_audio_string()));
    }

    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(s.SerializeToString(&summary_tensor->scalar<string>()()));
  }

 private:
  int max_outputs_;
  float sample_rate_;
};

REGISTER_KERNEL_BUILDER(Name("AudioSummary").Device(DEVICE_CPU),
                        SummaryAudioOp);

}  // namespace tensorflow
