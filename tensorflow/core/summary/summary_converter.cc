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
#include "tensorflow/core/summary/summary_converter.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/png/png_io.h"
#include "tensorflow/core/lib/wav/wav_io.h"

namespace tensorflow {
namespace {

template <typename T>
Status TensorValueAt(Tensor t, int64 i, T* out) {
#define CASE(I)                            \
  case DataTypeToEnum<I>::value:           \
    *out = static_cast<T>(t.flat<I>()(i)); \
    break;
#define COMPLEX_CASE(I)                           \
  case DataTypeToEnum<I>::value:                  \
    *out = static_cast<T>(t.flat<I>()(i).real()); \
    break;
  // clang-format off
  switch (t.dtype()) {
    TF_CALL_half(CASE)
    TF_CALL_float(CASE)
    TF_CALL_double(CASE)
    TF_CALL_int8(CASE)
    TF_CALL_int16(CASE)
    TF_CALL_int32(CASE)
    TF_CALL_int64(CASE)
    TF_CALL_uint8(CASE)
    TF_CALL_uint16(CASE)
    TF_CALL_uint32(CASE)
    TF_CALL_uint64(CASE)
    TF_CALL_complex64(COMPLEX_CASE)
    TF_CALL_complex128(COMPLEX_CASE)
    default:
        return errors::Unimplemented("SummaryFileWriter ",
                                     DataTypeString(t.dtype()),
                                     " not supported.");
  }
  // clang-format on
  return Status::OK();
#undef CASE
#undef COMPLEX_CASE
}

typedef Eigen::Tensor<uint8, 2, Eigen::RowMajor> Uint8Image;

// Add the sequence of images specified by ith_image to the summary.
//
// Factoring this loop out into a helper function lets ith_image behave
// differently in the float and uint8 cases: the float case needs a temporary
// buffer which can be shared across calls to ith_image, but the uint8 case
// does not.
Status AddImages(const string& tag, int max_images, int batch_size, int w,
                 int h, int depth,
                 const std::function<Uint8Image(int)>& ith_image, Summary* s) {
  const int N = std::min<int>(max_images, batch_size);
  for (int i = 0; i < N; ++i) {
    Summary::Value* v = s->add_value();
    // The tag depends on the number of requested images (not the number
    // produced.)
    //
    // Note that later on avisu uses "/" to figure out a consistent naming
    // convention for display, so we append "/image" to guarantee that the
    // image(s) won't be displayed in the global scope with no name.
    if (max_images > 1) {
      v->set_tag(strings::StrCat(tag, "/image/", i));
    } else {
      v->set_tag(strings::StrCat(tag, "/image"));
    }

    const auto image = ith_image(i);
    Summary::Image* si = v->mutable_image();
    si->set_height(h);
    si->set_width(w);
    si->set_colorspace(depth);
    const int channel_bits = 8;
    const int compression = -1;  // Use zlib default
    if (!png::WriteImageToBuffer(image.data(), w, h, w * depth, depth,
                                 channel_bits, compression,
                                 si->mutable_encoded_image_string(), nullptr)) {
      return errors::Internal("PNG encoding failed");
    }
  }
  return Status::OK();
}

template <class T>
void NormalizeFloatImage(int hw, int depth,
                         typename TTypes<T>::ConstMatrix values,
                         typename TTypes<uint8>::ConstVec bad_color,
                         Uint8Image* image) {
  if (!image->size()) return;  // Nothing to do for empty images

  // Rescale the image to uint8 range.
  //
  // We are trying to generate an RGB image from a float/half tensor.  We do
  // not have any info about the expected range of values in the tensor
  // but the generated image needs to have all RGB values within [0, 255].
  //
  // We use two different algorithms to generate these values.  If the
  // tensor has only positive values we scale them all by 255/max(values).
  // If the tensor has both negative and positive values we scale them by
  // the max of their absolute values and center them around 127.
  //
  // This works for most cases, but does not respect the relative dynamic
  // range across different instances of the tensor.

  // Compute min and max ignoring nonfinite pixels
  float image_min = std::numeric_limits<float>::infinity();
  float image_max = -image_min;
  for (int i = 0; i < hw; i++) {
    bool finite = true;
    for (int j = 0; j < depth; j++) {
      if (!Eigen::numext::isfinite(values(i, j))) {
        finite = false;
        break;
      }
    }
    if (finite) {
      for (int j = 0; j < depth; j++) {
        float value(values(i, j));
        image_min = std::min(image_min, value);
        image_max = std::max(image_max, value);
      }
    }
  }

  // Pick an affine transform into uint8
  const float kZeroThreshold = 1e-6;
  T scale, offset;
  if (image_min < 0) {
    const float max_val = std::max(std::abs(image_min), std::abs(image_max));
    scale = T(max_val < kZeroThreshold ? 0.0f : 127.0f / max_val);
    offset = T(128.0f);
  } else {
    scale = T(image_max < kZeroThreshold ? 0.0f : 255.0f / image_max);
    offset = T(0.0f);
  }

  // Transform image, turning nonfinite values to bad_color
  for (int i = 0; i < hw; i++) {
    bool finite = true;
    for (int j = 0; j < depth; j++) {
      if (!Eigen::numext::isfinite(values(i, j))) {
        finite = false;
        break;
      }
    }
    if (finite) {
      image->chip<0>(i) =
          (values.template chip<0>(i) * scale + offset).template cast<uint8>();
    } else {
      image->chip<0>(i) = bad_color;
    }
  }
}

template <class T>
Status NormalizeAndAddImages(const Tensor& tensor, int max_images, int h, int w,
                             int hw, int depth, int batch_size,
                             const string& base_tag, Tensor bad_color_tensor,
                             Summary* s) {
  // For float and half images, nans and infs are replaced with bad_color.
  if (bad_color_tensor.dim_size(0) < depth) {
    return errors::InvalidArgument(
        "expected depth <= bad_color.size, got depth = ", depth,
        ", bad_color.size = ", bad_color_tensor.dim_size(0));
  }
  auto bad_color_full = bad_color_tensor.vec<uint8>();
  typename TTypes<uint8>::ConstVec bad_color(bad_color_full.data(), depth);

  // Float images must be scaled and translated.
  Uint8Image image(hw, depth);
  auto ith_image = [&tensor, &image, bad_color, batch_size, hw, depth](int i) {
    auto tensor_eigen = tensor.template shaped<T, 3>({batch_size, hw, depth});
    typename TTypes<T>::ConstMatrix values(
        &tensor_eigen(i, 0, 0), Eigen::DSizes<Eigen::DenseIndex, 2>(hw, depth));
    NormalizeFloatImage<T>(hw, depth, values, bad_color, &image);
    return image;
  };
  return AddImages(base_tag, max_images, batch_size, w, h, depth, ith_image, s);
}

}  // namespace

Status AddTensorAsScalarToSummary(const Tensor& t, const string& tag,
                                  Summary* s) {
  Summary::Value* v = s->add_value();
  v->set_tag(tag);
  float value;
  TF_RETURN_IF_ERROR(TensorValueAt<float>(t, 0, &value));
  v->set_simple_value(value);
  return Status::OK();
}

Status AddTensorAsHistogramToSummary(const Tensor& t, const string& tag,
                                     Summary* s) {
  Summary::Value* v = s->add_value();
  v->set_tag(tag);
  histogram::Histogram histo;
  for (int64 i = 0; i < t.NumElements(); i++) {
    double double_val;
    TF_RETURN_IF_ERROR(TensorValueAt<double>(t, i, &double_val));
    if (Eigen::numext::isnan(double_val)) {
      return errors::InvalidArgument("Nan in summary histogram for: ", tag);
    } else if (Eigen::numext::isinf(double_val)) {
      return errors::InvalidArgument("Infinity in summary histogram for: ",
                                     tag);
    }
    histo.Add(double_val);
  }
  histo.EncodeToProto(v->mutable_histo(), false /* Drop zero buckets */);
  return Status::OK();
}

Status AddTensorAsImageToSummary(const Tensor& tensor, const string& tag,
                                 int max_images, const Tensor& bad_color,
                                 Summary* s) {
  if (!(tensor.dims() == 4 &&
        (tensor.dim_size(3) == 1 || tensor.dim_size(3) == 3 ||
         tensor.dim_size(3) == 4))) {
    return errors::InvalidArgument(
        "Tensor must be 4-D with last dim 1, 3, or 4, not ",
        tensor.shape().DebugString());
  }
  if (!(tensor.dim_size(0) < (1LL << 31) && tensor.dim_size(1) < (1LL << 31) &&
        tensor.dim_size(2) < (1LL << 31) &&
        (tensor.dim_size(1) * tensor.dim_size(2)) < (1LL << 29))) {
    return errors::InvalidArgument("Tensor too large for summary ",
                                   tensor.shape().DebugString());
  }
  // The casts and h * w cannot overflow because of the limits above.
  const int batch_size = static_cast<int>(tensor.dim_size(0));
  const int h = static_cast<int>(tensor.dim_size(1));
  const int w = static_cast<int>(tensor.dim_size(2));
  const int hw = h * w;  // Compact these two dims for simplicity
  const int depth = static_cast<int>(tensor.dim_size(3));
  if (tensor.dtype() == DT_UINT8) {
    // For uint8 input, no normalization is necessary
    auto ith_image = [&tensor, batch_size, hw, depth](int i) {
      auto values = tensor.shaped<uint8, 3>({batch_size, hw, depth});
      return typename TTypes<uint8>::ConstMatrix(
          &values(i, 0, 0), Eigen::DSizes<Eigen::DenseIndex, 2>(hw, depth));
    };
    TF_RETURN_IF_ERROR(
        AddImages(tag, max_images, batch_size, w, h, depth, ith_image, s));
  } else if (tensor.dtype() == DT_HALF) {
    TF_RETURN_IF_ERROR(NormalizeAndAddImages<Eigen::half>(
        tensor, max_images, h, w, hw, depth, batch_size, tag, bad_color, s));
  } else if (tensor.dtype() == DT_FLOAT) {
    TF_RETURN_IF_ERROR(NormalizeAndAddImages<float>(
        tensor, max_images, h, w, hw, depth, batch_size, tag, bad_color, s));
  } else {
    return errors::InvalidArgument(
        "Only DT_INT8, DT_HALF, and DT_FLOAT images are supported. Got ",
        DataTypeString(tensor.dtype()));
  }
  return Status::OK();
}

Status AddTensorAsAudioToSummary(const Tensor& tensor, const string& tag,
                                 int max_outputs, float sample_rate,
                                 Summary* s) {
  if (sample_rate <= 0.0f) {
    return errors::InvalidArgument("sample_rate must be > 0");
  }
  const int batch_size = tensor.dim_size(0);
  const int64 length_frames = tensor.dim_size(1);
  const int64 num_channels =
      tensor.dims() == 2 ? 1 : tensor.dim_size(tensor.dims() - 1);
  const int N = std::min<int>(max_outputs, batch_size);
  for (int i = 0; i < N; ++i) {
    Summary::Value* v = s->add_value();
    if (max_outputs > 1) {
      v->set_tag(strings::StrCat(tag, "/audio/", i));
    } else {
      v->set_tag(strings::StrCat(tag, "/audio"));
    }

    Summary::Audio* sa = v->mutable_audio();
    sa->set_sample_rate(sample_rate);
    sa->set_num_channels(num_channels);
    sa->set_length_frames(length_frames);
    sa->set_content_type("audio/wav");

    auto values =
        tensor.shaped<float, 3>({batch_size, length_frames, num_channels});
    auto channels_by_frames = typename TTypes<float>::ConstMatrix(
        &values(i, 0, 0),
        Eigen::DSizes<Eigen::DenseIndex, 2>(length_frames, num_channels));
    size_t sample_rate_truncated = lrintf(sample_rate);
    if (sample_rate_truncated == 0) {
      sample_rate_truncated = 1;
    }
    TF_RETURN_IF_ERROR(wav::EncodeAudioAsS16LEWav(
        channels_by_frames.data(), sample_rate_truncated, num_channels,
        length_frames, sa->mutable_encoded_audio_string()));
  }
  return Status::OK();
}

}  // namespace tensorflow
