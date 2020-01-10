// Operators that deal with SummaryProtos (encoded as DT_STRING tensors) as
// inputs or outputs in various ways.

// See docs in ../ops/summary_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/png/png_io.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

class SummaryImageOp : public OpKernel {
 public:
  explicit SummaryImageOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("max_images", &max_images_));
    const TensorProto* proto;
    OP_REQUIRES_OK(context, context->GetAttr("bad_color", &proto));
    OP_REQUIRES_OK(context, context->device()->MakeTensorFromProto(
                                *proto, AllocatorAttributes(), &bad_color_));
    OP_REQUIRES(context, bad_color_.dtype() == DT_UINT8,
                errors::InvalidArgument("bad_color must be uint8, got ",
                                        DataTypeString(bad_color_.dtype())));
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(bad_color_.shape()),
        errors::InvalidArgument("bad_color must be a vector, got shape ",
                                bad_color_.shape().ShortDebugString()));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& tags = c->input(0);
    const Tensor& tensor = c->input(1);
    OP_REQUIRES(c, TensorShapeUtils::IsLegacyScalar(tags.shape()),
                errors::InvalidArgument("Tags must have be a scalar"));
    OP_REQUIRES(c, tensor.dims() == 4 &&
                       (tensor.dim_size(3) == 1 || tensor.dim_size(3) == 3 ||
                        tensor.dim_size(3) == 4),
                errors::InvalidArgument(
                    "Tensor must be 4-D with last dim 1, 3, or 4, not ",
                    tensor.shape().DebugString()));
    const string& base_tag = tags.scalar<string>()();

    const int batch_size = tensor.dim_size(0);
    const int h = tensor.dim_size(1);
    const int w = tensor.dim_size(2);
    const int hw = h * w;  // Compact these two dims for simplicity
    const int depth = tensor.dim_size(3);
    auto tensor_eigen = tensor.shaped<float, 3>({batch_size, hw, depth});

    OP_REQUIRES(c, bad_color_.dim_size(0) >= depth,
                errors::InvalidArgument(
                    "expected depth <= bad_color.size, got depth = ", depth,
                    ", bad_color.size = ", bad_color_.dim_size(0)));
    auto bad_color_full = bad_color_.vec<uint8>();
    typename TTypes<uint8>::Vec bad_color(bad_color_full.data(), depth);

    // RGB (or gray or RGBA) is last dimension
    Eigen::Tensor<uint8, 2, Eigen::RowMajor> image(hw, depth);

    Summary s;
    const int N = std::min<int>(max_images_, batch_size);
    for (int i = 0; i < N; ++i) {
      Summary::Value* v = s.add_value();
      // The tag depends on the number of requested images (not the number
      // produced.)
      //
      // Note that later on avisu uses "/" to figure out a consistent naming
      // convention for display, so we append "/image" to guarantee that the
      // image(s) won't be displayed in the global scope with no name.
      if (max_images_ > 1) {
        v->set_tag(strings::StrCat(base_tag, "/image/", i));
      } else {
        v->set_tag(strings::StrCat(base_tag, "/image"));
      }

      if (image.size()) {
        typename TTypes<float>::ConstMatrix values(
            &tensor_eigen(i, 0, 0),
            Eigen::DSizes<Eigen::DenseIndex, 2>(hw, depth));

        // Rescale the image to uint8 range.
        //
        // We are trying to generate an RCG image from a float tensor.  We do
        // not have any info about the expected range of values in the tensor
        // but the generated image needs to have all RGB values within [0, 255].
        //
        // We use two different algorithms to generate these values.  If the
        // tensor has only positive values we scale them all by 255/max(values).
        // If the tensor has both negative and positive values we scale them by
        // the max of their absolute values and center them around 127.
        //
        // This works for most cases, but has the incovenient of not respecting
        // the relative dynamic range across different instances of the tensor.

        // Compute min and max ignoring nonfinite pixels
        float image_min = std::numeric_limits<float>::infinity();
        float image_max = -image_min;
        for (int i = 0; i < hw; i++) {
          bool finite = true;
          for (int j = 0; j < depth; j++) {
            if (!std::isfinite(values(i, j))) {
              finite = false;
              break;
            }
          }
          if (finite) {
            for (int j = 0; j < depth; j++) {
              float value = values(i, j);
              image_min = std::min(image_min, value);
              image_max = std::max(image_max, value);
            }
          }
        }

        // Pick an affine transform into uint8
        const float kZeroThreshold = 1e-6;
        float scale, offset;
        if (image_min < 0) {
          float max_val = std::max(std::abs(image_min), std::abs(image_max));
          scale = max_val < kZeroThreshold ? 0.0f : 127.0f / max_val;
          offset = 128.0f;
        } else {
          scale = image_max < kZeroThreshold ? 0.0f : 255.0f / image_max;
          offset = 0.0f;
        }

        // Transform image, turning nonfinite values to bad_color
        for (int i = 0; i < hw; i++) {
          bool finite = true;
          for (int j = 0; j < depth; j++) {
            if (!std::isfinite(values(i, j))) {
              finite = false;
              break;
            }
          }
          if (finite) {
            image.chip<0>(i) =
                (values.chip<0>(i) * scale + offset).cast<uint8>();
          } else {
            image.chip<0>(i) = bad_color;
          }
        }
      }

      Summary::Image* si = v->mutable_image();
      si->set_height(h);
      si->set_width(w);
      si->set_colorspace(depth);
      OP_REQUIRES(c, png::WriteImageToBuffer(
                         image.data(), w, h, w * depth, depth, 8, -1,
                         si->mutable_encoded_image_string(), nullptr),
                  errors::Internal("PNG encoding failed"));
    }

    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(s.SerializeToString(&summary_tensor->scalar<string>()()));
  }

 private:
  int64 max_images_;
  Tensor bad_color_;
};

REGISTER_KERNEL_BUILDER(Name("ImageSummary").Device(DEVICE_CPU),
                        SummaryImageOp);

}  // namespace tensorflow
