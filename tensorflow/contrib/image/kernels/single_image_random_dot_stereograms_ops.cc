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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

template <typename T>
class SingleImageRandomDotStereogramsOp : public OpKernel {
 private:
  int E2Epixels;  // Pixels from eye to eye = eye_to_eye_inches * DPI

  int input_Xvalue;  // X value of input Z values (width)
  int input_Yvalue;  // Y value of input Z values (height)

  int output_Ximage;  // X value of output image (width)
  int output_Yimage;  // Y value of output image (height)
  int output_Cimage;  // color value of output image (color, 1 or 3)  (3 not
                      // implemented)

  int data_box_left;    // X starting value for DATA window
  int data_box_top;     // Y starting value for DATA window
  int data_box_width;   // width of scan line
  int data_box_height;  // hight of image

  int converge_dot_box_end;  // Row convergences dots end on

  uint8* outputImage;  // Output Image flat as a buffer (Tensor Connection)
  double* ZBuffer;     // For internal use, allow for MASK, etc later, actual Z
                       // used for Stereogram, XxY (X is the row index, y is col
                       // index like a screen)
                       // 0 (far) -> 1.0(near) range
  bool hidden_surface_removal;
  int convergence_dots_size;
  int dots_per_inch;
  float eye_separation;
  float mu;
  bool normalize;
  float normalize_max;
  float normalize_min;
  float border_level;
  int number_colors;
  ::tensorflow::PartialTensorShape output_image_shape;
  ::tensorflow::PartialTensorShape output_data_window;

  uint8 Cblack = 0;
  uint8 Cwhite = 255;

  int indexMode = 0;  // 0 - truncate XY, 1 - round XY, 2 - Interpolate XY (not
                      // implemented yet, keep default of 0)
  int interp_x, interp_y;  // 1 - yes, 0 - no  interpolation directions (not
                           // implemented yet)

  bool debugging = false;

  inline int separation(double z) {
    return (std::round((1 - mu * z) * E2Epixels / (2 - mu * z)));
  }

  inline int get_far_width() { return (separation(0.0)); }
  inline int get_near_width() { return (separation(1.0)); }

 public:
  explicit SingleImageRandomDotStereogramsOp(OpKernelConstruction* context)
      : OpKernel(context) {  // Constructor
    OP_REQUIRES_OK(context, context->GetAttr("hidden_surface_removal",
                                             &hidden_surface_removal));
    OP_REQUIRES_OK(context, context->GetAttr("convergence_dots_size",
                                             &convergence_dots_size));
    OP_REQUIRES_OK(context, context->GetAttr("dots_per_inch", &dots_per_inch));
    OP_REQUIRES_OK(context,
                   context->GetAttr("eye_separation", &eye_separation));
    OP_REQUIRES_OK(context, context->GetAttr("mu", &mu));
    OP_REQUIRES_OK(context, context->GetAttr("normalize", &normalize));
    OP_REQUIRES_OK(context, context->GetAttr("normalize_max", &normalize_max));
    OP_REQUIRES_OK(context, context->GetAttr("normalize_min", &normalize_min));
    OP_REQUIRES_OK(context, context->GetAttr("border_level", &border_level));
    OP_REQUIRES_OK(context, context->GetAttr("number_colors", &number_colors));
    OP_REQUIRES_OK(context,
                   context->GetAttr("output_image_shape", &output_image_shape));
    OP_REQUIRES_OK(context,
                   context->GetAttr("output_data_window", &output_data_window));

    E2Epixels =
        eye_separation * dots_per_inch;  // Initialize pixels from eye to eye
  }

  ~SingleImageRandomDotStereogramsOp() override {  // Destructor
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    input_Xvalue = input_tensor.shape().dim_size(
        1);  // X value is the number of columns of the input matrix
    input_Yvalue =
        input_tensor.shape().dim_size(0);  // Y value is the number of rows

    output_Ximage = output_image_shape.dim_size(0);
    output_Yimage = output_image_shape.dim_size(1);
    output_Cimage = output_image_shape.dim_size(2);

    if (number_colors > 256)  // Go to full color image
      output_Cimage = 3;

    int data_Xwindow = output_data_window.dim_size(0);
    int data_Ywindow = output_data_window.dim_size(1);

    int deltaX_border_image = output_Ximage - data_Xwindow;
    int deltaY_border_image = output_Yimage - data_Ywindow;

    if (convergence_dots_size >
        0)  // 3 frame sections in Y direction due to DOTS
    {
      deltaY_border_image =
          deltaY_border_image -
          convergence_dots_size;  // Take off space for Convergence Dots
      deltaY_border_image = std::max(0, deltaY_border_image);
      data_box_top = deltaY_border_image / 3;

      if (deltaY_border_image >= 0) {
        converge_dot_box_end = output_Yimage - 1 - data_box_top;
      } else {
        converge_dot_box_end = output_Yimage - 1;
      }
    } else  // Otherwise only 2, no convergence dot
    {
      data_box_top = deltaY_border_image / 2;  // Center DATA in Y dimension
      converge_dot_box_end = output_Yimage - 1;
    }

    data_box_left = deltaX_border_image / 2;  // Center DATA in X dimension
    data_box_width = data_Xwindow;            // width of scan line
    data_box_height = data_Ywindow;           // hight of image

    const T* inputZ = input_tensor.flat<T>().data();  // Flatten input Z buffer

    BuildZBuffer(inputZ);

    // Output a scalar string.
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({output_Yimage, output_Ximage, output_Cimage}),
            &output_tensor));

    outputImage = output_tensor->flat<uint8>().data();

    generate_stereogram();

    delete[] ZBuffer;
  }

  //***************************************************************************
  //***************************************************************************
  // Move input into standard Z format to reduce complexity of algorithm
  //
  void BuildZBuffer(const T* Z, bool log = false) {
    double MaxValue = 1.0;
    double MinValue = 0.0;
    ZBuffer = new double[input_Xvalue * input_Yvalue];  // Used to computer
                                                        // final Z values before
                                                        // rendering to output

    if (normalize) {
      // Init Min/Max to first value
      if (normalize_max < normalize_min)  // Autoscale if MIN>MAX
      {
        MaxValue = *Z;
        MinValue = *Z;

        for (int y = 0; y < input_Yvalue; ++y)
          for (int x = 0; x < input_Xvalue; ++x) {
            double value = getZfromInputImage(Z, x, y);
            if (value > MaxValue) MaxValue = value;
            if (value < MinValue) MinValue = value;
          }
      } else {
        MaxValue = normalize_max;
        MinValue = normalize_min;
      }
    }

    for (int y = 0; y < input_Yvalue; ++y)
      for (int x = 0; x < input_Xvalue; ++x) {
        double value = getZfromInputImage(Z, x, y);

        if (normalize) {
          value = (value - MinValue) / (MaxValue - MinValue);
        }

        if (value > 1.0) value = 1.0;
        if (value < 0.0) value = 0.0;

        *(ZBuffer + (input_Xvalue * y + x)) = value;
      }
  }

  //***************************************************************************
  //***************************************************************************
  double getZfromInputImage(const T* Z, int x, int y) {
    return *(Z + input_Xvalue * y + x);
  }

  //***************************************************************************
  //***************************************************************************
  // All normalized, not checking required
  // Possible Projection issue if DATA is bigger or smaller than Input
  //  Modes include:
  //         Truncate value (Default)
  //         Round-off value
  //         Interpolate between values
  //
  double getZfromZbuffer(double x, double y) {
    int xi, yi;

    switch (indexMode) {
      case 0:  // Truncate
        xi = int(x);
        yi = int(y);
        return (*(ZBuffer + (xi + input_Xvalue * yi)));
        break;
      case 1:  // Round-off
        xi = std::round(x);
        yi = std::round(y);
        return (*(ZBuffer + (xi + input_Xvalue * yi)));
        break;
      case 2:  // Interpolate (Not implemented yet, will need 4 points
               // [x,y],[x+1,y],[x,y+1],[x+1,y+1], then interpolate)
        xi = int(x);
        yi = int(y);
        return (*(ZBuffer + (xi + input_Xvalue * yi)));
        break;
      default:  // Round-off is the default
        xi = int(x + 0.5);
        yi = int(y + 0.5);
        return (*(ZBuffer + (xi + input_Xvalue * yi)));
        break;
    }
  }

  //***************************************************************************
  //***************************************************************************

  int getOutputImageIndex(int x, int y,
                          int channel) {  // No error checking for some
                                          // optimization, calling routine
                                          // required to make sure there is no
                                          // violation
    return ((output_Ximage * output_Cimage) * y + x * output_Cimage + channel);
  }

  //***************************************************************************
  //***************************************************************************

  double getZFromOutputPixel(int x, int y) {
    // Convert pixel units to Z units, do this as "double"
    double xofz = static_cast<double>(input_Xvalue) * (x - data_box_left) /
                  (static_cast<double>(data_box_width));
    double yofz = static_cast<double>(input_Yvalue) * (y - data_box_top) /
                  (static_cast<double>(data_box_height));

    if ((xofz < 0) || (yofz < 0) || (yofz >= input_Yvalue) ||
        (xofz >= input_Xvalue)) {  // Top of left side border hit or Right
                                   // side or bottom border hit,
                                   // send BORDER Z value
      return border_level;
    } else {
      return getZfromZbuffer(xofz, yofz);
    }
  }

  //***************************************************************************
  //***************************************************************************

  void generate_stereogram() {
    int s, left, right, visible, t, l;
    double zt, gz;
    // Scan line
    uint8* pix;  // Scan row color for each pixel
    int* same;   // Used to determine if Pixel needs to be the same as another
                 // pixel in the row

    pix = new uint8[output_Ximage * output_Cimage];
    same = new int[output_Ximage];

    for (int y = 0; y < output_Yimage; ++y) {
      // Set no dependencies on any pixels, tie each one back to itself
      for (int x = 0; x < output_Ximage; ++x) same[x] = x;

      for (int x = 0; x < output_Ximage; ++x) {
        gz = getZFromOutputPixel(x, y);
        s = separation(gz);
        left = x - s / 2;
        right = left + s;

        if ((left >= 0) && (right < output_Ximage)) {
          t = 1;
          visible = 1;
          if (hidden_surface_removal) do {
              zt = gz + 2 * (2 - mu * gz) * t / (mu * E2Epixels);
              visible = (getZFromOutputPixel(x - t, y) < zt) &&
                        (getZFromOutputPixel(x + t, y) < zt);
              ++t;
            } while ((visible) && (zt < 1));

          if (visible) {
            l = same[left];
            while ((l != left) && (l != right))
              if (l < right) {
                left = l;
                l = same[left];
              } else {
                same[left] = right;
                left = right;
                l = same[left];
                right = l;
              }
            same[left] = right;
          }
        }
      }
      // Set colors for scan row, use channels and number_colors
      for (int x = output_Ximage - 1; x >= 0; x--) {
        for (int channel = 0; channel < output_Cimage; ++channel) {
          if (same[x] == x) {  // Pick a random color
            if (number_colors == 2) {
              if ((rand() % 2) == 0) {
                pix[x * output_Cimage + channel] = Cblack;
              } else {
                pix[x * output_Cimage + channel] = Cwhite;
              }
            } else {
              pix[x * output_Cimage + channel] = rand() % 256;
            }
          } else
            pix[x * output_Cimage + channel] =
                pix[same[x] * output_Cimage + channel];

          setpixel(x, y, channel, pix[x * output_Cimage + channel]);
        }
      }
    }

    draw_convergence_dots();

    delete[] pix;
    delete[] same;
  }

  //***************************************************************************
  //***************************************************************************

  void draw_convergence_dots() {
    int x1, x2;  // center position for convergence dots

    if (convergence_dots_size == 0)  // No dot, return
      return;

    x1 = output_Ximage / 2 - get_far_width() / 2;
    x2 = output_Ximage / 2 + get_far_width() / 2;

    for (int lloop = 0; lloop < convergence_dots_size; ++lloop)
      for (int wloop = 0; wloop < convergence_dots_size; ++wloop)
        for (int channel = 0; channel < output_Cimage; ++channel) {
          setpixel(x1 - (convergence_dots_size / 2) + wloop,
                   converge_dot_box_end - lloop, channel, Cblack);
          setpixel(x2 - (convergence_dots_size / 2) + wloop,
                   converge_dot_box_end - lloop, channel, Cblack);
        }
  }

  //***************************************************************************
  //***************************************************************************

  void setpixel(int x, int y, int channel, uint8 color) {
    *(outputImage + getOutputImageIndex(x, y, channel)) = color;
  }
};

#define REGISTER_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(Name("SingleImageRandomDotStereograms") \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<T>("T"),            \
                          SingleImageRandomDotStereogramsOp<T>);

REGISTER_KERNEL(int32);
REGISTER_KERNEL(int64);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL

}  // end namespace tensorflow
