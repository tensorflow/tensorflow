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

#ifndef TENSORFLOW_CORE_KERNELS_CROP_RESIZE_BILINEAR_CORE_H_
#define TENSORFLOW_CORE_KERNELS_CROP_RESIZE_BILINEAR_CORE_H_

namespace tensorflow {
namespace {
// Compute the interpolation indices only once.
struct CachedInterpolation {
  int lower;  // Lower source index used in the interpolation
  int upper;  // Upper source index used in the interpolation
  // 1-D linear iterpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  float lerp;
};

inline bool compute_single_interpolation_weight(
                const int in_size,
                const float out2in_scale,
                const float out2in_start,
                const bool clip,
		const int i,
                int& lower,
		int& upper,
		float& lerp) {
  const float in = i * out2in_scale + out2in_start;
  lower = (int)floor(in);
  upper = (int)ceil(in);
  lerp = (float)(in - (float)lower);
  if (clip) {
    if (lower < 0) lower = 0;
    else if (lower >= in_size) lower = in_size - 1;
    if (upper < 0) upper = 0;
    else if (upper >= in_size) upper = in_size - 1;
    return true;
  } else {
    return (lower >= 0 && upper < in_size) ? true : false;
  }
}
/**
 * Compute interpolation values for output indexes in range [out_start,out_start+out_size-1].
 * Returns true if all output indexes have lower and upper (input) indexes within range [0,in_size-1].
 */
inline bool compute_interpolation_weights(
                const int min_i,
                const int max_i,
		const int in_size,
		const float out2in_scale,
                const float out2in_start,
                const bool clip,
		CachedInterpolation* interpolation) {
  bool rval = true;
  int num_i = max_i - min_i + 1;
  for (int i = 0;  i < num_i;  ++i) {
    if (!compute_single_interpolation_weight(
      in_size,out2in_scale,out2in_start,clip,
      i+min_i,
      interpolation[i].lower,interpolation[i].upper,interpolation[i].lerp)) {
      rval = false;
    }
  }
  return rval;
}
/**
 * Compatibility method for resize_bilinear_op.cc
 */
inline void compute_interpolation_weights(
                const int out_size,
                const int in_size,
                const float out2in_scale,
                CachedInterpolation* interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  const bool clip = true;
  if (!compute_interpolation_weights(0,out_size-1,in_size,out2in_scale,0.0f,clip,interpolation)) {
    // Should never happen, check for it anyway
    printf("Warning! Interpolation values have lower,upper indexes outside of range [0,in_size-1]\n");
  }
}
/**
 * Compute minimum and maximum (output) i where both lower and upper (input) is in range [0,in_size-1]
 * If no values of i satisfy condition, min_i = in_size, max_i = -1 and method returns false.
 * Returns true if min_i >= max_i.
 */
inline bool compute_minmax_indexes(
		const int out_size,
                const int in_size,
                const float out2in_scale,
                const float out2in_start,
		int& min_i,
		int& max_i) {
  min_i = out_size;
  max_i = -1;
  int lower, upper;
  float lerp;
  for (int i = 0;  i < out_size;  ++i) {
    if (compute_single_interpolation_weight(in_size,out2in_scale,out2in_start,false,i,lower,upper,lerp)) {
      if (i < min_i) min_i = i;
      if (i > max_i) max_i = i;
    }
  }
  return (min_i <= max_i) ? true : false;
}
/**
 * Compute interpolation weights for crop_and_resize_op.cc
 * Also computes extrapolation areas.
 * Returns true if at least one point requires interpolation, false otherwise.
 */
inline bool compute_interpolation_weights(
		const int out_size,
		const int in_size,
		const float x1,		// lower bounding box, crop region starts at in_size*x1
		const float x2,		// upper bounding box, crop region ends at in_size*x2
		int& min_i,
		int& max_i,
		CachedInterpolation*& interpolation) {
  float out2in_start = out_size > 1 ? (float)(in_size-1) * (float)x1 : (float)(in_size-1) * (float)(x1 + x2) / 2.0f;
  float out2in_scale = out_size > 1 ? (float)(x2-x1) * (float)(in_size-1) / (float)(out_size-1) : 0.0f;
  if (compute_minmax_indexes(out_size,in_size,out2in_scale,out2in_start,min_i,max_i)) {
    interpolation = new CachedInterpolation[max_i-min_i+1];
    bool all_inputs_ok = compute_interpolation_weights(min_i,max_i,in_size,out2in_scale,out2in_start,false,interpolation);
    if (!all_inputs_ok) {
      // should never happen, purpose of compute_minmax_indexes is to ensure that all inputs are ok.
      printf("Error! compute_interpolation_weights returned input indexes outside valid range - SEGV will likely ensue.\n");
    }
    return true;
  } else {
    interpolation = 0l;
    return false;
  }
}

template <typename U> U cast_to(float v, float min_val, float max_val, U u_min_val, U u_max_val);
template <typename U> 
inline U cast_to(float v, float min_val, float max_val, U u_min_val, U u_max_val) {
  if (v < min_val)
    return u_min_val;
  else if (v > max_val)
    return u_max_val;
  else
    return static_cast<U>(v);
}
template<>
inline float cast_to<float>(float v, float min_val, float max_val, float u_min_val, float u_max_val) {
  return v;
}

inline float compute_lerp(const float top_left, const float top_right,
    const float bottom_left, const float bottom_right,
    const float x_lerp, const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 * Accepts input tensors of type T and produces output tensors of type U.
 * Optionally flips horizontal and/or vertical axis.
 */
template <typename T, typename U>
void crop_resize_single_image(
    const T* image,
    const int64 in_height, const int64 in_width,
    const int64 out_height, const int64 out_width,
    const int channels,
    const int min_ix, const int max_ix,
    const CachedInterpolation* xs,
    const int min_iy, const int max_iy,
    const CachedInterpolation* ys,
    const float extrapolated_value,
    const bool flip_x,
    const bool flip_y,
    U* output) TF_ATTRIBUTE_NOINLINE;
template <typename T, typename U>
void crop_resize_single_image(
    const T* image,
    const int64 in_height, const int64 in_width,
    const int64 out_height, const int64 out_width,
    const int channels,
    const int min_ix, const int max_ix,
    const CachedInterpolation* xs,
    const int min_iy, const int max_iy,
    const CachedInterpolation* ys,
    const float extrapolated_value,
    const bool flip_x,
    const bool flip_y,
    U* output) {
  const int64 in_row_size = in_width * channels;
  const int64 out_row_size = out_width * channels;
  U u_min_val = std::numeric_limits<U>::min();
  U u_max_val = std::numeric_limits<U>::max();
  float min_val = static_cast<float>(u_min_val);
  float max_val = static_cast<float>(u_max_val);
  U uEx = cast_to<U>(extrapolated_value,min_val,max_val,u_min_val,u_max_val);
  // low y extrapolation zone
  if (min_iy > 0) {
    U* p = flip_y ? output + out_row_size * (out_height - min_iy)  : output;
    int64 nn = out_row_size * (int64)min_iy;
    for (int64 i = 0;  i < nn;  ++i) p[i] = uEx;
  }
  // high y extrapolation zone
  if (max_iy < out_height-1) {
    U* p = flip_y ? output : output + out_row_size * (max_iy + 1);
    int64 nn = out_row_size * (int64)(out_height - 1 - max_iy);
    for (int64 i = 0;  i < nn;  ++i) p[i] = uEx;
  }
  // low x extrapolation zone
  if (min_ix > 0) {
    for (int iy = min_iy;  iy <= max_iy;  ++iy) {
      int xx0 = flip_x ? (out_width-min_ix)*channels : 0;
      int nxx = min_ix*channels;
      U* p = output + xx0 + out_row_size * (int64)(flip_y ? out_height - 1 - iy : iy);
      for (int ix = 0;  ix < nxx;  ++ix) {
        p[ix] = uEx;
      }
    }
  }
  // high x extrapolation zone
  if (max_ix < out_width-1) {
    for (int iy = min_iy;  iy <= max_iy;  ++iy) {
      int xx0 = flip_x ? 0 : (max_ix+1)*channels;
      int nxx = (out_width-1-max_ix)*channels;
      U* p = output + xx0 + out_row_size * (int64)(flip_y ? out_height - 1 - iy : iy);
      for (int ix = 0;  ix < nxx;  ++ix) {
        p[ix] = uEx;
      }
    }
  }
  U* output_y_ptr = output + out_row_size * (int64)(flip_y ? out_height - 1 - min_iy : min_iy);
  // interpolation zone
  if (channels == 1) {
    for (int y = min_iy; y <= max_iy; ++y) {
      const int iy = y - min_iy;
      const T* ys_input_lower_ptr = image + ys[iy].lower * in_row_size;
      const T* ys_input_upper_ptr = image + ys[iy].upper * in_row_size;
      const float ys_lerp = ys[iy].lerp;
      const int x0 = flip_x ? out_width - 1 - max_ix : min_ix;
      const int x1 = flip_x ? out_width - 1 - min_ix : max_ix;
      for (int x = x0; x <= x1; ++x) {
        const int ix = flip_x ? out_width - 1 - min_ix - x : x - min_ix;
	const int64 xs_lower = xs[ix].lower;
	const int64 xs_upper = xs[ix].upper;
	const float xs_lerp = xs[ix].lerp;

	// Read channel 0.
	const float top_left0(ys_input_lower_ptr[xs_lower]);
	const float top_right0(ys_input_lower_ptr[xs_upper]);
	const float bottom_left0(ys_input_upper_ptr[xs_lower]);
	const float bottom_right0(ys_input_upper_ptr[xs_upper]);

	// Compute output.
	float result0 = compute_lerp(top_left0, top_right0, bottom_left0, bottom_right0,xs_lerp, ys_lerp);
	output_y_ptr[x] = cast_to<U>(result0,min_val,max_val,u_min_val,u_max_val);
      }
      output_y_ptr = flip_y ? output_y_ptr - out_row_size : output_y_ptr + out_row_size;
    }
  } else if (channels == 2) {
    for (int y = min_iy; y <= max_iy; ++y) {
      const int iy = y - min_iy;
      const T* ys_input_lower_ptr = image + ys[iy].lower * in_row_size;
      const T* ys_input_upper_ptr = image + ys[iy].upper * in_row_size;
      const float ys_lerp = ys[iy].lerp;
      const int x0 = flip_x ? out_width - 1 - max_ix : min_ix;
      const int x1 = flip_x ? out_width - 1 - min_ix : max_ix;
      for (int x = x0; x <= x1; ++x) {
        const int ix = flip_x ? out_width - 1 - min_ix - x : x - min_ix;
	const int64 xs_lower = xs[ix].lower;
	const int64 xs_upper = xs[ix].upper;
	const float xs_lerp = xs[ix].lerp;

	// Read channel 0.
	const float top_left0(ys_input_lower_ptr[xs_lower + 0]);
	const float top_right0(ys_input_lower_ptr[xs_upper + 0]);
	const float bottom_left0(ys_input_upper_ptr[xs_lower + 0]);
	const float bottom_right0(ys_input_upper_ptr[xs_upper + 0]);

	// Read channel 1.
	const float top_left1(ys_input_lower_ptr[xs_lower + 1]);
	const float top_right1(ys_input_lower_ptr[xs_upper + 1]);
	const float bottom_left1(ys_input_upper_ptr[xs_lower + 1]);
	const float bottom_right1(ys_input_upper_ptr[xs_upper + 1]);

	// Compute output.
	float result0 = compute_lerp(top_left0, top_right0, bottom_left0, bottom_right0,
	    xs_lerp, ys_lerp);
	float result1 = compute_lerp(top_left1, top_right1, bottom_left1, bottom_right1,
	    xs_lerp, ys_lerp);
	output_y_ptr[x*2 + 0] = cast_to<U>(result0,min_val,max_val,u_min_val,u_max_val);
	output_y_ptr[x*2 + 1] = cast_to<U>(result1,min_val,max_val,u_min_val,u_max_val);
      }
      output_y_ptr = flip_y ? output_y_ptr - out_row_size : output_y_ptr + out_row_size;
    }
  } else if (channels == 3) {
    for (int y = min_iy; y <= max_iy; ++y) {
      const int iy = y - min_iy;
      const T* ys_input_lower_ptr = image + ys[iy].lower * in_row_size;
      const T* ys_input_upper_ptr = image + ys[iy].upper * in_row_size;
      const float ys_lerp = ys[iy].lerp;
      const int x0 = flip_x ? out_width - 1 - max_ix : min_ix;
      const int x1 = flip_x ? out_width - 1 - min_ix : max_ix;
      for (int x = x0; x <= x1; ++x) {
        const int ix = flip_x ? out_width - 1 - min_ix - x : x - min_ix;
	const int64 xs_lower = xs[ix].lower;
	const int64 xs_upper = xs[ix].upper;
	const float xs_lerp = xs[ix].lerp;

	// Read channel 0.
	const float top_left0(ys_input_lower_ptr[xs_lower + 0]);
	const float top_right0(ys_input_lower_ptr[xs_upper + 0]);
	const float bottom_left0(ys_input_upper_ptr[xs_lower + 0]);
	const float bottom_right0(ys_input_upper_ptr[xs_upper + 0]);

	// Read channel 1.
	const float top_left1(ys_input_lower_ptr[xs_lower + 1]);
	const float top_right1(ys_input_lower_ptr[xs_upper + 1]);
	const float bottom_left1(ys_input_upper_ptr[xs_lower + 1]);
	const float bottom_right1(ys_input_upper_ptr[xs_upper + 1]);

	// Read channel 2.
	const float top_left2(ys_input_lower_ptr[xs_lower + 2]);
	const float top_right2(ys_input_lower_ptr[xs_upper + 2]);
	const float bottom_left2(ys_input_upper_ptr[xs_lower + 2]);
	const float bottom_right2(ys_input_upper_ptr[xs_upper + 2]);

	// Compute output.
	float result0 = compute_lerp(top_left0, top_right0, bottom_left0, bottom_right0,
	    xs_lerp, ys_lerp);
	float result1 = compute_lerp(top_left1, top_right1, bottom_left1, bottom_right1,
	    xs_lerp, ys_lerp);
	float result2 = compute_lerp(top_left2, top_right2, bottom_left2, bottom_right2,
	    xs_lerp, ys_lerp);
	output_y_ptr[x*3 + 0] = cast_to<U>(result0,min_val,max_val,u_min_val,u_max_val);
	output_y_ptr[x*3 + 1] = cast_to<U>(result1,min_val,max_val,u_min_val,u_max_val);
	output_y_ptr[x*3 + 2] = cast_to<U>(result2,min_val,max_val,u_min_val,u_max_val);
      }
      output_y_ptr = flip_y ? output_y_ptr - out_row_size : output_y_ptr + out_row_size;
    }
  } else if (channels == 4) {
    for (int y = min_iy; y <= max_iy; ++y) {
      const int iy = y - min_iy;
      const T* ys_input_lower_ptr = image + ys[iy].lower * in_row_size;
      const T* ys_input_upper_ptr = image + ys[iy].upper * in_row_size;
      const float ys_lerp = ys[iy].lerp;
      const int x0 = flip_x ? out_width - 1 - max_ix : min_ix;
      const int x1 = flip_x ? out_width - 1 - min_ix : max_ix;
      for (int x = x0; x <= x1; ++x) {
        const int ix = flip_x ? out_width - 1 - min_ix - x : x - min_ix;
	const int64 xs_lower = xs[ix].lower;
	const int64 xs_upper = xs[ix].upper;
	const float xs_lerp = xs[ix].lerp;

	// Read channel 0.
	const float top_left0(ys_input_lower_ptr[xs_lower + 0]);
	const float top_right0(ys_input_lower_ptr[xs_upper + 0]);
	const float bottom_left0(ys_input_upper_ptr[xs_lower + 0]);
	const float bottom_right0(ys_input_upper_ptr[xs_upper + 0]);

	// Read channel 1.
	const float top_left1(ys_input_lower_ptr[xs_lower + 1]);
	const float top_right1(ys_input_lower_ptr[xs_upper + 1]);
	const float bottom_left1(ys_input_upper_ptr[xs_lower + 1]);
	const float bottom_right1(ys_input_upper_ptr[xs_upper + 1]);

	// Read channel 2.
	const float top_left2(ys_input_lower_ptr[xs_lower + 2]);
	const float top_right2(ys_input_lower_ptr[xs_upper + 2]);
	const float bottom_left2(ys_input_upper_ptr[xs_lower + 2]);
	const float bottom_right2(ys_input_upper_ptr[xs_upper + 2]);

	// Read channel 3.
	const float top_left3(ys_input_lower_ptr[xs_lower + 3]);
	const float top_right3(ys_input_lower_ptr[xs_upper + 3]);
	const float bottom_left3(ys_input_upper_ptr[xs_lower + 3]);
	const float bottom_right3(ys_input_upper_ptr[xs_upper + 3]);

	// Compute output.
	float result0 = compute_lerp(top_left0, top_right0, bottom_left0, bottom_right0,
	    xs_lerp, ys_lerp);
	float result1 = compute_lerp(top_left1, top_right1, bottom_left1, bottom_right1,
	    xs_lerp, ys_lerp);
	float result2 = compute_lerp(top_left2, top_right2, bottom_left2, bottom_right2,
	    xs_lerp, ys_lerp);
	float result3 = compute_lerp(top_left3, top_right3, bottom_left3, bottom_right3,
	    xs_lerp, ys_lerp);
	output_y_ptr[x*4 + 0] = cast_to<U>(result0,min_val,max_val,u_min_val,u_max_val);
	output_y_ptr[x*4 + 1] = cast_to<U>(result1,min_val,max_val,u_min_val,u_max_val);
	output_y_ptr[x*4 + 2] = cast_to<U>(result2,min_val,max_val,u_min_val,u_max_val);
	output_y_ptr[x*4 + 3] = cast_to<U>(result3,min_val,max_val,u_min_val,u_max_val);
      }
      output_y_ptr = flip_y ? output_y_ptr - out_row_size : output_y_ptr + out_row_size;
    }
  } else {
    for (int y = min_iy; y <= max_iy; ++y) {
      const int iy = y - min_iy;
      const T* ys_input_lower_ptr = image + ys[iy].lower * in_row_size;
      const T* ys_input_upper_ptr = image + ys[iy].upper * in_row_size;
      const float ys_lerp = ys[iy].lerp;
      const int x0 = flip_x ? out_width - 1 - max_ix : min_ix;
      const int x1 = flip_x ? out_width - 1 - min_ix : max_ix;
      for (int x = x0; x <= x1; ++x) {
        const int ix = flip_x ? out_width - 1 - min_ix - x : x - min_ix;
	const int64 xs_lower = xs[ix].lower;
	const int64 xs_upper = xs[ix].upper;
	const float xs_lerp = xs[ix].lerp;
        for (int ichan = 0;  ichan < channels;  ++ichan) {
	  const float top_left0(ys_input_lower_ptr[xs_lower + ichan]);
	  const float top_right0(ys_input_lower_ptr[xs_upper + ichan]);
	  const float bottom_left0(ys_input_upper_ptr[xs_lower + ichan]);
	  const float bottom_right0(ys_input_upper_ptr[xs_upper + ichan]);
	  float result0 = compute_lerp(top_left0, top_right0, bottom_left0, bottom_right0,
	      xs_lerp, ys_lerp);
	  output_y_ptr[x*channels + ichan] = cast_to<U>(result0,min_val,max_val,u_min_val,u_max_val);
	}
      }
      output_y_ptr = flip_y ? output_y_ptr - out_row_size : output_y_ptr + out_row_size;
    }
  }
}
}  // namespace
}  // namespace tensorflow
#endif
