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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_SPRITE_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_SPRITE_H_

#ifdef __RENDER_OPENGL__

#include <GLES/gl.h>
#include <GLES/glext.h>

#include "tensorflow/examples/android/jni/object_tracking/image-inl.h"
#include "tensorflow/examples/android/jni/object_tracking/image.h"

namespace tf_tracking {

// This class encapsulates the logic necessary to load an render image data
// at the same aspect ratio as the original source.
class Sprite {
 public:
  // Only create Sprites when you have an OpenGl context.
  explicit Sprite(const Image<uint8_t>& image) { LoadTexture(image, NULL); }

  Sprite(const Image<uint8_t>& image, const BoundingBox* const area) {
    LoadTexture(image, area);
  }

  // Also, try to only delete a Sprite when holding an OpenGl context.
  ~Sprite() {
    glDeleteTextures(1, &texture_);
  }

  inline int GetWidth() const {
    return actual_width_;
  }

  inline int GetHeight() const {
    return actual_height_;
  }

  // Draw the sprite at 0,0 - original width/height in the current reference
  // frame. Any transformations desired must be applied before calling this
  // function.
  void Draw() const {
    const float float_width = static_cast<float>(actual_width_);
    const float float_height = static_cast<float>(actual_height_);

    // Where it gets rendered to.
    const float vertices[] = { 0.0f, 0.0f, 0.0f,
                               0.0f, float_height, 0.0f,
                               float_width, 0.0f, 0.0f,
                               float_width, float_height, 0.0f,
                               };

    // The coordinates the texture gets drawn from.
    const float max_x = float_width / texture_width_;
    const float max_y = float_height / texture_height_;
    const float textureVertices[] = {
        0, 0,
        0, max_y,
        max_x, 0,
        max_x, max_y,
    };

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture_);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glVertexPointer(3, GL_FLOAT, 0, vertices);
    glTexCoordPointer(2, GL_FLOAT, 0, textureVertices);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  }

 private:
  inline int GetNextPowerOfTwo(const int number) const {
    int power_of_two = 1;
    while (power_of_two < number) {
      power_of_two *= 2;
    }
    return power_of_two;
  }

  // TODO(andrewharp): Allow sprites to have their textures reloaded.
  void LoadTexture(const Image<uint8_t>& texture_source,
                   const BoundingBox* const area) {
    glEnable(GL_TEXTURE_2D);

    glGenTextures(1, &texture_);

    glBindTexture(GL_TEXTURE_2D, texture_);

    int left = 0;
    int top = 0;

    if (area != NULL) {
      // If a sub-region was provided to pull the texture from, use that.
      left = area->left_;
      top = area->top_;
      actual_width_ = area->GetWidth();
      actual_height_ = area->GetHeight();
    } else {
      actual_width_ = texture_source.GetWidth();
      actual_height_ = texture_source.GetHeight();
    }

    // The textures must be a power of two, so find the sizes that are large
    // enough to contain the image data.
    texture_width_ = GetNextPowerOfTwo(actual_width_);
    texture_height_ = GetNextPowerOfTwo(actual_height_);

    bool allocated_data = false;
    uint8_t* texture_data;

    // Except in the lucky case where we're not using a sub-region of the
    // original image AND the source data has dimensions that are power of two,
    // care must be taken to copy data at the appropriate source and destination
    // strides so that the final block can be copied directly into texture
    // memory.
    // TODO(andrewharp): Figure out if data can be pulled directly from the
    // source image with some alignment modifications.
    if (left != 0 || top != 0 ||
        actual_width_ != texture_source.GetWidth() ||
        actual_height_ != texture_source.GetHeight()) {
      texture_data = new uint8_t[actual_width_ * actual_height_];

      for (int y = 0; y < actual_height_; ++y) {
        memcpy(texture_data + actual_width_ * y, texture_source[top + y] + left,
               actual_width_ * sizeof(uint8_t));
      }
      allocated_data = true;
    } else {
      // Cast away const-ness because for some reason glTexSubImage2D wants
      // a non-const data pointer.
      texture_data = const_cast<uint8_t*>(texture_source.data());
    }

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_LUMINANCE,
                 texture_width_,
                 texture_height_,
                 0,
                 GL_LUMINANCE,
                 GL_UNSIGNED_BYTE,
                 NULL);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    actual_width_,
                    actual_height_,
                    GL_LUMINANCE,
                    GL_UNSIGNED_BYTE,
                    texture_data);

    if (allocated_data) {
      delete(texture_data);
    }

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  }

  // The id for the texture on the GPU.
  GLuint texture_;

  // The width and height to be used for display purposes, referring to the
  // dimensions of the original texture.
  int actual_width_;
  int actual_height_;

  // The allocated dimensions of the texture data, which must be powers of 2.
  int texture_width_;
  int texture_height_;

  TF_DISALLOW_COPY_AND_ASSIGN(Sprite);
};

}  // namespace tf_tracking

#endif  // __RENDER_OPENGL__

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_SPRITE_H_
