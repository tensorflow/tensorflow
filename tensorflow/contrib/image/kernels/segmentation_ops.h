/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_IMAGE_KERNELS_SEGMENTATION_OPS_H_
#define TENSORFLOW_CONTRIB_IMAGE_KERNELS_SEGMENTATION_OPS_H_

// Connected component analysis. The op is described in ../ops/image_ops.cc. A
// description of the algorithm appears below.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

namespace functor {

template <typename T>
bool is_nonzero(T value) {
  return value != T(0);
}

template <>
bool is_nonzero(string value) {
  return value.size() != 0;
}

// Processes each pixel of an image for union-find, in parallel blocks. This is
// loosely based on the algorithm in "GPU Computing Gems" by Ondrej Stava and
// Bedrich Benes, available here:
// http://hpcg.purdue.edu/bbenes/papers/Stava2011CCL.pdf
// The bulk of the process uses blocks of each image, which have each been
// processed separately. As long as there are multiple blocks in the image, we
// double the height and width of the blocks, creating new blocks which each
// consist of 2x2 previous sub-blocks. On each new block, we process adjacent
// pixels from the previous sub-blocks serially. However, the new blocks are not
// connected, so we can process each block in parallel.
// The GPU algorithm first processes blocks of a fixed size in GPU shared
// memory, with one image block per CUDA thread block. On the CPU, we just start
// with a block size of a single pixel, and borrow the rest of the algorithm
// unchanged.
template <typename T>
class BlockedImageUnionFindFunctor {
 public:
  using OutputType = int64;

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE BlockedImageUnionFindFunctor(
      const T* images, const int64 num_rows, const int64 num_cols,
      OutputType* forest, OutputType* rank)
      : images_(images),
        num_rows_(num_rows),
        num_cols_(num_cols),
        block_height_(1),
        block_width_(1),
        forest_(forest),
        rank_(rank) {}

  // Returns the root of the tree that the pixel at the given index belongs to.
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE OutputType
  find(OutputType index) const {
    while (forest_[index] != index) {
      index = forest_[index];
    }
    return index;
  }

  // Returns the number of blocks along the y axis.
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int64 num_blocks_vertically() const {
    return (num_rows_ + block_height_ - 1) / block_height_;
  }

  // Returns the number of blocks along the x axis.
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int64 num_blocks_horizontally() const {
    return (num_cols_ + block_width_ - 1) / block_width_;
  }

  // Returns the total number of blocks in each image.
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int64 num_blocks() const {
    return num_blocks_vertically() * num_blocks_horizontally();
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int64 block_height() const {
    return block_height_;
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int64 block_width() const {
    return block_width_;
  }

  // Returns whether we may merge again (the image contains more than one
  // block).
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool can_merge() const {
    return block_height_ < num_rows_ || block_width_ < num_cols_;
  }

  // Doubles the block size. After this method, you must call
  // `merge_internal_block_edges` for each image and each *new* block's xy
  // coordinates (typically in parallel).
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void merge_blocks() {
    block_height_ *= 2;
    block_width_ *= 2;
  }

  // Processes pairs of pixels within the block which were adjacent in the four
  // sub-blocks. This must be done at each stage so that the connected
  // components in each block are joined correctly.
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void merge_internal_block_edges(
      int64 image_index, int64 block_vertical_index,
      int64 block_horizontal_index) const {
    int64 block_start_y = block_vertical_index * block_height_;
    int64 block_start_x = block_horizontal_index * block_width_;
    // Merge the 4 sub-blocks horizontally (fixing the vertical seam).
    int64 block_center_x = block_start_x + block_width_ / 2 - 1;
    if (0 <= block_center_x && block_center_x + 1 < num_cols_) {
      int64 merge_blocks_limit_y =
          std::min(num_rows_, block_start_y + block_height_);
      for (int64 y = block_start_y; y < merge_blocks_limit_y; y++) {
        union_right(image_index, y, block_center_x);
      }
    }
    // Merge the 4 sub-blocks vertically (fixing the horizontal seam).
    int64 block_center_y = block_start_y + block_height_ / 2 - 1;
    if (0 <= block_center_y && block_center_y + 1 < num_rows_) {
      int64 merge_blocks_limit_x =
          std::min(num_cols_, block_start_x + block_width_);
      for (int64 x = block_start_x; x < merge_blocks_limit_x; x++) {
        union_down(image_index, block_center_y, x);
      }
    }
  }

 private:
  // The input image(s).
  const T* const images_;
  const int64 num_rows_;
  const int64 num_cols_;
  // Current height of each sub-block of the image.
  int64 block_height_;
  // Current width of each sub-block of the image.
  int64 block_width_;
  // Union-find forest. This has the same size as `images_`, and each entry
  // holds the index of its parent in `images_` (roots hold their own index).
  // Cycles should not occur.
  OutputType* const forest_;
  // Union-find rank of each pixel.
  OutputType* const rank_;

  // Unions the pixel with the pixel below it if applicable (both pixels are
  // true, and the pixel is not in the last row).
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void union_down(OutputType batch,
                                                        OutputType row,
                                                        OutputType col) const {
    T pixel = read_pixel(batch, row, col);
    if (is_nonzero<T>(pixel)) {
      const int64 index_a = col + num_cols_ * (row + num_rows_ * batch);
      if (row + 1 < num_rows_ && read_pixel(batch, row + 1, col) == pixel) {
        const int64 index_b = col + num_cols_ * (row + 1 + num_rows_ * batch);
        do_union(index_a, index_b);
      }
    }
  }

  // Unions the pixel with the pixel to the right of it if applicable.
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void union_right(OutputType batch,
                                                         OutputType row,
                                                         OutputType col) const {
    T pixel = read_pixel(batch, row, col);
    if (is_nonzero<T>(pixel)) {
      const int64 index_a = col + num_cols_ * (row + num_rows_ * batch);
      if (col + 1 < num_cols_ && read_pixel(batch, row, col + 1) == pixel) {
        const int64 index_b = col + 1 + num_cols_ * (row + num_rows_ * batch);
        do_union(index_a, index_b);
      }
    }
  }

  // Reads a pixel value in the images.
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  read_pixel(const OutputType batch, const OutputType row,
             const OutputType col) const {
    return images_[col + num_cols_ * (row + num_rows_ * batch)];
  }

  // Unions the trees that the two pixels belong to, using their index in the
  // `images_` array.
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void do_union(
      OutputType index_a, OutputType index_b) const {
    // Find the roots of index_a and index_b in the forest, and make one the
    // child of the other.
    index_a = find(index_a);
    index_b = find(index_b);
    const OutputType rank_a = rank_[index_a];
    const OutputType rank_b = rank_[index_b];
    OutputType parent, child;
    if (index_a == index_b) {
      return;
    } else if (rank_a < rank_b) {
      parent = index_a;
      child = index_b;
    } else {
      parent = index_b;
      child = index_a;
      rank_[parent]++;
    }
    forest_[child] = parent;
  }
};

// Runs the ImageUnionFindFunctor on all pixels. Will require different CPU and
// GPU implementations.
template <typename Device, typename T>
class ImageConnectedComponentsFunctor {
 public:
  using OutputType = typename BlockedImageUnionFindFunctor<T>::OutputType;

  void operator()(OpKernelContext* ctx,
                  typename TTypes<T, 3>::ConstTensor images,
                  typename TTypes<OutputType, 3>::Tensor forest,
                  typename TTypes<OutputType, 3>::Tensor rank);
};

// Fills a flat Tensor with indices from 0 to n - 1.
template <typename Device>
class TensorRangeFunctor {
 public:
  using OutputType = typename BlockedImageUnionFindFunctor<bool>::OutputType;

  void operator()(const Device& device,
                  typename TTypes<OutputType>::Flat tensor) {
    tensor.device(device) = tensor.generate(TensorRangeGenerator());
  }

 private:
  class TensorRangeGenerator {
   public:
    EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE OutputType
    operator()(const Eigen::array<Eigen::DenseIndex, 1>& coords) const {
      return coords[0];
    }
  };
};

// Given the union-find forest, generates the root index for each node. This
// gives us arbitrary, usually non-consecutive ids for each connected component.
// The ids are massaged in Python to get deterministic, consecutive ids.
template <typename Device, typename T>
class FindRootFunctor {
 public:
  using OutputType = typename BlockedImageUnionFindFunctor<T>::OutputType;

  void operator()(const Device& device,
                  typename TTypes<OutputType>::Flat component_ids,
                  const T* images,
                  const BlockedImageUnionFindFunctor<T>& union_find) {
    component_ids.device(device) =
        component_ids.generate(FindRootGenerator(images, union_find));
  }

 private:
  class FindRootGenerator {
    const T* const images_;
    const BlockedImageUnionFindFunctor<T> union_find_;

   public:
    EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE FindRootGenerator(
        const T* images, BlockedImageUnionFindFunctor<T> union_find)
        : images_(images), union_find_(union_find) {}

    EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE OutputType
    operator()(const Eigen::array<Eigen::DenseIndex, 1>& coords) const {
      if (is_nonzero<T>(images_[coords[0]])) {
        // True pixels have an arbitrary segment id > 0. The segment ids will be
        // made contiguous later.
        return union_find_.find(coords[0]) + 1;
      } else {
        // False pixels have a segment of 0.
        return 0;
      }
    }
  };
};

}  // end namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_IMAGE_KERNELS_SEGMENTATION_OPS_H_
