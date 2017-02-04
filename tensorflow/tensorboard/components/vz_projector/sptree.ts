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

/** N-dimensional point. Usually 2D or 3D. */
export type Point = number[];

export interface BBox {
  center: Point;
  halfDim: number;
}

/** A node in a space-partitioning tree. */
export interface SPNode {
  /** The children of this node. */
  children?: SPNode[];
  /** The bounding box of the region this node occupies. */
  box: BBox;
  /** One or more points this node has. */
  point: Point;
}

/**
 * A Space-partitioning tree (https://en.wikipedia.org/wiki/Space_partitioning)
 * that recursively divides the space into regions of equal sizes. This data
 * structure can act both as a Quad tree and an Octree when the data is 2 or
 * 3 dimensional respectively. One usage is in t-SNE in order to do Barnes-Hut
 * approximation.
 */
export class SPTree {
  root: SPNode;

  private masks: number[];
  private dim: number;

  /**
   * Constructs a new tree with the provided data.
   *
   * @param data List of n-dimensional data points.
   * @param capacity Number of data points to store in a single node.
   */
  constructor(data: Point[]) {
    if (data.length < 1) {
      throw new Error('There should be at least 1 data point');
    }
    // Make a bounding box based on the extent of the data.
    this.dim = data[0].length;
    // Each node has 2^d children, where d is the dimension of the space.
    // Binary masks (e.g. 000, 001, ... 111 in 3D) are used to determine in
    // which child (e.g. quadron in 2D) the new point is going to be assigned.
    // For more details, see the insert() method and its comments.
    this.masks = new Array(Math.pow(2, this.dim));
    for (let d = 0; d < this.masks.length; ++d) {
      this.masks[d] = (1 << d);
    }
    let min: Point = new Array(this.dim);
    fillArray(min, Number.POSITIVE_INFINITY);
    let max: Point = new Array(this.dim);
    fillArray(max, Number.NEGATIVE_INFINITY);

    for (let i = 0; i < data.length; ++i) {
      // For each dim get the min and max.
      // E.g. For 2-D, get the x_min, x_max, y_min, y_max.
      for (let d = 0; d < this.dim; ++d) {
        min[d] = Math.min(min[d], data[i][d]);
        max[d] = Math.max(max[d], data[i][d]);
      }
    }
    // Create a bounding box with the center of the largest span.
    let center: Point = new Array(this.dim);
    let halfDim = 0;
    for (let d = 0; d < this.dim; ++d) {
      let span = max[d] - min[d];
      center[d] = min[d] + span / 2;
      halfDim = Math.max(halfDim, span / 2);
    }
    this.root = {box: {center: center, halfDim: halfDim}, point: data[0]};
    for (let i = 1; i < data.length; ++i) {
      this.insert(this.root, data[i]);
    }
  }

  /**
   * Visits every node in the tree. Each node can store 1 or more points,
   * depending on the node capacity provided in the constructor.
   *
   * @param accessor Method that takes the currently visited node, and the
   * low and high point of the region that this node occupies. E.g. in 2D,
   * the low and high points will be the lower-left corner and the upper-right
   * corner.
   */
  visit(
      accessor: (node: SPNode, lowPoint: Point, highPoint: Point) => boolean,
      noBox = false) {
    this.visitNode(this.root, accessor, noBox);
  }

  private visitNode(
      node: SPNode,
      accessor: (node: SPNode, lowPoint?: Point, highPoint?: Point) => boolean,
      noBox: boolean) {
    let skipChildren: boolean;
    if (noBox) {
      skipChildren = accessor(node);
    } else {
      let lowPoint = new Array(this.dim);
      let highPoint = new Array(this.dim);
      for (let d = 0; d < this.dim; ++d) {
        lowPoint[d] = node.box.center[d] - node.box.halfDim;
        highPoint[d] = node.box.center[d] + node.box.halfDim;
      }
      skipChildren = accessor(node, lowPoint, highPoint);
    }
    if (!node.children || skipChildren) {
      return;
    }
    for (let i = 0; i < node.children.length; ++i) {
      let child = node.children[i];
      if (child) {
        this.visitNode(child, accessor, noBox);
      }
    }
  }

  private insert(node: SPNode, p: Point) {
    // Subdivide and then add the point to whichever node will accept it.
    if (node.children == null) {
      node.children = new Array(this.masks.length);
    }

    // Decide which child will get the new point by constructing a D-bits binary
    // signature (D=3 for 3D) where the k-th bit is 1 if the point's k-th
    // coordinate is greater than the node's k-th coordinate, 0 otherwise.
    // Then the binary signature in decimal system gives us the index of the
    // child where the new point should be.
    let index = 0;
    for (let d = 0; d < this.dim; ++d) {
      if (p[d] > node.box.center[d]) {
        index |= this.masks[d];
      }
    }
    if (node.children[index] == null) {
      this.makeChild(node, index, p);
    } else {
      this.insert(node.children[index], p);
    }
  }

  private makeChild(node: SPNode, index: number, p: Point): void {
    let oldC = node.box.center;
    let h = node.box.halfDim / 2;
    let newC: Point = new Array(this.dim);
    for (let d = 0; d < this.dim; ++d) {
      newC[d] = (index & (1 << d)) ? oldC[d] + h : oldC[d] - h;
    }
    node.children[index] = {box: {center: newC, halfDim: h}, point: p};
  }
}

function fillArray<T>(arr: T[], value: T): void {
  for (let i = 0; i < arr.length; ++i) {
    arr[i] = value;
  }
}
