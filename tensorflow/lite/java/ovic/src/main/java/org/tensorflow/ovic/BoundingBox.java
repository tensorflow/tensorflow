/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package org.tensorflow.ovic;

/** Class for holding a detection bounding box with category and confidence. */
public class BoundingBox {
  // Upper left point.
  public float x1;
  public float y1;

  // Lower right point.
  public float x2;
  public float y2;

  // The area of the box
  public float area;

  // The object category
  public int category;

  // The confidence of the detection
  public float score;

  public BoundingBox(float x1, float y1, float x2, float y2, int category, float score) {
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
    this.category = category;
    this.score = score;
    // -1 stands for area not initialized
    this.area = -1;
  }

  // The intersection area of two bounding boxes
  public float intersect(BoundingBox bbx) {
    return Math.max(0, Math.min(x2, bbx.x2) - Math.max(x1, bbx.x1))
        * Math.max(0, Math.min(y2, bbx.y2) - Math.max(y1, bbx.y1));
  }

  // The union area of two bounding boxes
  public float union(BoundingBox bbx) {
    return bbx.getArea() + this.getArea() - this.intersect(bbx);
  }

  public float getArea() {
    if (area < 0) {
      area = (x2 - x1) * (y2 - y1);
    }
    return area;
  }

  public float computeIoU(BoundingBox bbx) {
    return (float) (this.intersect(bbx) * 1.0 / this.union(bbx));
  }
}
