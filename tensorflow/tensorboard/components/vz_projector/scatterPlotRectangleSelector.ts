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

const FILL = '#dddddd';
const FILL_OPACITY = .2;
const STROKE = '#aaaaaa';
const STROKE_WIDTH = 2;
const STROKE_DASHARRAY = '10 5';

export interface BoundingBox {
  // The bounding box (x, y) position refers to the bottom left corner of the
  // rect.
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * A class that manages and renders a data selection rectangle.
 */
export class ScatterPlotRectangleSelector {
  private svgElement: SVGElement;
  private rectElement: SVGRectElement;

  private isMouseDown: boolean;
  private startCoordinates: [number, number];
  private lastBoundingBox: BoundingBox;

  private selectionCallback: (boundingBox: BoundingBox) => void;

  /**
   * @param container The container HTML element that the selection SVG rect
   *     will be a child of.
   * @param selectionCallback The callback that accepts a bounding box to be
   *     called when selection changes. Currently, we only call the callback on
   *     mouseUp.
   */
  constructor(
      container: HTMLElement,
      selectionCallback: (boundingBox: BoundingBox) => void) {
    this.svgElement = container.querySelector('#selector') as SVGElement;
    this.rectElement =
        document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    this.rectElement.style.stroke = STROKE;
    this.rectElement.style.strokeDasharray = STROKE_DASHARRAY;
    this.rectElement.style.strokeWidth = '' + STROKE_WIDTH;
    this.rectElement.style.fill = FILL;
    this.rectElement.style.fillOpacity = '' + FILL_OPACITY;
    this.svgElement.appendChild(this.rectElement);

    this.selectionCallback = selectionCallback;
    this.isMouseDown = false;
  }

  onMouseDown(offsetX: number, offsetY: number) {
    this.isMouseDown = true;
    this.rectElement.style.display = 'block';

    this.startCoordinates = [offsetX, offsetY];
    this.lastBoundingBox = {
      x: this.startCoordinates[0],
      y: this.startCoordinates[1],
      width: 1,
      height: 1
    };
  }

  onMouseMove(offsetX: number, offsetY: number) {
    if (!this.isMouseDown) {
      return;
    }

    this.lastBoundingBox.x = Math.min(offsetX, this.startCoordinates[0]);
    this.lastBoundingBox.y = Math.max(offsetY, this.startCoordinates[1]);
    this.lastBoundingBox.width =
        Math.max(offsetX, this.startCoordinates[0]) - this.lastBoundingBox.x;
    this.lastBoundingBox.height =
        this.lastBoundingBox.y - Math.min(offsetY, this.startCoordinates[1]);

    this.rectElement.setAttribute('x', '' + this.lastBoundingBox.x);
    this.rectElement.setAttribute(
        'y', '' + (this.lastBoundingBox.y - this.lastBoundingBox.height));
    this.rectElement.setAttribute('width', '' + this.lastBoundingBox.width);
    this.rectElement.setAttribute('height', '' + this.lastBoundingBox.height);
  }

  onMouseUp() {
    this.isMouseDown = false;
    this.rectElement.style.display = 'none';
    this.rectElement.setAttribute('width', '0');
    this.rectElement.setAttribute('height', '0');
    this.selectionCallback(this.lastBoundingBox);
  }
}
