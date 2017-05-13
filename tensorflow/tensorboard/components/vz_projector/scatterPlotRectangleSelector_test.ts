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

import {BoundingBox, ScatterPlotRectangleSelector} from './scatterPlotRectangleSelector';

describe('selector callbacks make bounding box start bottom left', () => {
  let containerElement: HTMLElement;
  let selectionCallback: (boundingBox: BoundingBox) => void;
  let selection: ScatterPlotRectangleSelector;

  beforeEach(() => {
    containerElement = document.createElement('div');
    const selector = document.createElement('svg');
    selector.id = 'selector';
    containerElement.appendChild(selector);

    selectionCallback = jasmine.createSpy('selectionCallback');
    selection =
        new ScatterPlotRectangleSelector(containerElement, selectionCallback);
  });

  it('Simple mouse event starting top left', () => {
    selection.onMouseDown(0, 0);
    selection.onMouseMove(10, 10);
    selection.onMouseUp();

    expect(selectionCallback)
        .toHaveBeenCalledWith({x: 0, y: 10, width: 10, height: 10});
  });

  it('Simple mouse event starting bottom left', () => {
    selection.onMouseDown(0, 10);
    selection.onMouseMove(10, 0);
    selection.onMouseUp();

    expect(selectionCallback)
        .toHaveBeenCalledWith({x: 0, y: 10, width: 10, height: 10});
  });

  it('Simple mouse event starting top right', () => {
    selection.onMouseDown(10, 0);
    selection.onMouseMove(0, 10);
    selection.onMouseUp();

    expect(selectionCallback)
        .toHaveBeenCalledWith({x: 0, y: 10, width: 10, height: 10});
  });

  it('Simple mouse event starting bottom right', () => {
    selection.onMouseDown(10, 10);
    selection.onMouseMove(0, 0);
    selection.onMouseUp();

    expect(selectionCallback)
        .toHaveBeenCalledWith({x: 0, y: 10, width: 10, height: 10});
  });
});
