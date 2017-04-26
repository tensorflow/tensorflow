/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


import * as d3 from 'd3';  // from //third_party/javascript/typings/d3_v4
import * as Plottable from 'Plottable/plottable';  // from //third_party/javascript/plottable/v3


export class DragZoomLayer extends Plottable.Components.SelectionBoxLayer {
  private _dragInteraction: Plottable.Interactions.Drag;
  private _doubleClickInteraction: Plottable.Interactions.Click;
  private isZoomed = false;
  private easeFn: (t: number) => number = d3.easeCubicInOut;
  private _animationTime = 750;
  private onStart: Function;
  private onEnd: Function;
  private unzoomMethod: Function;

  /**
   * Constructs a SelectionBoxLayer with an attached DragInteraction and
   * ClickInteraction. On drag, it triggers an animated zoom into the box
   * that was dragged. On double click, it zooms back out to the original
   * view, before any zooming.
   * The zoom animation uses an easing function (default
   * d3.ease('cubic-in-out')) and is customizable.
   * Usage: Construct the selection box layer and attach x and y scales,
   * and then add the layer over the plot you are zooming on using a
   * Component Group.
   * TODO(danmane) - merge this into Plottable
   */
  constructor(
      xScale: Plottable.QuantitativeScale<number|{valueOf(): number}>,
      yScale: Plottable.QuantitativeScale<number|{valueOf(): number}>,
      unzoomMethod: Function) {
    super();
    this.xScale(xScale);
    this.yScale(yScale);
    this._dragInteraction = new Plottable.Interactions.Drag();
    this._dragInteraction.attachTo(this);
    this._doubleClickInteraction = new Plottable.Interactions.Click();
    this._doubleClickInteraction.attachTo(this);
    this.setupCallbacks();
    this.unzoomMethod = unzoomMethod;
  }

  /**
   * Register a method that calls when the DragZoom interaction starts.
   */
  public interactionStart(cb: Function) {
    this.onStart = cb;
  }

  /**
   * Register a method that calls when the DragZoom interaction ends.
   */
  public interactionEnd(cb: Function) {
    this.onEnd = cb;
  }

  private setupCallbacks() {
    let dragging = false;
    this._dragInteraction.onDragStart((startPoint: Plottable.Point) => {
      this.bounds({
        topLeft: startPoint,
        bottomRight: startPoint,
      });
      this.onStart();
    });
    this._dragInteraction.onDrag((startPoint, endPoint) => {
      this.bounds({topLeft: startPoint, bottomRight: endPoint});
      this.boxVisible(true);
      dragging = true;
    });
    this._dragInteraction.onDragEnd((startPoint, endPoint) => {
      this.boxVisible(false);
      this.bounds({topLeft: startPoint, bottomRight: endPoint});
      if (dragging) {
        this.zoom();
      } else {
        this.onEnd();
      }
      dragging = false;
    });

    this._doubleClickInteraction.onDoubleClick(this.unzoom.bind(this));
  }

  /* Set the time (in ms) over which the zoom will interpolate.
   * 0 implies no interpolation. (ie zoom is instant)
   */
  public animationTime(): number;
  public animationTime(animationTime: number): DragZoomLayer;
  public animationTime(animationTime?: number): any {
    if (animationTime == null) {
      return this._animationTime;
    }
    if (animationTime < 0) {
      throw new Error('animationTime cannot be negative');
    }
    this._animationTime = animationTime;
    return this;
  }

  /**
   * Set the easing function, which determines how the zoom interpolates
   * over time.
   */
  public ease(fn: (t: number) => number): DragZoomLayer {
    if (typeof(fn) !== 'function') {
      throw new Error('ease function must be a function');
    }
    if (fn(0) !== 0 || fn(1) !== 1) {
      Plottable.Utils.Window.warn(
          'Easing function does not maintain invariant ' +
          'f(0)==0 && f(1)==1. Bad behavior may result.');
    }
    this.easeFn = fn;
    return this;
  }

  // Zoom into extent of the selection box bounds
  private zoom() {
    let x0: number = this.xExtent()[0].valueOf();
    let x1: number = this.xExtent()[1].valueOf();
    let y0: number = this.yExtent()[1].valueOf();
    let y1: number = this.yExtent()[0].valueOf();

    if (x0 === x1 || y0 === y1) {
      return;
    }

    if (!this.isZoomed) {
      this.isZoomed = true;
    }
    this.interpolateZoom(x0, x1, y0, y1);
  }

  // Restore the scales to their state before any zoom
  private unzoom() {
    if (!this.isZoomed) {
      return;
    }
    this.isZoomed = false;
    let xScale = this.xScale() as any;
    xScale._domainMin = null;
    xScale._domainMax = null;
    let xDomain = xScale._getExtent();
    this.xScale().domain(xDomain);
    this.unzoomMethod();
  }

  // If we are zooming, disable interactions, to avoid contention
  private isZooming(isZooming: boolean) {
    this._dragInteraction.enabled(!isZooming);
    this._doubleClickInteraction.enabled(!isZooming);
  }

  private interpolateZoom(x0f: number, x1f: number, y0f: number, y1f: number) {
    let x0s: number = this.xScale().domain()[0].valueOf();
    let x1s: number = this.xScale().domain()[1].valueOf();
    let y0s: number = this.yScale().domain()[0].valueOf();
    let y1s: number = this.yScale().domain()[1].valueOf();

    // Copy a ref to the ease fn, so that changing ease wont affect zooms in
    // progress.
    let ease = this.easeFn;
    let interpolator = (a: number, b: number, p: number) =>
        d3.interpolateNumber(a, b)(ease(p));

    this.isZooming(true);
    let start = Date.now();
    let draw = () => {
      let now = Date.now();
      let passed = now - start;
      let p = this._animationTime === 0 ?
          1 :
          Math.min(1, passed / this._animationTime);
      let x0 = interpolator(x0s, x0f, p);
      let x1 = interpolator(x1s, x1f, p);
      let y0 = interpolator(y0s, y0f, p);
      let y1 = interpolator(y1s, y1f, p);
      this.xScale().domain([x0, x1]);
      this.yScale().domain([y0, y1]);
      if (p < 1) {
        Plottable.Utils.DOM.requestAnimationFramePolyfill(draw);
      } else {
        this.onEnd();
        this.isZooming(false);
      }
    };
    draw();
  }
}
