/* Copyright 2015 Google Inc. All Rights Reserved.

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

module TF {
  interface NodeData<T> {
    value: T;
    node: Element | HTMLElement;
  }

  export interface ScanResponse<T> {
    visible: T[];
    almost: T[];
    hidden: T[];
  }

  /**
   * Scans HTMLElements to see if they are positioned in view in the browser
   * window, in reference to some other "framing" node. The framing Node does not
   * need to be a parent of the nodes that are monitored. All visibility
   * calculations are simply made with reference to the frameNode's client
   * bounding box.
   *
   * Note: the radar currently only has the circuitry to detect vertical
   * visibility and movements.
   *
   *
   * When NodeRadar.scan is called, all nodes are placed into one of four states.
   * visible = The node is visible on the screen (fully or partially).
   * almost  = within container's height away from becoming visible.
   * hidden  = The node is not visible or close to being visible.
   *
   * Scanning updates need to be maintained manually, for instance by calling
   * radar.scan() in a function that is bound to scroll and resize events.
   */
  export class NodeRadar<T> {

    private _frameNode: Element | HTMLElement;
    private _nodes: NodeData<T>[] = [];

    /**
     * Construct a NodeRadar.
     * @param frameNode The containing Element that all scans are relative to.
     * The frameNode doesn't need to be a parent of monitored nodes.
     */
    constructor(frameNode: Element | HTMLElement) {
      this._frameNode = frameNode;
    }

    /**
     * Add a node to be monitored, and an identifying value.
     * When "scan" is called, the value will be returned with metadata
     * about the state of the associated node.
     * @param node The Element/HTMLElement to track visibility for.
     * @param value The identity value associated with that node. Must be unique
     * wrt strict equality.
     * An exception will be thrown if the value is not unique.
     */
    public add(node: Element | HTMLElement, value: T): this {
      if (this._nodes.filter(n => n.value === value).length) {
        throw new Error("Values in NodeRadar must be unique");
      }
      this._nodes.push({
        value: value,
        node: node,
      });
      return this;
    }

    /** Remove a value and its associated node from the NodeRadar.
     * No exception is thrown if the value is not found.
     */
    public remove(value: T): this {
      this._nodes = this._nodes.filter(function(n) { return n.value !== value; });
      return this;
    }

    /**
     * Scan the DOM and determine the visibility of each node relative to the
     * nodeRadar's frameNode. Return a ScansResponse<T> that gives metadata
     * on the visibility of each Node, by returning the associated value.
     */
    public scan(): ScanResponse<T> {
      var containerBox = this._frameNode.getBoundingClientRect();
      var visibleC: T[] = [];
      var almostC: T[] = [];
      var hiddenC: T[] = [];

      this._nodes.forEach(function(n) {
        var box = n.node.getBoundingClientRect();
        var visible = (box.bottom > containerBox.top
          && box.top < containerBox.bottom);
        var almost = (box.bottom > (containerBox.top - containerBox.height)
          && box.top < (containerBox.bottom + containerBox.height));
        var target = visible ? visibleC : almost ? almostC : hiddenC;
        target.push(n.value);
      });
      return {visible: visibleC, almost: almostC, hidden: hiddenC};
    }

  }
}
