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
 * Note: Callbacks only get fired when one of the three visibility states
 * change. They are all fired once on startup.
 *
 * It calculates three states for each node.
 * full    = All of the node is visible. Other states are also set to true in
 *           this state.
 * partial = At least some part of the node is visible. "almost" is also set to
 *           true in this case.
 * almost  = within one container's height away from becoming partially visible.
 *         = Other states are necessarily false in this case.
 *
 * Scanning updates need to be maintained manually, for instance by calling
 * radar.scan() in a function that is bound to scroll and resize events.
 */

interface NodeVisibility {
  node: HTMLElement;
  partial: Boolean;
  full: Boolean;
  almost: Boolean;
}

interface NodeData {
  callback: NodeRadarCallback;
  visibility: NodeVisibility;
}

interface NodeRadarCallback {
  (nodeVisibility: NodeVisibility);
}

class NodeRadar {

  private _frameNode: HTMLElement;
  private _nodes: Array<NodeData> = [];

  // The frameNode does not need to be a parent of the nodes that are monitored.
  constructor(frameNode: HTMLElement) {
    this._frameNode = frameNode;
  }

  // Public API for adding nodes to be monitored. The callback is called on a
  // per node basis when any of it's visibility states
  public add(node: Element | HTMLCollection | NodeList, callback: Function) {
    var result = Object.prototype.toString.call(node);
    if ( result === "[object NodeList]" || result === "[object HTMLCollection]") {
      Array.prototype.forEach.call(node, (n) => {
        this._push(n, callback);
      });
    } else {
      this._push(node, callback);
    }
    this.scan();
  }

  // Adds a node to our internal nodes array
  private _push(node, callback) {
    this._nodes.push({
      callback: callback,
      visibility: {
        node: node,
        full: false,
        partial: false,
        almost: false
      }
    });
  }

  public remove(node: Element) {
    this._nodes = this._nodes.filter(function(n) { return n.visibility.node !== node; });
  }

  public checkVisibility(node: Element) {
    var matches = this._nodes.filter(function(n) { return n.visibility.node === node; });
    if (matches.length > 0) {
      return matches[0].visibility;
    } else {
      throw new Error("Couldn't find node to check visibility.");
    }
  }

  public getNodes(): Array<NodeData> {
    return this._nodes;
  }

  // Scans the DOM and determines the visible state of each node. Fires the
  // callback on each node that has a change in visible state.
  public scan() {
    var containerBox = this._frameNode.getBoundingClientRect();
    this._nodes.forEach(function(n) {
      var box = n.visibility.node.getBoundingClientRect();
      var partial = (box.bottom > containerBox.top
        && box.top < containerBox.bottom);
      var full = (box.top > containerBox.top
        && box.bottom < containerBox.bottom);
      var almost = (box.top > (containerBox.top - containerBox.height)
        && box.bottom < (containerBox.bottom + containerBox.height));
      if (n.visibility.partial !== partial || n.visibility.full !== full || n.visibility.almost !== almost) {
        n.visibility.partial = partial;
        n.visibility.full = full;
        n.visibility.almost = almost;
        n.callback(n.visibility);
      }
    });
  }

}
