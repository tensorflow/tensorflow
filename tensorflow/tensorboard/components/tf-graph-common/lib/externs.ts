/* Copyright 2015 Google Inc. All Rights Reserved.

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

/**
 * @fileoverview Extern declarations for tensorflow graph visualizer.
 *     This file contains compiler stubs for external dependencies whos
 *     implementations are defined at runtime.
 */

declare module graphlib {
  interface GraphOptions {
    name?: string;
    /**
     * Direction for rank nodes. Can be TB, BT, LR, or RL, where T = top,
     * B = bottom, L = left, and R = right.
     */
    rankdir?: string;
    type?: string|number;
    /** Number of pixels between each rank in the layout. */
    ranksep?: number;
    /** Number of pixels that separate nodes horizontally in the layout. */
    nodesep?: number;
    /** Number of pixels that separate edges horizontally in the layout */
    edgesep?: number;
  }

  export interface EdgeObject {
    v: string;
    w: string;
    name?: string;
  }

  export class Graph<N, E> {
    constructor(opt?: Object);
    setNode(name: string, value?: N): void;
    hasNode(name: string): boolean;
    setEdge(fromName: string, toName: string, value?: E): void;
    hasEdge(fromName: string, toName: string): boolean;
    edge(fromName: string, toName: string): E;
    edge(edgeObject: EdgeObject): E;
    removeEdge(v: string, w: string): void;
    nodes(): string[];
    node(name: string): N;
    removeNode(name: string): void;
    setGraph(graphOptions: GraphOptions): void;
    graph(): GraphOptions;
    nodeCount(): number;
    neighbors(name: string): string[];
    successors(name: string): string[];
    predecessors(name: string): string[];
    edges(): EdgeObject[];
    outEdges(name: string): E[];
    inEdges(name: string): E[];
    /**
     * Returns those nodes in the graph that have no in-edges.
     * Takes O(|V|) time.
     */
    sources(): string[];
    /**
     * Remove the node with the id v in the graph or do nothing if
     * the node is not in the graph. If the node was removed this
     * function also removes any incident edges. Returns the graph,
     * allowing this to be chained with other functions. Takes O(|E|) time.
     */
    removeNode(name: string): Graph<N, E>;
    setParent(name: string, parentName: string): void;
  }
}

/**
 * Declaring dagre var used for dagre layout.
 */
declare var dagre: {layout(graph: graphlib.Graph<any, any>): void;};
