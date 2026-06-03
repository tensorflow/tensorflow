/**
 * @license
 * Copyright 2026 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @fileoverview WebGL-based renderer for large HLO Execution DAGs.
 */

import {EDGE_FS, EDGE_VS, NODE_FS, NODE_FS_SIMPLE, NODE_VS} from './shaders';

/**
 * Data structure for a node in the HLO graph.
 */
export declare interface NodeData {
  id: number;
  x: number;
  y: number;
  diffScore: number;
  key: string;
  anchorId: number;
}

/**
 * Data structure for an edge in the HLO graph.
 */
export declare interface EdgeData {
  supplierId: number;
  consumerId: number;
}

/**
 * Data structure for the entire HLO graph.
 */
export declare interface GraphData {
  nodes: NodeData[];
  edges: EdgeData[];
}

/**
 * Interface for the HLO graph renderer.
 *
 * This interface is used to allow for different implementations of the
 * renderer, such as WebGL and SVG.
 */
export declare interface HloGraphRendererInterface {
  render(): void;
  setOnClick(callback: (node: NodeData) => void): void;
  setOnHover(callback: (node: NodeData|null) => void): void;
  fitToView(): void;
  zoomIn(): void;
  zoomOut(): void;
  selectNodeByAnchorId(anchorId: number): void;
}

// Performance and rendering quality configuration limits
const MAX_VISIBLE_EDGES = 30000;
const MAX_NEIGHBORS_TO_EXPLORE = 5000;
const EDGE_SEGMENTS_HIGH_QUALITY = 20;
const EDGE_SEGMENTS_MEDIUM_QUALITY = 5;
const EDGE_SEGMENTS_LOW_QUALITY = 3;
const SEGMENT_DEGRADATION_THRESHOLD_MEDIUM = 10000;
const SEGMENT_DEGRADATION_THRESHOLD_LOW = 20000;

interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

class QuadTreeNode {
  private readonly nodes: NodeData[] = [];
  private children?: [QuadTreeNode, QuadTreeNode, QuadTreeNode, QuadTreeNode]|
      null = null;
  private static readonly MAX_NODES = 16;
  private static readonly MAX_DEPTH = 8;

  constructor(
      private readonly bounds: Bounds,
      private readonly depth = 0,
  ) {}

  insert(node: NodeData): boolean {
    if (!this.contains(node)) return false;

    if (this.children) {
      for (const child of this.children) {
        if (child.insert(node)) return true;
      }
    }

    this.nodes.push(node);

    if (!this.children && this.nodes.length > QuadTreeNode.MAX_NODES &&
        this.depth < QuadTreeNode.MAX_DEPTH) {
      this.split();
      let i = 0;
      while (i < this.nodes.length) {
        let inserted = false;
        for (const child of this.children!) {
          if (child.insert(this.nodes[i])) {
            this.nodes.splice(i, 1);
            inserted = true;
            break;
          }
        }
        if (!inserted) i++;
      }
    }
    return true;
  }

  private split() {
    const midX = (this.bounds.minX + this.bounds.maxX) / 2;
    const midY = (this.bounds.minY + this.bounds.maxY) / 2;

    this.children = [
      new QuadTreeNode(
          {
            minX: this.bounds.minX,
            maxX: midX,
            minY: this.bounds.minY,
            maxY: midY,
          },
          this.depth + 1,
          ),
      new QuadTreeNode(
          {
            minX: midX,
            maxX: this.bounds.maxX,
            minY: this.bounds.minY,
            maxY: midY,
          },
          this.depth + 1,
          ),
      new QuadTreeNode(
          {
            minX: this.bounds.minX,
            maxX: midX,
            minY: midY,
            maxY: this.bounds.maxY,
          },
          this.depth + 1,
          ),
      new QuadTreeNode(
          {
            minX: midX,
            maxX: this.bounds.maxX,
            minY: midY,
            maxY: this.bounds.maxY,
          },
          this.depth + 1,
          ),
    ];
  }

  query(searchBounds: Bounds, results: NodeData[]) {
    if (!this.intersects(searchBounds)) return;

    for (const node of this.nodes) {
      if (this.intersectsNode(searchBounds, node)) {
        results.push(node);
      }
    }

    if (this.children) {
      for (const child of this.children) {
        child.query(searchBounds, results);
      }
    }
  }

  private contains(node: NodeData): boolean {
    return (
        node.x >= this.bounds.minX && node.x <= this.bounds.maxX &&
        node.y >= this.bounds.minY && node.y <= this.bounds.maxY);
  }

  private intersects(other: Bounds): boolean {
    return !(
        other.maxX < this.bounds.minX || other.minX > this.bounds.maxX ||
        other.maxY < this.bounds.minY || other.minY > this.bounds.maxY);
  }

  private intersectsNode(searchBounds: Bounds, node: NodeData): boolean {
    return (
        node.x >= searchBounds.minX && node.x <= searchBounds.maxX &&
        node.y >= searchBounds.minY && node.y <= searchBounds.maxY);
  }
}

/**
 * WebGL-based renderer for large HLO Execution DAGs.
 *
 * This class is responsible for rendering the graph nodes and edges using
 * WebGL. It also handles user interactions such as panning, zooming, and node
 * selection.
 */
export class HloGraphRenderer implements HloGraphRendererInterface {
  private readonly gl: WebGL2RenderingContext;
  private zoom = 1.0;
  private minZoom = 0.0;
  private panX = 0.0;
  private panY = 0.0;
  private segmentsPerEdge = 20;

  private nodeProgram: WebGLProgram|null = null;
  private nodeProgramSimple: WebGLProgram|null = null;
  private edgeProgram: WebGLProgram|null = null;

  private quadBuffer: WebGLBuffer|null = null;
  private nodeCenterBuffer: WebGLBuffer|null = null;
  private nodeDiffScoreBuffer: WebGLBuffer|null = null;
  private edgeTemplateBuffer: WebGLBuffer|null = null;
  private edgeHighQualityTemplateBuffer: WebGLBuffer|null = null;
  private edgeEndpointsBuffer: WebGLBuffer|null = null;
  private highlightedEdgeEndpointsBuffer: WebGLBuffer|null = null;
  private highlightedEdgeRadiiBuffer: WebGLBuffer|null = null;
  private readonly edgeEndpointsScratch = new Float32Array(
      MAX_VISIBLE_EDGES * 4,
  );

  private isDragging = false;
  private startX = 0;
  private startY = 0;
  private startClientX = 0;
  private startClientY = 0;

  private highlightedEdgeCount = 0;

  private readonly nodeMap = new Map<number, NodeData>();
  private readonly nodeEdges = new Map<number, EdgeData[]>();
  private readonly layerIndex = new Map<number, NodeData[]>();

  private selectedNode: NodeData|null = null;

  private nodeSelectionStateBuffer: WebGLBuffer|null = null;
  private selectionStates: Float32Array|null = null;
  private readonly nodeIdToIndex = new Map<number, number>();

  private quadTree: QuadTreeNode|null = null;

  private graphBounds: Bounds|null = null;

  private onClickCallback: ((node: NodeData) => void)|null = null;

  private onHoverCallback: ((node: NodeData|null) => void)|null = null;

  private overlayCanvas: HTMLCanvasElement|null = null;
  private scopeIntervals:
      Array<{name: string; level: number; minX: number; maxX: number;}> = [];

  constructor(
      private readonly canvas: HTMLCanvasElement,
      private readonly data: GraphData,
  ) {
    const gl = canvas.getContext('webgl2');
    if (!gl) {
      throw new Error('WebGL 2 not supported');
    }
    this.gl = gl;
    this.init();
    this.setupEvents();
  }

  setOnClick(callback: (node: NodeData) => void) {
    this.onClickCallback = callback;
  }

  setOnHover(callback: (node: NodeData|null) => void) {
    this.onHoverCallback = callback;
  }

  private init() {
    this.initShaders();
    this.initBuffers();
    this.gl.clearColor(1, 1, 1, 1.0);
    this.gl.enable(this.gl.DEPTH_TEST);
    this.gl.depthFunc(this.gl.LEQUAL);
    this.gl.enable(this.gl.BLEND);
    this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

    this.fitToView();
  }

  private initShaders() {
    this.nodeProgram = this.createProgram(NODE_VS, NODE_FS);
    this.nodeProgramSimple = this.createProgram(NODE_VS, NODE_FS_SIMPLE);
    this.edgeProgram = this.createProgram(EDGE_VS, EDGE_FS);
  }

  private createProgram(vsSource: string, fsSource: string): WebGLProgram {
    const gl = this.gl;
    const vs = this.loadShader(gl.VERTEX_SHADER, vsSource);
    const fs = this.loadShader(gl.FRAGMENT_SHADER, fsSource);
    const program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(
          'Unable to initialize the shader program: ' +
              gl.getProgramInfoLog(program)!,
      );
    }
    return program;
  }

  private loadShader(type: number, source: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`An error occurred compiling the shaders: ${info}`);
    }
    return shader;
  }

  private initBuffers() {
    const gl = this.gl;

    // Quad for instancing
    const quadVertices = new Float32Array([
      -0.5,
      -0.5,
      0.5,
      -0.5,
      -0.5,
      0.5,
      0.5,
      0.5,
    ]);
    this.quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);

    // Node Instanced Data
    const numNodes = this.data.nodes.length;

    const centers = new Float32Array(numNodes * 2);
    const diffScores = new Float32Array(numNodes);
    const selectionStates = new Float32Array(numNodes);

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (let i = 0; i < numNodes; ++i) {
      const node = this.data.nodes[i];

      this.nodeMap.set(node.id, node);
      this.nodeIdToIndex.set(node.id, i);

      // Index by Layer
      const x = Math.round(node.x);
      let layer = this.layerIndex.get(x);
      if (!layer) {
        layer = [];
        this.layerIndex.set(x, layer);
      }
      layer.push(node);

      if (node.x < minX) minX = node.x;
      if (node.x > maxX) maxX = node.x;
      if (node.y < minY) minY = node.y;
      if (node.y > maxY) maxY = node.y;

      centers[i * 2] = node.x;
      centers[i * 2 + 1] = node.y;

      diffScores[i] = node.diffScore;
    }

    this.nodeCenterBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nodeCenterBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, centers, gl.STATIC_DRAW);

    this.nodeDiffScoreBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nodeDiffScoreBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, diffScores, gl.STATIC_DRAW);

    this.nodeSelectionStateBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.nodeSelectionStateBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, selectionStates, gl.DYNAMIC_DRAW);
    this.selectionStates = selectionStates;

    // Edges
    for (const edge of this.data.edges) {
      let edges = this.nodeEdges.get(edge.supplierId);
      if (!edges) {
        edges = [];
        this.nodeEdges.set(edge.supplierId, edges);
      }
      edges.push(edge);

      edges = this.nodeEdges.get(edge.consumerId);
      if (!edges) {
        edges = [];
        this.nodeEdges.set(edge.consumerId, edges);
      }
      edges.push(edge);
    }

    // Template Buffer (T values)
    const segmentsPerEdge = this.segmentsPerEdge;
    const tValues = new Float32Array(segmentsPerEdge + 1);
    for (let i = 0; i <= segmentsPerEdge; i++) {
      tValues[i] = i / segmentsPerEdge;
    }
    this.edgeTemplateBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeTemplateBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, tValues, gl.STATIC_DRAW);

    const hqSegments = EDGE_SEGMENTS_HIGH_QUALITY;
    const hqTValues = new Float32Array(hqSegments + 1);
    for (let i = 0; i <= hqSegments; i++) {
      hqTValues[i] = i / hqSegments;
    }
    this.edgeHighQualityTemplateBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeHighQualityTemplateBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, hqTValues, gl.STATIC_DRAW);

    // Endpoints Buffer (Allocated for max visible edges)
    this.edgeEndpointsBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeEndpointsBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, MAX_VISIBLE_EDGES * 4 * 4, gl.DYNAMIC_DRAW);

    if (numNodes === 0) {
      minX = 0;
      maxX = 0;
      minY = 0;
      maxY = 0;
    }

    this.graphBounds = {minX, maxX, minY, maxY};

    // Build QuadTree
    this.quadTree = new QuadTreeNode(this.graphBounds);
    const scopeMap = new Map < string, {
      name: string;
      level: number;
      minX: number;
      maxX: number
    }
    >();
    for (const node of this.data.nodes) {
      this.quadTree.insert(node);
      const parts = node.key.split('/');
      // The last element is the instruction itself, so scopes are
      // parts.slice(0, parts.length - 1)
      let prefix = '';
      for (let i = 0; i < parts.length - 1; i++) {
        prefix = prefix ? prefix + '/' + parts[i] : parts[i];
        let scope = scopeMap.get(prefix);
        if (!scope) {
          scope = {name: parts[i], level: i, minX: Infinity, maxX: -Infinity};
          scopeMap.set(prefix, scope);
        }
        scope.minX = Math.min(scope.minX, node.x);
        scope.maxX = Math.max(scope.maxX, node.x);
      }
    }
    this.scopeIntervals = Array.from(scopeMap.values());
  }

  private setupEvents() {
    this.canvas.addEventListener('wheel', (e) => {
      this.handleWheel(e);
    });
    this.canvas.addEventListener('mousedown', (e) => {
      this.handleMouseDown(e);
    });
    window.addEventListener('mousemove', (e) => {
      this.handleMouseMove(e);
    });
    window.addEventListener('mouseup', (e) => {
      this.handleMouseUp(e);
    });

    const resizeObserver = new ResizeObserver(() => {
      this.canvas.width = this.canvas.clientWidth;
      this.canvas.height = this.canvas.clientHeight;
      this.clampPan();
      this.render();
    });
    resizeObserver.observe(this.canvas);
  }

  private zoomByFactor(zoomFactor: number, mouseX: number, mouseY: number) {
    const worldXBefore = mouseX / this.zoom - this.panX;
    const worldYBefore = mouseY / this.zoom - this.panY;

    let newZoom = this.zoom * zoomFactor;
    newZoom = Math.max(this.minZoom, Math.min(100.0, newZoom));

    this.panX = mouseX / newZoom - worldXBefore;
    this.panY = mouseY / newZoom - worldYBefore;
    this.zoom = newZoom;

    this.clampPan();
    this.render();
  }

  zoomIn() {
    this.zoomByFactor(1.2, this.canvas.width / 2, this.canvas.height / 2);
  }

  zoomOut() {
    this.zoomByFactor(1.0 / 1.2, this.canvas.width / 2, this.canvas.height / 2);
  }

  selectNodeByAnchorId(anchorId: number) {
    for (const node of this.data.nodes) {
      if (node.anchorId === anchorId) {
        this.selectedNode = node;
        this.updateHighlightedEdges();

        this.panX = this.canvas.width / (2 * this.zoom) - node.x;
        this.panY = this.canvas.height / (2 * this.zoom) - node.y;
        this.clampPan();

        this.render();
        break;
      }
    }
  }

  private handleWheel(e: WheelEvent) {
    e.preventDefault();
    const zoomFactor = Math.exp(-e.deltaY / 400.0);
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    this.zoomByFactor(zoomFactor, mouseX, mouseY);
  }

  private handleMouseDown(e: MouseEvent) {
    this.isDragging = true;
    this.startX = e.clientX;
    this.startY = e.clientY;
    this.startClientX = e.clientX;
    this.startClientY = e.clientY;
  }

  private handleMouseMove(e: MouseEvent) {
    if (this.isDragging) {
      const deltaX = e.clientX - this.startX;
      const deltaY = e.clientY - this.startY;

      this.panX += deltaX / this.zoom;
      this.panY += deltaY / this.zoom;

      this.startX = e.clientX;
      this.startY = e.clientY;

      this.clampPan();
      this.render();
    } else {
      // Hover picking
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const node = this.pickNode(mouseX, mouseY);
      if (this.onHoverCallback) {
        this.onHoverCallback(node);
      }
    }
  }

  private handleMouseUp(e: MouseEvent) {
    this.isDragging = false;

    const dx = e.clientX - this.startClientX;
    const dy = e.clientY - this.startClientY;
    if (Math.sqrt(dx * dx + dy * dy) < 15) {
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const node = this.pickNode(mouseX, mouseY);
      if (node) {
        this.selectedNode = node;
        this.updateHighlightedEdges();
        if (this.onClickCallback) {
          this.onClickCallback(node);
        }
        this.render();
      }
    }
  }

  private updateHighlightedEdges() {
    if (this.selectionStates && this.nodeSelectionStateBuffer) {
      this.selectionStates.fill(0);
      if (this.selectedNode) {
        const selectedIndex = this.nodeIdToIndex.get(this.selectedNode.id);
        if (selectedIndex !== undefined) {
          this.selectionStates[selectedIndex] = 1.0;
        }

        const edges = this.nodeEdges.get(this.selectedNode.id) || [];
        for (const edge of edges) {
          const neighborId = edge.supplierId === this.selectedNode.id ?
              edge.consumerId :
              edge.supplierId;
          const neighborIndex = this.nodeIdToIndex.get(neighborId);
          if (neighborIndex !== undefined) {
            this.selectionStates[neighborIndex] = 2.0;
          }
        }
      }
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.nodeSelectionStateBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, this.selectionStates, gl.DYNAMIC_DRAW);
    }

    // Highlighted edges are now drawn on top, no need to exclude them from main
    // buffer.
    if (!this.selectedNode) {
      this.highlightedEdgeCount = 0;
      return;
    }

    const edges = this.nodeEdges.get(this.selectedNode.id) || [];
    const endpoints = new Float32Array(edges.length * 4);
    const radii = new Float32Array(edges.length * 2);

    const strokeThickness = Math.min(
        3.0,
        Math.max(1.0, 1.0 + ((this.zoom - 1.0) / 99.0) * 2.0),
    );
    const strokeWorld = strokeThickness / this.zoom;

    let validCount = 0;
    for (const edge of edges) {
      const u = this.nodeMap.get(edge.supplierId);
      const v = this.nodeMap.get(edge.consumerId);
      if (u && v) {
        endpoints[validCount * 4] = u.x;
        endpoints[validCount * 4 + 1] = u.y;
        endpoints[validCount * 4 + 2] = v.x;
        endpoints[validCount * 4 + 3] = v.y;

        let uRadius = 0.3;
        let vRadius = 0.3;
        if (u.id === this.selectedNode.id) {
          uRadius = 0.45;
        } else {
          uRadius = 0.3 + strokeWorld;
        }
        if (v.id === this.selectedNode.id) {
          vRadius = 0.45;
        } else {
          vRadius = 0.3 + strokeWorld;
        }
        radii[validCount * 2] = uRadius;
        radii[validCount * 2 + 1] = vRadius;

        validCount++;
      }
    }
    this.highlightedEdgeCount = validCount;

    const gl = this.gl;
    if (!this.highlightedEdgeEndpointsBuffer) {
      this.highlightedEdgeEndpointsBuffer = gl.createBuffer();
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.highlightedEdgeEndpointsBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, endpoints, gl.DYNAMIC_DRAW);

    if (!this.highlightedEdgeRadiiBuffer) {
      this.highlightedEdgeRadiiBuffer = gl.createBuffer();
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.highlightedEdgeRadiiBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, radii, gl.DYNAMIC_DRAW);
  }

  private pickNode(mouseX: number, mouseY: number): NodeData|null {
    const worldX = mouseX / this.zoom - this.panX;
    const worldY = mouseY / this.zoom - this.panY;

    // Node size in world units
    const maxBaseNodeSize = 0.8;
    const minPixelSize = 4.0;
    const nodeSize = Math.max(maxBaseNodeSize, minPixelSize / this.zoom);
    const searchRadius = nodeSize / 2;

    const searchBounds = {
      minX: worldX - searchRadius,
      maxX: worldX + searchRadius,
      minY: worldY - searchRadius,
      maxY: worldY + searchRadius,
    };

    let pickedNode: NodeData|null = null;

    const strokeThickness = Math.min(
        3.0,
        Math.max(1.0, 1.0 + ((this.zoom - 1.0) / 99.0) * 2.0),
    );
    const strokeWorld = strokeThickness / this.zoom;

    const results: NodeData[] = [];
    this.quadTree?.query(searchBounds, results);

    for (const node of results) {
      const index = this.nodeIdToIndex.get(node.id);
      const state = index !== undefined && this.selectionStates ?
          this.selectionStates[index] :
          0.0;

      let radius = 0.3;
      if (state > 0.5 && state < 1.5) {
        radius = 0.45;
      } else if (state > 1.5) {
        radius = 0.3 + strokeWorld;
      }

      const dx = node.x - worldX;
      const dy = node.y - worldY;
      const distSq = dx * dx + dy * dy;
      if (distSq <= radius * radius) {
        if (!pickedNode || node.diffScore > pickedNode.diffScore) {
          pickedNode = node;
        }
      }
    }
    return pickedNode;
  }

  private clampPan() {
    if (!this.graphBounds) return;
    const {minX, maxX, minY, maxY} = this.graphBounds;

    const w = this.canvas.width / this.zoom;
    const h = this.canvas.height / this.zoom;

    this.panX = Math.max(-maxX, Math.min(w - minX, this.panX));
    this.panY = Math.max(-maxY, Math.min(h - minY, this.panY));
  }

  fitToView() {
    if (this.data.nodes.length === 0) return;
    if (!this.graphBounds) return;

    this.canvas.width = this.canvas.clientWidth;
    this.canvas.height = this.canvas.clientHeight;

    if (this.canvas.width === 0 || this.canvas.height === 0) return;

    const {minX, maxX, minY, maxY} = this.graphBounds;
    const graphWidth = maxX - minX;
    const graphHeight = maxY - minY;

    const padding = 20;  // pixels
    const w = graphWidth > 0 ? graphWidth : 1;
    const h = graphHeight > 0 ? graphHeight : 1;

    const zoomX = (this.canvas.width - 2 * padding) / w;
    const zoomY = (this.canvas.height - 2 * padding) / h;
    const calculatedZoom = Math.min(zoomX, zoomY);

    // Minimum zoom is set to 1/10th of the fit-to-view zoom to allow zooming
    // out.
    this.minZoom = Math.min(100.0, calculatedZoom) / 10.0;
    // Initial zoom is capped at 100.
    this.zoom = Math.min(100.0, calculatedZoom);

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    this.panX = this.canvas.width / (2 * this.zoom) - centerX;
    this.panY = this.canvas.height / (2 * this.zoom) - centerY;

    this.clampPan();
    this.render();
  }

  private getMatrix(): Float32Array {
    const width = this.canvas.width;
    const height = this.canvas.height;

    const m00 = (2 * this.zoom) / width;
    const m01 = 0;
    const m02 = (2 * this.panX * this.zoom) / width - 1;

    const m10 = 0;
    const m11 = (-2 * this.zoom) / height;
    const m12 = 1 - (2 * this.panY * this.zoom) / height;

    return new Float32Array([m00, m10, 0, m01, m11, 0, m02, m12, 1]);
  }

  /**
   * Renders the graph to the canvas.
   */
  render() {
    console.log(
        'Renderer: render() called. Zoom:',
        this.zoom,
        'Pan:',
        this.panX,
        this.panY,
    );
    const gl = this.gl;
    if (gl.isContextLost()) {
      console.warn('Renderer: WebGL Context is Lost!');
      return;
    }

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const matrix = this.getMatrix();
    const strokeThickness = Math.min(
        3.0,
        Math.max(1.0, 1.0 + ((this.zoom - 1.0) / 99.0) * 2.0),
    );

    const isLargeGraph = this.data.nodes.length > 50000;

    // Calculate viewport bounds
    const viewportBounds: Bounds = {
      minX: -this.panX,
      maxX: -this.panX + this.canvas.width / this.zoom,
      minY: -this.panY,
      maxY: -this.panY + this.canvas.height / this.zoom,
    };

    console.log('Renderer: Viewport Bounds:', viewportBounds);

    // Query visible nodes
    let visibleNodes: NodeData[] = [];
    this.quadTree?.query(viewportBounds, visibleNodes);

    const uniqueNodes = new Map<number, NodeData>();
    for (const node of visibleNodes) {
      uniqueNodes.set(node.id, node);
    }
    visibleNodes = Array.from(uniqueNodes.values());

    // 1. Draw Edges
    if (this.edgeProgram) {
      console.log('Renderer: Visible Nodes:', visibleNodes.length);

      // Collect edges
      const candidateEdges = new Set<EdgeData>();

      // 1. Prioritize selected node
      if (this.selectedNode) {
        // Do not add old unhighlighted edges of the selected node to
        // visibleEdges so they are cleaned up and do not overlap with the new
        // highlighted edges.
      }

      // 2. Add edges of visible nodes
      for (const node of visibleNodes) {
        const edges = this.nodeEdges.get(node.id);
        if (edges) {
          for (const edge of edges) {
            if (this.selectedNode &&
                (edge.supplierId === this.selectedNode.id ||
                 edge.consumerId === this.selectedNode.id)) {
              continue;
            }
            candidateEdges.add(edge);
          }
        }
      }

      // 3. Add edges of neighbors of visible nodes (heuristic for pass-through)
      const visibleNodeIds = new Set<number>(visibleNodes.map((n) => n.id));
      const neighbors = new Set<number>();
      for (const node of visibleNodes) {
        const edges = this.nodeEdges.get(node.id);
        if (edges) {
          for (const edge of edges) {
            if (this.selectedNode &&
                (edge.supplierId === this.selectedNode.id ||
                 edge.consumerId === this.selectedNode.id)) {
              continue;
            }
            const neighborId =
                edge.supplierId === node.id ? edge.consumerId : edge.supplierId;
            if (!visibleNodeIds.has(neighborId)) {
              neighbors.add(neighborId);
            }
          }
        }
        // Cap neighbors to explore to avoid huge loops
        if (neighbors.size >= MAX_NEIGHBORS_TO_EXPLORE) break;
      }

      for (const neighborId of neighbors) {
        const neighborEdges = this.nodeEdges.get(neighborId);
        if (neighborEdges) {
          for (const nEdge of neighborEdges) {
            if (this.selectedNode &&
                (nEdge.supplierId === this.selectedNode.id ||
                 nEdge.consumerId === this.selectedNode.id)) {
              continue;
            }
            candidateEdges.add(nEdge);
          }
        }
      }

      const candidateEdgesArray = Array.from(candidateEdges);
      const visibleEdges: EdgeData[] = [];
      const cap = MAX_VISIBLE_EDGES;
      if (candidateEdgesArray.length <= cap) {
        for (const edge of candidateEdgesArray) {
          visibleEdges.push(edge);
        }
      } else {
        const step = candidateEdgesArray.length / cap;
        for (let i = 0; i < cap; i++) {
          visibleEdges.push(candidateEdgesArray[Math.floor(i * step)]);
        }
      }

      const numEdgesToDraw = visibleEdges.length;
      const activeEdgesCount = numEdgesToDraw + this.highlightedEdgeCount;
      let desiredSegments = EDGE_SEGMENTS_HIGH_QUALITY;
      if (activeEdgesCount > SEGMENT_DEGRADATION_THRESHOLD_LOW) {
        desiredSegments = EDGE_SEGMENTS_LOW_QUALITY;
      } else if (activeEdgesCount > SEGMENT_DEGRADATION_THRESHOLD_MEDIUM) {
        desiredSegments = EDGE_SEGMENTS_MEDIUM_QUALITY;
      }
      if (this.segmentsPerEdge !== desiredSegments) {
        this.segmentsPerEdge = desiredSegments;
        const tValues = new Float32Array(desiredSegments + 1);
        for (let i = 0; i <= desiredSegments; i++) {
          tValues[i] = i / desiredSegments;
        }
        if (!this.edgeTemplateBuffer) {
          this.edgeTemplateBuffer = gl.createBuffer();
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeTemplateBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, tValues, gl.STATIC_DRAW);
      }
      console.log('Renderer: Edges to draw:', numEdgesToDraw);
      if (numEdgesToDraw > 0) {
        // Fill endpoints buffer
        let i = 0;
        for (const edge of visibleEdges) {
          const u = this.nodeMap.get(edge.supplierId);
          const v = this.nodeMap.get(edge.consumerId);
          if (u && v) {
            this.edgeEndpointsScratch[i * 4] = u.x;
            this.edgeEndpointsScratch[i * 4 + 1] = u.y;
            this.edgeEndpointsScratch[i * 4 + 2] = v.x;
            this.edgeEndpointsScratch[i * 4 + 3] = v.y;
          }
          i++;
        }

        // Upload to GPU
        gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeEndpointsBuffer);
        // We only upload the active portion of the buffer
        gl.bufferSubData(
            gl.ARRAY_BUFFER,
            0,
            this.edgeEndpointsScratch.subarray(0, numEdgesToDraw * 4),
        );

        gl.useProgram(this.edgeProgram);
        const uMatrixLocation = gl.getUniformLocation(
            this.edgeProgram,
            'uMatrix',
        );
        gl.uniformMatrix3fv(uMatrixLocation, false, matrix);

        const uEdgeColorLocation = gl.getUniformLocation(
            this.edgeProgram,
            'uEdgeColor',
        );

        // Draw All Edges
        gl.uniform4f(
            uEdgeColorLocation, 0.0, 0.0, 0.0, 0.3);  // Half-transparent black
        gl.lineWidth(strokeThickness);                // Thicker edges

        const aTLocation = gl.getAttribLocation(this.edgeProgram, 'aT');
        const aEndpointULocation = gl.getAttribLocation(
            this.edgeProgram,
            'aEndpointU',
        );
        const aEndpointVLocation = gl.getAttribLocation(
            this.edgeProgram,
            'aEndpointV',
        );
        const aRadiiLocation = gl.getAttribLocation(this.edgeProgram, 'aRadii');

        // T values (Template)
        if (this.edgeTemplateBuffer) {
          gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeTemplateBuffer);
          gl.enableVertexAttribArray(aTLocation);
          gl.vertexAttribPointer(aTLocation, 1, gl.FLOAT, false, 0, 0);
          gl.vertexAttribDivisor(aTLocation, 0);  // Varies per vertex
        }

        // Endpoints (Instanced)
        if (this.edgeEndpointsBuffer) {
          gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeEndpointsBuffer);
          gl.enableVertexAttribArray(aEndpointULocation);
          gl.vertexAttribPointer(
              aEndpointULocation,
              2,
              gl.FLOAT,
              false,
              4 * 4,
              0,
          );
          gl.vertexAttribDivisor(aEndpointULocation, 1);  // Varies per instance

          gl.enableVertexAttribArray(aEndpointVLocation);
          gl.vertexAttribPointer(
              aEndpointVLocation,
              2,
              gl.FLOAT,
              false,
              4 * 4,
              2 * 4,
          );
          gl.vertexAttribDivisor(aEndpointVLocation, 1);  // Varies per instance
        }

        // Radii (Constant for main draw call)
        gl.disableVertexAttribArray(aRadiiLocation);
        gl.vertexAttrib2f(aRadiiLocation, 0.3, 0.3);

        gl.drawArraysInstanced(
            gl.LINE_STRIP,
            0,
            this.segmentsPerEdge + 1,
            numEdgesToDraw,
        );

        const err = gl.getError();
        if (err !== gl.NO_ERROR) {
          console.error('Renderer: WebGL Error after edge draw:', err);
        }

        // Reset divisors to avoid leaking to other draw calls (e.g. node
        // drawing)
        gl.vertexAttribDivisor(aEndpointULocation, 0);
        gl.vertexAttribDivisor(aEndpointVLocation, 0);

        // Disable arrays to avoid leaking or out-of-bounds reads in other draw
        // calls
        gl.disableVertexAttribArray(aTLocation);
        gl.disableVertexAttribArray(aEndpointULocation);
        gl.disableVertexAttribArray(aEndpointVLocation);
      }
    }
    // Highlighted edges moved to end to be drawn above nodes.

    // 2. Draw Nodes
    const useSimpleShader =
        (isLargeGraph && this.zoom < 1.0) || this.zoom < 0.05;
    const nodeProgram =
        useSimpleShader ? this.nodeProgramSimple : this.nodeProgram;

    if (nodeProgram) {
      gl.useProgram(nodeProgram);
      const uMatrixLocation = gl.getUniformLocation(nodeProgram, 'uMatrix');
      gl.uniformMatrix3fv(uMatrixLocation, false, matrix);

      const uNodeSizeLocation = gl.getUniformLocation(nodeProgram, 'uNodeSize');
      const baseNodeSize = 0.6;                       // World units
      const minPixelSize = isLargeGraph ? 1.5 : 4.0;  // Pixels
      const nodeSize = Math.max(baseNodeSize, minPixelSize / this.zoom);
      gl.uniform1f(uNodeSizeLocation, nodeSize);

      const uZoomLocation = gl.getUniformLocation(nodeProgram, 'uZoom');
      gl.uniform1f(uZoomLocation, this.zoom);

      const uStrokeThicknessPixelsLocation = gl.getUniformLocation(
          nodeProgram,
          'uStrokeThicknessPixels',
      );
      gl.uniform1f(uStrokeThicknessPixelsLocation, strokeThickness);

      // Quad
      const aPositionLocation = gl.getAttribLocation(nodeProgram, 'aPosition');
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
      gl.enableVertexAttribArray(aPositionLocation);
      gl.vertexAttribPointer(aPositionLocation, 2, gl.FLOAT, false, 0, 0);
      // Explicitly set divisor to 0 to ensure vertices advance per vertex (not
      // instance), preventing degenerate quads if state was leaked.
      gl.vertexAttribDivisor(aPositionLocation, 0);

      // Centers (Instanced)
      // Centers (Instanced - using relative coordinates to prevent float32
      // overflow)
      const focusX = -this.panX;
      const focusY = -this.panY;

      const count = visibleNodes.length;
      const relativeCenters = new Float32Array(count * 2);
      const dynamicDiffScores = new Float32Array(count);
      const dynamicSelectionStates = new Float32Array(count);

      for (let i = 0; i < count; i++) {
        const node = visibleNodes[i];
        relativeCenters[i * 2] = node.x - focusX;
        relativeCenters[i * 2 + 1] = node.y - focusY;
        dynamicDiffScores[i] = node.diffScore;
        const idx = this.nodeIdToIndex.get(node.id);
        dynamicSelectionStates[i] = idx !== undefined && this.selectionStates ?
            this.selectionStates[idx] :
            0;
      }

      // Use modified matrix with translation exactly matching the focus shift
      const width = this.canvas.width;
      const height = this.canvas.height;
      const m00 = (2 * this.zoom) / width;
      const m11 = (-2 * this.zoom) / height;
      const relativeMatrix = new Float32Array([m00, 0, 0, 0, m11, 0, -1, 1, 1]);
      gl.uniformMatrix3fv(uMatrixLocation, false, relativeMatrix);

      const dynamicCenterBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, dynamicCenterBuf);
      gl.bufferData(gl.ARRAY_BUFFER, relativeCenters, gl.DYNAMIC_DRAW);

      const aCenterLocation = gl.getAttribLocation(nodeProgram, 'aCenter');
      gl.enableVertexAttribArray(aCenterLocation);
      gl.vertexAttribPointer(aCenterLocation, 2, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(aCenterLocation, 1);

      // Diff Scores (Instanced)
      const dynamicDiffBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, dynamicDiffBuf);
      gl.bufferData(gl.ARRAY_BUFFER, dynamicDiffScores, gl.DYNAMIC_DRAW);

      const aDiffScoreLocation = gl.getAttribLocation(
          nodeProgram,
          'aDiffScore',
      );
      gl.enableVertexAttribArray(aDiffScoreLocation);
      gl.vertexAttribPointer(aDiffScoreLocation, 1, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(aDiffScoreLocation, 1);

      // Selection States (Instanced)
      const dynamicSelBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, dynamicSelBuf);
      gl.bufferData(gl.ARRAY_BUFFER, dynamicSelectionStates, gl.DYNAMIC_DRAW);

      const aSelectionStateLocation = gl.getAttribLocation(
          nodeProgram,
          'aSelectionState',
      );
      if (aSelectionStateLocation >= 0) {
        gl.enableVertexAttribArray(aSelectionStateLocation);
        gl.vertexAttribPointer(
            aSelectionStateLocation,
            1,
            gl.FLOAT,
            false,
            0,
            0,
        );
        gl.vertexAttribDivisor(aSelectionStateLocation, 1);
      }

      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);

      // Cleanup dynamic buffers
      gl.deleteBuffer(dynamicCenterBuf);
      gl.deleteBuffer(dynamicDiffBuf);
      gl.deleteBuffer(dynamicSelBuf);

      const err = gl.getError();
      if (err !== gl.NO_ERROR) {
        console.error('Renderer: WebGL Error after node draw:', err);
      }

      // Reset divisors
      gl.vertexAttribDivisor(aCenterLocation, 0);
      gl.vertexAttribDivisor(aDiffScoreLocation, 0);
      if (aSelectionStateLocation >= 0) {
        gl.vertexAttribDivisor(aSelectionStateLocation, 0);
      }

      // Disable arrays to avoid leaking to other draw calls (e.g. edge drawing
      // in next frame)
      gl.disableVertexAttribArray(aPositionLocation);
      gl.disableVertexAttribArray(aCenterLocation);
      gl.disableVertexAttribArray(aDiffScoreLocation);
      if (aSelectionStateLocation >= 0) {
        gl.disableVertexAttribArray(aSelectionStateLocation);
      }
    }

    // 3. Draw Highlighted Edges (Above Nodes)
    if (this.edgeProgram && this.highlightedEdgeCount > 0) {
      gl.disable(gl.DEPTH_TEST);  // Disable depth test to draw above nodes
      gl.useProgram(this.edgeProgram);
      const uMatrixLocation = gl.getUniformLocation(
          this.edgeProgram,
          'uMatrix',
      );
      gl.uniformMatrix3fv(uMatrixLocation, false, matrix);

      const uEdgeColorLocation = gl.getUniformLocation(
          this.edgeProgram,
          'uEdgeColor',
      );
      gl.uniform4f(uEdgeColorLocation, 0, 0, 0, 1.0);  // Darker

      const aTLocation = gl.getAttribLocation(this.edgeProgram, 'aT');
      const aEndpointULocation = gl.getAttribLocation(
          this.edgeProgram,
          'aEndpointU',
      );
      const aEndpointVLocation = gl.getAttribLocation(
          this.edgeProgram,
          'aEndpointV',
      );
      const aRadiiLocation = gl.getAttribLocation(this.edgeProgram, 'aRadii');

      // T values (Template)
      if (this.edgeHighQualityTemplateBuffer) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.edgeHighQualityTemplateBuffer);
        gl.enableVertexAttribArray(aTLocation);
        gl.vertexAttribPointer(aTLocation, 1, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(aTLocation, 0);
      }

      // Endpoints (Instanced)
      if (this.highlightedEdgeEndpointsBuffer) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.highlightedEdgeEndpointsBuffer);
        gl.enableVertexAttribArray(aEndpointULocation);
        gl.vertexAttribPointer(
            aEndpointULocation,
            2,
            gl.FLOAT,
            false,
            4 * 4,
            0,
        );
        gl.vertexAttribDivisor(aEndpointULocation, 1);

        gl.enableVertexAttribArray(aEndpointVLocation);
        gl.vertexAttribPointer(
            aEndpointVLocation,
            2,
            gl.FLOAT,
            false,
            4 * 4,
            2 * 4,
        );
        gl.vertexAttribDivisor(aEndpointVLocation, 1);
      }

      // Radii (Instanced for highlighted)
      if (this.highlightedEdgeRadiiBuffer) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.highlightedEdgeRadiiBuffer);
        gl.enableVertexAttribArray(aRadiiLocation);
        gl.vertexAttribPointer(aRadiiLocation, 2, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(aRadiiLocation, 1);
      }

      gl.lineWidth(strokeThickness);
      gl.drawArraysInstanced(
          gl.LINE_STRIP,
          0,
          EDGE_SEGMENTS_HIGH_QUALITY + 1,
          this.highlightedEdgeCount,
      );
      gl.lineWidth(1.0);

      // Reset divisors for edge attributes
      gl.vertexAttribDivisor(aEndpointULocation, 0);
      gl.vertexAttribDivisor(aEndpointVLocation, 0);
      gl.vertexAttribDivisor(aRadiiLocation, 0);

      // Disable arrays
      gl.disableVertexAttribArray(aTLocation);
      gl.disableVertexAttribArray(aEndpointULocation);
      gl.disableVertexAttribArray(aEndpointVLocation);
      gl.disableVertexAttribArray(aRadiiLocation);

      gl.enable(gl.DEPTH_TEST);  // Re-enable depth test
    }

    if (!this.overlayCanvas) {
      this.overlayCanvas = document.createElement('canvas');
      this.overlayCanvas.style.position = 'absolute';
      this.overlayCanvas.style.left = '0';
      this.overlayCanvas.style.top = '0';
      this.overlayCanvas.style.pointerEvents = 'none';
      if (this.canvas.parentElement) {
        this.canvas.parentElement.style.position = 'relative';
        this.canvas.parentElement.appendChild(this.overlayCanvas);
      }
    }
    this.overlayCanvas.width = this.canvas.width;
    this.overlayCanvas.height = this.canvas.height;
    const ctx = this.overlayCanvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
      ctx.font = '12px sans-serif';

      for (const interval of this.scopeIntervals) {
        const screenMinX = (interval.minX - 0.45 + this.panX) * this.zoom;
        const screenMaxX = (interval.maxX + 0.45 + this.panX) * this.zoom;

        if (screenMaxX < 0 || screenMinX > this.overlayCanvas.width) {
          continue;
        }

        const y = 14 + interval.level * 14;

        ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.lineWidth = 1;

        if (screenMinX >= 0 && screenMinX <= this.overlayCanvas.width) {
          ctx.beginPath();
          ctx.moveTo(screenMinX, y);
          ctx.lineTo(screenMinX, this.overlayCanvas.height);
          ctx.stroke();
        }

        if (screenMaxX >= 0 && screenMaxX <= this.overlayCanvas.width) {
          ctx.beginPath();
          ctx.moveTo(screenMaxX, y);
          ctx.lineTo(screenMaxX, this.overlayCanvas.height);
          ctx.stroke();
        }

        ctx.beginPath();
        ctx.moveTo(Math.max(0, screenMinX), y);
        ctx.lineTo(Math.min(this.overlayCanvas.width, screenMaxX), y);
        ctx.stroke();

        const textX = Math.max(2, screenMinX + 2);
        const availableWidth = screenMaxX - textX;
        if (availableWidth > 10) {
          let text = interval.name;
          if (ctx.measureText(text).width > availableWidth) {
            while (text.length > 0 &&
                   ctx.measureText(text + '...').width > availableWidth) {
              text = text.substring(0, text.length - 1);
            }
            if (text.length > 0) {
              text += '...';
            }
          }
          if (text.length > 0) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.fillText(text, textX, y - 2);
          }
        }
      }

      if (visibleNodes.length < 50 && this.zoom >= 23.0) {
        ctx.save();
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#000000';

        for (const node of visibleNodes) {
          const screenX = (node.x + this.panX) * this.zoom;
          const screenY = (node.y + this.panY) * this.zoom;

          if (screenX < -20 || screenX > this.overlayCanvas.width + 20 ||
              screenY < -20 || screenY > this.overlayCanvas.height + 20) {
            continue;
          }

          const parts = node.key.split('/');
          const nodeName = parts[parts.length - 1];

          ctx.fillText(nodeName, screenX, screenY);
        }
        ctx.restore();
      }
    }
  }
}

// tslint:disable-next-line:no-any
(window as any)['HloGraphRenderer'] = HloGraphRenderer;

/**
 * Parses the base64 encoded binary graph data and returns the GraphData object.
 *
 * @param base64Str The base64 encoded binary graph data.
 * @return The GraphData object.
 */
export async function parseBinaryGraphData(
    base64Str: string,
    ): Promise<GraphData> {
  const binaryString = atob(base64Str);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  const ds = new DecompressionStream('gzip');
  const stream = new Response(bytes).body;
  if (!stream) {
    throw new Error('Failed to create stream from bytes');
  }
  const decompressedStream = stream.pipeThrough(ds);
  const buffer = await new Response(decompressedStream).arrayBuffer();
  const uncompressed = new Uint8Array(buffer);

  const view = new DataView(
      uncompressed.buffer,
      uncompressed.byteOffset,
      uncompressed.byteLength,
  );

  function readInt64(v: DataView, o: number): number {
    const low = v.getUint32(o, true);
    const high = v.getInt32(o + 4, true);
    return high * 4294967296 + low;
  }

  let offset = 0;

  const nodeCount = view.getUint32(offset, true);
  offset += 4;

  const nodes: NodeData[] = [];
  for (let i = 0; i < nodeCount; i++) {
    const id = readInt64(view, offset);
    offset += 8;
    const x = view.getFloat32(offset, true);
    offset += 4;
    const y = view.getFloat32(offset, true);
    offset += 4;
    const diffScore = view.getFloat32(offset, true);
    offset += 4;
    const anchorId = readInt64(view, offset);
    offset += 8;
    const keyLen = view.getUint16(offset, true);
    offset += 2;

    const keyBytes = uncompressed.subarray(offset, offset + keyLen);
    const key = new TextDecoder().decode(keyBytes);
    offset += keyLen;

    nodes.push({id, x, y, diffScore, key, anchorId});
  }

  const edgeCount = view.getUint32(offset, true);
  offset += 4;

  const edges: EdgeData[] = [];
  for (let i = 0; i < edgeCount; i++) {
    const supplierId = readInt64(view, offset);
    offset += 8;
    const consumerId = readInt64(view, offset);
    offset += 8;
    edges.push({supplierId, consumerId});
  }

  return {nodes, edges};
}

// tslint:disable-next-line:no-any
(window as any)['parseBinaryGraphData'] = parseBinaryGraphData;
// tslint:disable-next-line:no-any
(window as any)['HloGraphRenderer'] = HloGraphRenderer;
// tslint:disable-next-line:no-any
(window as any)['__force_shaders_retention'] = [
  NODE_VS,
  NODE_FS,
  NODE_FS_SIMPLE,
  EDGE_VS,
  EDGE_FS,
];
