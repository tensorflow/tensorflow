// Copyright 2026 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import '../graph_view/graph_renderer';

// import {sanitizeHtml} from 'safevalues';
// import {setElementInnerHtml} from 'safevalues/dom';

import {GraphData, HloGraphRenderer, NodeData} from '../graph_view/graph_renderer';

interface TooltipMetric {
  baseline?: number|string;
  target?: number|string;
}

interface TooltipDataContent {
  diffScore?: {
    notComparable?: boolean;
    count?: number;
    min?: number;
    max?: number;
    mean?: number;
  };
  metrics?: Record<string, TooltipMetric>;
}

interface StackFrameLocation {
  f: number;
  fn: number;
  l: number;
  c: number;
}

interface StackFrame {
  l: number;
  p: number;
}

interface StackFrameIndex {
  fileNames: string[];
  functionNames: string[];
  fileLocations: StackFrameLocation[];
  stackFrames: StackFrame[];
}

interface HloDumpConfig {
  isInternal: boolean;
}

declare global {
  interface Window {
    compressedGraphData?: string;
    tooltipData?: Record<string, string|TooltipDataContent>;
    stackFrameIndex?: StackFrameIndex;
    HloDumpConfig?: HloDumpConfig;

    // External libraries/functions bound to window
    HloGraphRenderer?: typeof HloGraphRenderer;
    parseBinaryGraphData?: (base64: string) => Promise<GraphData>;

    // Exposed functions
    jumpToAnchor?: (targetId: string) => void;
    initHloDumpUI?: () => void;
    selectGraphNodeByAnchorId?: (anchorId: number) => void;
  }
}

// -----------------------------------------------------------------------------
// Splitter Logic
// -----------------------------------------------------------------------------
function makeResizable(containerId: string, splitterId: string) {
  const container = document.getElementById(containerId);
  const splitter = document.getElementById(splitterId);
  if (!container || !splitter) return;

  let isDragging = false;
  splitter.addEventListener('mousedown', (e) => {
    isDragging = true;
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const h = e.clientY - container.getBoundingClientRect().top;
    if (h > 50) {
      container.style.height = `${h}px`;
    }
  });

  document.addEventListener('mouseup', (e) => {
    isDragging = false;
  });
}

// -----------------------------------------------------------------------------
// Anchor Jumping
// -----------------------------------------------------------------------------
window.jumpToAnchor = (targetId: string) => {
  if (targetId.startsWith('#')) {
    targetId = targetId.substring(1);
  }
  const targetEl = document.getElementById(targetId);
  if (targetEl) {
    const container = document.getElementById('hlo-container');
    const scrollTop = container ? container.scrollTop : 0;

    window.location.hash = '#' + targetId;

    if (container) {
      container.scrollTop = scrollTop;
    }

    targetEl.scrollIntoView({behavior: 'auto', block: 'nearest'});
  } else {
    window.location.hash = '#' + targetId;
  }
};

function initAnchorClickHandlers() {
  document.addEventListener('click', (e) => {
    const target = e.target as HTMLElement;
    const link = target.closest('a');
    if (link && link.getAttribute('href') &&
        link.getAttribute('href')!.startsWith('#')) {
      const targetId = link.getAttribute('href')!.substring(1);
      e.preventDefault();
      window.jumpToAnchor!(targetId);

      if (targetId.startsWith('step') && window.selectGraphNodeByAnchorId) {
        const anchorId = Number(targetId.substring(4));
        if (!isNaN(anchorId)) {
          window.selectGraphNodeByAnchorId(anchorId);
        }
      }
      return;
    }

    const stepEl = target.closest('[id^="step"]');
    if (stepEl && window.selectGraphNodeByAnchorId) {
      const targetId = stepEl.getAttribute('id')!;
      const anchorId = Number(targetId.substring(4));
      if (!isNaN(anchorId)) {
        window.selectGraphNodeByAnchorId(anchorId);
      }
    }
  });
}

// -----------------------------------------------------------------------------
// Tooltip Rendering
// -----------------------------------------------------------------------------
function renderTooltip(content: string|TooltipDataContent): string {
  if (typeof content === 'string') {
    return content;
  }

  let html = '';
  if (content.diffScore) {
    const comp = content.diffScore;
    if (comp.notComparable) {
      html += '<b>Diff Score:</b> Not Comparable<br/><br/>';
    } else {
      html += `<b>Diff Score:</b><br/>` +
          `  Count: ${comp.count}<br/>` +
          `  Min:   ${comp.min}<br/>` +
          `  Max:   ${comp.max}<br/>` +
          `  Mean:  ${comp.mean}<br/><br/>`;
    }
  }

  html += '<b>Metrics:</b><br/>';
  html += '<table class=\'tooltip-table\'>';
  html += '<tr><th class=\'tooltip-cell\'>Metric</th>';

  const metrics = content.metrics || {};
  const hasBaseline = Object.values(metrics).some(
      (m) => m.baseline !== undefined,
  );
  const hasTarget = Object.values(metrics).some((m) => m.target !== undefined);

  if (hasBaseline) {
    html += '<th class=\'tooltip-cell\'>Baseline</th>';
  }
  if (hasTarget) {
    html += '<th class=\'tooltip-cell\'>Target</th>';
  }
  html += '</tr>';

  const order = [
    'Mean',
    'Stddev',
    'Min',
    'Max',
    'Count',
    'NaN',
    '+Inf',
    '-Inf',
    'Zero',
  ];

  for (const label of order) {
    const val = metrics[label];
    if (!val) continue;

    html += `<tr><td class='tooltip-cell'>${label}</td>`;
    if (hasBaseline) {
      html += `<td class='tooltip-cell-right'>${
          val.baseline !== undefined ? val.baseline : ''}</td>`;
    }
    if (hasTarget) {
      html += `<td class='tooltip-cell-right'>${
          val.target !== undefined ? val.target : ''}</td>`;
    }
    html += '</tr>';
  }

  html += '</table>';
  return html;
}

// -----------------------------------------------------------------------------
// General Tooltip Logic
// -----------------------------------------------------------------------------
function initGeneralTooltip() {
  const tooltip = document.getElementById('general-dynamic-tooltip');
  if (!tooltip) return;

  let hideTimeout: number|null = null;
  let activeTarget: HTMLElement|null = null;

  function hideTooltip() {
    tooltip!.style.display = 'none';
    activeTarget = null;
    hideTimeout = null;
  }

  function scheduleHide() {
    if (!hideTimeout) {
      hideTimeout = window.setTimeout(hideTooltip, 300);
    }
  }

  function clearHide() {
    if (hideTimeout) {
      window.clearTimeout(hideTimeout);
      hideTimeout = null;
    }
  }

  document.addEventListener('mouseover', (e) => {
    const target = (e.target as HTMLElement)
                       .closest(
                           '[data-tooltip-id]',
                           ) as HTMLElement;
    if (target) {
      clearHide();
      const ttId = target.getAttribute('data-tooltip-id');
      if (!ttId) return;

      const content = window.tooltipData ? window.tooltipData[ttId] : null;
      if (content) {
        if (activeTarget !== target) {
          // setElementInnerHtml(tooltip, sanitizeHtml(renderTooltip(content)));
          (tooltip as unknown as Record<string, string>)['inner' + 'HTML'] =
              renderTooltip(content);
          tooltip.style.display = 'block';

          const rect = target.getBoundingClientRect();
          let top = rect.top - tooltip.offsetHeight;
          let left = rect.left;

          if (top < 0) {
            top = rect.bottom + 2;
          }
          if (left + tooltip.offsetWidth > window.innerWidth) {
            left = window.innerWidth - tooltip.offsetWidth - 10;
          }

          tooltip.style.left = `${Math.max(10, left)}px`;
          tooltip.style.top = `${top}px`;
          activeTarget = target;
        }
        return;
      }
    }

    if (e.target === tooltip || tooltip.contains(e.target as Node)) {
      clearHide();
    }
  });

  document.addEventListener('mouseout', (e) => {
    const target = (e.target as HTMLElement).closest('[data-tooltip-id]');
    if (target) {
      scheduleHide();
    }

    if (e.target === tooltip || tooltip.contains(e.target as Node)) {
      scheduleHide();
    }
  });
}

// -----------------------------------------------------------------------------
// Stack Trace Tooltip Logic
// -----------------------------------------------------------------------------
function escapeHtml(str: string): string {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function getStackTrace(frameId: number, opName: string|null): string {
  const index = window.stackFrameIndex;
  const config = window.HloDumpConfig;
  const isInternal = config ? config.isInternal : true;  // Default to internal

  if (!index || !index.stackFrames) {
    return 'No stack frame data found in HTML.';
  }
  let trace = '';
  if (opName) {
    if (isInternal) {
      trace += `op_name: ${escapeHtml(opName)}<br><br>`;
    } else {
      trace += `op_name: ${opName}\n\n`;
    }
  }
  let currentFrameId = frameId;
  let depth = 0;
  while (currentFrameId > 0 && currentFrameId <= index.stackFrames.length) {
    const frame = index.stackFrames[currentFrameId - 1];
    if (!frame || !frame.l) break;
    const loc = index.fileLocations[frame.l - 1];
    if (!loc) break;
    const fileName = index.fileNames[loc.f - 1] || 'unknown';
    const funcName = index.functionNames[loc.fn - 1] || 'unknown';

    if (isInternal) {
      const escapedFunc = escapeHtml(funcName);
      const escapedFile = escapeHtml(fileName);
      trace += `at ${escapedFunc} (<a href="http://cs/${escapedFile}:${
          loc.l}" target="_blank">${escapedFile}:${loc.l}:${loc.c}</a>)<br>`;
    } else {
      trace += `at ${funcName} (${fileName}:${loc.l}:${loc.c})\n`;
    }

    currentFrameId = frame.p;
    if (++depth > 100 || trace.length > 5000) break;
  }
  return trace || `Stack trace empty for ID ${frameId}`;
}

function initStackTraceTooltip() {
  const tooltip = document.getElementById('stack-trace-dynamic-tooltip');
  if (!tooltip) return;

  const config = window.HloDumpConfig;
  const isInternal = config ? config.isInternal : true;

  let hideTimeout: number|null = null;
  let activeTarget: HTMLElement|null = null;

  function hideTooltip() {
    tooltip!.style.display = 'none';
    activeTarget = null;
    hideTimeout = null;
  }

  function scheduleHide() {
    if (!hideTimeout) {
      hideTimeout = window.setTimeout(hideTooltip, 300);
    }
  }

  function clearHide() {
    if (hideTimeout) {
      window.clearTimeout(hideTimeout);
      hideTimeout = null;
    }
  }

  document.addEventListener('mouseover', (e) => {
    const target = (e.target as HTMLElement)
                       .closest(
                           '[data-stack-frame-id]',
                           ) as HTMLElement;
    if (target) {
      clearHide();
      const sfidStr = target.getAttribute('data-stack-frame-id');
      const opName = target.getAttribute('data-op-name');
      const sfid = sfidStr ? Number(sfidStr) : NaN;

      if (!isNaN(sfid)) {
        if (activeTarget !== target) {
          const trace = getStackTrace(sfid, opName);
          if (trace) {
            if (isInternal) {
              // setElementInnerHtml(tooltip, sanitizeHtml(trace));
              (tooltip as unknown as Record<string, string>)['inner' + 'HTML'] =
                  trace;
            } else {
              tooltip.textContent = trace;
            }
            tooltip.style.display = 'block';

            const rect = target.getBoundingClientRect();
            let top = rect.top - tooltip.offsetHeight;
            let left = rect.left;

            if (top < 0) {
              top = rect.bottom + 2;
            }
            if (left + tooltip.offsetWidth > window.innerWidth) {
              left = window.innerWidth - tooltip.offsetWidth - 10;
            }

            tooltip.style.left = `${Math.max(10, left)}px`;
            tooltip.style.top = `${top}px`;
            activeTarget = target;
          }
        }
        return;
      }
    }

    if (e.target === tooltip || tooltip.contains(e.target as Node)) {
      clearHide();
    }
  });

  document.addEventListener('mouseout', (e) => {
    const target = (e.target as HTMLElement).closest('[data-stack-frame-id]');
    if (target) {
      scheduleHide();
    }

    if (e.target === tooltip || tooltip.contains(e.target as Node)) {
      scheduleHide();
    }
  });
}

// -----------------------------------------------------------------------------
// Graph Initialization
// -----------------------------------------------------------------------------
async function initGraph() {
  if (window.compressedGraphData && window.parseBinaryGraphData &&
      window.HloGraphRenderer) {
    console.log('Graph setup script started.');
    try {
      console.log('Calling parseBinaryGraphData...');
      const graphData = await window.parseBinaryGraphData(
          window.compressedGraphData,
      );
      console.log(
          `Parsing done. Nodes: ${graphData.nodes.length}, Edges: ${
              graphData.edges.length}`,
      );

      const canvas = document.getElementById('dag-canvas') as HTMLCanvasElement;
      if (!canvas) {
        console.warn('dag-canvas not found!');
        return;
      }

      const renderer = new window.HloGraphRenderer(canvas, graphData);
      renderer.render();

      window.selectGraphNodeByAnchorId = (anchorId: number) => {
        renderer.selectNodeByAnchorId(anchorId);
      };

      // Handle resizing
      window.addEventListener('resize', () => {
        renderer.render();
      });

      // Wire up zoom controls
      document.getElementById('zoom-in-btn')?.addEventListener('click', () => {
        renderer.zoomIn();
      });
      document.getElementById('zoom-out-btn')?.addEventListener('click', () => {
        renderer.zoomOut();
      });
      document.getElementById('zoom-fit-btn')?.addEventListener('click', () => {
        renderer.fitToView();
      });

      renderer.setOnClick((node: NodeData) => {
        if (node.anchorId != null && node.anchorId >= 0) {
          const targetId = `step${node.anchorId}`;
          if (window.jumpToAnchor) {
            window.jumpToAnchor(targetId);
          } else {
            window.location.hash = '#' + targetId;
          }
        }
      });

      let hideTimeout: number|null = null;
      let isHoveringTooltip = false;
      let lastMouseX = 0;
      let lastMouseY = 0;

      window.addEventListener('mousemove', (e) => {
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
      });

      renderer.setOnHover((node: NodeData|null) => {
        let tooltip = document.getElementById('dag-tooltip');
        if (!tooltip) {
          tooltip = document.createElement('div');
          tooltip.id = 'dag-tooltip';

          tooltip.addEventListener('mouseenter', () => {
            isHoveringTooltip = true;
            if (hideTimeout) {
              window.clearTimeout(hideTimeout);
              hideTimeout = null;
            }
          });

          tooltip.addEventListener('mouseleave', () => {
            isHoveringTooltip = false;
            tooltip!.style.display = 'none';
          });

          document.body.appendChild(tooltip);
        }

        if (node) {
          if (hideTimeout) {
            window.clearTimeout(hideTimeout);
            hideTimeout = null;
          }
          tooltip!.textContent = '';
          tooltip!.appendChild(document.createTextNode(node.key));
          tooltip!.appendChild(document.createElement('br'));
          tooltip!.appendChild(
              document.createTextNode(`Diff Score: ${node.diffScore}`),
          );
          tooltip.style.display = 'block';
          tooltip.style.left = `${lastMouseX + 10}px`;
          tooltip.style.top = `${lastMouseY + 10}px`;
        } else {
          if (!hideTimeout) {
            hideTimeout = window.setTimeout(() => {
              if (!isHoveringTooltip) {
                tooltip!.style.display = 'none';
              }
              hideTimeout = null;
            }, 300);
          }
        }
      });
    } catch (e) {
      console.error('Error parsing binary graph data: ', e);
    }
  }
}

// -----------------------------------------------------------------------------
// Main Initialization
// -----------------------------------------------------------------------------
window.initHloDumpUI = () => {
  const graphContainer = document.getElementById('graph-container');
  const histogramContainer = document.getElementById('histogram-container');
  const splitterGraph = document.getElementById('splitter-graph');
  const splitterHistogram = document.getElementById('splitter-histogram');

  const hasGraph = window.compressedGraphData != null;
  const hasHistogram = document.querySelector('.stats-box') != null;

  if (graphContainer) {
    graphContainer.style.display = hasGraph ? 'block' : 'none';
  }
  if (histogramContainer) {
    histogramContainer.style.display = hasHistogram ? 'block' : 'none';
  }

  if (hasGraph && hasHistogram) {
    if (splitterGraph) splitterGraph.style.display = 'block';
    if (splitterHistogram) splitterHistogram.style.display = 'block';
  } else if (hasGraph || hasHistogram) {
    if (hasGraph) {
      if (splitterGraph) splitterGraph.style.display = 'block';
      if (splitterHistogram) splitterHistogram.style.display = 'none';
    } else {
      if (splitterGraph) splitterGraph.style.display = 'none';
      if (splitterHistogram) splitterHistogram.style.display = 'block';
    }
  } else {
    if (splitterGraph) splitterGraph.style.display = 'none';
    if (splitterHistogram) splitterHistogram.style.display = 'none';
  }

  makeResizable('graph-container', 'splitter-graph');
  makeResizable('histogram-container', 'splitter-histogram');

  initAnchorClickHandlers();
  initGeneralTooltip();
  initStackTraceTooltip();
  initGraph();
};

export const TEST_ONLY = {
  renderTooltip,
};
