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

module tf {
/**
 * Recommended delay (ms) when running an expensive task asynchronously
 * that gives enough time for the progress bar to update its UI.
 */
const ASYNC_TASK_DELAY = 20;

export function time<T>(msg: string, task: () => T) {
    let start = Date.now();
    let result = task();
    /* tslint:disable */
    console.log(msg, ":", Date.now() - start, "ms");
    /* tslint:enable */
    return result;
}

/**
 * Tracks task progress. Each task being passed a progress tracker needs
 * to call the below-defined methods to notify the caller about the gradual
 * progress of the task.
 */
export interface ProgressTracker {
  updateProgress(incrementValue: number): void;
  setMessage(msg: string): void;
  reportError(msg: string, err: Error): void;
}

/**
 * Creates a tracker that sets the progress property of the
 * provided polymer component. The provided component must have
 * a property called 'progress' that is not read-only. The progress
 * property is an object with a numerical 'value' property and a
 * string 'msg' property.
 */
export function getTracker(polymerComponent: any) {
  return {
    setMessage: function(msg) {
      polymerComponent.set("progress", {
        value: polymerComponent.progress.value,
        msg: msg
      });
    },
    updateProgress: function(value) {
      polymerComponent.set("progress", {
        value: polymerComponent.progress.value + value,
        msg: polymerComponent.progress.msg
      });
    },
    reportError: function(msg: string, err) {
      // Log the stack trace in the console.
      console.error(err.stack);
      // And send a user-friendly message to the UI.
      polymerComponent.set("progress", {
        value: polymerComponent.progress.value,
        msg: msg,
        error: true
      });
    },
  };
}

/**
 * Creates a tracker for a subtask given the parent tracker, the total progress
 * of the subtask and the subtask message. The parent task should pass a
 * subtracker to its subtasks. The subtask reports its own progress which
 * becames relative to the main task.
 */
export function getSubtaskTracker(parentTracker: ProgressTracker,
    impactOnTotalProgress: number, subtaskMsg: string): ProgressTracker {
  return {
    setMessage: function(progressMsg) {
      // The parent should show a concatenation of its message along with
      // its subtask tracker message.
      parentTracker.setMessage(subtaskMsg + ": " + progressMsg);
    },
    updateProgress: function(incrementValue) {
      // Update the parent progress relative to the child progress.
      // For example, if the sub-task progresses by 30%, and the impact on the
      // total progress is 50%, then the task progresses by 30% * 50% = 15%.
      parentTracker
          .updateProgress(incrementValue * impactOnTotalProgress / 100);
    },
    reportError: function(msg: string, err: Error) {
      // The parent should show a concatenation of its message along with
      // its subtask error message.
      parentTracker.reportError(subtaskMsg + ": " + msg, err);
    }
  };
}

/**
 * Runs an expensive task and return the result.
 */
export function runTask<T>(msg: string, incProgressValue: number,
    task: () => T, tracker: ProgressTracker): T {
  // Update the progress message to say the current running task.
  tracker.setMessage(msg);
  // Run the expensive task with a delay that gives enough time for the
  // UI to update.
  try {
    var result = tf.time(msg, task);
    // Update the progress value.
    tracker.updateProgress(incProgressValue);
    // Return the result to be used by other tasks.
    return result;
  } catch (e) {
    // Errors that happen inside asynchronous tasks are
    // reported to the tracker using a user-friendly message.
    tracker.reportError("Failed " + msg, e);
  }
}

/**
 * Runs an expensive task asynchronously and returns a promise of the result.
 */
export function runAsyncTask<T>(msg: string, incProgressValue: number,
    task: () => T, tracker: ProgressTracker): Promise<T> {
  return new Promise((resolve, reject) => {
    // Update the progress message to say the current running task.
    tracker.setMessage(msg);
    // Run the expensive task with a delay that gives enough time for the
    // UI to update.
    setTimeout(function() {
      try {
        var result = tf.time(msg, task);
        // Update the progress value.
        tracker.updateProgress(incProgressValue);
        // Return the result to be used by other tasks.
        resolve(result);
      } catch (e) {
        // Errors that happen inside asynchronous tasks are
        // reported to the tracker using a user-friendly message.
        tracker.reportError("Failed " + msg, e);
      }
    }, ASYNC_TASK_DELAY);
  });
}

/**
 * Returns a query selector with escaped special characters that are not
 * allowed in a query selector.
 */
export function escapeQuerySelector(querySelector: string): string {
  return querySelector.replace( /([:.\[\],/\\\(\)])/g, "\\$1" );
}

/**
 * TensorFlow node definition as defined in the graph proto file.
 */
export interface TFNode {
  /** Name of the node */
  name: string;
  /** List of nodes that are inputs for this node. */
  input: string[];
  /** The name of the device where the computation will run. */
  device: string;
  /** The name of the operation associated with this node. */
  op: string;
  /** List of attributes that describe/modify the operation. */
  attr: {key: string, value: Object}[];
}

/**
 * TensorFlow stats file definition as defined in the stats proto file.
 */
export interface StepStats {
  dev_stats: {device: string, node_stats: NodeStats[]}[];
}

/**
 * TensorFlow stats for a node as defined in the stats proto file.
 */
export interface NodeStats {
  node_name: string;
  // The next 4 properties are currently stored as string in json
  // and must be parsed.
  all_start_micros: number;
  op_start_rel_micros: number;
  op_end_rel_micros: number;
  all_end_rel_micros: number;
  memory: {
    allocator_name: string;
    total_bytes: number; // Stored as string in json and should be parsed.
    peak_bytes: number; // Stored as string in json and should be parsed.
  }[];
  /** Output sizes recorded for a single execution of a graph node */
  output: TFNodeOutput[];
  timeline_label: string;
  scheduled_micros: string;
  thread_id: string;
}

/**
 * Description for the output tensor(s) of an operation in the graph.
 */
export interface TFNodeOutput {
  slot: number; // Stored as string in json and should be parsed.
  tensor_description: {
    /** Data type of tensor elements */
    dtype: string;
    /** Shape of the tensor */
    shape: {
      /**
       * Dimensions of the tensor, such as [{name: "input", size: 30},
       * {name: "output", size: 40}] for a 30 x 40 2D tensor.  The names
       * are optional. The order of entries in "dim" matters: It indicates
       * the layout of the values in the tensor in-memory representation.
       */
      dim: {
        /** Size of the tensor in that dimension */
        size: number, // Stored as string in json and should be parsed.
        /** Optional name of the tensor dimension */
        name?: string
      }[];
    };
    /** Information about the size and allocator used for the data */
    allocation_description: {
      // The next 2 properties are stored as string in json and
      // should be parsed.
      /** Total number of bytes requested */
      requested_bytes: number;
      /** Total number of bytes allocated, if known */
      allocated_bytes?: number;
      /** Name of the allocator used */
      allocator_name: string;
    };
  };
}
} // close module tf

/**
 * Declaring dagre var used for dagre layout.
 */
declare var dagre: { layout(graph: graphlib.Graph<any, any>): void; };
