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

suite('graph', () => {
  let assert = chai.assert;

  test('graphlib exists', () => { assert.isTrue(graphlib != null); });

  test('simple graph contruction', done => {
    let pbtxt = tf.graph.test.util.stringToArrayBuffer(`
      node {
        name: "Q"
        op: "Input"
      }
      node {
        name: "W"
        op: "Input"
      }
      node {
        name: "X"
        op: "MatMul"
        input: "Q:2"
        input: "W"
      }`);
    let statsPbtxt = tf.graph.test.util.stringToArrayBuffer(`step_stats {
      dev_stats {
        device: "cpu"
        node_stats {
          node_name: "Q"
          all_start_micros: 10
          all_end_rel_micros: 4
        }
        node_stats {
          node_name: "Q"
          all_start_micros: 12
          all_end_rel_micros: 4
        }
      }
    }`);

    let buildParams: tf.graph.BuildParams = {
      enableEmbedding: true,
      inEmbeddingTypes: ['Const'],
      outEmbeddingTypes: ['^[a-zA-Z]+Summary$'],
      refEdges: {}
    };
    let dummyTracker =
        tf.graph.util.getTracker({set: () => { return; }, progress: 0});
    tf.graph.parser.parseGraphPbTxt(pbtxt).then(nodes => {
      tf.graph.build(nodes, buildParams, dummyTracker)
          .then((slimGraph: tf.graph.SlimGraph) => {
            assert.isTrue(slimGraph.nodes['X'] != null);
            assert.isTrue(slimGraph.nodes['W'] != null);
            assert.isTrue(slimGraph.nodes['Q'] != null);

            let firstInputOfX = slimGraph.nodes['X'].inputs[0];
            assert.equal(firstInputOfX.name, 'Q');
            assert.equal(firstInputOfX.outputTensorIndex, 2);

            let secondInputOfX = slimGraph.nodes['X'].inputs[1];
            assert.equal(secondInputOfX.name, 'W');
            assert.equal(secondInputOfX.outputTensorIndex, 0);

            tf.graph.parser.parseStatsPbTxt(statsPbtxt).then(stepStats => {
              tf.graph.joinStatsInfoWithGraph(slimGraph, stepStats);
              assert.equal(slimGraph.nodes['Q'].stats.totalMicros, 6);
              done();
            });
          });
    });
  });

  test('health pill numbers round correctly', () => {
    // Integers are rounded to the ones place.
    assert.equal(tf.graph.scene.humanizeHealthPillStat(42.0, true), '42');

    // Numbers with magnitude >= 1 are rounded to the tenths place.
    assert.equal(tf.graph.scene.humanizeHealthPillStat(1, false), '1.0');
    assert.equal(tf.graph.scene.humanizeHealthPillStat(42.42, false), '42.4');
    assert.equal(tf.graph.scene.humanizeHealthPillStat(-42.42, false), '-42.4');

    // Numbers with magnitude < 1 are written in scientific notation rounded to
    // the tenths place.
    assert.equal(tf.graph.scene.humanizeHealthPillStat(0, false), '0.0e+0');
    assert.equal(tf.graph.scene.humanizeHealthPillStat(0.42, false), '4.2e-1');
    assert.equal(
        tf.graph.scene.humanizeHealthPillStat(-0.042, false), '-4.2e-2');
  });

  // TODO(bp): write tests.
});
