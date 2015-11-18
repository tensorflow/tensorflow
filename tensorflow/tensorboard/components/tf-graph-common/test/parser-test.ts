suite("parser", () => {
let assert = chai.assert;

test("simple pbtxt", () => {
  let pbtxt =
    `node {
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
       input: "Q"
       input: "W"
     }`;
  let result = tf.graph.parser.parsePbtxt(pbtxt);
  assert.isTrue(result != null);
});

test("d3 exists", () => {
  assert.isTrue(d3 != null);
});

// TODO(bp): write tests.

});
