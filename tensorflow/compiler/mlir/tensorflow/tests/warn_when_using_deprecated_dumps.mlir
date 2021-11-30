// RUN: tf-opt -split-input-file -verify-diagnostics %s

// Test warning on using deprecated attribute or type in old debug dump.

func @main() {
  // expected-error@+1 {{#tf_type.shape}}
  "tf.foo"() { shape = #tf.shape<?>} : () -> ()
  return
}

// -----

func @main() {
  // expected-error@+1 {{!tf_type.string}}
  "tf.foo"() : () -> (tensor<*x!tf.string>)
  return
}