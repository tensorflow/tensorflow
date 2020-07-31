// RUN: tf-opt %s -split-input-file -verify-diagnostics

// Tests invalid #tf.func attributes.

// expected-error@+1 {{invalid TensorFlow func attribute: func}}
func @main() attributes {tf._implements = #tf.func} {
  return
}

// -----

// expected-error@+1 {{invalid TensorFlow func attribute: func<>}}
func @main() attributes {tf._implements = #tf.func<>} {
  return
}

// -----

// expected-error@+1 {{invalid TensorFlow func attribute: func<@symbol>}}
func @main() attributes {tf._implements = #tf.func<@symbol>} {
  return
}

// -----

// expected-error@+1 {{invalid TensorFlow func attribute: func<{}>}}
func @main() attributes {tf._implements = #tf.func<{}>} {
  return
}

// -----

// expected-error@+1 {{invalid TensorFlow func attribute: func<"test", {}>}}
func @main() attributes {tf._implements = #tf.func<"test", {}>} {
  return
}

// -----

// expected-error@+1 {{invalid TensorFlow func attribute: func<@symbol, "">}}
func @main() attributes {tf._implements = #tf.func<@symbol, "">} {
  return
}

// -----

// expected-error@+1 {{invalid TensorFlow func attribute: func<@symbol, {}, "">}}
func @main() attributes {tf._implements = #tf.func<@symbol, {}, "">} {
  return
}
