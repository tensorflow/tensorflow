// RUN: tf-opt %s -split-input-file -verify-diagnostics

// Tests invalid #tf.func attributes.

// expected-error@+1 {{expected '<'}}
func @main() attributes {tf._implements = #tf.func} {
  return
}

// -----

// expected-error@+2 {{expected non-function type}}
// expected-error@+1 {{expected symbol while parsing tf.func attribute}}
func @main() attributes {tf._implements = #tf.func<>} {
  return
}

// -----

// expected-error@+1 {{expected ','}}
func @main() attributes {tf._implements = #tf.func<@symbol>} {
  return
}

// -----

// expected-error@+1 {{expected symbol while parsing tf.func attribute}}
func @main() attributes {tf._implements = #tf.func<{}>} {
  return
}

// -----

// expected-error@+1 {{expected empty string or symbol while parsing tf.func attribute}}
func @main() attributes {tf._implements = #tf.func<"test", {}>} {
  return
}

// -----

// expected-error@+1 {{expected Dictionary attribute while parsing tf.func attribute}}
func @main() attributes {tf._implements = #tf.func<@symbol, "">} {
  return
}

// -----

// expected-error@+1 {{expected '>'}}
func @main() attributes {tf._implements = #tf.func<@symbol, {}, "">} {
  return
}
