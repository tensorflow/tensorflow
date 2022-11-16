// RUN: tf-opt %s -split-input-file -verify-diagnostics

// Tests invalid #tf_type.func attributes.

// expected-error@+1 {{expected '<'}}
func.func @main() attributes {tf._implements = #tf_type.func} {
  func.return
}

// -----

// expected-error@+2 {{expected attribute value}}
// expected-error@+1 {{expected symbol while parsing tf.func attribute}}
func.func @main() attributes {tf._implements = #tf_type.func<>} {
  func.return
}

// -----

// expected-error@+1 {{expected ','}}
func.func @main() attributes {tf._implements = #tf_type.func<@symbol>} {
  func.return
}

// -----

// expected-error@+1 {{expected symbol while parsing tf.func attribute}}
func.func @main() attributes {tf._implements = #tf_type.func<{}>} {
  func.return
}

// -----

// expected-error@+1 {{expected empty string or symbol while parsing tf.func attribute}}
func.func @main() attributes {tf._implements = #tf_type.func<"test", {}>} {
  func.return
}

// -----

// expected-error@+1 {{expected Dictionary attribute while parsing tf.func attribute}}
func.func @main() attributes {tf._implements = #tf_type.func<@symbol, "">} {
  func.return
}

// -----

// expected-error@+1 {{expected '>'}}
func.func @main() attributes {tf._implements = #tf_type.func<@symbol, {}, "">} {
  func.return
}
