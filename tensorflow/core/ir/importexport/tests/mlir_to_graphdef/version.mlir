// RUN: tfg-translate -mlir-to-graphdef %s | FileCheck %s

tfg.graph #tf_type.version<producer = 42, min_consumer = 21, bad_consumers = [1, 2, 5, 12]> {
}

// CHECK: producer: 42
// CHECK: min_consumer: 21
// CHECK: bad_consumers: 1
// CHECK: bad_consumers: 2
// CHECK: bad_consumers: 5
// CHECK: bad_consumers: 12
