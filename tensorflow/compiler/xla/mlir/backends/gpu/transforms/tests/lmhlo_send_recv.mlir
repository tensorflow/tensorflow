// RUN: xla-gpu-opt %s -split-input-file -xla-lmhlo-to-gpu-runtime \
// RUN:   | FileCheck %s

// CHECK: func @send(
// CHECK:   %[[ARG0:[a-z0-9]+]]: memref<4xf32>
// CHECK: )
func.func @send(%arg0: memref<4xf32>) {
  // CHECK: call @xla.gpu.send_host(%[[ARG0]]) {
  // CHECK-SAME:   channel_handle = #mhlo.channel_handle<handle = 1, type = 2>,
  // CHECK-SAME:   frontend_attributes = {
  // CHECK-SAME:     _xla_dcn_recv_channel = "2",
  // CHECK-SAME:     _xla_host_transfer_handler_name = "undef",
  // CHECK-SAME:     _xla_host_transfer_is_lower_bits = "false",
  // CHECK-SAME:     _xla_host_transfer_original_type = "f32",
  // CHECK-SAME:     _xla_host_transfer_rendezvous = "undef"
  // CHECK-SAME:   }} : (memref<4xf32>) -> ()
  "lmhlo.send"(%arg0) {
    channel_handle = #mhlo.channel_handle<handle = 1, type = 2>,
    frontend_attributes = {_xla_dcn_recv_channel = "2",
                           _xla_host_transfer_handler_name = "undef",
                           _xla_host_transfer_is_lower_bits = "false",
                           _xla_host_transfer_original_type = "f32",
                           _xla_host_transfer_rendezvous = "undef"},
    is_host_transfer = true
  } : (memref<4xf32>) -> !mhlo.token
  return
}

// CHECK: func private @xla.gpu.send_host(memref<4xf32>)
// CHECK-SAME: attributes {rt.custom_call = "xla.gpu.send_host"}

// -----

// CHECK: func @recv(
// CHECK:   %[[ARG0:[a-z0-9]+]]: memref<4xf32>
// CHECK: )
func.func @recv(%arg0: memref<4xf32>) {
  // CHECK: call @xla.gpu.recv_host(%[[ARG0]]) {
  // CHECK-SAME:   channel_handle = #mhlo.channel_handle<handle = 1, type = 2>,
  // CHECK-SAME:   frontend_attributes = {
  // CHECK-SAME:     _xla_host_transfer_handler_name = "undef",
  // CHECK-SAME:     _xla_host_transfer_is_lower_bits = "false",
  // CHECK-SAME:     _xla_host_transfer_original_type = "f32",
  // CHECK-SAME:     _xla_host_transfer_rendezvous = "undef"
  // CHECK-SAME:   }} : (memref<4xf32>) -> ()
  "lmhlo.recv"(%arg0) {
    channel_handle = #mhlo.channel_handle<handle = 1, type = 2>,
    frontend_attributes = {_xla_host_transfer_handler_name = "undef",
                           _xla_host_transfer_is_lower_bits = "false",
                           _xla_host_transfer_original_type = "f32",
                           _xla_host_transfer_rendezvous = "undef"},
    is_host_transfer = true
  } : (memref<4xf32>) -> !mhlo.token
  return
}

// CHECK: func private @xla.gpu.recv_host(memref<4xf32>)
// CHECK-SAME: attributes {rt.custom_call = "xla.gpu.recv_host"}

// -----

// CHECK: func @send_done(
// CHECK:   %[[ARG0:[a-z0-9]+]]: !mhlo.token
// CHECK: )
func.func @send_done(%arg0: !mhlo.token) {
  // CHECK: call @xla.gpu.send_done_host() {
  // CHECK-SAME:   channel_handle = #mhlo.channel_handle<handle = 1, type = 2>
  // CHECK-SAME: } : () -> ()
  "lmhlo.send_done"(%arg0) {
    channel_handle = #mhlo.channel_handle<handle = 1, type = 2>,
    is_host_transfer = true
  } : (!mhlo.token) -> ()
  return
}

// CHECK: func private @xla.gpu.send_done_host()
// CHECK-SAME: attributes {rt.custom_call = "xla.gpu.send_done_host"}

// -----

// CHECK: func @recv_done(
// CHECK:   %[[ARG0:[a-z0-9]+]]: !mhlo.token
// CHECK: )
func.func @recv_done(%arg0: !mhlo.token) {
  // CHECK: call @xla.gpu.recv_done_host() {
  // CHECK-SAME:   channel_handle = #mhlo.channel_handle<handle = 1, type = 2>
  // CHECK-SAME: } : () -> ()
  "lmhlo.recv_done"(%arg0) {
    channel_handle = #mhlo.channel_handle<handle = 1, type = 2>,
    is_host_transfer = true
  } : (!mhlo.token) -> ()
  return
}

// CHECK: func private @xla.gpu.recv_done_host()
// CHECK-SAME: attributes {rt.custom_call = "xla.gpu.recv_done_host"}
