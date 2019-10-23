/*
 * Copyright (C) XMOS Limited 2013
 */

#ifndef OVERLAY_DESCRIPTOR_H_
#define OVERLAY_DESCRIPTOR_H_

typedef struct overlay_descriptor_t {
  /// The virtual address of the overlay.
  unsigned virtual_address;
  /// The physical address of the overlay.
  unsigned physical_address;
  /// The size of the overlay in bytes.
  unsigned size;
} overlay_descriptor_t;

#endif // OVERLAY_DESCRIPTOR_H_
