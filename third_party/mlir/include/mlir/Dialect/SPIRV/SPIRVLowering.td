//===- SPIRVBase.td - MLIR SPIR-V Op Definitions Base file -*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the base file for supporting lowering to SPIR-V dialect. This
// file defines SPIR-V attributes used for specifying the shader
// interface or ABI. This is because SPIR-V module is expected to work in
// an execution environment as specified by a client API. A SPIR-V module
// needs to "link" correctly with the execution environment regarding the
// resources that are used in the SPIR-V module and get populated with
// data via the client API. The shader interface (or ABI) is passed into
// SPIR-V lowering path via attributes defined in this file. A
// compilation flow targeting SPIR-V is expected to attach such
// attributes to resources and other suitable places.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LOWERING
#define SPIRV_LOWERING

include "mlir/Dialect/SPIRV/SPIRVBase.td"

// For arguments that eventually map to spv.globalVariable for the
// shader interface, this attribute specifies the information regarding
// the global variable :
// 1) Descriptor Set.
// 2) Binding number.
// 3) Storage class.
def SPV_InterfaceVarABIAttr:
    StructAttr<"InterfaceVarABIAttr", SPV_Dialect,
               [StructFieldAttr<"descriptor_set", I32Attr>,
                StructFieldAttr<"binding", I32Attr>,
                StructFieldAttr<"storage_class", SPV_StorageClassAttr>]>;

// For entry functions, this attribute specifies information related to entry
// points in the generated SPIR-V module:
// 1) WorkGroup Size.
def SPV_EntryPointABIAttr:
    StructAttr<"EntryPointABIAttr", SPV_Dialect,
               [StructFieldAttr<"local_size", I32ElementsAttr>]>;

#endif // SPIRV_LOWERING
