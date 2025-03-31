#ifndef PROTON_ATTRDEFS
#define PROTON_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "ProtonDialect.td"

class Proton_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<Proton_Dialect, name, traits, baseCppClass> {
}

#endif // PROTON_ATTRDEFS
