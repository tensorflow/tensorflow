// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2 | FileCheck %s -check-prefix=TILE-2
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=0,2 | FileCheck %s -check-prefix=TILE-02
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=0,0,2 | FileCheck %s -check-prefix=TILE-002
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2,3,4 | FileCheck %s -check-prefix=TILE-234

//   TILE-2-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
//  TILE-02-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
// TILE-002-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
// TILE-234-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
// TILE-234-DAG: #[[UB1:.*]] = (d0) -> (d0 + 3)
// TILE-234-DAG: #[[UB2:.*]] = (d0) -> (d0 + 4)

func @matmul(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %I = linalg.range %c0:%arg1:%c1 : !linalg.range
  %J = linalg.range %c0:%arg2:%c1 : !linalg.range
  %K = linalg.range %c0:%arg3:%c1 : !linalg.range
  %A = linalg.view %arg0[%I, %K] : !linalg.view<?x?xf32>
  %B = linalg.view %arg0[%K, %J] : !linalg.view<?x?xf32>
  %C = linalg.view %arg0[%I, %J] : !linalg.view<?x?xf32>
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return
}
// TILE-2-LABEL: func @matmul(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-2: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-2-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-2-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       TILE-2: %[[M:.*]] = linalg.dim %[[A]], 0 : !linalg.view<?x?xf32>
//       TILE-2: linalg.for %i0 = %c0{{.*}} to %[[M]] step %c2 {
//  TILE-2-NEXT:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[M:.*]] = linalg.dim %[[A]], 0 : !linalg.view<?x?xf32>
//  TILE-2-NEXT:   %[[cmpuba:.*]] = cmpi "slt", %[[M]], %[[a]] : index
//  TILE-2-NEXT:   %[[uba:.*]] = select %[[cmpuba]], %[[M]], %[[a]] : index
//  TILE-2-NEXT:   %[[ra:.*]] = linalg.range %i0:%[[uba]]:%c1 : !linalg.range
//       TILE-2:   %[[sAi:.*]] = linalg.slice %[[A]][%[[ra]], {{.*}}] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//
//       TILE-2:   %[[c:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[M:.*]] = linalg.dim %[[C]], 0 : !linalg.view<?x?xf32>
//  TILE-2-NEXT:   %[[cmpubc:.*]] = cmpi "slt", %[[M]], %[[c]] : index
//  TILE-2-NEXT:   %[[ubc:.*]] = select %[[cmpubc]], %[[M]], %[[c]] : index
//  TILE-2-NEXT:   %[[rc:.*]] = linalg.range %i0:%[[ubc]]:%c1 : !linalg.range
//       TILE-2:   %[[sCi:.*]] = linalg.slice %[[C]][%[[rc]], {{.*}}] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//
//  TILE-2-NEXT:   linalg.matmul(%[[sAi]], %[[B]], %[[sCi]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

// TILE-02-LABEL: func @matmul(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-02: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-02-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-02-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       TILE-02: %[[N:.*]] = linalg.dim %[[B]], 1 : !linalg.view<?x?xf32>
//       TILE-02: linalg.for %i0 = %c0 to %[[N]] step %c2 {
//       TILE-02:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-02-NEXT:   %[[N:.*]] = linalg.dim %[[B]], 1 : !linalg.view<?x?xf32>
//  TILE-02-NEXT:   %[[cmpubb:.*]] = cmpi "slt", %[[N]], %[[b]] : index
//  TILE-02-NEXT:   %[[ubb:.*]] = select %[[cmpubb]], %[[N]], %[[b]] : index
//  TILE-02-NEXT:   %[[rb:.*]] = linalg.range %i0:%[[ubb]]:%c1 : !linalg.range
//       TILE-02:   %[[sBj:.*]] = linalg.slice %[[B]][%{{.*}}, %[[rb]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//
//       TILE-02:   %[[c:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-02:   %[[N:.*]] = linalg.dim %[[C]], 1 : !linalg.view<?x?xf32>
//  TILE-02-NEXT:   %[[cmpubc:.*]] = cmpi "slt", %[[N]], %[[c]] : index
//  TILE-02-NEXT:   %[[ubc:.*]] = select %[[cmpubc]], %[[N]], %[[c]] : index
//  TILE-02-NEXT:   %[[rc:.*]] = linalg.range %i0:%[[ubc]]:%c1 : !linalg.range
//  TILE-02-NEXT:   %[[sCj:.*]] = linalg.slice %[[C]][%{{.*}}, %[[rc]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//
//       TILE-02:   linalg.matmul(%[[A]], %[[sBj]], %[[sCj]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

// TILE-002-LABEL: func @matmul(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-002: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-002-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-002-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       TILE-002: %[[K:.*]] = linalg.dim %[[A]], 1 : !linalg.view<?x?xf32>
//       TILE-002: linalg.for %i0 = %c0{{.*}} to %[[K]] step %c2 {
//       TILE-002:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-002-NEXT:   %[[K:.*]] = linalg.dim %[[A]], 1 : !linalg.view<?x?xf32>
//  TILE-002-NEXT:   %[[cmpuba:.*]] = cmpi "slt", %[[K]], %[[a]] : index
//  TILE-002-NEXT:   %[[uba:.*]] = select %[[cmpuba]], %[[K]], %[[a]] : index
//  TILE-002-NEXT:   %[[ra:.*]] = linalg.range %i0:%[[uba]]:%c1 : !linalg.range
//  TILE-002-NEXT:   %[[sAj:.*]] = linalg.slice %[[A]][%{{.*}}, %[[ra]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//
//       TILE-002:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-002-NEXT:   %[[K:.*]] = linalg.dim %[[B]], 0 : !linalg.view<?x?xf32>
//  TILE-002-NEXT:   %[[cmpubb:.*]] = cmpi "slt", %[[K]], %[[b]] : index
//  TILE-002-NEXT:   %[[ubb:.*]] = select %[[cmpubb]], %[[K]], %[[b]] : index
//  TILE-002-NEXT:   %[[rb:.*]] = linalg.range %i0:%[[ubb]]:%c1 : !linalg.range
//       TILE-002:   %[[sBj:.*]] = linalg.slice %[[B]][%[[rb]], %{{.*}}] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//
//       TILE-002:   linalg.matmul(%[[sAj]], %[[sBj]], %[[C]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

// TILE-234-LABEL: func @matmul(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-234: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-234-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-234-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       TILE-234: %[[M:.*]] = linalg.dim %[[A]], 0 : !linalg.view<?x?xf32>
//       TILE-234: %[[K:.*]] = linalg.dim %[[A]], 1 : !linalg.view<?x?xf32>
//       TILE-234: %[[N:.*]] = linalg.dim %[[B]], 1 : !linalg.view<?x?xf32>
//       TILE-234:  linalg.for %i0 = %c0{{.*}} to %[[M]] step %c2 {
//  TILE-234-NEXT:    linalg.for %i1 = %c0{{.*}} to %[[N]] step %c3 {
//  TILE-234-NEXT:      linalg.for %i2 = %c0{{.*}} to %[[K]] step %c4 {
//  TILE-234-NEXT:        %[[ai:.*]]  = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:        %[[M:.*]] = linalg.dim %[[A]], 0 : !linalg.view<?x?xf32>
//  TILE-234-NEXT:        %[[cmpubai:.*]] = cmpi "slt", %[[M]], %[[ai]] : index
//  TILE-234-NEXT:        %[[ubai:.*]] = select %[[cmpubai]], %[[M]], %[[ai]] : index
//  TILE-234-NEXT:        %[[rai:.*]] = linalg.range %i0:%[[ubai]]:%c1 : !linalg.range
//
//  TILE-234-NEXT:        %[[ak:.*]] = affine.apply #[[UB2]](%i2)
//  TILE-234-NEXT:        %[[K:.*]] = linalg.dim %[[A]], 1 : !linalg.view<?x?xf32>
//  TILE-234-NEXT:        %[[cmpubak:.*]] = cmpi "slt", %[[K]], %[[ak]] : index
//  TILE-234-NEXT:        %[[ubak:.*]] = select %[[cmpubak]], %[[K]], %[[ak]] : index
//  TILE-234-NEXT:        %[[rak:.*]] = linalg.range %i2:%[[ubak]]:%c1 : !linalg.range
//  TILE-234-NEXT:        %[[sAik:.*]] = linalg.slice %[[A]][%[[rai]], %[[rak]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//
//  TILE-234-NEXT:        %[[bk:.*]] = affine.apply #[[UB2]](%i2)
//  TILE-234-NEXT:        %[[K:.*]] = linalg.dim %[[B]], 0 : !linalg.view<?x?xf32>
//  TILE-234-NEXT:        %[[cmpubbk:.*]] = cmpi "slt", %[[K]], %[[bk]] : index
//  TILE-234-NEXT:        %[[ubbk:.*]] = select %[[cmpubbk]], %[[K]], %[[bk]] : index
//  TILE-234-NEXT:        %[[rbk:.*]] = linalg.range %i2:%[[ubbk]]:%c1 : !linalg.range
//
//  TILE-234-NEXT:        %[[bj:.*]] = affine.apply #[[UB1]](%i1)
//  TILE-234-NEXT:        %[[N:.*]] = linalg.dim %[[B]], 1 : !linalg.view<?x?xf32>
//  TILE-234-NEXT:        %[[cmpubbj:.*]] = cmpi "slt", %[[N]], %[[bj]] : index
//  TILE-234-NEXT:        %[[ubbj:.*]] = select %[[cmpubbj]], %[[N]], %[[bj]] : index
//  TILE-234-NEXT:        %[[rbj:.*]] = linalg.range %i1:%[[ubbj]]:%c1 : !linalg.range
//  TILE-234-NEXT:        %[[sBkj:.*]] = linalg.slice %[[B]][%[[rbk]], %[[rbj]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//
//  TILE-234-NEXT:        %[[ci:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:        %[[M:.*]] = linalg.dim %[[C]], 0 : !linalg.view<?x?xf32>
//  TILE-234-NEXT:        %[[cmpubci:.*]] = cmpi "slt", %[[M]], %[[ci]] : index
//  TILE-234-NEXT:        %[[ubci:.*]] = select %[[cmpubci]], %[[M]], %[[ci]] : index
//  TILE-234-NEXT:        %[[rci:.*]] = linalg.range %i0:%[[ubci]]:%c1 : !linalg.range
//
//  TILE-234-NEXT:        %[[cj:.*]] = affine.apply #[[UB1]](%i1)
//  TILE-234-NEXT:        %[[N:.*]] = linalg.dim %[[C]], 1 : !linalg.view<?x?xf32>
//  TILE-234-NEXT:        %[[cmpubcj:.*]] = cmpi "slt", %[[N]], %[[cj]] : index
//  TILE-234-NEXT:        %[[ubcj:.*]] = select %[[cmpubcj]], %[[N]], %[[cj]] : index
//  TILE-234-NEXT:        %[[rcj:.*]] = linalg.range %i1:%[[ubcj]]:%c1 : !linalg.range
//  TILE-234-NEXT:        %[[sCij:.*]] = linalg.slice %[[C]][%[[rci]], %[[rcj]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//
//  TILE-234-NEXT:        linalg.matmul(%[[sAik]], %[[sBkj]], %[[sCij]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

func @matvec(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %I = linalg.range %c0:%arg1:%c1 : !linalg.range
  %J = linalg.range %c0:%arg2:%c1 : !linalg.range
  %2 = linalg.view %arg0[%I, %J] : !linalg.view<?x?xf32>
  %3 = linalg.view %arg0[%J] : !linalg.view<?xf32>
  %4 = linalg.view %arg0[%I] : !linalg.view<?xf32>
  linalg.matvec(%2, %3, %4) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>
  return
}
// TILE-2-LABEL: func @matvec(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-2: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-2-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//  TILE-2-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//       TILE-2: %[[M:.*]] = linalg.dim %[[A]], 0 : !linalg.view<?x?xf32>
//       TILE-2: linalg.for %i0 = %c0{{.*}} to %[[M]] step %c2 {
//  TILE-2-NEXT:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[M:.*]] = linalg.dim %[[A]], 0 : !linalg.view<?x?xf32>
//  TILE-2-NEXT:   %[[cmpuba:.*]] = cmpi "slt", %[[M]], %[[a]] : index
//  TILE-2-NEXT:   %[[uba:.*]] = select %[[cmpuba]], %[[M]], %[[a]] : index
//  TILE-2-NEXT:   %[[ra:.*]] = linalg.range %i0:%[[uba]]:%c1 : !linalg.range
//       TILE-2:   %[[sAi:.*]] = linalg.slice %[[A]][%[[ra]], %{{.*}}] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//
//       TILE-2:   %[[c:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[M:.*]] = linalg.dim %[[C]], 0 : !linalg.view<?xf32>
//  TILE-2-NEXT:   %[[cmpubc:.*]] = cmpi "slt", %[[M]], %[[c]] : index
//  TILE-2-NEXT:   %[[ubc:.*]] = select %[[cmpubc]], %[[M]], %[[c]] : index
//  TILE-2-NEXT:   %[[rc:.*]] = linalg.range %i0:%[[ubc]]:%c1 : !linalg.range
//       TILE-2:   %[[sCi:.*]] = linalg.slice %[[C]][%[[rc]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//
//  TILE-2-NEXT:   linalg.matvec(%[[sAi]], %[[B]], %[[sCi]]) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>

// TILE-02-LABEL: func @matvec(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-02: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-02-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//  TILE-02-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//       TILE-02: %[[K:.*]] = linalg.dim %[[A]], 1 : !linalg.view<?x?xf32>
//       TILE-02: linalg.for %i0 = %c0{{.*}} to %[[K]] step %c2 {
//       TILE-02:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-02-NEXT:   %[[K:.*]] = linalg.dim %[[A]], 1 : !linalg.view<?x?xf32>
//  TILE-02-NEXT:   %[[cmpuba:.*]] = cmpi "slt", %[[K]], %[[a]] : index
//  TILE-02-NEXT:   %[[uba:.*]] = select %[[cmpuba]], %[[K]], %[[a]] : index
//  TILE-02-NEXT:   %[[ra:.*]] = linalg.range %i0:%[[uba]]:%c1 : !linalg.range
//  TILE-02-NEXT:   %[[sAj:.*]] = linalg.slice %[[A]][%{{.*}}, %[[ra]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//
//  TILE-02-NEXT:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-02-NEXT:   %[[K:.*]] = linalg.dim %[[B]], 0 : !linalg.view<?xf32>
//  TILE-02-NEXT:   %[[cmpubb:.*]] = cmpi "slt", %[[K]], %[[b]] : index
//  TILE-02-NEXT:   %[[ubb:.*]] = select %[[cmpubb]], %[[K]], %[[b]] : index
//  TILE-02-NEXT:   %[[rb:.*]] = linalg.range %i0:%[[ubb]]:%c1 : !linalg.range
//  TILE-02-NEXT:   %[[sBj:.*]] = linalg.slice %[[B]][%[[rb]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//
//       TILE-02:   linalg.matvec(%[[sAj]], %[[sBj]], %[[C]]) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>

// TILE-002-LABEL: func @matvec(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//   TILE-002-NOT: linalg.for

// TILE-234-LABEL: func @matvec(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-234: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-234-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//  TILE-234-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//       TILE-234: %[[M:.*]] = linalg.dim %[[A]], 0 : !linalg.view<?x?xf32>
//       TILE-234: %[[K:.*]] = linalg.dim %[[A]], 1 : !linalg.view<?x?xf32>
//       TILE-234:  linalg.for %i0 = %c0{{.*}} to %[[M]] step %c2 {
//  TILE-234-NEXT:    linalg.for %i1 = %c0{{.*}} to %[[K]] step %c3 {
//  TILE-234-NEXT:      %[[ai:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:      %[[M:.*]] = linalg.dim %[[A]], 0 : !linalg.view<?x?xf32>
//  TILE-234-NEXT:      %[[cmpubai:.*]] = cmpi "slt", %[[M]], %[[ai]] : index
//  TILE-234-NEXT:      %[[ubai:.*]] = select %[[cmpubai]], %[[M]], %[[ai]] : index
//  TILE-234-NEXT:      %[[rai:.*]] = linalg.range %i0:%[[ubai]]:%c1 : !linalg.range
//
//  TILE-234-NEXT:      %[[aj:.*]] = affine.apply #[[UB1]](%i1)
//  TILE-234-NEXT:      %[[K:.*]] = linalg.dim %[[A]], 1 : !linalg.view<?x?xf32>
//  TILE-234-NEXT:      %[[cmpubaj:.*]] = cmpi "slt", %[[K]], %[[aj]] : index
//  TILE-234-NEXT:      %[[ubaj:.*]] = select %[[cmpubaj]], %[[K]], %[[aj]] : index
//  TILE-234-NEXT:      %[[raj:.*]] = linalg.range %i1:%[[ubaj]]:%c1 : !linalg.range
//  TILE-234-NEXT:      %[[sAij:.*]] = linalg.slice %[[A]][%[[rai]], %[[raj]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//
//  TILE-234-NEXT:      %[[b:.*]] = affine.apply #[[UB1]](%i1)
//  TILE-234-NEXT:      %[[K:.*]] = linalg.dim %[[B]], 0 : !linalg.view<?xf32>
//  TILE-234-NEXT:      %[[cmpubb:.*]] = cmpi "slt", %[[K]], %[[b]] : index
//  TILE-234-NEXT:      %[[ubb:.*]] = select %[[cmpubb]], %[[K]], %[[b]] : index
//  TILE-234-NEXT:      %[[rb:.*]] = linalg.range %i1:%[[ubb]]:%c1 : !linalg.range
//  TILE-234-NEXT:      %[[sB:.*]] = linalg.slice %[[B]][%[[rb]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//
//  TILE-234-NEXT:      %[[c:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:      %[[M:.*]] = linalg.dim %[[C]], 0 : !linalg.view<?xf32>
//  TILE-234-NEXT:      %[[cmpubc:.*]] = cmpi "slt", %[[M]], %[[c]] : index
//  TILE-234-NEXT:      %[[ubc:.*]] = select %[[cmpubc]], %[[M]], %[[c]] : index
//  TILE-234-NEXT:      %[[rc:.*]] = linalg.range %i0:%[[ubc]]:%c1 : !linalg.range
//  TILE-234-NEXT:      %[[sC:.*]] = linalg.slice %[[C]][%[[rc]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//
//  TILE-234-NEXT:      linalg.matvec(%[[sAij]], %[[sB]], %[[sC]]) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>

func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  return
}
// TILE-2-LABEL: func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
//       TILE-2: %[[M:.*]] = linalg.dim %arg0, 0 : !linalg.view<?xf32>
//       TILE-2: linalg.for %i0 = %c0{{.*}} to %[[M]] step %c2 {
//  TILE-2-NEXT:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[M:.*]] = linalg.dim %arg0, 0 : !linalg.view<?xf32>
//  TILE-2-NEXT:   %[[cmpuba:.*]] = cmpi "slt", %[[M]], %[[a]] : index
//  TILE-2-NEXT:   %[[uba:.*]] = select %[[cmpuba]], %[[M]], %[[a]] : index
//  TILE-2-NEXT:   %[[ra:.*]] = linalg.range %i0:%[[uba]]:%c1 : !linalg.range
//       TILE-2:   %[[sAi:.*]] = linalg.slice %arg0[%[[ra]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//
//  TILE-2-NEXT:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[K:.*]] = linalg.dim %arg1, 0 : !linalg.view<?xf32>
//  TILE-2-NEXT:   %[[cmpubb:.*]] = cmpi "slt", %[[K]], %[[b]] : index
//  TILE-2-NEXT:   %[[ubb:.*]] = select %[[cmpubb]], %[[K]], %[[b]] : index
//  TILE-2-NEXT:   %[[rb:.*]] = linalg.range %i0:%[[ubb]]:%c1 : !linalg.range
//  TILE-2-NEXT:   %[[sBi:.*]] = linalg.slice %arg1[%[[rb]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//
//  TILE-2-NEXT:   linalg.dot(%[[sAi]], %[[sBi]], {{.*}}) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>

// TILE-02-LABEL: func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
//   TILE-02-NOT: linalg.for

// TILE-002-LABEL: func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
//   TILE-002-NOT: linalg.for

// TILE-234-LABEL: func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
//       TILE-234: %[[K:.*]] = linalg.dim %arg0, 0 : !linalg.view<?xf32>
//       TILE-234:  linalg.for %i0 = %c0{{.*}} to %[[K]] step %c2 {
//  TILE-234-NEXT:    %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:    %[[K:.*]] = linalg.dim %arg0, 0 : !linalg.view<?xf32>
//  TILE-234-NEXT:    %[[cmpuba:.*]] = cmpi "slt", %[[K]], %[[a]] : index
//  TILE-234-NEXT:    %[[uba:.*]] = select %[[cmpuba]], %[[K]], %[[a]] : index
//  TILE-234-NEXT:    %[[ra:.*]] = linalg.range %i0:%[[uba]]:%c1 : !linalg.range
//  TILE-234-NEXT:    %[[sA:.*]] = linalg.slice %arg0[%[[ra]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//
//  TILE-234-NEXT:    %[[b:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:    %[[K:.*]] = linalg.dim %arg1, 0 : !linalg.view<?xf32>
//  TILE-234-NEXT:    %[[cmpubb:.*]] = cmpi "slt", %[[K]], %[[b]] : index
//  TILE-234-NEXT:    %[[ubb:.*]] = select %[[cmpubb]], %[[K]], %[[b]] : index
//  TILE-234-NEXT:    %[[rb:.*]] = linalg.range %i0:%[[ubb]]:%c1 : !linalg.range
//  TILE-234-NEXT:    %[[sB:.*]] = linalg.slice %arg1[%[[rb]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//
//  TILE-234-NEXT:    linalg.dot(%[[sA]], %[[sB]], %arg2) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
