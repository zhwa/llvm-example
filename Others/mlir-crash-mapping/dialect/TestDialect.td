#ifndef TESTDIALECT_TD
#define TESTDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Modelica dialect
//===----------------------------------------------------------------------===//

def Test_Dialect : Dialect {
    let name = "test";
    let summary = "Test dialect.";

    let description = [{
        Test dialect.
    }];

    let cppNamespace = "::mlir::test";
}

#endif // TESTDIALECT_TD
