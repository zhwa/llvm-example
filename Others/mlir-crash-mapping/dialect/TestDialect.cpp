#include "TestDialect.h"
#include "Ops.h"
#include "mlir/IR/DialectImplementation.h"

using namespace ::mlir;

#include "dialect/TestDialect.cpp.inc"

namespace mlir::test
{
    //===----------------------------------------------------------------------===//
    // Modelica dialect
    //===----------------------------------------------------------------------===//

    void TestDialect::initialize()
    {
        addOperations<
#define GET_OP_LIST
#include "dialect/Test.cpp.inc"
        >();
    }
}
