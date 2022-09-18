#include "TestDialect.h"
#include "Ops.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir::test;

#define GET_OP_CLASSES
#include "dialect/Test.cpp.inc"
