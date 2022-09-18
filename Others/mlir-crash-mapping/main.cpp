#include "dialect/TestDialect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include <iostream>

int main() {
    mlir::MLIRContext context;
    context.loadDialect<mlir::arith::ArithmeticDialect>();
    context.loadDialect<mlir::test::TestDialect>();

    mlir::OpBuilder builder(&context);
    mlir::Location loc = builder.getUnknownLoc();

    mlir::Value intOperand = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(57));
    auto testOp = builder.create<mlir::test::TestOp>(loc, intOperand.getType(), intOperand);

    auto floatValue = builder.create<mlir::arith::ConstantOp>(loc, builder.getF64FloatAttr(57));

    mlir::BlockAndValueMapping mapping;
    mapping.map(intOperand, floatValue);

    mapping.lookup(testOp.typedOperand()).dump();

    // Workaround:
    //mapping.lookup(mlir::Value(testOp.typedOperand())).dump();

    return 0;
}
