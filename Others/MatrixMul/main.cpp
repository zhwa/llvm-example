#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/InitAllDialects.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::arith;
using namespace mlir::func;

int main() {
    // Define the MLIR context.
    DialectRegistry registry;
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();

    MLIRContext context(registry);

    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<linalg::LinalgDialect>();
    context.allowUnregisteredDialects(true);
    context.printOpOnDiagnostic(true);

    // Create a MLIR module.
    auto module = ModuleOp::create(UnknownLoc::get(&context));

    // Define the 'linalg.matmul' operation.
    OpBuilder builder(&context);

    // Define a MLIR function that multiplies two matrices using the linalg dialect.
    auto func = FuncOp::create(builder.getUnknownLoc(), "matmul",
    builder.getFunctionType({RankedTensorType::get({2, 2}, builder.getF32Type()),
                             RankedTensorType::get({2, 2}, builder.getF32Type())},
                            {RankedTensorType::get({2, 2}, builder.getF32Type())}));

    // Define the input matrices.
    DenseElementsAttr lhs = DenseElementsAttr::get(RankedTensorType::get({2, 2}, IntegerType::get(&context, 32)), ArrayRef<int32_t>{1, 2, 3, 4}); 
    DenseElementsAttr rhs = DenseElementsAttr::get(RankedTensorType::get({2, 2}, IntegerType::get(&context, 32)), ArrayRef<int32_t>{5, 6, 7, 8});

    // Define the output matrix.
    RankedTensorType resultType = RankedTensorType::get({2, 2}, IntegerType::get(&context, 32));
    DenseElementsAttr result = DenseElementsAttr::get(resultType, 0);


    auto lhsOp = builder.create<mlir::arith::ConstantOp>(UnknownLoc::get(&context), RankedTensorType::get({2, 2}, IntegerType::get(&context, 32)), lhs);
    auto rhsOp = builder.create<mlir::arith::ConstantOp>(UnknownLoc::get(&context), RankedTensorType::get({2, 2}, IntegerType::get(&context, 32)), rhs);
    auto resultOp = builder.create<mlir::arith::ConstantOp>(UnknownLoc::get(&context), RankedTensorType::get({2, 2}, IntegerType::get(&context, 32)), result);

    auto matmulOp = builder.create<linalg::MatmulOp>(UnknownLoc::get(&context), ValueRange{lhsOp, rhsOp}, ValueRange{resultOp});

    // Get the output value and print IR.
    Value lhsOutput = lhsOp.getResult();
    Value rhsOutput = rhsOp.getResult();
    Value matmulOutput = matmulOp.getResult(0);
    lhsOutput.print(llvm::outs());
    rhsOutput.print(llvm::outs());
    matmulOutput.print(llvm::outs());

    return 0;
}