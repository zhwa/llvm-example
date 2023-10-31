#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/InitAllDialects.h"

using namespace mlir;
using namespace mlir::linalg;

int main() {
    // Define the MLIR context.
    MLIRContext context;

    context.allowUnregisteredDialects(true);
    context.printOpOnDiagnostic(true);


    /*
    // Define the input matrices.
    DenseElementsAttr lhs = DenseElementsAttr::get(
        RankedTensorType::get({2, 2}, IntegerType::get(&context, 32)),
        ArrayRef<int32_t>{1, 2, 3, 4});
    DenseElementsAttr rhs = DenseElementsAttr::get(
        RankedTensorType::get({2, 2}, IntegerType::get(&context, 32)),
        ArrayRef<int32_t>{5, 6, 7, 8});

    // Define the output matrix.
    RankedTensorType resultType =
        RankedTensorType::get({2, 2}, IntegerType::get(&context, 32));
    DenseElementsAttr result = DenseElementsAttr::get(resultType, 0);

    // Define the 'linalg.matmul' operation.
    OpBuilder builder(&context);
    auto matmulOp = builder.create<linalg::MatmulOp>(UnknownLoc::get(&context), ValueRange{lhs, rhs}, ValueRange{result});

    // Print the IR.
    matmulOp.print(llvm::outs());
    */

    // Define the input matrix.
    DenseElementsAttr input = DenseElementsAttr::get(
        RankedTensorType::get({2, 2}, IntegerType::get(&context, 32)),
        ArrayRef<int32_t>{1, 2, 3, 4});

    // Define the 'linalg.matmul' operation.
    OpBuilder builder(&context);
    auto constantOp = builder.create<ConstantOp>(
        /*location=*/UnknownLoc::get(&context),
        /*resultType=*/RankedTensorType::get({2, 2},
            IntegerType::get(&context, 32)),
        /*value=*/input);

    // Get the output value.
    Value output = constantOp.getResult();

    // Print the IR.
    output.print(llvm::outs());

    return 0;
}