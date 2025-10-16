//===- gemm_example.cpp - Linalg GEMM Example ----------------------------===//
//
// This file demonstrates how to use MLIR's Linalg dialect for matrix
// multiplication (GEMM - General Matrix Multiply).
//
// Learning objectives:
// 1. Create linalg.matmul operations
// 2. Understand structured operations in Linalg
// 3. Lower Linalg operations to affine loops
// 4. Apply optimizations (tiling, fusion)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// Helper: Create a simple matmul function
//===----------------------------------------------------------------------===//

/// Creates a function that performs C = A * B using linalg.matmul
/// 
/// Generated MLIR:
/// ```
/// func.func @matmul(%A: tensor<MxKxf32>, %B: tensor<KxNxf32>, 
///                   %C: tensor<MxNxf32>) -> tensor<MxNxf32> {
///   %result = linalg.matmul ins(%A, %B : tensor<MxKxf32>, tensor<KxNxf32>)
///                           outs(%C : tensor<MxNxf32>) -> tensor<MxNxf32>
///   return %result : tensor<MxNxf32>
/// }
/// ```
func::FuncOp createMatmulFunction(OpBuilder &builder, Location loc, int64_t M, int64_t K, int64_t N)
{
  auto f32Type = builder.getF32Type();

  // Create tensor types for matrices
  auto matrixAType = RankedTensorType::get({M, K}, f32Type);
  auto matrixBType = RankedTensorType::get({K, N}, f32Type);
  auto matrixCType = RankedTensorType::get({M, N}, f32Type);

  // Create function type: (A, B, C_init) -> C_result
  auto funcType = builder.getFunctionType(
      {matrixAType, matrixBType, matrixCType},
      {matrixCType}
  );

  // Create function
  auto func = builder.create<func::FuncOp>(loc, "matmul", funcType);

  // Create function body
  Block* entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Get function arguments
  Value A = entryBlock->getArgument(0);
  Value B = entryBlock->getArgument(1);
  Value C = entryBlock->getArgument(2);

  // Create linalg.matmul operation
  // This is the high-level operation that represents:
  // C[i,j] = Σ_k A[i,k] * B[k,j]
  auto matmulOp = builder.create<linalg::MatmulOp>(
      loc,
      TypeRange{matrixCType},  // Result type
      ValueRange{A, B},         // Inputs (ins)
      ValueRange{C}             // Output (outs)
  );

  // Return result
  builder.create<func::ReturnOp>(loc, matmulOp.getResult(0));
  return func;
}

//===----------------------------------------------------------------------===//
// Helper: Create a batched matmul function
//===----------------------------------------------------------------------===//

/// Creates a function for batched matrix multiplication
/// C[b,i,j] = Σ_k A[b,i,k] * B[b,k,j] for each batch b
func::FuncOp createBatchMatmulFunction(OpBuilder &builder, Location loc,
                                        int64_t batch, int64_t M, int64_t K, int64_t N) {
  auto f32Type = builder.getF32Type();

  // 3D tensors for batched operations
  auto matrixAType = RankedTensorType::get({batch, M, K}, f32Type);
  auto matrixBType = RankedTensorType::get({batch, K, N}, f32Type);
  auto matrixCType = RankedTensorType::get({batch, M, N}, f32Type);

  auto funcType = builder.getFunctionType(
      {matrixAType, matrixBType, matrixCType},
      {matrixCType}
  );

  auto func = builder.create<func::FuncOp>(loc, "batch_matmul", funcType);
  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  Value A = entryBlock->getArgument(0);
  Value B = entryBlock->getArgument(1);
  Value C = entryBlock->getArgument(2);

  // linalg.batch_matmul handles the batch dimension automatically
  auto batchMatmulOp = builder.create<linalg::BatchMatmulOp>(
      loc,
      TypeRange{matrixCType},
      ValueRange{A, B},
      ValueRange{C}
  );

  builder.create<func::ReturnOp>(loc, batchMatmulOp.getResult(0));

  return func;
}

//===----------------------------------------------------------------------===//
// Helper: Create a generic linalg operation (manual matmul)
//===----------------------------------------------------------------------===//

/// Creates a matmul using linalg.generic (more flexible, lower-level)
/// This shows the structure of linalg operations explicitly
func::FuncOp createGenericMatmulFunction(OpBuilder &builder, Location loc,
                                          int64_t M, int64_t K, int64_t N) {
  auto f32Type = builder.getF32Type();

  auto matrixAType = RankedTensorType::get({M, K}, f32Type);
  auto matrixBType = RankedTensorType::get({K, N}, f32Type);
  auto matrixCType = RankedTensorType::get({M, N}, f32Type);

  auto funcType = builder.getFunctionType(
      {matrixAType, matrixBType, matrixCType},
      {matrixCType}
  );

  auto func = builder.create<func::FuncOp>(loc, "generic_matmul", funcType);
  Block *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  Value A = entryBlock->getArgument(0);
  Value B = entryBlock->getArgument(1);
  Value C = entryBlock->getArgument(2);

  // Define indexing maps for matmul: C[i,j] = Σ_k A[i,k] * B[k,j]
  // - Map 0: (i, j, k) -> (i, k)  for A
  // - Map 1: (i, j, k) -> (k, j)  for B
  // - Map 2: (i, j, k) -> (i, j)  for C
  auto context = builder.getContext();

  auto map0 = AffineMap::get(3, 0, {builder.getAffineDimExpr(0), 
                                     builder.getAffineDimExpr(2)}, context);
  auto map1 = AffineMap::get(3, 0, {builder.getAffineDimExpr(2), 
                                     builder.getAffineDimExpr(1)}, context);
  auto map2 = AffineMap::get(3, 0, {builder.getAffineDimExpr(0), 
                                     builder.getAffineDimExpr(1)}, context);

  SmallVector<AffineMap> indexingMaps = {map0, map1, map2};

  // Iterator types: parallel, parallel, reduction
  SmallVector<utils::IteratorType> iteratorTypes = {
      utils::IteratorType::parallel,
      utils::IteratorType::parallel,
      utils::IteratorType::reduction
  };

  // Create linalg.generic operation
  auto genericOp = builder.create<linalg::GenericOp>(
      loc,
      TypeRange{matrixCType},           // Result type
      ValueRange{A, B},                  // Inputs
      ValueRange{C},                     // Outputs
      indexingMaps,                      // Indexing maps
      iteratorTypes,                     // Iterator types
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Body: multiply and accumulate
        // args[0] = A[i,k], args[1] = B[k,j], args[2] = C[i,j]
        Value mul = b.create<arith::MulFOp>(loc, args[0], args[1]);
        Value add = b.create<arith::AddFOp>(loc, args[2], mul);
        b.create<linalg::YieldOp>(loc, add);
      }
  );

  builder.create<func::ReturnOp>(loc, genericOp.getResult(0));

  return func;
}

//===----------------------------------------------------------------------===//
// Main: Demonstrate Linalg GEMM
//===----------------------------------------------------------------------===//

int main() {
  // 1. Setup MLIR context and dialects
  llvm::outs() << "=== Linalg GEMM Example ===\n\n";

  MLIRContext context;
  context.loadDialect<arith::ArithDialect, 
                      func::FuncDialect,
                      linalg::LinalgDialect,
                      memref::MemRefDialect,
                      tensor::TensorDialect>();

  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  // 2. Create module
  auto module = ModuleOp::create(loc);

  // 3. Create different matmul variants
  llvm::outs() << "Creating matmul functions...\n";

  // Standard matmul: 4x8 * 8x4 = 4x4
  // Set insertion point to module body before each function
  builder.setInsertionPointToEnd(module.getBody());
  createMatmulFunction(builder, loc, 4, 8, 4);

  // Batched matmul: [2, 4x8] * [2, 8x4] = [2, 4x4]
  builder.setInsertionPointToEnd(module.getBody());
  createBatchMatmulFunction(builder, loc, 2, 4, 8, 4);

  // Generic matmul (explicit structure): 4x8 * 8x4 = 4x4
  builder.setInsertionPointToEnd(module.getBody());
  createGenericMatmulFunction(builder, loc, 4, 8, 4);

  // 4. Print the generated MLIR
  llvm::outs() << "\n=== Generated MLIR (High-Level) ===\n\n";
  module.print(llvm::outs());

  // 5. Apply transformations (optional - shows optimization pipeline)
  llvm::outs() << "\n\n=== Applying Linalg Optimizations ===\n";

  PassManager pm(&context);

  // Add some standard passes
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Note: To see loop lowering, you would add:
  // pm.addPass(createConvertLinalgToLoopsPass());
  // pm.addPass(createConvertLinalgToAffineLoopsPass());

  if (failed(pm.run(module))) {
    llvm::errs() << "Pass pipeline failed\n";
    return 1;
  }

  llvm::outs() << "\n=== Optimized MLIR ===\n\n";
  module.print(llvm::outs());

  llvm::outs() << "\n\n=== Key Concepts Demonstrated ===\n";
  llvm::outs() << "1. linalg.matmul: High-level matrix multiplication\n";
  llvm::outs() << "2. linalg.batch_matmul: Batched operations\n";
  llvm::outs() << "3. linalg.generic: Flexible structured operations\n";
  llvm::outs() << "4. Indexing maps: Define access patterns\n";
  llvm::outs() << "5. Iterator types: parallel vs reduction dimensions\n";
  llvm::outs() << "\n=== Next Steps ===\n";
  llvm::outs() << "- Add tiling transformation for better cache locality\n";
  llvm::outs() << "- Lower to affine loops or SCF loops\n";
  llvm::outs() << "- Vectorize inner loops\n";
  llvm::outs() << "- Generate executable code\n";

  return 0;
}