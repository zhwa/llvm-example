# Linalg GEMM Example

This example demonstrates how to use MLIR's **Linalg dialect** for matrix multiplication (GEMM - General Matrix Multiply).

## What You'll Learn

1. **linalg.matmul**: High-level structured operation for matrix multiplication
2. **linalg.batch_matmul**: Batched matrix operations
3. **linalg.generic**: Flexible way to express structured computations
4. **Indexing maps**: How to define memory access patterns
5. **Iterator types**: Parallel vs reduction dimensions

## Key Concepts

### Linalg Dialect Philosophy

The Linalg (Linear Algebra) dialect provides:
- **Structured operations**: Operations with well-defined mathematical semantics
- **Transformation infrastructure**: Rich set of transformations (tiling, fusion, etc.)
- **Target independence**: Can lower to loops, vectorized code, or library calls
- **Composability**: Works well with other MLIR dialects

### Matrix Multiplication

```mlir
// High-level operation
%C = linalg.matmul ins(%A, %B : tensor<MxKxf32>, tensor<KxNxf32>)
                   outs(%C_init : tensor<MxNxf32>) -> tensor<MxNxf32>
```

This represents: `C[i,j] = Σ_k A[i,k] * B[k,j]`

### Lowering Path

```
linalg.matmul (high-level)
      ↓
affine.for (polyhedral loops)
      ↓
scf.for (structured control flow)
      ↓
cf.br (control flow)
      ↓
LLVM dialect
```

## Building and Running

```powershell
# Build the project
cmake --build build --target linalg-gemm-example

# Run
.\build\bin\linalg-gemm-example.exe
```

## Example Output

The program generates MLIR code showing three ways to express matrix multiplication:

1. **Standard matmul**: Simple 2D matrix multiplication
2. **Batched matmul**: Multiple matrices multiplied in parallel
3. **Generic matmul**: Explicit structure with indexing maps

## Next Steps

After understanding this example, try:

1. **Add tiling**: Break large matrices into smaller tiles
   ```cpp
   pm.addPass(createLinalgTilingPass({tileSize, tileSize}));
   ```

2. **Lower to loops**: See the actual loop structure
   ```cpp
   pm.addPass(createConvertLinalgToLoopsPass());
   ```

3. **Vectorize**: Generate SIMD code
   ```cpp
   pm.addPass(createLinalgVectorizationPass());
   ```

4. **Generate code**: Compile to executable
   ```cpp
   pm.addPass(createConvertLinalgToLLVMPass());
   ```

## Related Chapters

- **Chapter 6**: Partial lowering to Affine (linalg → affine)
- **Chapter 7**: Complete lowering to LLVM
- **Chapter 9**: Designing custom dialects (Linalg as case study)

## References

- [Linalg Dialect Documentation](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [Linalg Transformations](https://mlir.llvm.org/docs/Dialects/Linalg/#transformations)
- [Structured Operations](https://mlir.llvm.org/docs/Dialects/Linalg/#structured-operations)
