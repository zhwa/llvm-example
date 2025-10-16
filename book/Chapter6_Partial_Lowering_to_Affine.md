# Chapter 6: Partial Lowering to Affine Dialect

> *"There are only two hard things in Computer Science: cache invalidation and naming things and off-by-one errors." - Phil Karlton (with additions)*

## Introduction

Up until now, we've worked entirely within the Toy dialect:
- **Chapter 3**: Created Toy operations
- **Chapter 4**: Transformed Toy to better Toy
- **Chapter 5**: Inferred shapes within Toy

But eventually, we need to **execute** our code. This means transforming high-level Toy operations into something a CPU (or GPU) can run.

Enter **lowering** - the process of transforming from one dialect to a lower-level dialect.

In this chapter, you'll learn:

- What lowering means and why it's gradual
- The Affine dialect and why it's special
- **Dialect conversion framework** - the infrastructure for lowering
- How to lower Toy operations to loops and memory operations
- Partial lowering - mixing dialects at different levels

By the end, you'll see MLIR's **multi-level** philosophy in action!

Let's begin the descent from high-level to machine code! 🚀

---

## 6.1 The Multi-Level Philosophy

### What Is Lowering?

**Lowering** = transforming from higher abstraction to lower abstraction.

**Example progression:**
```
High Level:  c = a * b                (Toy: semantic operations)
              ↓
Mid Level:   for i, j: c[i,j] = ...  (Affine: loops + memory)
              ↓
Low Level:   %addr = add %base, %off  (LLVM: instructions)
              ↓
Machine:     imul rax, rbx            (x86: CPU instructions)
```

Each level is easier to analyze and optimize than the one below it.

### Why NOT Lower All at Once?

You might wonder: why not go directly from Toy to machine code?

**Consider matrix multiply:**

```toy
var c = a * b;
```

**Direct lowering would be a mess:**
- Have to think about loops AND registers AND memory layout simultaneously
- Can't reuse optimizations (each operation needs custom logic)
- Hard to reason about correctness
- Lose opportunities for high-level optimizations

**Progressive lowering is cleaner:**

```
Toy → Affine → SCF → LLVM → Machine Code
```

Each stage:
- ✅ Has clear responsibilities
- ✅ Enables stage-specific optimizations
- ✅ Is independently testable
- ✅ Can be reused for different source languages

### The Abstraction Ladder

Think of dialects as rungs on a ladder:

```
┌─────────────┐
│ Toy Dialect │  ← High-level: multiply, transpose, reshape
├─────────────┤
│   Affine    │  ← Loop-level: perfect nests, compile-time analysis
├─────────────┤
│     SCF     │  ← Control-flow: arbitrary loops, conditionals
├─────────────┤
│    LLVM     │  ← ISA-agnostic: virtual registers, basic blocks
├─────────────┤
│   Target    │  ← Machine code: x86, ARM, RISC-V
└─────────────┘
```

**Each rung:**
- More concrete than above
- More abstract than below
- Bridges the gap between levels

This chapter: **Toy → Affine**

---

## 6.2 The Affine Dialect

### What Is Affine?

The **Affine dialect** represents loops with special properties that enable powerful compile-time analysis.

**Key concepts:**
- **Affine loops**: Loops with linear induction variables
- **Affine memory**: Accesses with affine subscripts
- **Compile-time bounds**: Loop bounds known at compile time (or parametric)

### Why Affine Matters

Affine loops are **analyzable**. The compiler can:
- Determine if memory accesses overlap
- Reorder loops safely
- Fuse loops together
- Vectorize iterations
- Generate parallel code

**Example:**
```mlir
affine.for %i = 0 to 10 {
  affine.for %j = 0 to 20 {
    %v = affine.load %A[%i, %j] : memref<10x20xf64>
    %result = arith.mulf %v, %c : f64
    affine.store %result, %B[%i, %j] : memref<10x20xf64>
  }
}
```

The compiler **knows**:
- Loop bounds: `i ∈ [0, 10)`, `j ∈ [0, 20)`
- Memory access pattern: `A[i, j]` and `B[i, j]`
- No aliasing: `A` and `B` don't overlap (different memrefs)
- Perfect nesting: No statements between loop levels

This enables **polyhedral optimization** - powerful loop transformations!

### Affine Operations

**Key operations in Affine dialect:**

```mlir
// Loops
affine.for %i = <lb> to <ub> step <step> {
  // body
}

// Memory access
%v = affine.load %memref[%i, %j] : memref<NxMxf64>
affine.store %value, %memref[%i, %j] : memref<NxMxf64>

// Conditionals (with affine conditions)
affine.if #set(%i, %j) {
  // then branch
} else {
  // else branch
}

// Apply affine expressions
%v = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%i, %j)
```

### What's Affine About Them?

An **affine expression** is a linear function:

$$f(i_1, i_2, ..., i_n) = c_0 + c_1 \cdot i_1 + c_2 \cdot i_2 + ... + c_n \cdot i_n$$

**Examples:**
- ✅ `i + j`
- ✅ `2 * i - 3 * j + 5`
- ✅ `i * 4 + j * 2 + 10`
- ❌ `i * j` (not linear!)
- ❌ `i * i` (not linear!)
- ❌ `sqrt(i)` (not linear!)

**Why restrict to affine?**
Affine expressions can be analyzed efficiently using integer linear programming and polyhedral mathematics.

### From Tensors to MemRefs

Lowering to Affine also changes **types**:

**Before (Toy):**
```mlir
%result = toy.mul %a, %b : tensor<2x3xf64>
```

**After (Affine):**
```mlir
%a_mem = memref.alloc() : memref<2x3xf64>
%b_mem = memref.alloc() : memref<2x3xf64>
%result_mem = memref.alloc() : memref<2x3xf64>
// ... populate memrefs ...
memref.dealloc %result_mem : memref<2x3xf64>
```

**Tensor vs. MemRef:**

| Tensor | MemRef |
|--------|--------|
| Value semantics | Reference semantics |
| Immutable | Mutable |
| SSA (single assignment) | Can be updated |
| Abstract | Concrete memory |
| No aliasing | Can alias |

**Analogy:**
- **Tensor** = `const int x = 5;` (value)
- **MemRef** = `int* ptr = malloc(...);` (pointer)

---

## 6.3 The Dialect Conversion Framework

MLIR provides infrastructure for lowering: the **Dialect Conversion Framework**.

### Key Concepts

**1. ConversionTarget**
Specifies which operations are legal after conversion:
```cpp
ConversionTarget target(getContext());
target.addLegalDialect<affine::AffineDialect>();
target.addIllegalDialect<toy::ToyDialect>();
```

**2. ConversionPattern**
Defines how to transform operations:
```cpp
struct AddOpLowering : public ConversionPattern {
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const;
};
```

**3. TypeConverter**
Handles type transformations (e.g., tensor → memref):
```cpp
MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}
```

### The Conversion Process

```
1. Define target: What's legal after conversion?
2. Register patterns: How to transform each operation
3. Apply conversion: Run the transformation
4. Verify: Check all illegal ops were converted
```

If any "illegal" operation remains, the conversion **fails**. This ensures completeness!

---

## 6.4 Lowering Strategy

Let's plan our lowering from Toy to Affine.

### What to Lower

**Operations we'll lower:**
- ✅ `toy.add` → affine loops + `arith.addf`
- ✅ `toy.mul` → affine loops + `arith.mulf`
- ✅ `toy.transpose` → affine loops with swapped indices
- ✅ `toy.constant` → memref allocation + stores
- ✅ `toy.func` → `func.func` (standard function)
- ✅ `toy.return` → `func.return`

**Operations we'll keep (for now):**
- ⏸️ `toy.print` → update operands, but keep operation

### The General Pattern

Most Toy operations follow this template:

```
1. Allocate memref for result
2. Create nested affine loops (one per dimension)
3. In innermost loop:
   - Load operands at current indices
   - Compute result
   - Store result at current indices
4. Replace Toy operation with the memref
```

**Visual representation:**
```
toy.add %a, %b : tensor<2x3xf64>
           ↓
%result = memref.alloc() : memref<2x3xf64>
affine.for %i = 0 to 2 {
  affine.for %j = 0 to 3 {
    %lhs = affine.load %a[%i, %j]
    %rhs = affine.load %b[%i, %j]
    %sum = arith.addf %lhs, %rhs
    affine.store %sum, %result[%i, %j]
  }
}
```

From **one operation** to **~10 operations + nested structure**!

---

## 6.5 Lowering Binary Operations

Let's start with the simplest: `toy.add` and `toy.mul`.

### The Template Pattern

Both operations are element-wise, so they share the same structure:

```cpp
template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     // Get the operands
                     typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

                     // Load left and right operands
                     auto loadedLhs = builder.create<affine::AffineLoadOp>(
                         loc, binaryAdaptor.getLhs(), loopIvs);
                     auto loadedRhs = builder.create<affine::AffineLoadOp>(
                         loc, binaryAdaptor.getRhs(), loopIvs);

                     // Create the arithmetic operation
                     return builder.create<LoweredBinaryOp>(loc, loadedLhs,
                                                            loadedRhs);
                   });
    return success();
  }
};
```

**Breaking it down:**

### Step 1: The Template

```cpp
template <typename BinaryOp, typename LoweredBinaryOp>
```

This is a **C++ template** - one pattern works for multiple ops!

**Instantiations:**
```cpp
using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;
```

### Step 2: The Adaptor

```cpp
typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);
```

**What's an Adaptor?**
- Generated by TableGen
- Provides named accessors for operands
- Handles type changes (tensor → memref)

**Without adaptor:**
```cpp
Value lhs = memRefOperands[0];  // Which one is lhs? Have to remember!
Value rhs = memRefOperands[1];
```

**With adaptor:**
```cpp
Value lhs = binaryAdaptor.getLhs();  // Clear and self-documenting!
Value rhs = binaryAdaptor.getRhs();
```

### Step 3: Load Operands

```cpp
auto loadedLhs = builder.create<affine::AffineLoadOp>(
    loc, binaryAdaptor.getLhs(), loopIvs);
```

**What this does:**
```mlir
%lhs = affine.load %memref_lhs[%i, %j] : memref<2x3xf64>
```

- `binaryAdaptor.getLhs()`: The memref to load from
- `loopIvs`: The indices (`[%i, %j]`)
- Returns: The scalar value at that location

### Step 4: Compute Result

```cpp
return builder.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
```

**For AddOp:**
```mlir
%result = arith.addf %loadedLhs, %loadedRhs : f64
```

**For MulOp:**
```mlir
%result = arith.mulf %loadedLhs, %loadedRhs : f64
```

This is the **scalar** operation on individual elements!

### The Helper: lowerOpToLoops

The template calls `lowerOpToLoops`. Let's see what it does:

```cpp
static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = cast<RankedTensorType>((*op->result_type_begin()));
  auto loc = op->getLoc();

  // 1. Allocate memref for result
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // 2. Create nested affine loops
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), 0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // 3. Call the callback to process each iteration
        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        
        // 4. Store the result
        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc, ivs);
      });

  // 5. Replace original operation with the memref
  rewriter.replaceOp(op, alloc);
}
```

**Step by step:**

#### 1. Allocate MemRef

```cpp
auto memRefType = convertTensorToMemRef(tensorType);
auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
```

**Generated code:**
```mlir
%result = memref.alloc() : memref<2x3xf64>
// ... computation ...
memref.dealloc %result : memref<2x3xf64>
```

**Memory management:**
- `alloc` placed at **beginning** of block
- `dealloc` placed at **end** of block
- Simple but correct (Toy has no control flow)

#### 2. Build Loop Nest

```cpp
SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), 0);
SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);
affine::buildAffineLoopNest(rewriter, loc, lowerBounds, 
                            tensorType.getShape(), steps, ...);
```

**For tensor<2x3xf64>:**
- Rank = 2
- Lower bounds = `[0, 0]`
- Upper bounds = `[2, 3]`
- Steps = `[1, 1]`

**Generated:**
```mlir
affine.for %i = 0 to 2 {
  affine.for %j = 0 to 3 {
    // body
  }
}
```

#### 3. Process Each Iteration

```cpp
Value valueToStore = processIteration(nestedBuilder, operands, ivs);
```

This calls the **callback** provided by the specific operation lowering. For AddOp:
```cpp
[loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs) {
  // Load, add, return result
}
```

#### 4. Store Result

```cpp
nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc, ivs);
```

**Generated:**
```mlir
affine.store %valueToStore, %result[%i, %j] : memref<2x3xf64>
```

### Complete Example: toy.add

**Before:**
```mlir
%result = toy.add %a, %b : tensor<2x3xf64>
```

**After:**
```mlir
%result = memref.alloc() : memref<2x3xf64>
affine.for %i = 0 to 2 {
  affine.for %j = 0 to 3 {
    %lhs = affine.load %a[%i, %j] : memref<2x3xf64>
    %rhs = affine.load %b[%i, %j] : memref<2x3xf64>
    %sum = arith.addf %lhs, %rhs : f64
    affine.store %sum, %result[%i, %j] : memref<2x3xf64>
  }
}
// ... later ...
memref.dealloc %result : memref<2x3xf64>
```

From **1 operation** to **a full loop nest** with loads, compute, and stores!

---

## 6.6 Lowering Transpose

Transpose is more interesting - we need to **swap indices**.

### The Pattern

```cpp
struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     // Get the input memref
                     toy::TransposeOpAdaptor transposeAdaptor(memRefOperands);
                     Value input = transposeAdaptor.getInput();

                     // Transpose = load with reversed indices!
                     SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                     return builder.create<affine::AffineLoadOp>(loc, input,
                                                                 reverseIvs);
                   });
    return success();
  }
};
```

### The Key Insight

```cpp
SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
return builder.create<affine::AffineLoadOp>(loc, input, reverseIvs);
```

**What's happening:**

**Input:** `tensor<2x3xf64>` → stored as `memref<2x3xf64>`
**Output:** `tensor<3x2xf64>` → stored as `memref<3x2xf64>`

**Loop indices:**
- Outer loop: `%i = 0 to 3` (output rows)
- Inner loop: `%j = 0 to 2` (output columns)

**Store at:** `output[i][j]`
**Load from:** `input[j][i]` ← **reversed!**

### Complete Example

**Before:**
```mlir
%result = toy.transpose(%input : tensor<2x3xf64>) to tensor<3x2xf64>
```

**After:**
```mlir
%result = memref.alloc() : memref<3x2xf64>
affine.for %i = 0 to 3 {      // output rows
  affine.for %j = 0 to 2 {    // output columns
    // Load from [j, i] (reversed!)
    %v = affine.load %input[%j, %i] : memref<2x3xf64>
    // Store at [i, j] (normal)
    affine.store %v, %result[%i, %j] : memref<3x2xf64>
  }
}
memref.dealloc %result : memref<3x2xf64>
```

**Tracing through iterations:**

| i | j | Load from | Store to |
|---|---|-----------|----------|
| 0 | 0 | input[0,0] | result[0,0] |
| 0 | 1 | input[1,0] | result[0,1] |
| 1 | 0 | input[0,1] | result[1,0] |
| 1 | 1 | input[1,1] | result[1,1] |
| 2 | 0 | input[0,2] | result[2,0] |
| 2 | 1 | input[1,2] | result[2,1] |

Perfect! That's exactly what transpose does!

---

## 6.7 Lowering Constants

Constants are special - they don't take operands, they **produce** data.

### The Pattern

```cpp
struct ConstantOpLowering : public OpRewritePattern<toy::ConstantOp> {
  using OpRewritePattern<toy::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    // 1. Allocate memref
    auto tensorType = cast<RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // 2. Generate stores for each element
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    // Pre-create constant indices
    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
               0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, i));
    }

    // Recursively generate stores
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      if (dimension == valueShape.size()) {
        // Base case: store the element
        rewriter.create<affine::AffineStoreOp>(
            loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
            ArrayRef(indices));
        return;
      }

      // Recursive case: iterate over current dimension
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    storeElements(0);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};
```

### Breaking It Down

**The challenge:** A constant like `[[1, 2, 3], [4, 5, 6]]` needs to become individual stores.

#### Step 1: Pre-create Index Constants

```cpp
for (auto i : llvm::seq<int64_t>(0, max_dimension))
  constantIndices.push_back(
      rewriter.create<arith::ConstantIndexOp>(loc, i));
```

**Why?** Avoid creating duplicate constants. Instead of:
```mlir
%c0 = arith.constant 0 : index
%c0_dup = arith.constant 0 : index  // Duplicate!
```

We create once and reuse.

#### Step 2: Recursive Store Generation

```cpp
std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
  if (dimension == valueShape.size()) {
    // Base case: store element
    rewriter.create<affine::AffineStoreOp>(...);
    return;
  }
  
  // Recursive case: iterate over dimension
  for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
    indices.push_back(constantIndices[i]);
    storeElements(dimension + 1);
    indices.pop_back();
  }
};
```

**This generates all index combinations!**

**For shape [2, 3]:**
```
storeElements(0):
  i=0: indices=[0], call storeElements(1)
         i=0: indices=[0,0], call storeElements(2) → STORE at [0,0]
         i=1: indices=[0,1], call storeElements(2) → STORE at [0,1]
         i=2: indices=[0,2], call storeElements(2) → STORE at [0,2]
  i=1: indices=[1], call storeElements(1)
         i=0: indices=[1,0], call storeElements(2) → STORE at [1,0]
         i=1: indices=[1,1], call storeElements(2) → STORE at [1,1]
         i=2: indices=[1,2], call storeElements(2) → STORE at [1,2]
```

### Complete Example

**Before:**
```mlir
%0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
```

**After:**
```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index

%alloc = memref.alloc() : memref<2x3xf64>

%cst_1 = arith.constant 1.0 : f64
affine.store %cst_1, %alloc[%c0, %c0] : memref<2x3xf64>

%cst_2 = arith.constant 2.0 : f64
affine.store %cst_2, %alloc[%c0, %c1] : memref<2x3xf64>

%cst_3 = arith.constant 3.0 : f64
affine.store %cst_3, %alloc[%c0, %c2] : memref<2x3xf64>

%cst_4 = arith.constant 4.0 : f64
affine.store %cst_4, %alloc[%c1, %c0] : memref<2x3xf64>

%cst_5 = arith.constant 5.0 : f64
affine.store %cst_5, %alloc[%c1, %c1] : memref<2x3xf64>

%cst_6 = arith.constant 6.0 : f64
affine.store %cst_6, %alloc[%c1, %c2] : memref<2x3xf64>

memref.dealloc %alloc : memref<2x3xf64>
```

Each constant element gets its own store operation!

---

## 6.8 Lowering Functions and Control Flow

### FuncOp Lowering

We convert `toy.func` to `func.func` (standard MLIR function):

```cpp
struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
  using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Only lower main (others should be inlined)
    if (op.getName() != "main")
      return failure();

    // Verify main has no inputs/results
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    // Create new func.func with same region
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};
```

**Key points:**
- Only lower `main` (other functions should be inlined already)
- Verify it has no arguments (Toy main is special)
- Move the region from `toy.func` to `func.func`

### ReturnOp Lowering

```cpp
struct ReturnOpLowering : public OpRewritePattern<toy::ReturnOp> {
  using OpRewritePattern<toy::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // Main should return void (no operands)
    if (op.hasOperand())
      return failure();

    // Replace with func.return
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};
```

Simple transformation: `toy.return` → `func.return`

### PrintOp Lowering (Partial)

```cpp
struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Don't lower toy.print, just update its operands
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};
```

**Why keep toy.print?**
- We don't have a standard "print" in MLIR
- It's not a computational operation
- Later chapters will lower it to runtime calls

**But we update operands:**
- Operands change from `tensor<...>` to `memref<...>`
- Must reflect this in the operation

---

## 6.9 The Conversion Target

Now we tie it all together with the **ConversionTarget**.

### Defining What's Legal

```cpp
void ToyToAffineLoweringPass::runOnOperation() {
  // 1. Define conversion target
  ConversionTarget target(getContext());

  // Legal dialects (allowed after conversion)
  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect>();

  // Illegal dialect (must be converted)
  target.addIllegalDialect<toy::ToyDialect>();

  // Exception: toy.print is legal if operands are memrefs
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return isa<TensorType>(type); });
  });

  // 2. Register conversion patterns
  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, 
               MulOpLowering, PrintOpLowering, ReturnOpLowering,
               TransposeOpLowering>(&getContext());

  // 3. Apply conversion
  if (failed(applyPartialConversion(getOperation(), target, 
                                    std::move(patterns))))
    signalPassFailure();
}
```

### Understanding Legality

**addLegalDialect:**
```cpp
target.addLegalDialect<affine::AffineDialect>();
```
All operations from this dialect are OK in the output.

**addIllegalDialect:**
```cpp
target.addIllegalDialect<toy::ToyDialect>();
```
No operations from this dialect should remain (except exceptions).

**addDynamicallyLegalOp:**
```cpp
target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
  return llvm::none_of(op->getOperandTypes(),
                       [](Type type) { return isa<TensorType>(type); });
});
```

Conditional legality:
- `toy.print` is legal **only if** it has no tensor operands
- If it has tensor operands, it must be converted
- After conversion, operands become memrefs → legal!

### Partial vs. Full Conversion

**Partial conversion:**
```cpp
applyPartialConversion(getOperation(), target, patterns);
```
Some ops may remain unconverted (like `toy.print`).

**Full conversion:**
```cpp
applyFullConversion(getOperation(), target, patterns);
```
ALL operations must match the target (stricter).

We use **partial** because we intentionally keep `toy.print`.

---

## 6.10 Running the Complete Pipeline

Let's see the full transformation!

### The Pass Pipeline

From `toy/Ch5/toyc.cpp`:

```cpp
if (enableOpt || isLoweringToAffine) {
  // Inline all functions
  pm.addPass(mlir::createInlinerPass());

  // Infer shapes
  mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
  optPM.addPass(mlir::toy::createShapeInferencePass());
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());
}

if (isLoweringToAffine) {
  // Lower to Affine
  pm.addPass(mlir::toy::createLowerToAffinePass());

  // Cleanup after lowering
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());

  // Affine optimizations
  if (enableOpt) {
    optPM.addPass(mlir::affine::createLoopFusionPass());
    optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
  }
}
```

**Pipeline stages:**
1. **Inline** - Make everything one function
2. **Shape inference** - Resolve all shapes
3. **Canonicalize + CSE** - Cleanup
4. **Lower to Affine** - THE TRANSFORMATION
5. **Cleanup** - Simplify generated code
6. **Affine optimizations** - Fuse loops, etc.

### Building and Running

```powershell
# Build
cmake --build build --target toyc-ch5
```

### Test File

Create `test_affine.toy`:

```toy
def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var c = transpose(a);
  var d = c + b;
  print(d);
}
```

### Without Lowering

```powershell
cd toy\Ch5
..\..\build\toy\Ch5\toyc-ch5.exe -emit=mlir -opt test_affine.toy
```

**Output (Toy dialect):**
```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %2 = toy.add %1, %0 : tensor<3x2xf64>  // Error: shape mismatch!
    toy.print %2 : tensor<3x2xf64>
    toy.return
  }
}
```

Wait, there's a shape error in our example! Let me fix it.

Actually, let's use a correct example:

```toy
def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b = transpose(a);
  var c = b * b;
  print(c);
}
```

### With Lowering to Affine

```powershell
..\..\build\toy\Ch5\toyc-ch5.exe -emit=mlir-affine -opt test_affine.toy
```

**Output (Affine + MemRef):**
```mlir
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    
    // Allocate constant
    %alloc = memref.alloc() : memref<2x3xf64>
    %cst = arith.constant 1.000000e+00 : f64
    affine.store %cst, %alloc[%c0, %c0] : memref<2x3xf64>
    %cst_0 = arith.constant 2.000000e+00 : f64
    affine.store %cst_0, %alloc[%c0, %c1] : memref<2x3xf64>
    // ... more stores ...
    
    // Transpose
    %alloc_1 = memref.alloc() : memref<3x2xf64>
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %1 = affine.load %alloc[%arg1, %arg0] : memref<2x3xf64>
        affine.store %1, %alloc_1[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    
    // Multiply
    %alloc_2 = memref.alloc() : memref<3x2xf64>
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %2 = affine.load %alloc_1[%arg0, %arg1] : memref<3x2xf64>
        %3 = affine.load %alloc_1[%arg0, %arg1] : memref<3x2xf64>
        %4 = arith.mulf %2, %3 : f64
        affine.store %4, %alloc_2[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    
    toy.print %alloc_2 : memref<3x2xf64>
    
    memref.dealloc %alloc_2 : memref<3x2xf64>
    memref.dealloc %alloc_1 : memref<3x2xf64>
    memref.dealloc %alloc : memref<2x3xf64>
    func.return
  }
}
```

**Observe:**
- ✅ All `toy.*` operations converted to loops (except print)
- ✅ `tensor<...>` types became `memref<...>`
- ✅ Nested `affine.for` loops
- ✅ Explicit memory management (alloc/dealloc)
- ✅ Element-wise scalar operations (`arith.mulf`)

### With Affine Optimizations

```powershell
..\..\build\toy\Ch5\toyc-ch5.exe -emit=mlir-affine -opt test_affine.toy
```

Now MLIR can apply **loop fusion** and **scalar replacement**:
- Fuse loops that access the same data
- Eliminate temporary allocations when possible
- Optimize memory access patterns

---

## 6.11 Benefits of Affine Lowering

### 1. Enable Loop Optimizations

Affine loops can be optimized in ways impossible at Toy level:

**Loop fusion:**
```mlir
// Before
affine.for %i = 0 to N {
  B[i] = A[i] + 1
}
affine.for %i = 0 to N {
  C[i] = B[i] * 2
}

// After fusion
affine.for %i = 0 to N {
  B[i] = A[i] + 1
  C[i] = B[i] * 2
}
```

**Loop interchange:**
```mlir
// Before (bad cache locality)
affine.for %i = 0 to M {
  affine.for %j = 0 to N {
    C[j][i] = ...
  }
}

// After (better cache locality)
affine.for %j = 0 to N {
  affine.for %i = 0 to M {
    C[j][i] = ...
  }
}
```

### 2. Explicit Memory Management

MemRefs make memory layout explicit:
- Control allocation strategy
- Specify memory spaces (stack, heap, GPU)
- Reason about aliasing

### 3. Target-Specific Optimization

Affine code can be specialized for hardware:
- Vectorization (SIMD instructions)
- Parallelization (multi-threading)
- GPU mapping (CUDA, OpenCL)

### 4. Correctness Analysis

Affine structure enables verification:
- Prove no buffer overflows
- Verify data dependencies
- Check race conditions

---

## 6.12 Design Patterns in Lowering

### Pattern 1: Progressive Lowering

Don't lower everything at once:
```
Toy → Affine → SCF → LLVM
```

Each stage is simpler and more focused.

### Pattern 2: Type Conversion

Types change with abstraction level:
```
tensor<2x3xf64>  (Toy - value semantics)
    ↓
memref<2x3xf64>  (Affine - memory semantics)
    ↓
!llvm.ptr        (LLVM - raw pointers)
```

### Pattern 3: Template Lowering

Reuse patterns via templates:
```cpp
template <typename HighOp, typename LowOp>
struct GenericLowering { ... };
```

### Pattern 4: Helper Functions

Extract common logic:
```cpp
lowerOpToLoops(...)  // Nested loop generation
insertAllocAndDealloc(...)  // Memory management
```

### Pattern 5: Partial Lowering

Some operations stay high-level:
```mlir
toy.print %memref : memref<2x3xf64>
```

Lower it later when you have runtime support.

---

## Summary

Let's recap what we've learned:

### Key Concepts

1. **Progressive Lowering**
   - Transform from high to low abstraction gradually
   - Each level enables specific optimizations
   - Toy → Affine → (later) LLVM

2. **Affine Dialect**
   - Loop nests with affine bounds
   - Memory operations (load/store)
   - Enables polyhedral optimization
   - Perfect for array/tensor computations

3. **Dialect Conversion Framework**
   - `ConversionTarget`: Define legal operations
   - `ConversionPattern`: How to transform
   - `TypeConverter`: Handle type changes
   - Partial vs. full conversion

4. **Type Changes**
   - `tensor<...>` → `memref<...>`
   - Value semantics → reference semantics
   - Immutable → mutable

5. **Lowering Patterns**
   - Binary ops: Element-wise loops + scalar arithmetic
   - Transpose: Swapped indices
   - Constants: Recursive store generation
   - Functions: Dialect change + region movement

### What We Built

- ✅ Complete lowering from Toy to Affine
- ✅ Patterns for all computational operations
- ✅ Memory management (alloc/dealloc)
- ✅ Loop nest generation
- ✅ Integration with optimization passes

### Design Philosophy

**"Each IR level has a purpose"**

- **High level** (Toy): Express algorithm intent
- **Mid level** (Affine): Enable loop optimization
- **Low level** (LLVM): Enable code generation

MLIR's power is in having **all levels** and transforming between them!

---

## What's Next

In **Chapter 7**, we'll complete the journey! We'll lower from Affine all the way to **LLVM IR**:

- Convert affine loops to control flow (`scf.for`)
- Lower memref operations to LLVM memory operations
- Transform to LLVM dialect
- Generate executable code!

This is where we finally **run** our Toy programs!

---

## Exercises

### Exercise 1: Analyze Loop Nests

Given this Toy code:
```toy
def main() {
  var a<3, 4> = [...];
  var b<3, 4> = [...];
  var c = a + b;
  var d = transpose(c);
  print(d);
}
```

**Tasks:**
1. Draw the loop nests after lowering
2. Count total iterations for each loop
3. How many loads/stores total?
4. Could any loops be fused?

### Exercise 2: Write a Lowering Pattern

Implement lowering for a hypothetical `toy.negate` operation:

```cpp
struct NegateOpLowering : public ConversionPattern {
  // TODO: Implement matchAndRewrite
  // Hint: Similar to transpose, but apply negation
};
```

**The operation:**
```mlir
%result = toy.negate(%input : tensor<2x3xf64>) to tensor<2x3xf64>
```

**Should become:**
```mlir
// Loops that load, negate, and store
```

### Exercise 3: Optimize Memory

The current constant lowering creates many `arith.constant` operations. How could you optimize this?

**Hint:** Look at how `constantIndices` are pre-created and reused.

### Exercise 4: Trace the Conversion

Given:
```mlir
%0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
%1 = toy.transpose(%0 : tensor<2x2xf64>) to tensor<2x2xf64>
%2 = toy.add %0, %1 : tensor<2x2xf64>
```

**Tasks:**
1. Show the IR after constant lowering
2. Show the IR after transpose lowering
3. Show the IR after add lowering
4. Count memory operations (alloc/dealloc/load/store)

---

## Further Reading

### MLIR Documentation

- **Affine Dialect**: [https://mlir.llvm.org/docs/Dialects/Affine/](https://mlir.llvm.org/docs/Dialects/Affine/)
- **Dialect Conversion**: [https://mlir.llvm.org/docs/DialectConversion/](https://mlir.llvm.org/docs/DialectConversion/)
- **Toy Tutorial Chapter 5**: [https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/)

### Polyhedral Optimization

- **Polyhedral Compilation**: Understanding affine transformations
- **Loop Optimization**: Tiling, fusion, interchange
- **Pluto Algorithm**: Automatic parallelization

### Memory Models

- **SSA vs. Memory**: Understanding the semantic shift
- **Alias Analysis**: Reasoning about pointer relationships
- **Memory Hierarchy**: Cache-aware optimization

---

## Reflection Questions

Before moving to Chapter 7, consider:

1. **Why multiple IRs?**
   - Could we design one IR that works for all levels?
   - What would be the trade-offs?
   - When is abstraction beneficial vs. harmful?

2. **Correctness of lowering**
   - How do you prove a lowering is correct?
   - What could go wrong?
   - How do you test lowering transformations?

3. **Performance implications**
   - Does lowering always improve performance?
   - When might high-level operations be better?
   - How do you decide when to lower?

4. **Design alternatives**
   - What if we used direct code generation instead?
   - Could we skip the Affine level?
   - What would we lose?

These questions touch on fundamental compiler design choices!

---

**Next up:** Chapter 7, where we complete the lowering to LLVM and finally **execute** our Toy programs! ⚡
