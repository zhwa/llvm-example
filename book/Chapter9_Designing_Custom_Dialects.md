# Chapter 9: Designing Custom Dialects

> *"Design is not just what it looks like and feels like. Design is how it works." - Steve Jobs*

## Introduction

We've spent eight chapters learning MLIR's mechanics:
- **Chapter 1**: Motivation and philosophy
- **Chapter 2-3**: Building blocks (AST, operations, dialects)
- **Chapter 4-5**: Transformations and abstractions (patterns, interfaces)
- **Chapter 6-7**: Lowering and execution (progressive lowering, JIT)
- **Chapter 8**: Tooling (TableGen mastery)

Now comes the **synthesis**: How do you design a **custom dialect** from scratch?

This isn't about TableGen syntax or API calls. It's about:
- **Architecture**: How to structure your dialect
- **Abstraction level**: What operations to include
- **Composability**: How to integrate with other dialects
- **Evolution**: How to maintain and extend your dialect

By the end of this chapter, you'll understand:
- The design principles behind MLIR dialects
- When to create a new dialect vs. reusing existing ones
- How to layer dialects for progressive lowering
- Real-world design patterns and case studies

Let's design some dialects! 🎨

---

## 9.1 The Dialect Design Space

### What Is a Dialect, Really?

At its core, a **dialect** is:

**Definition**: A **coherent collection of operations** that represent computations at a **specific level of abstraction**.

**Key words:**
- **Coherent**: Operations work together, share design principles
- **Specific level**: Not too abstract, not too concrete
- **Collection**: Multiple operations, not just one

**Analogy:**
- **Natural languages**: English, Spanish, Mandarin (different ways to express ideas)
- **MLIR dialects**: Toy, Affine, LLVM (different ways to express computation)

### Dimensions of Dialect Design

Every dialect exists along multiple dimensions:

#### 1. Abstraction Level

```
High (Domain-Specific)
    ↑
    │  TensorFlow dialect (ML operations)
    │  Toy dialect (tensor computations)
    │  Linalg dialect (linear algebra)
    │  Affine dialect (loop nests with affine bounds)
    │  SCF dialect (structured control flow)
    │  CF dialect (basic control flow)
    │  LLVM dialect (LLVM IR)
    ↓
Low (Machine-Level)
```

**Trade-offs:**
- **Higher abstraction**: More semantic information, harder to lower
- **Lower abstraction**: Easier to codegen, less optimization opportunity

#### 2. Domain Specificity

```
General Purpose ←──────────────────────→ Domain Specific

LLVM dialect                               TensorFlow dialect
CF dialect                                 GPU dialect  
Arith dialect                              Vector dialect
```

**Trade-offs:**
- **General purpose**: Reusable, but may not capture domain semantics
- **Domain specific**: Optimized for domain, but limited scope

#### 3. Target Architecture

```
Architecture Independent ←─────────────→ Architecture Specific

Linalg dialect                           NVVM dialect (NVIDIA GPUs)
Affine dialect                           ROCDL dialect (AMD GPUs)
                                         X86Vector dialect
```

**Trade-offs:**
- **Independent**: Portable, single implementation
- **Specific**: Can exploit hardware features, requires multiple backends

#### 4. Semantic Density

```
Low (Explicit) ←──────────────────────────→ High (Implicit)

LLVM dialect                                ML Graph dialect
(every memory access explicit)              (whole operations atomic)

arith.addi %a, %b                          tf.MatMul %a, %b
memref.load %ptr[%i]                       (implicitly N² operations)
```

**Trade-offs:**
- **Explicit**: Fine-grained control, verbose
- **Implicit**: Concise, less control over details

---

## 9.2 When to Create a New Dialect

### The Golden Rule

**Create a new dialect when**:
1. **No existing dialect** fits your abstraction level
2. You have a **coherent set** of operations (5+ related ops)
3. The operations **share common properties** (traits, interfaces)
4. You need **domain-specific optimizations**

**Don't create a new dialect when**:
1. You only need 1-2 operations (add to existing dialect)
2. Operations are at **different abstraction levels** (split into multiple dialects)
3. An existing dialect already covers 80%+ of your needs (extend it)

### Decision Tree

```
Do you need new operations?
    │
    ├─ No → Use existing dialects
    │
    └─ Yes → Do existing dialects cover this abstraction?
            │
            ├─ Yes (80%+) → Extend existing dialect
            │
            └─ No → How many operations?
                    │
                    ├─ 1-2 ops → Add to closest dialect
                    │
                    └─ 5+ ops → Are they coherent?
                                │
                                ├─ No → Split into multiple dialects
                                │
                                └─ Yes → CREATE NEW DIALECT ✓
```

### Real-World Examples

#### Example 1: Vector Dialect (Good Decision)

**Problem**: Need SIMD operations (not covered by Arith or LLVM)

**Decision**: Create `vector` dialect
- Operations: `vector.broadcast`, `vector.extract`, `vector.fma`, etc.
- Level: Between Affine (loops) and LLVM (scalar/memory)
- Coherent: All about vector/SIMD computations

**Result**: Successful! Used in many MLIR pipelines.

#### Example 2: Memref Dialect (Extracted)

**Problem**: Originally, memory operations scattered across dialects

**Decision**: Extract into dedicated `memref` dialect
- Operations: `memref.alloc`, `memref.load`, `memref.store`, etc.
- Level: Abstract memory operations (independent of allocation strategy)
- Coherent: All about memory management

**Result**: Better organization, easier to maintain.

#### Example 3: Extending Existing (Counter-example)

**Problem**: Need complex number arithmetic

**Decision**: DON'T create "complex" dialect
- Only need ~5 operations (add, mul, div, conj, abs)
- Same abstraction level as `arith` dialect
- Better: Add operations to `arith` dialect as `arith.complex.*`

---

## 9.3 Design Principles for Dialects

### Principle 1: Single Level of Abstraction

**Each dialect should operate at ONE abstraction level.**

**Bad: Mixed abstraction**
```tablegen
// DON'T: Mix high-level and low-level in same dialect
def MatMulOp : MyDialect_Op<"matmul"> { ... }    // High-level
def AddPointerOp : MyDialect_Op<"add_ptr"> { ... } // Low-level
```

**Good: Consistent abstraction**
```tablegen
// DO: Keep operations at same level
def MatMulOp : LinAlg_Op<"matmul"> { ... }       // LinAlg dialect
def ConvOp : LinAlg_Op<"conv"> { ... }           // LinAlg dialect
def DotOp : LinAlg_Op<"dot"> { ... }             // LinAlg dialect
```

**Why?**: Makes lowering paths clear and predictable.

### Principle 2: Minimal and Complete

**Minimal**: Only include operations you actually need.

**Complete**: Cover all common operations in your domain.

**Example: Toy dialect**
```tablegen
// Minimal: Only 8 operations
ConstantOp, AddOp, MulOp,         // Arithmetic
FuncOp, ReturnOp, GenericCallOp,  // Control flow
TransposeOp, ReshapeOp, PrintOp   // Tensor manipulation
```

Not included (because not needed for Toy language):
- Subtraction, division (can be added later if needed)
- Broadcasting (shapes must match)
- Control flow (if, while) - Toy is purely functional

**Balance**: Start minimal, expand based on actual requirements.

### Principle 3: Composability

**Design operations to compose with other dialects.**

**Bad: Monolithic operation**
```tablegen
// DON'T: Operation does too much
def MatMulWithBiasAndReluOp : MyDialect_Op<"fused_matmul"> {
  let arguments = (ins Tensor:$lhs, Tensor:$rhs, Tensor:$bias);
  let results = (outs Tensor);
  // Hard-codes specific optimization (matmul + bias + relu)
}
```

**Good: Composable operations**
```tablegen
// DO: Separate operations that can be composed
def MatMulOp : LinAlg_Op<"matmul"> { ... }
def AddOp : Arith_Op<"addf"> { ... }
def MaxOp : Arith_Op<"maximumf"> { ... }  // ReLU = max(x, 0)

// Users compose: relu(matmul(A, B) + bias)
```

**Benefits**:
- Pattern matching can fuse when profitable
- Unfused version still valid
- Works with different combinations (matmul alone, matmul+bias, etc.)

### Principle 4: Progressive Lowering

**Design dialects to support incremental lowering.**

**Example: Linalg → Affine → SCF → CF → LLVM**

```mlir
// High-level: Linalg
linalg.matmul ins(%A, %B : ...) outs(%C : ...)

// ↓ Lower to Affine loops

affine.for %i = 0 to %M {
  affine.for %j = 0 to %N {
    affine.for %k = 0 to %K {
      // Body
    }
  }
}

// ↓ Lower to SCF (structured control flow)

scf.for %i = %c0 to %M step %c1 {
  scf.for %j = %c0 to %N step %c1 {
    scf.for %k = %c0 to %K step %c1 {
      // Body
    }
  }
}

// ↓ Lower to CF (basic blocks)

^bb0(%i: index):
  %cond = arith.cmpi slt, %i, %M
  cf.cond_br %cond, ^bb1, ^bb_exit
^bb1:
  // Loop body
  cf.br ^bb0

// ↓ Lower to LLVM

%i = llvm.phi [%c0, ^entry], [%i_next, ^loop]
%cond = llvm.icmp "slt" %i, %M
llvm.cond_br %cond, ^loop, ^exit
```

**Key point**: Each step is a **small, manageable transformation**.

### Principle 5: Verifiability

**Operations should be verifiable at each stage.**

**Build verification into operations:**

```tablegen
def MatMulOp : LinAlg_Op<"matmul"> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  
  let hasVerifier = 1;  // Enable verification
}
```

**Implement verification:**
```cpp
LogicalResult MatMulOp::verify() {
  auto lhsType = getLhs().getType().cast<RankedTensorType>();
  auto rhsType = getRhs().getType().cast<RankedTensorType>();
  auto resultType = getResult().getType().cast<RankedTensorType>();
  
  // Check rank
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2)
    return emitError("matmul requires 2D tensors");
  
  // Check dimensions
  if (lhsType.getDimSize(1) != rhsType.getDimSize(0))
    return emitError("matmul dimension mismatch");
  
  // Check result shape
  if (resultType.getDimSize(0) != lhsType.getDimSize(0) ||
      resultType.getDimSize(1) != rhsType.getDimSize(1))
    return emitError("matmul result shape mismatch");
  
  return success();
}
```

**Benefits**: Catch errors early, before lowering or execution.

---

## 9.4 Dialect Architecture Patterns

### Pattern 1: Single-Purpose Dialect

**Description**: Dialect focused on ONE specific domain.

**Example: Vector Dialect**
```
vector.broadcast    - Splat scalar to vector
vector.extract      - Extract element
vector.insert       - Insert element
vector.fma          - Fused multiply-add
vector.transpose    - Transpose vector
vector.contract     - Generalized contraction
```

**Characteristics**:
- 10-30 operations
- All operations related
- Clear scope and boundaries

**When to use**: Well-defined domain with specific operations.

### Pattern 2: Utility Dialect

**Description**: Common operations used across many dialects.

**Example: Arith Dialect**
```
arith.addi    - Integer addition
arith.addf    - Float addition
arith.muli    - Integer multiplication
arith.divsi   - Signed integer division
arith.cmpi    - Integer comparison
arith.cmpf    - Float comparison
// ... ~40 basic arithmetic operations
```

**Characteristics**:
- Many operations (20-50+)
- General purpose
- No domain-specific semantics
- Used by most other dialects

**When to use**: Building blocks needed everywhere.

### Pattern 3: Layered Dialect Family

**Description**: Multiple dialects at different abstraction levels.

**Example: Control Flow**
```
High Level:   structured control flow (scf)
              ├─ scf.if, scf.for, scf.while
              ↓
Mid Level:    affine control flow (affine)
              ├─ affine.if, affine.for
              ↓
Low Level:    basic blocks (cf)
              └─ cf.br, cf.cond_br, cf.switch
```

**Characteristics**:
- Multiple dialects
- Clear hierarchy
- Well-defined lowering between levels

**When to use**: Complex domain with multiple abstraction levels.

### Pattern 4: Interface-Based Dialect

**Description**: Dialect organized around interfaces, not operations.

**Example: Linalg Dialect**

All operations implement `LinalgOp` interface:
```cpp
// Interface defines common behavior
class LinalgOp {
  virtual ArrayRef<AffineMap> getIndexingMaps();
  virtual StringRef getLibraryCallName();
  // ... other methods
};

// Operations implement interface
linalg.matmul  → Implements LinalgOp
linalg.conv    → Implements LinalgOp
linalg.reduce  → Implements LinalgOp
```

**Characteristics**:
- Few operations, powerful interfaces
- Generic algorithms operate on any LinalgOp
- Extensible (add new ops without changing passes)

**When to use**: Operations share common structure/behavior.

---

## 9.5 Case Study: Designing a DSP Dialect

Let's design a **Digital Signal Processing (DSP)** dialect from scratch!

### Step 1: Requirements Analysis

**Target domain**: Audio/signal processing
- FFT (Fast Fourier Transform)
- Filtering (FIR, IIR)
- Convolution
- Windowing
- Delay lines

**Abstraction level**: Above Linalg (domain operations), below application logic

**Target hardware**: CPUs with SIMD, DSPs, FPGAs

### Step 2: Identify Core Operations

**Core operations** (minimal set):

1. **Transform operations**
   - `dsp.fft` - Fast Fourier Transform
   - `dsp.ifft` - Inverse FFT
   - `dsp.dct` - Discrete Cosine Transform

2. **Filter operations**
   - `dsp.fir` - Finite Impulse Response filter
   - `dsp.iir` - Infinite Impulse Response filter
   - `dsp.convolve` - Convolution

3. **Window operations**
   - `dsp.window` - Apply window function (Hamming, Hann, etc.)

4. **Buffer operations**
   - `dsp.delay` - Delay line
   - `dsp.circular_buffer` - Ring buffer

### Step 3: Define Operations (TableGen)

```tablegen
//===- DSPOps.td - DSP dialect operations ------------------*- tablegen -*-===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def DSP_Dialect : Dialect {
  let name = "dsp";
  let cppNamespace = "::mlir::dsp";
  let description = [{
    The DSP dialect provides operations for digital signal processing.
    Operations are designed to be lowered to optimized implementations
    for CPUs (SIMD), DSPs, and FPGAs.
  }];
}

// Base class for DSP operations
class DSP_Op<string mnemonic, list<Trait> traits = []> :
    Op<DSP_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FFT Operation
//===----------------------------------------------------------------------===//

def FFTOp : DSP_Op<"fft", [Pure]> {
  let summary = "Fast Fourier Transform";
  let description = [{
    Computes the Fast Fourier Transform of the input signal.
    
    The input must be a 1D tensor of complex or real values.
    For real inputs, only positive frequencies are returned.
    
    Example:
      %freq = dsp.fft %time : tensor<1024xcomplex<f32>>
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    OptionalAttr<I64Attr>:$n  // Optional: FFT size (zero-padding)
  );
  
  let results = (outs AnyRankedTensor:$output);
  
  let assemblyFormat = [{
    $input (`size` `=` $n^)? attr-dict `:` type($input) `to` type($output)
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FIR Filter Operation
//===----------------------------------------------------------------------===//

def FIRFilterOp : DSP_Op<"fir", [Pure]> {
  let summary = "Finite Impulse Response filter";
  let description = [{
    Applies a FIR filter with the given coefficients to the input signal.
    
    Output[n] = Σ(coeff[k] * input[n-k]) for k = 0 to N-1
    
    Example:
      %filtered = dsp.fir %signal, %coefficients 
                    : tensor<?xf32>, tensor<64xf32> -> tensor<?xf32>
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$coefficients
  );
  
  let results = (outs AnyRankedTensor:$output);
  
  let assemblyFormat = [{
    $input `,` $coefficients attr-dict 
    `:` type($input) `,` type($coefficients) `->` type($output)
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Window Operation
//===----------------------------------------------------------------------===//

def WindowOp : DSP_Op<"window", [Pure]> {
  let summary = "Apply window function to signal";
  let description = [{
    Multiplies the input signal by a window function.
    
    Supported windows: hamming, hann, blackman, kaiser
    
    Example:
      %windowed = dsp.window %signal, "hamming" : tensor<1024xf32>
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    StrAttr:$window_type  // "hamming", "hann", etc.
  );
  
  let results = (outs AnyRankedTensor:$output);
  
  let assemblyFormat = [{
    $input `,` $window_type attr-dict `:` type($input)
  }];
  
  let builders = [
    OpBuilder<(ins "Value":$input, "StringRef":$windowType)>
  ];
}

//===----------------------------------------------------------------------===//
// Delay Line Operation
//===----------------------------------------------------------------------===//

def DelayOp : DSP_Op<"delay"> {
  let summary = "Delay signal by N samples";
  let description = [{
    Delays the input signal by the specified number of samples.
    Maintains internal state (not a pure operation).
    
    Example:
      %delayed = dsp.delay %signal by 100 : tensor<?xf32>
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$delay_samples
  );
  
  let results = (outs AnyRankedTensor:$output);
  
  let assemblyFormat = [{
    $input `by` $delay_samples attr-dict `:` type($input)
  }];
  
  // Note: NOT Pure (has internal state)
}
```

### Step 4: Define Interfaces

```tablegen
//===- DSPInterfaces.td - DSP dialect interfaces -----------*- tablegen -*-===//

def FrequencyDomainOp : OpInterface<"FrequencyDomainOp"> {
  let description = [{
    Interface for operations that work in the frequency domain.
    Enables automatic domain conversion optimizations.
  }];
  
  let methods = [
    InterfaceMethod<
      "Get whether operation expects frequency domain input",
      "bool", "expectsFrequencyDomain"
    >,
    InterfaceMethod<
      "Get whether operation produces frequency domain output",
      "bool", "producesFrequencyDomain"
    >
  ];
}

def StreamableOp : OpInterface<"StreamableOp"> {
  let description = [{
    Interface for operations that can process streaming data
    (block-by-block processing).
  }];
  
  let methods = [
    InterfaceMethod<
      "Get minimum block size for streaming",
      "int64_t", "getMinBlockSize"
    >,
    InterfaceMethod<
      "Get whether operation requires state between blocks",
      "bool", "requiresState"
    >
  ];
}
```

### Step 5: Define Lowering Strategy

**DSP → Linalg/Affine**
```
dsp.fft
  ↓
linalg.generic + affine.for  (explicit butterfly operations)

dsp.fir
  ↓
affine.for + arith.mulf + arith.addf  (direct convolution)

dsp.window
  ↓
affine.for + arith.mulf  (element-wise multiply)
```

**Linalg/Affine → Vectorized**
```
affine.for (FFT butterfly)
  ↓
vector.fma (SIMD butterfly operations)

affine.for (FIR)
  ↓
vector.fma (SIMD multiply-accumulate)
```

**Vectorized → LLVM**
```
vector.fma
  ↓
llvm.intr.fma (hardware FMA instruction)
```

### Step 6: Implement Transformations

**Optimization pattern: FFT → IFFT fusion**

```tablegen
// In DSPCombine.td
def FoldFFTIFFT : Pat<
  (IFFTOp (FFTOp $input)),
  (replaceWithValue $input)
>;
```

**Optimization pattern: Frequency domain fusion**

```cpp
// In DSPOptimizations.cpp
// Pattern: Convert time → freq → time to keep in frequency domain
// Before: ifft(filter_freq(fft(signal)))
// After:  filter_time(signal)  (if profitable)

struct FrequencyDomainFusion : public OpRewritePattern<IFFTOp> {
  LogicalResult matchAndRewrite(IFFTOp op,
                                 PatternRewriter &rewriter) const override {
    // Check if input comes from frequency domain operations
    auto producer = op.getInput().getDefiningOp();
    if (auto fftOp = dyn_cast_or_null<FFTOp>(producer)) {
      // Found FFT → IFFT, can eliminate
      rewriter.replaceOp(op, fftOp.getInput());
      return success();
    }
    return failure();
  }
};
```

### Step 7: Create Pass Infrastructure

```cpp
// In Passes.h
namespace mlir::dsp {

std::unique_ptr<Pass> createDSPOptimizationPass();
std::unique_ptr<Pass> createLowerToLinalgPass();
std::unique_ptr<Pass> createVectorizePass();

} // namespace mlir::dsp
```

**Pass pipeline:**
```cpp
void buildDSPPipeline(OpPassManager &pm) {
  // 1. DSP-level optimizations
  pm.addPass(createDSPOptimizationPass());
  
  // 2. Lower to Linalg/Affine
  pm.addPass(createLowerToLinalgPass());
  
  // 3. Vectorize for SIMD
  pm.addPass(createVectorizePass());
  
  // 4. Standard MLIR lowering
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createConvertToLLVMPass());
}
```

---

## 9.6 Design Patterns and Anti-Patterns

### ✅ Pattern: Start High-Level

**DO**: Start with high-level operations, lower progressively.

```mlir
// Good: High-level DSP operation
%output = dsp.fft %input : tensor<1024xcomplex<f32>>

// Lowers to mid-level
affine.for %stage = 0 to 10 {
  affine.for %i = 0 to 512 {
    // Butterfly operations
  }
}

// Lowers to low-level
llvm.intr.fma %a, %b, %c : vector<4xf32>
```

**Benefits**: 
- High-level preserves semantics
- More optimization opportunities
- Progressive refinement

### ❌ Anti-Pattern: Premature Lowering

**DON'T**: Start too low-level.

```mlir
// Bad: Starting with low-level operations
%ptr = llvm.alloca %size x f32
llvm.store %value, %ptr
// Lost all domain knowledge!
```

**Problems**:
- Can't apply domain-specific optimizations
- Hard to understand intent
- Difficult to retarget

### ✅ Pattern: Use Traits for Properties

**DO**: Use traits to declare operation properties.

```tablegen
def FFTOp : DSP_Op<"fft", [
    Pure,                    // No side effects
    DeclareOpInterfaceMethods<FrequencyDomainOp>
  ]> {
  // ...
}
```

**Benefits**:
- Optimization passes can query traits
- Verification is automatic
- Documentation is clear

### ❌ Anti-Pattern: Ad-Hoc Properties

**DON'T**: Use attributes for structural properties.

```tablegen
// Bad: Using attribute for semantic property
def BadOp : DSP_Op<"bad"> {
  let arguments = (ins 
    ...,
    BoolAttr:$is_pure  // Should be a trait!
  );
}
```

**Problems**:
- Inconsistent across codebase
- Can't query systematically
- Runtime overhead

### ✅ Pattern: Composable Operations

**DO**: Design operations that compose naturally.

```mlir
// Good: Separate operations
%windowed = dsp.window %signal, "hamming"
%freq = dsp.fft %windowed
%filtered = dsp.filter_freq %freq, %response
%time = dsp.ifft %filtered

// Can be fused by patterns when profitable
```

### ❌ Anti-Pattern: Mega Operations

**DON'T**: Create operations that do too much.

```tablegen
// Bad: Operation with too many features
def MegaSignalProcessOp : DSP_Op<"mega_process"> {
  let arguments = (ins
    AnyTensor:$input,
    BoolAttr:$apply_window,
    StrAttr:$window_type,
    BoolAttr:$apply_fft,
    BoolAttr:$apply_filter,
    AnyTensor:$filter_coeffs,
    BoolAttr:$apply_ifft
    // Too many options!
  );
}
```

**Problems**:
- Hard to optimize
- Inflexible
- Difficult to understand

### ✅ Pattern: Interface-Based Abstractions

**DO**: Use interfaces for generic algorithms.

```cpp
// Good: Generic pass works on any FrequencyDomainOp
void optimizeFrequencyDomain(Operation *op) {
  if (auto freqOp = dyn_cast<FrequencyDomainOp>(op)) {
    if (freqOp.expectsFrequencyDomain()) {
      // Apply frequency-domain optimizations
    }
  }
}
```

### ❌ Anti-Pattern: Hardcoded Operation Names

**DON'T**: Check operation names directly.

```cpp
// Bad: Hardcoded operation names
if (op->getName().getStringRef() == "dsp.fft" ||
    op->getName().getStringRef() == "dsp.ifft") {
  // Brittle! Need to update for every new operation
}
```

---

## 9.7 Dialect Composition and Integration

### Composing Multiple Dialects

Real-world MLIR uses **multiple dialects together**:

```mlir
func.func @audio_process(%signal: tensor<1024xf32>) -> tensor<1024xf32> {
  // Control flow: func dialect
  
  // DSP operations: dsp dialect
  %windowed = dsp.window %signal, "hamming" : tensor<1024xf32>
  %freq = dsp.fft %windowed : tensor<1024xf32> to tensor<1024xcomplex<f32>>
  
  // Arithmetic: arith dialect
  %threshold = arith.constant 0.1 : f32
  
  // Comparison: arith dialect
  %mask = arith.cmpf ogt, %freq_magnitude, %threshold : tensor<1024xf32>
  
  // Selection: arith dialect
  %filtered = arith.select %mask, %freq, %zero : tensor<1024xcomplex<f32>>
  
  // More DSP: dsp dialect
  %result = dsp.ifft %filtered : tensor<1024xcomplex<f32>> to tensor<1024xf32>
  
  return %result : tensor<1024xf32>
}
```

**Multiple dialects, single program!**

### Dialect Dependencies

**Organize dialects in layers:**

```
Application Layer
    ↓ uses
Domain Dialects (DSP, ML, Graphics)
    ↓ uses
Mid-Level Dialects (Linalg, Affine, Vector)
    ↓ uses
Low-Level Dialects (Arith, Memref, CF)
    ↓ uses
LLVM Dialect
```

**Dependencies:**
- Higher dialects depend on lower dialects
- Lower dialects are independent
- Circular dependencies are **forbidden**

**Example: DSP dialect dependencies**
```cpp
// In DSPDialect.cpp
void DSPDialect::initialize() {
  // DSP depends on these dialects
  getContext()->loadDialect<arith::ArithDialect>();
  getContext()->loadDialect<func::FuncDialect>();
  getContext()->loadDialect<tensor::TensorDialect>();
}
```

### Cross-Dialect Patterns

**Patterns can match across dialects:**

```cpp
// Pattern: dsp.convolve → linalg.conv
struct LowerConvolveToLinalg : public OpRewritePattern<dsp::ConvolveOp> {
  LogicalResult matchAndRewrite(dsp::ConvolveOp op,
                                 PatternRewriter &rewriter) const override {
    // Create equivalent linalg.conv operation
    auto convOp = rewriter.create<linalg::ConvOp>(
        op.getLoc(),
        op.getInput(),
        op.getKernel(),
        op.getResult().getType()
    );
    
    rewriter.replaceOp(op, convOp.getResult());
    return success();
  }
};
```

---

## 9.8 Evolution and Maintenance

### Versioning Strategies

**Dialects evolve**. How to maintain compatibility?

#### Strategy 1: Careful Extension

**Add new operations/attributes, don't change existing ones.**

```tablegen
// Version 1.0
def FFTOp : DSP_Op<"fft"> {
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$output);
}

// Version 1.1 (backward compatible)
def FFTOp : DSP_Op<"fft"> {
  let arguments = (ins 
    AnyRankedTensor:$input,
    OptionalAttr<I64Attr>:$n  // NEW: optional attribute
  );
  let results = (outs AnyRankedTensor:$output);
}
```

**Old IR still parses correctly!**

#### Strategy 2: Deprecation Path

**Mark old operations as deprecated, provide migration path.**

```tablegen
// Old operation (deprecated)
def OldFFTOp : DSP_Op<"fft_old"> {
  let summary = "DEPRECATED: Use dsp.fft instead";
  // ...
}

// New operation
def FFTOp : DSP_Op<"fft"> {
  // ...
}
```

**Provide automatic upgrade:**
```cpp
void upgradeDSPDialect(ModuleOp module) {
  module.walk([](dsp::OldFFTOp op) {
    OpBuilder builder(op);
    auto newOp = builder.create<dsp::FFTOp>(op.getLoc(), op.getInput());
    op.replaceAllUsesWith(newOp);
    op.erase();
  });
}
```

#### Strategy 3: Dialect Versioning

**Include version in dialect definition.**

```tablegen
def DSP_Dialect : Dialect {
  let name = "dsp";
  let cppNamespace = "::mlir::dsp";
  
  // Version info
  let extraClassDeclaration = [{
    static constexpr int kMajorVersion = 2;
    static constexpr int kMinorVersion = 0;
  }];
}
```

**Check version during parsing:**
```cpp
LogicalResult DSPDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  if (attr.getName() == "dsp.version") {
    auto version = attr.getValue().cast<IntegerAttr>().getInt();
    if (version > kMajorVersion) {
      return op->emitError("unsupported DSP dialect version");
    }
  }
  return success();
}
```

### Testing Strategies

**Test each component of your dialect:**

#### 1. Operation Verification Tests

```mlir
// valid.mlir - Should parse and verify successfully
func.func @test_fft(%input: tensor<1024xf32>) -> tensor<1024xcomplex<f32>> {
  %result = dsp.fft %input : tensor<1024xf32> to tensor<1024xcomplex<f32>>
  return %result : tensor<1024xcomplex<f32>>
}

// invalid.mlir - Should fail verification
func.func @test_fft_invalid(%input: tensor<1024xf32>) -> tensor<512xf32> {
  // ERROR: Wrong result type
  %result = dsp.fft %input : tensor<1024xf32> to tensor<512xf32>
  return %result : tensor<512xf32>
}
```

**Test command:**
```powershell
mlir-opt valid.mlir --verify-diagnostics
mlir-opt invalid.mlir --verify-diagnostics --expected-error="FFT output size mismatch"
```

#### 2. Transformation Tests

```mlir
// RUN: mlir-opt %s -dsp-optimize | FileCheck %s

func.func @test_fft_ifft(%input: tensor<1024xf32>) -> tensor<1024xf32> {
  %freq = dsp.fft %input : tensor<1024xf32> to tensor<1024xcomplex<f32>>
  %result = dsp.ifft %freq : tensor<1024xcomplex<f32>> to tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// CHECK-LABEL: func.func @test_fft_ifft
// CHECK-SAME: (%[[INPUT:.*]]: tensor<1024xf32>)
// CHECK-NOT: dsp.fft
// CHECK-NOT: dsp.ifft
// CHECK: return %[[INPUT]]
```

#### 3. Lowering Tests

```mlir
// RUN: mlir-opt %s -lower-dsp-to-linalg | FileCheck %s

func.func @test_lower_fft(%input: tensor<1024xf32>) -> tensor<1024xcomplex<f32>> {
  %result = dsp.fft %input : tensor<1024xf32> to tensor<1024xcomplex<f32>>
  return %result : tensor<1024xcomplex<f32>>
}

// CHECK-LABEL: func.func @test_lower_fft
// CHECK: affine.for
// CHECK: linalg.generic
```

#### 4. End-to-End Tests

```cpp
// Test full compilation pipeline
TEST(DSPDialect, EndToEnd) {
  MLIRContext context;
  context.loadDialect<dsp::DSPDialect, func::FuncDialect>();
  
  // Create IR
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @main(%input: tensor<1024xf32>) -> tensor<1024xf32> {
      %w = dsp.window %input, "hamming" : tensor<1024xf32>
      %f = dsp.fft %w : tensor<1024xf32> to tensor<1024xcomplex<f32>>
      %r = dsp.ifft %f : tensor<1024xcomplex<f32>> to tensor<1024xf32>
      return %r : tensor<1024xf32>
    }
  )mlir", &context);
  
  // Apply passes
  PassManager pm(&context);
  pm.addPass(createDSPOptimizationPass());
  pm.addPass(createLowerToLinalgPass());
  
  ASSERT_TRUE(succeeded(pm.run(*module)));
  
  // Verify result
  // ...
}
```

---

## 9.9 Real-World Case Studies

### Case Study 1: TensorFlow Dialect

**Domain**: Machine learning computations

**Design decisions**:
- **High-level operations**: `tf.MatMul`, `tf.Conv2D`, `tf.Softmax`
- **Graph semantics**: Operations represent computation graph nodes
- **Attributes**: Extensive use for hyperparameters (stride, padding, etc.)
- **Types**: Custom tensor types with dynamic shapes

**Lowering strategy**:
```
TF dialect
  ↓
TFLite dialect (mobile optimization)
  ↓
Linalg dialect (generic linear algebra)
  ↓
Affine/SCF (loops)
  ↓
LLVM dialect
```

**Key insight**: Multiple abstraction levels enable targeting CPUs, GPUs, TPUs from single source!

### Case Study 2: GPU Dialect

**Domain**: GPU programming (CUDA/ROCm)

**Design decisions**:
- **Execution model**: Explicit grids, blocks, threads
- **Memory hierarchy**: Global, shared, local memory
- **Operations**: `gpu.launch`, `gpu.barrier`, `gpu.thread_id`
- **Target-specific**: NVVM dialect (NVIDIA), ROCDL dialect (AMD)

**Lowering strategy**:
```
GPU dialect (target-independent)
  ↓
NVVM dialect (NVIDIA-specific)  OR  ROCDL dialect (AMD-specific)
  ↓
LLVM PTX (NVIDIA)  OR  LLVM AMDGPU
  ↓
Native GPU code
```

**Key insight**: Abstract common GPU concepts, lower to specific targets!

### Case Study 3: Affine Dialect

**Domain**: Loop transformations and polyhedral optimization

**Design decisions**:
- **Affine constraints**: Loop bounds are affine expressions
- **Memory access**: Affine indexing only
- **Operations**: `affine.for`, `affine.if`, `affine.load`, `affine.store`
- **Analysis**: Dependence analysis, polyhedral optimization

**Why restricted?**
- **Analyzability**: Can prove properties (no aliasing, etc.)
- **Optimization**: Enables powerful loop transformations (tiling, fusion, etc.)
- **Verification**: Easier to verify correctness

**Key insight**: Restrictions enable powerful optimizations!

---

## Summary

Let's recap the key principles for designing custom dialects:

### Design Principles

1. **Single Level of Abstraction**
   - Each dialect operates at ONE abstraction level
   - Mixed levels make lowering unpredictable

2. **Minimal and Complete**
   - Only include operations you need
   - But cover all common operations in your domain

3. **Composability**
   - Operations should compose with other dialects
   - Avoid monolithic operations

4. **Progressive Lowering**
   - Design for incremental transformation
   - Each step should be small and verifiable

5. **Verifiability**
   - Operations should be verifiable at each stage
   - Catch errors early

### When to Create a Dialect

✅ **Create a new dialect when**:
- No existing dialect fits your abstraction level
- You have 5+ coherent, related operations
- You need domain-specific optimizations
- Operations share common properties (traits, interfaces)

❌ **Don't create a new dialect when**:
- You only need 1-2 operations
- Operations are at different abstraction levels
- An existing dialect covers 80%+ of your needs

### Architecture Patterns

1. **Single-Purpose Dialect**: Focused on one domain
2. **Utility Dialect**: Common operations used everywhere
3. **Layered Dialect Family**: Multiple dialects at different levels
4. **Interface-Based Dialect**: Organized around interfaces

### Anti-Patterns to Avoid

❌ Premature lowering
❌ Mega operations
❌ Ad-hoc properties
❌ Hardcoded operation names
❌ Mixed abstraction levels

### Key Takeaways

- **Dialects are about abstraction**: Choose the right level for your domain
- **Composition is key**: Design operations that work with other dialects
- **Think in layers**: Progressive lowering from high to low level
- **Interfaces enable reuse**: Generic algorithms work across dialects
- **Evolution requires planning**: Version carefully, provide migration paths

**You now understand how to architect robust, maintainable MLIR dialects!** 🏗️

---

## What's Next

In **Chapter 10** (our final chapter!), we'll explore the **MLIR Ecosystem**:

- Integration with existing compilers (LLVM, GCC)
- Building complete compilation pipelines
- Multi-target code generation
- Production deployment strategies
- Tools and debugging
- Community and resources

---

## Exercises

### Exercise 1: Design a Graphics Dialect

Design a dialect for 2D/3D graphics operations:
- What operations would you include?
- What abstraction level?
- How would it compose with existing dialects?
- What lowering strategy?

**Hints**:
- Consider operations: transforms, primitives, shaders
- Think about GPU vs CPU execution
- How to represent pipelines?

### Exercise 2: Extend the DSP Dialect

Add these operations to the DSP dialect:
1. `dsp.downsample` - Reduce sampling rate
2. `dsp.upsample` - Increase sampling rate
3. `dsp.biquad` - Biquad filter (common IIR structure)

Write TableGen definitions including:
- Operation definitions
- Verification logic
- Assembly format

### Exercise 3: Design Optimization Patterns

For the DSP dialect, design patterns for:
1. **Cascaded filters**: Combine multiple FIR filters into one
2. **Windowing fusion**: Fuse window + FFT into optimized kernel
3. **Domain optimization**: Keep operations in frequency domain when possible

Write DRR or C++ patterns.

### Exercise 4: Create a Lowering Pass

Implement a pass to lower `dsp.window` to affine loops:

```cpp
void LowerWindowToAffine::runOnOperation() {
  // For each dsp.window operation:
  // 1. Get window coefficients
  // 2. Generate affine.for loop
  // 3. Multiply input by coefficients
  // 4. Replace dsp.window with loop
}
```

### Exercise 5: Dialect Versioning

Design a versioning strategy for a dialect that needs to:
- Add new optional attributes
- Deprecate an old operation
- Change default behavior

How would you maintain backward compatibility?

---

## Further Reading

### MLIR Dialect Documentation

- **Builtin Dialects**: [https://mlir.llvm.org/docs/Dialects/](https://mlir.llvm.org/docs/Dialects/)
- **Defining Dialects**: [https://mlir.llvm.org/docs/DefiningDialects/](https://mlir.llvm.org/docs/DefiningDialects/)
- **Dialect Conversion**: [https://mlir.llvm.org/docs/DialectConversion/](https://mlir.llvm.org/docs/DialectConversion/)

### Design Philosophy

- **Lattner's MLIR Primer**: Original design rationale
- **Progressive Lowering**: Multi-level IR philosophy
- **Domain-Specific Optimization**: Why high-level abstractions matter

### Case Studies

- **TensorFlow/MLIR**: ML compilation
- **IREE**: End-to-end ML deployment
- **Flang**: Fortran compiler using MLIR
- **CIRCT**: Circuit IR (hardware design)

---

## Reflection Questions

Before moving to the final chapter, consider:

1. **Abstraction Level**
   - How do you choose the right abstraction level?
   - When is too high-level a problem?
   - When is too low-level a problem?

2. **Dialect Boundaries**
   - What makes operations belong to the same dialect?
   - How do you decide when to split into multiple dialects?
   - How do you prevent dialect proliferation?

3. **Optimization Strategy**
   - Where should optimizations happen (which dialect level)?
   - How do you balance early vs late optimization?
   - When to preserve vs discard information?

4. **Evolution**
   - How to design for future extensions?
   - How to maintain backward compatibility?
   - When is breaking compatibility acceptable?

5. **Reuse vs. Specialization**
   - When to extend existing dialects vs create new ones?
   - How much should dialects share?
   - What belongs in common utilities vs dialect-specific code?

These questions touch on fundamental software architecture trade-offs!

---

**You've mastered dialect design - the art and science of MLIR architecture!** 🎨

**Next up:** Chapter 10, the final chapter on the MLIR ecosystem, tooling, and production deployment! 🚀
