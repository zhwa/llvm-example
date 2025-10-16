# Chapter 1: The Compiler Problem Space

> *"The best way to predict the future is to invent it." - Alan Kay*

## Introduction

If you've ever written code in Python, C++, JavaScript, or any programming language, you've benefited from a compiler or interpreter. But have you ever wondered what happens between the moment you write `x = a + b` and when your computer actually adds two numbers?

This chapter isn't about MLIR yet. Instead, we're going to explore the *problem space* that MLIR was designed to solve. By understanding the challenges of modern compiler design, you'll see why MLIR exists and why it's gaining adoption across the industry—from machine learning frameworks like TensorFlow and PyTorch to traditional compilers like Flang (Fortran).

Think of this chapter as the "origin story" of MLIR. We'll start with the basics, build up to modern challenges, and end with a preview of MLIR's elegant solution.

---

## 1.1 The Traditional Compiler Pipeline

### What Does a Compiler Actually Do?

At its core, a compiler is a translator. It takes code written in a high-level language (like C++ or Python) and translates it into something your computer can execute. This translation happens in stages, like a factory assembly line.

Here's the traditional pipeline:

```
Source Code → [Lexer] → Tokens → [Parser] → AST → [Semantic Analysis] → 
→ [IR Generation] → IR → [Optimization] → Optimized IR → 
→ [Code Generation] → Assembly/Machine Code
```

Let's break this down with a simple example:

**Source Code:**
```c++
int square(int x) {
    return x * x;
}
```

**Step 1: Lexing (Tokenization)**
The lexer breaks the text into meaningful chunks called tokens:
```
[INT] [IDENTIFIER:square] [LPAREN] [INT] [IDENTIFIER:x] [RPAREN]
[LBRACE] [RETURN] [IDENTIFIER:x] [STAR] [IDENTIFIER:x] [SEMICOLON]
[RBRACE]
```

**Step 2: Parsing (Building an AST)**
The parser organizes tokens into a tree structure called an Abstract Syntax Tree (AST):
```
FunctionDecl: square
├── ReturnType: int
├── Parameter: int x
└── Body:
    └── ReturnStmt
        └── BinaryOp: *
            ├── VarRef: x
            └── VarRef: x
```

This tree captures the *structure* of your code—what it means, not just what characters appear in which order.

**Step 3: Intermediate Representation (IR)**
The AST is converted into an intermediate representation. For example, in LLVM IR:
```llvm
define i32 @square(i32 %x) {
entry:
  %mul = mul nsw i32 %x, %x
  ret i32 %mul
}
```

This is where things get interesting. Let's talk about why IR matters.

### Why Do We Need Intermediate Representations?

Imagine you're building a translation service that translates between 10 different languages. The naive approach would be to build 10×9 = 90 different translators (each language to every other language). That's a lot of work!

A smarter approach: create one common "pivot language" that everyone translates to and from. Now you only need 10×2 = 20 translators. This is exactly what IR does for compilers.

**Benefits of IR:**

1. **Language Independence**: Many source languages (C, C++, Rust, Swift) can compile to the same IR
2. **Target Independence**: One IR can generate code for many targets (x86, ARM, RISC-V)
3. **Optimization**: Write optimizations once, benefit all languages and targets
4. **Analysis**: Easier to analyze and transform than raw source code

The most famous IR is LLVM IR, which powers compilers for dozens of languages.

### The Structure of IR

Good IR has several key properties:

**1. It's Explicit**
Unlike source code, IR makes everything explicit. No hidden assumptions.

Source code:
```c++
x + y
```

LLVM IR:
```llvm
%1 = load i32, i32* %x      ; Load value of x
%2 = load i32, i32* %y      ; Load value of y
%3 = add nsw i32 %1, %2     ; Add them (nsw = no signed wrap)
```

**2. It's in SSA Form (Static Single Assignment)**
Each variable is assigned exactly once. Instead of reusing variable names, we create new ones:

Bad (non-SSA):
```
x = 1
x = x + 2
x = x * 3
```

Good (SSA):
```
x1 = 1
x2 = x1 + 2
x3 = x2 * 3
```

This makes optimizations much easier because you know exactly where each value comes from.

**3. It's Suitable for Analysis**
Compilers need to answer questions like:
- "Can I safely move this operation?"
- "What values can this variable hold?"
- "Is this memory access safe?"

IR is designed to make these questions answerable.

---

## 1.2 The Multi-Level Problem

Now we reach the heart of the matter: **One IR is not enough for modern computing.**

### The Rise of Heterogeneous Computing

When LLVM was created in the early 2000s, the world was simpler:
- Most code ran on CPUs
- CPUs were getting faster through clock speed increases
- One size fit most needs

Today's world is radically different:
- **CPUs** with complex instruction sets
- **GPUs** for parallel computation
- **TPUs** (Tensor Processing Units) for machine learning
- **FPGAs** (Field-Programmable Gate Arrays) for custom hardware
- **DSPs** (Digital Signal Processors) for signal processing
- **Neural Processing Units** for AI inference

Each of these processors has:
- Different programming models
- Different optimization opportunities
- Different constraints and capabilities

### The Abstraction Gap

Here's the problem: LLVM IR operates at roughly the level of assembly language. It's great for CPUs, but terrible for representing high-level concepts.

**Example: Matrix Multiplication**

At a high level, you want to express:
```python
C = A @ B  # Matrix multiplication
```

This could be:
- Optimized with tiling and cache blocking
- Parallelized across cores
- Mapped to GPU operations
- Compiled to specialized hardware

But in LLVM IR, all you have are loops and memory operations:
```llvm
; Nested loops with thousands of load/store operations
for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  ; ... hundreds of lines of low-level code ...
```

**The problem**: By the time you're at LLVM IR, you've lost the high-level structure. You can't easily say "this is a matrix multiplication" and apply matrix-specific optimizations.

### Domain-Specific Languages (DSLs)

Many domains have created their own specialized languages:

- **Machine Learning**: TensorFlow's graph format, PyTorch's TorchScript
- **Databases**: SQL query plans
- **Graphics**: Shader languages (GLSL, HLSL)
- **Scientific Computing**: High-performance Fortran constructs

Each of these wants to:
1. Represent domain-specific operations naturally
2. Apply domain-specific optimizations
3. Eventually compile to efficient machine code

The traditional approach? Each domain builds its own complete compiler infrastructure from scratch. This means:
- Reimplementing basic compiler passes
- Reinventing optimization techniques
- Duplicating testing and tooling
- Limited interoperability

**There has to be a better way.**

---

## 1.3 Enter MLIR: The Philosophy

### What Does "Multi-Level" Mean?

MLIR stands for **Multi-Level Intermediate Representation**. The key word is "Multi-Level."

Instead of having one IR at one abstraction level (like LLVM IR), MLIR lets you define *multiple IRs at different abstraction levels* that can all coexist and interoperate.

Think of it like this:

**Traditional approach (LLVM):**
```
High-Level Language → [BIG JUMP] → LLVM IR (low-level) → Machine Code
```

**MLIR approach:**
```
High-Level Language → High-Level IR → Mid-Level IR → Low-Level IR → Machine Code
                       ↓                ↓              ↓
                    (optimize)      (optimize)    (optimize)
```

Each level can use the right abstractions for its optimization opportunities.

### The Dialect System

MLIR's killer feature is **dialects**. A dialect is like a vocabulary for a specific domain or abstraction level.

Some dialects in MLIR:

**High-Level Dialects:**
- `tensor` - Operations on multi-dimensional arrays
- `linalg` - Linear algebra operations (matrix multiply, convolutions)

**Mid-Level Dialects:**
- `affine` - Structured loop nests with mathematical properties
- `scf` - Structured control flow (for, while, if)

**Low-Level Dialects:**
- `arith` - Basic arithmetic operations
- `memref` - Memory references and operations
- `llvm` - LLVM IR represented in MLIR

**Domain-Specific Dialects:**
- `toy` - Our tutorial language (which we'll explore)
- `tosa` - Tensor Operator Set Architecture for ML
- `spirv` - For GPU shaders
- `nvvm` - NVIDIA GPU operations

### Progressive Lowering

The magic happens through **progressive lowering** - gradually transforming from high-level to low-level representations:

```
Toy Dialect:
  toy.mul %tensor1, %tensor2 : tensor<2x3xf64>
    ↓ (lower to linear algebra)
    
Linalg Dialect:
  linalg.generic [...] : tensor<2x3xf64>
    ↓ (lower to loops)
    
Affine Dialect:
  affine.for %i = 0 to 2
    affine.for %j = 0 to 3
      %v = affine.load %mem[%i, %j]
    ↓ (lower to standard operations)
    
Standard Dialects (arith, memref):
  scf.for %i = 0 to 2
    %v = memref.load %mem[%i, %j]
    ↓ (lower to LLVM)
    
LLVM Dialect:
  %addr = llvm.getelementptr %ptr[%i, %j]
  %v = llvm.load %addr
    ↓ (translate to LLVM IR)
    
LLVM IR & Machine Code
```

At each level:
- You can apply optimizations appropriate to that abstraction
- You preserve semantic information as long as possible
- You can mix operations from multiple dialects

### Reusability Through Infrastructure

MLIR provides infrastructure that all dialects can use:

**1. Pass Infrastructure**
Write transformations once, apply to any dialect:
- Canonicalization (simplification)
- Dead code elimination
- Inlining
- Loop optimizations

**2. Type System**
Dialects can define custom types, but share common type infrastructure:
- Integer types: `i32`, `i64`
- Floating point: `f32`, `f64`
- Tensors: `tensor<2x3xf64>`
- Custom types: `!mydialect.mytype`

**3. Attribute System**
Attach compile-time constants and metadata to operations

**4. Region System**
Operations can contain nested regions of code (like function bodies)

**5. Interfaces**
Define common behavior across dialects (we'll explore this in Chapter 5)

### Why This Matters

Let's see a concrete example of MLIR's power.

**Scenario**: You're building a machine learning framework. You want to:
1. Parse a high-level model definition
2. Apply ML-specific optimizations (operator fusion, layout transforms)
3. Map to different hardware (CPU, GPU, TPU)
4. Generate efficient code

**Without MLIR**: Build everything from scratch
- Custom IR for ML operations
- Custom optimization passes
- Custom code generation for each target
- Thousands of person-hours of work

**With MLIR**: Compose existing pieces
- Use `tensor` and `linalg` dialects for ML operations (already exist)
- Use built-in optimization infrastructure
- Lower to `gpu` dialect for GPUs, `llvm` dialect for CPUs
- Reuse LLVM's code generation
- Focus on your unique value-add

---

## 1.4 Overview of the Toy Language Journey

### What Is the Toy Language?

Throughout this book, we'll be working with a language called "Toy." It's a simple language designed for teaching MLIR concepts. Don't let the name fool you—it demonstrates real compiler techniques used in production systems.

**Toy Language Features:**
- Variables and functions
- Multi-dimensional arrays (tensors)
- Operations: multiply, transpose, reshape
- No control flow (no if/while) - keeps things simple

**Example Toy Program:**
```toy
# User defined function
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  print(c);
}
```

This simple program demonstrates:
- Function definitions
- Tensor literals
- Shape annotations (`<2, 3>`)
- Function calls
- Generic functions (work with any shape)

### The Seven-Chapter Journey

Each chapter in the Toy tutorial (which we'll explore in this book) introduces exactly ONE major concept:

**Chapter 2: Building the AST**
- Traditional lexing and parsing
- Constructing an Abstract Syntax Tree
- *Why it matters*: Foundation—start with what you know
- *Code walkthrough*: Lexer, Parser, AST nodes

**Chapter 3: Your First MLIR Dialect**
- Defining operations with TableGen
- Converting AST to MLIR
- Understanding dialects and operations
- *Why it matters*: First exposure to MLIR's core concepts
- *Code walkthrough*: `Ops.td`, `MLIRGen.cpp`

**Chapter 4: Transformations and Optimization**
- Pattern matching and rewriting
- Declarative Rewrite Rules (DRR)
- Canonicalization
- *Why it matters*: Making code better through transformations
- *Code walkthrough*: `ToyCombine.td`, `ToyCombine.cpp`

**Chapter 5: Interfaces and Reusability**
- Defining custom interfaces
- Shape inference pass
- Generic algorithms across operations
- *Why it matters*: Write once, use everywhere
- *Code walkthrough*: `ShapeInferenceInterface.td`, `ShapeInferencePass.cpp`

**Chapter 6: Partial Lowering to Affine**
- Progressive lowering concept
- Mixing dialects
- From tensors to loops and memory
- *Why it matters*: First step toward executable code
- *Code walkthrough*: `LowerToAffineLoops.cpp`

**Chapter 7: Complete Lowering to LLVM**
- Multi-stage lowering
- Standard conversion passes
- From MLIR to LLVM IR
- *Why it matters*: Complete the compilation pipeline
- *Code walkthrough*: `LowerToLLVM.cpp`

**Chapter 8: JIT Execution** *(We'll cover the concepts, less focus on code)*
- Just-In-Time compilation
- Executing MLIR code directly
- Integration with native code

### The Big Picture

Here's what the complete pipeline looks like:

```
Toy Source Code (.toy file)
    ↓
[Chapter 2: Lexer & Parser]
    ↓
Abstract Syntax Tree (AST)
    ↓
[Chapter 3: MLIRGen]
    ↓
Toy Dialect IR (high-level)
    ↓
[Chapter 4: Optimizations]
    ↓
Optimized Toy IR
    ↓
[Chapter 5: Shape Inference]
    ↓
Fully-typed Toy IR
    ↓
[Chapter 6: Lower to Affine]
    ↓
Affine + MemRef IR (loops and memory)
    ↓
[Chapter 7: Lower to LLVM]
    ↓
LLVM Dialect IR
    ↓
[LLVM Translation]
    ↓
LLVM IR
    ↓
[LLVM Optimization & CodeGen]
    ↓
Executable Machine Code
```

### What Makes This Approach Special?

**1. Incremental Complexity**
Each chapter builds on the previous one, adding exactly one new concept. You're never overwhelmed.

**2. Concrete Before Abstract**
We start with familiar concepts (AST, parsing) before diving into MLIR-specific features.

**3. Real Code, Real Concepts**
The Toy tutorial uses the same techniques as production compilers. The only difference is scale, not sophistication.

**4. Hands-On Learning**
Every chapter includes runnable code. You can experiment, break things, and learn by doing.

### A Note on Scope

The Toy language is intentionally simple. It lacks:
- Control flow (if, while, for)
- Multiple data types (just f64)
- Complex type system
- Error handling
- Standard library

This isn't a limitation—it's a feature. By keeping the language simple, we can focus on MLIR concepts without getting distracted by language design questions.

---

## 1.5 Why MLIR Matters for You

Before we dive into code in the next chapter, let's talk about why learning MLIR is valuable.

### The Industry Is Adopting MLIR

**Machine Learning:**
- **TensorFlow**: MLIR powers TensorFlow's compiler infrastructure
- **PyTorch**: PyTorch 2.0 uses Torch-MLIR for compilation
- **IREE**: Machine learning deployment runtime built on MLIR
- **TVM**: Apache TVM is integrating MLIR

**Traditional Compilers:**
- **Flang**: LLVM's Fortran compiler uses MLIR
- **Polygeist**: C/C++ to MLIR translation
- **CIRCT**: Circuit IR Compilers and Tools (hardware design)

**Domain-Specific:**
- **PlaidML**: GPU programming
- **NPComp**: NumPy compiler
- **Enzyme**: Automatic differentiation

### Skills You'll Gain

By learning MLIR, you'll develop:

**1. Compiler Literacy**
Understand how code transforms from source to executable. This makes you a better:
- Performance engineer (know what the compiler can/can't do)
- Language designer (understand implementation implications)
- Tools developer (build better developer tools)

**2. Abstraction Design**
MLIR teaches you to think in layers:
- What abstraction level is appropriate?
- How should abstractions compose?
- When to optimize vs. when to preserve structure?

**3. DSL Implementation**
If you need to build a domain-specific language (DSL), MLIR provides:
- Proven infrastructure
- Interoperability with existing tools
- Optimization capabilities
- Path to efficient execution

**4. Modern Compiler Techniques**
MLIR represents the state-of-the-art in:
- IR design
- Type systems
- Transformation infrastructure
- Extensibility mechanisms

### Real-World Applications

**Example 1: ML Model Optimization**
You write a TensorFlow model. MLIR:
- Fuses operations to reduce memory traffic
- Maps operations to optimized libraries
- Generates GPU or TPU code
- Produces a deployable artifact

**Example 2: Query Compilation**
Your database receives a SQL query. MLIR:
- Represents the query plan
- Applies database-specific optimizations
- Generates vectorized code
- Interfaces with storage engines

**Example 3: Custom Accelerator**
You're designing custom hardware. MLIR:
- Provides a language to describe operations
- Enables high-level optimizations
- Lowers to hardware-specific representations
- Generates configuration or RTL

### The MLIR Mindset

More than specific techniques, MLIR teaches a way of thinking:

**Progressive Refinement**: Start high-level, gradually get more concrete
**Composition**: Mix and match abstractions as needed
**Reusability**: Build on existing infrastructure
**Extensibility**: Add new concepts without modifying core infrastructure

These principles apply far beyond compilers—they're valuable for any complex software system.

---

## Summary

Let's recap what we've learned:

### Key Concepts

1. **Compilers translate code through multiple stages**: Lexing → Parsing → IR → Optimization → Code Generation

2. **Intermediate Representations (IR) are crucial**: They enable optimization and retargeting, sitting between source and machine code

3. **One IR isn't enough for modern computing**: Different domains and hardware need different abstractions

4. **MLIR provides multi-level IRs**: Different dialects at different abstraction levels, all interoperating

5. **Progressive lowering is the key**: Gradually transform from high-level to low-level, optimizing at each stage

6. **The Toy tutorial demonstrates real concepts**: Seven chapters, each introducing one major idea

### Why This Matters

- Modern computing is heterogeneous (CPUs, GPUs, TPUs, etc.)
- Domain-specific optimizations require domain-specific representations
- Building compilers from scratch is expensive and duplicates effort
- MLIR provides reusable infrastructure for extensible compiler development
- Major industry projects are adopting MLIR

### What's Next

In **Chapter 2**, we'll start writing code. We'll build:
- A lexer to tokenize Toy source code
- A parser to construct an Abstract Syntax Tree
- Our first complete (traditional) compiler frontend

This will give us a solid foundation before we introduce MLIR concepts in Chapter 3.

### Reflection Questions

Before moving on, consider:

1. **What domains or applications might benefit from custom IRs?**
   Think about areas where you work. Could high-level abstractions enable better optimization?

2. **Where have you seen the "abstraction gap" problem?**
   Have you worked with tools that forced you to operate at the wrong level?

3. **What would you want to compile?**
   If you could design a language or compiler, what would it do?

Keep these questions in mind as we explore MLIR. The concepts we'll learn apply broadly to software design, not just compilers.

---

## Further Reading

### Academic Papers
- **MLIR: A Compiler Infrastructure for the End of Moore's Law** (Lattner et al., 2020)
  - The foundational paper explaining MLIR's design
  - [https://arxiv.org/abs/2002.11054](https://arxiv.org/abs/2002.11054)

### Online Resources
- **MLIR Documentation**: [https://mlir.llvm.org/](https://mlir.llvm.org/)
- **LLVM Documentation**: [https://llvm.org/docs/](https://llvm.org/docs/)
- **MLIR Discourse Forum**: [https://discourse.llvm.org/c/mlir/](https://discourse.llvm.org/c/mlir/)

### Books
- **Engineering a Compiler** (Cooper & Torczon)
  - Comprehensive compiler textbook
- **Compilers: Principles, Techniques, and Tools** (Aho, Sethi, Ullman - "The Dragon Book")
  - Classic compiler theory

### Videos
- **2020 LLVM Developers' Meeting: MLIR Tutorial**
  - Official tutorial by MLIR creators
- **Chris Lattner on MLIR** (various conference talks)
  - Search YouTube for recent presentations

---

## Notes for the Curious

### Why "Toy"?

The Toy language is intentionally simple to teach concepts without distraction. Real MLIR dialects (like `linalg` or `gpu`) use the same techniques but at a larger scale.

### What About LLVM?

LLVM and MLIR are complementary, not competing:
- MLIR handles high- and mid-level representations
- MLIR can lower to LLVM IR
- LLVM handles low-level optimization and code generation
- Together they form a complete stack

### Is MLIR Only for ML?

Despite the name, MLIR is general-purpose. "ML" originally meant "Multi-Level" not "Machine Learning" (though it works great for ML). MLIR is used for traditional compilers (Flang), hardware design (CIRCT), and more.

### Can I Use MLIR Today?

Yes! MLIR is production-ready and used in major projects. The infrastructure is stable, though some dialects are still evolving. Check the MLIR documentation for dialect stability classifications.

---

**Ready to write some code?** In the next chapter, we'll build a lexer and parser for the Toy language, constructing our first Abstract Syntax Tree. We'll start with familiar territory before introducing MLIR-specific concepts.

Let's get started! 🚀
