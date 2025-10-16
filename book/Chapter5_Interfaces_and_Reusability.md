# Chapter 5: Interfaces and Reusability

> *"The power of abstraction is not just in hiding complexity, but in enabling reuse across different contexts." - Barbara Liskov*

## Introduction

In previous chapters, we've learned to:
- **Chapter 2**: Build an AST
- **Chapter 3**: Create a dialect with operations
- **Chapter 4**: Transform and optimize IR

But there's a problem. Every operation we wrote is **specific** to the Toy dialect. What if we want:
- A generic optimization that works across **multiple dialects**?
- A shape inference pass that works with **any** tensor operation?
- Reusable infrastructure that doesn't duplicate code?

This chapter introduces **interfaces** - MLIR's solution to writing generic, reusable code. You'll learn:

- What interfaces are and why they matter
- How to define interfaces in TableGen
- How to implement interfaces for operations
- How to write generic algorithms using interfaces
- A complete example: shape inference

By the end, you'll understand how MLIR achieves incredible reusability and extensibility.

Let's dive in! 🚀

---

## 5.1 The Interface Problem

### The Motivation

Imagine we want to write a **shape inference pass**. Without interfaces, we'd write:

```cpp
void inferShapesInModule(ModuleOp module) {
  module.walk([](Operation *op) {
    if (auto addOp = dyn_cast<toy::AddOp>(op)) {
      // Infer shape for AddOp
      addOp.getResult().setType(addOp.getLhs().getType());
    } else if (auto mulOp = dyn_cast<toy::MulOp>(op)) {
      // Infer shape for MulOp (same logic!)
      mulOp.getResult().setType(mulOp.getLhs().getType());
    } else if (auto transposeOp = dyn_cast<toy::TransposeOp>(op)) {
      // Infer shape for TransposeOp (different logic)
      auto inputType = cast<RankedTensorType>(transposeOp.getInput().getType());
      SmallVector<int64_t, 2> dims(llvm::reverse(inputType.getShape()));
      transposeOp.getResult().setType(RankedTensorType::get(dims, inputType.getElementType()));
    } else if ... // And on and on for every operation!
  });
}
```

**Problems:**
1. ❌ **Not extensible**: Adding a new operation requires modifying the pass
2. ❌ **Doesn't work for other dialects**: Hardcoded to Toy operations
3. ❌ **Duplicated code**: Many operations have similar shape inference logic
4. ❌ **Tight coupling**: Pass knows about every specific operation

### The Interface Solution

With interfaces:

```cpp
void inferShapesInModule(ModuleOp module) {
  module.walk([](Operation *op) {
    // Generic: works for ANY operation that implements ShapeInference!
    if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
      shapeOp.inferShapes();
    }
  });
}
```

**Benefits:**
1. ✅ **Extensible**: New operations just implement the interface
2. ✅ **Dialect-agnostic**: Works with any dialect's operations
3. ✅ **No duplication**: Each operation implements its own logic
4. ✅ **Loose coupling**: Pass doesn't know about specific operations

This is the power of **polymorphism** applied to IR!

---

## 5.2 What Are Interfaces?

### Conceptual Model

An interface is a **contract**. It says:

> "Any operation that implements me promises to provide these methods."

Think of it like this:

```
┌─────────────────────────────┐
│   ShapeInference Interface  │
│                             │
│  Method: inferShapes()      │
└─────────────────────────────┘
           ▲     ▲     ▲
           │     │     │
    ┌──────┘     │     └────────┐
    │            │              │
┌────────┐  ┌────────┐    ┌──────────┐
│ AddOp  │  │ MulOp  │    │ ReshapeOp│
│        │  │        │    │          │
│ ✓ impl │  │ ✓ impl │   │ ✓ impl   │
└────────┘  └────────┘    └──────────┘
```

Each operation implements the interface **differently**, but all provide the **same method signature**.

### Types of Interfaces in MLIR

MLIR has several interface types:

1. **OpInterface**: For operations (what we'll use)
2. **TypeInterface**: For custom types
3. **AttrInterface**: For custom attributes
4. **DialectInterface**: For entire dialects

We'll focus on `OpInterface` in this chapter.

### Interface vs. Trait

You might be confused with **traits** from earlier chapters. Here's the distinction:

**Traits:**
- Compile-time properties
- Static checks and constraints
- Example: `Pure` (operation has no side effects)
- No runtime behavior

**Interfaces:**
- Runtime polymorphism
- Dynamic method dispatch
- Example: `ShapeInference` (operation can infer shapes)
- Provides behavior (methods)

**Analogy:**
- **Trait** = "This operation IS pure" (property)
- **Interface** = "This operation CAN infer shapes" (capability)

---

## 5.3 Defining an Interface in TableGen

Let's define a shape inference interface step by step.

### Step 1: Create the TableGen File

Create `ShapeInferenceInterface.td`:

```tablegen
#ifndef SHAPE_INFERENCE_INTERFACE
#define SHAPE_INFERENCE_INTERFACE

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}

#endif // SHAPE_INFERENCE_INTERFACE
```

Let's break this down piece by piece.

### Step 2: Understanding OpInterface

```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
```

**What this does:**
- `def ShapeInferenceOpInterface`: Name used in TableGen (for reference)
- `: OpInterface<"ShapeInference">`: Inherits from OpInterface
- `"ShapeInference"`: The C++ class name that will be generated

**Generated C++ class hierarchy:**
```cpp
class ShapeInference {  // Abstract interface
  virtual void inferShapes() = 0;  // Pure virtual
};
```

### Step 3: Understanding InterfaceMethod

```tablegen
let methods = [
  InterfaceMethod<"Infer and set the output shape for the current operation.",
                  "void", "inferShapes">
];
```

**InterfaceMethod takes 3 parameters:**

1. **Description** (`"Infer and set..."`)
   - Documentation for the method
   - Shows up in generated docs

2. **Return type** (`"void"`)
   - C++ return type
   - Can be any valid C++ type

3. **Method name** (`"inferShapes"`)
   - Name of the method to implement
   - Must be a valid C++ identifier

**This generates:**
```cpp
class ShapeInference {
public:
  /// Infer and set the output shape for the current operation.
  virtual void inferShapes() = 0;
};
```

### Step 4: Interface with Arguments

Interfaces can have methods with arguments:

```tablegen
InterfaceMethod<
  "Check if two types are compatible",
  "bool", "areTypesCompatible",
  (ins "Type":$lhs, "Type":$rhs)
>
```

**This generates:**
```cpp
virtual bool areTypesCompatible(Type lhs, Type rhs) = 0;
```

The `(ins ...)` syntax is the same DAG syntax we saw in Chapter 3!

### Step 5: Default Implementations

Interfaces can provide **default implementations**:

```tablegen
InterfaceMethod<
  "Return the number of operands",
  "unsigned", "getNumOperands",
  (ins), [{
    return $_op.getNumOperands();  // Default implementation
  }]
>
```

**The extra argument is C++ code:**
- `$_op`: Special variable referring to the operation
- Code is injected into the generated class

Operations can **override** the default if needed, or use it as-is.

---

## 5.4 Declaring Interfaces on Operations

Now that we have an interface, let's attach it to operations!

### Method 1: Direct Declaration

In `Ops.td`:

```tablegen
def AddOp : Toy_Op<"add", [
    Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
  ]> {
  let summary = "element-wise addition operation";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  
  // ... rest of definition
}
```

**Key part:**
```tablegen
DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
```

**What this does:**
1. Declares that `AddOp` implements `ShapeInference` interface
2. Generates a declaration in the C++ header: `void inferShapes();`
3. You must provide the implementation in your `.cpp` file

**Generated C++ (in Ops.h):**
```cpp
class AddOp : public /* ... */ ShapeInference {
public:
  // Interface method declaration
  void inferShapes();
};
```

### Method 2: Selective Method Declaration

If an interface has multiple methods, you can selectively declare which ones to implement:

```tablegen
DeclareOpInterfaceMethods<MyInterface, ["method1", "method3"]>
```

This only declares `method1` and `method3`, using defaults for others.

### Method 3: Using Traits

The trait list is where you declare all compile-time and runtime behavior:

```tablegen
def MulOp : Toy_Op<"mul", [
    Pure,                                               // Trait
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>  // Interface
  ]> {
  // ...
}
```

**Traits and interfaces coexist** in the same list!

---

## 5.5 Implementing Interface Methods

Now we need to **implement** the interface methods in C++.

### Implementation Location

Interface implementations go in your dialect's `.cpp` file (e.g., `Dialect.cpp`).

### Example: AddOp Shape Inference

```cpp
/// Infer the output shape of the AddOp, this is required by the shape inference
/// interface.
void AddOp::inferShapes() {
  getResult().setType(getLhs().getType());
}
```

**Breaking it down:**

```cpp
void AddOp::inferShapes() {
  // AddOp performs element-wise addition
  // Output shape = input shape (both inputs must match)
  
  getResult().setType(      // Set the type of the result
    getLhs().getType()      // Use the left-hand side's type
  );
}
```

**Why this works:**
- Element-wise operations preserve shape
- If `lhs` is `tensor<2x3xf64>`, result is also `tensor<2x3xf64>`
- We could also use `getRhs().getType()` (they should match)

### Example: TransposeOp Shape Inference

Transpose needs different logic:

```cpp
void TransposeOp::inferShapes() {
  auto inputType = cast<RankedTensorType>(getOperand().getType());
  
  // Get the input dimensions
  SmallVector<int64_t, 2> dims(llvm::reverse(inputType.getShape()));
  
  // Create the transposed type
  getResult().setType(
    RankedTensorType::get(dims, inputType.getElementType())
  );
}
```

**Breaking it down:**

```cpp
// Input: tensor<2x3xf64>
auto inputType = cast<RankedTensorType>(getOperand().getType());

// dims = [2, 3]
SmallVector<int64_t, 2> dims(llvm::reverse(inputType.getShape()));
// After reverse: dims = [3, 2]

// Output: tensor<3x2xf64>
getResult().setType(RankedTensorType::get(dims, inputType.getElementType()));
```

**The process:**
1. Get input shape: `[2, 3]`
2. Reverse dimensions: `[3, 2]`
3. Create new type with reversed shape

### Example: ReshapeOp Shape Inference

Reshape is special - the output shape is an **attribute** of the operation:

```cpp
void ReshapeOp::inferShapes() {
  // The output shape is specified as an attribute
  // Just use it directly!
  getResult().setType(getType());
}
```

**Why this is trivial:**
```mlir
%1 = toy.reshape(%0 : tensor<6xf64>) to tensor<2x3xf64>
                                        ^^^^^^^^^^^^^^^^
                                        Output type is explicit!
```

The reshape operation already knows its output type (it's part of the operation syntax), so "inferring" just means using what's already there.

### Example: CastOp Shape Inference

Cast operations also have explicit types:

```cpp
void CastOp::inferShapes() {
  // Cast also has an explicit output type
  getResult().setType(getInput().getType());
}
```

---

## 5.6 The Shape Inference Pass

Now let's see how to **use** the interface in a generic pass!

### The Algorithm

From `toy/Ch4/mlir/ShapeInferencePass.cpp`:

```cpp
struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<toy::FuncOp>> {
  
  void runOnOperation() override {
    auto f = getOperation();

    // 1. Build worklist of operations needing shape inference
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op))
        opWorklist.insert(op);
    });

    // 2. Iteratively infer shapes
    while (!opWorklist.empty()) {
      // Find an operation whose operands are all inferred
      auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
      if (nextop == opWorklist.end())
        break;  // No progress possible

      Operation *op = *nextop;
      opWorklist.erase(op);

      // 3. Ask the operation to infer its shape
      if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
        shapeOp.inferShapes();
      } else {
        op->emitError("unable to infer shape of operation without shape "
                      "inference interface");
        return signalPassFailure();
      }
    }

    // 4. Check for success
    if (!opWorklist.empty()) {
      f.emitError("Shape inference failed, ")
          << opWorklist.size() << " operations couldn't be inferred\n";
      signalPassFailure();
    }
  }
};
```

Let's break down this algorithm step by step.

### Step 1: Build the Worklist

```cpp
llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
f.walk([&](mlir::Operation *op) {
  if (returnsDynamicShape(op))
    opWorklist.insert(op);
});
```

**What's a "dynamic shape"?**
```mlir
%0 = toy.add %arg0, %arg1 : tensor<*xf64>  ← Dynamic (unknown shape)
%1 = toy.mul %x, %y : tensor<2x3xf64>      ← Static (known shape)
```

**The helper function:**
```cpp
static bool returnsDynamicShape(Operation *op) {
  return llvm::any_of(op->getResultTypes(), [](Type resultType) {
    return !llvm::isa<RankedTensorType>(resultType);
  });
}
```

**Logic:**
- If any result is **not** a `RankedTensorType`, it needs inference
- `RankedTensorType` = tensor with known shape (e.g., `tensor<2x3xf64>`)
- `UnrankedTensorType` = tensor with unknown shape (e.g., `tensor<*xf64>`)

### Step 2: Process Ready Operations

```cpp
while (!opWorklist.empty()) {
  // Find operation where all operands have known shapes
  auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
  if (nextop == opWorklist.end())
    break;  // Stuck - no operation is ready

  Operation *op = *nextop;
  opWorklist.erase(op);
  
  // Infer shapes for this operation
  // ...
}
```

**Why this order matters:**
```mlir
%0 = toy.constant ...       : tensor<2x3xf64>   ← Known from start
%1 = toy.add %arg0, %arg1   : tensor<*xf64>     ← Unknown (needs args)
%2 = toy.mul %0, %1         : tensor<*xf64>     ← Depends on %1
```

**Processing order:**
1. Start with `%0` (constant has known shape)
2. Can't process `%1` yet (`%arg0` and `%arg1` unknown)
3. Wait until function arguments are inferred
4. Then process `%1`
5. Finally process `%2` (now `%1` is known)

**The helper function:**
```cpp
static bool allOperandsInferred(Operation *op) {
  return llvm::all_of(op->getOperandTypes(), [](Type operandType) {
    return llvm::isa<RankedTensorType>(operandType);
  });
}
```

Returns `true` only if **all** operands have ranked (known) types.

### Step 3: Invoke the Interface

```cpp
if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
  shapeOp.inferShapes();
} else {
  op->emitError("unable to infer shape of operation without shape "
                "inference interface");
  return signalPassFailure();
}
```

**This is the magic line:**
```cpp
if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
```

**What happens:**
1. Try to cast `Operation*` to `ShapeInference` interface
2. If operation implements the interface → cast succeeds
3. If operation doesn't implement → cast returns `nullptr`

**Then:**
```cpp
shapeOp.inferShapes();
```

Calls the **operation's specific implementation**! Polymorphism in action!

### Step 4: Verify Completion

```cpp
if (!opWorklist.empty()) {
  f.emitError("Shape inference failed, ")
      << opWorklist.size() << " operations couldn't be inferred\n";
  signalPassFailure();
}
```

If worklist isn't empty, we got stuck. This happens when:
- Operation doesn't implement interface (error in step 3)
- Circular dependencies (shouldn't happen in SSA)
- Missing information (e.g., function args not specialized)

---

## 5.7 Complete Example Walkthrough

Let's trace shape inference through a complete example!

### Input: Toy Code with Generic Calls

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var c = multiply_transpose(a, b);
  print(c);
}
```

### Before Shape Inference

Without specialization, `multiply_transpose` has generic types:

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) 
                          -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}

toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %2 = toy.generic_call @multiply_transpose(%0, %1) 
       : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  toy.print %2 : tensor<*xf64>
  toy.return
}
```

**Problems:**
- `multiply_transpose` has all `tensor<*xf64>` (unknown shapes)
- Can't optimize without knowing shapes
- Can't generate efficient code

### Step 1: Function Specialization (via Inlining)

First, the inliner pass copies `multiply_transpose` into `main`:

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  
  // Inlined multiply_transpose body:
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<*xf64>
  %3 = toy.transpose(%1 : tensor<2x3xf64>) to tensor<*xf64>
  %4 = toy.mul %2, %3 : tensor<*xf64>
  
  toy.print %4 : tensor<*xf64>
  toy.return
}
```

**Now:**
- Constants have known types: `tensor<2x3xf64>`
- But transposes and mul still unknown: `tensor<*xf64>`

### Step 2: Shape Inference Pass

**Initial worklist:**
```
Operations needing inference:
- %2 = toy.transpose (unknown output)
- %3 = toy.transpose (unknown output)
- %4 = toy.mul (unknown output)
```

**Iteration 1: Process %2**
```cpp
allOperandsInferred(%2) → true (operand %0 is tensor<2x3xf64>)
```

Call `%2.inferShapes()`:
```cpp
void TransposeOp::inferShapes() {
  // Input: tensor<2x3xf64>
  // Reverse dims: [3, 2]
  // Output: tensor<3x2xf64>
  getResult().setType(RankedTensorType::get({3, 2}, F64Type));
}
```

**After iteration 1:**
```mlir
%2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>  ✓
%3 = toy.transpose(%1 : tensor<2x3xf64>) to tensor<*xf64>
%4 = toy.mul %2, %3 : tensor<*xf64>
```

**Iteration 2: Process %3**
```cpp
allOperandsInferred(%3) → true (operand %1 is tensor<2x3xf64>)
```

Call `%3.inferShapes()`:
```cpp
// Same logic: tensor<2x3xf64> → tensor<3x2xf64>
```

**After iteration 2:**
```mlir
%2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>  ✓
%3 = toy.transpose(%1 : tensor<2x3xf64>) to tensor<3x2xf64>  ✓
%4 = toy.mul %2, %3 : tensor<*xf64>
```

**Iteration 3: Process %4**
```cpp
allOperandsInferred(%4) → true (both %2 and %3 are tensor<3x2xf64>)
```

Call `%4.inferShapes()`:
```cpp
void MulOp::inferShapes() {
  // Element-wise: output = input shape
  getResult().setType(getLhs().getType());  // tensor<3x2xf64>
}
```

**After iteration 3:**
```mlir
%2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>  ✓
%3 = toy.transpose(%1 : tensor<2x3xf64>) to tensor<3x2xf64>  ✓
%4 = toy.mul %2, %3 : tensor<3x2xf64>  ✓
```

**Worklist is empty → Success!**

### Final Output

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.transpose(%1 : tensor<2x3xf64>) to tensor<3x2xf64>
  %4 = toy.mul %2, %3 : tensor<3x2xf64>
  toy.print %4 : tensor<3x2xf64>
  toy.return
}
```

**All shapes known!** Now we can:
- Optimize (canonicalization knows exact shapes)
- Generate efficient code (allocate exact sizes)
- Verify correctness (check shape compatibility)

---

## 5.8 Running Chapter 4 with Shape Inference

Let's see this in action!

### Building

```powershell
cmake --build build --target toyc-ch4
```

### Creating a Test File

Create `test_shape.toy`:

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var c = multiply_transpose(a, b);
  print(c);
}
```

### Without Optimization

```powershell
cd toy\Ch4
..\..\build\toy\Ch4\toyc-ch4.exe -emit=mlir test_shape.toy
```

**Output:**
```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) 
                            -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.mul %0, %1 : tensor<*xf64>
    toy.return %2 : tensor<*xf64>
  }
  
  toy.func @main() {
    %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    %1 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    %2 = toy.generic_call @multiply_transpose(%0, %1) 
         : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    toy.print %2 : tensor<*xf64>
    toy.return
  }
}
```

Notice all the `tensor<*xf64>` (unknown shapes)!

### With Optimization (Shape Inference)

```powershell
..\..\build\toy\Ch4\toyc-ch4.exe -emit=mlir -opt test_shape.toy
```

**Output:**
```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %2 = toy.mul %1, %1 : tensor<3x2xf64>
    toy.print %2 : tensor<3x2xf64>
    toy.return
  }
}
```

**What happened:**
1. ✅ Inliner eliminated `multiply_transpose` function
2. ✅ Shape inference propagated all types to `tensor<3x2xf64>`
3. ✅ Canonicalizer optimized `transpose(a) * transpose(b)` with `a == b` → reused transpose
4. ✅ All shapes are concrete!

---

## 5.9 The Generated Code

Let's peek at what TableGen generates from our interface definition.

### The Interface Header

From the TableGen definition:
```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}
```

**TableGen generates** (in `ShapeInferenceOpInterfaces.h.inc`):

```cpp
namespace mlir {
namespace toy {
namespace detail {

/// Interface definition
class ShapeInferenceOpInterfaceInterfaceTraits {
public:
  /// The fallback model implementation.
  struct FallbackModel : public Concept {
    void inferShapes(::mlir::Operation *tablegen_opaque_val) const override {
      throw std::logic_error("inferShapes not implemented");
    }
  };
  
  /// The concept declaration.
  struct Concept {
    virtual ~Concept() = default;
    virtual void inferShapes(::mlir::Operation *tablegen_opaque_val) const = 0;
  };
  
  /// The model implementation.
  template <typename ConcreteOp>
  struct Model : public Concept {
    void inferShapes(::mlir::Operation *tablegen_opaque_val) const final {
      return llvm::cast<ConcreteOp>(tablegen_opaque_val).inferShapes();
    }
  };
};

} // namespace detail

/// The user-facing interface class
class ShapeInference : public ::mlir::OpInterface<ShapeInference, 
                                                   detail::ShapeInferenceOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ShapeInference, 
                             detail::ShapeInferenceOpInterfaceInterfaceTraits>::OpInterface;
  
  /// Infer and set the output shape for the current operation.
  void inferShapes() {
    return getImpl()->inferShapes(getOperation());
  }
};

} // namespace toy
} // namespace mlir
```

**This is complex C++ template magic that enables:**
1. Dynamic dispatch (`dyn_cast<ShapeInference>(op)`)
2. Type-safe method calls
3. Compile-time checking

**You don't need to understand all this!** The key takeaway:
- TableGen generates boilerplate
- You just use `ShapeInference` interface
- Polymorphism "just works"

---

## 5.10 Design Patterns with Interfaces

### Pattern 1: Query Interfaces

Check if an operation has an interface:

```cpp
if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
  // Operation implements ShapeInference
  shapeOp.inferShapes();
} else {
  // Operation doesn't implement ShapeInference
}
```

### Pattern 2: Require Interfaces

Fail if operation doesn't implement interface:

```cpp
auto shapeOp = dyn_cast<ShapeInference>(op);
if (!shapeOp) {
  op->emitError("operation must implement ShapeInference");
  return failure();
}
shapeOp.inferShapes();
```

### Pattern 3: Walk with Interface Filter

Process only operations with an interface:

```cpp
getOperation().walk([](ShapeInference shapeOp) {
  // This lambda only called for ops implementing ShapeInference!
  shapeOp.inferShapes();
});
```

**Note:** The walk automatically filters by interface!

### Pattern 4: Multiple Interfaces

Operations can implement multiple interfaces:

```tablegen
def CastOp : Toy_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>
  ]> {
  // ...
}
```

Then use both:

```cpp
if (auto castOp = dyn_cast<CastOpInterface>(op)) {
  if (castOp.areCastCompatible(inputs, outputs)) {
    // Valid cast
  }
}

if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
  shapeOp.inferShapes();
}
```

---

## 5.11 Other MLIR Interfaces

Shape inference is just one example. MLIR provides many built-in interfaces:

### CastOpInterface

For cast/conversion operations:

```cpp
class CastOpInterface {
  /// Returns true if types are compatible for this cast
  bool areCastCompatible(TypeRange inputs, TypeRange outputs);
};
```

**Used by:**
- Verification (check cast is valid)
- Optimization (eliminate redundant casts)

### CallOpInterface & CallableOpInterface

For function calls:

```cpp
class CallOpInterface {
  /// Returns the callee of this call operation
  CallInterfaceCallable getCallableForCallee();
  
  /// Returns the operands passed to the callee
  Operation::operand_range getArgOperands();
};
```

**Used by:**
- Inlining (knowing what to inline)
- Interprocedural analysis

### InferTypeOpInterface

Similar to shape inference but for types:

```cpp
class InferTypeOpInterface {
  /// Infer the result types from operand types
  LogicalResult inferReturnTypes(/* ... */);
};
```

**Used by:**
- Type inference during parsing
- Builder methods

### MemoryEffectOpInterface

Describes memory side effects:

```cpp
class MemoryEffectOpInterface {
  /// Get the memory effects of this operation
  void getEffects(SmallVectorImpl<MemoryEffect> &effects);
};
```

**Used by:**
- Dead code elimination (pure operations can be removed)
- Reordering optimizations

### LoopLikeOpInterface

For loop constructs:

```cpp
class LoopLikeOpInterface {
  /// Returns if this operation is a loop
  bool isDefinedOutsideOfLoop(Value value);
  
  /// Move out of loop if safe
  LogicalResult moveOutOfLoop(Operation *op);
};
```

**Used by:**
- Loop-invariant code motion
- Loop analysis and transformations

---

## 5.12 Benefits of the Interface System

Let's reflect on what interfaces give us:

### 1. Extensibility

**Without interfaces:**
```cpp
// Must modify this code for every new operation!
if (isa<AddOp>(op)) {
  // ...
} else if (isa<MulOp>(op)) {
  // ...
} else if (isa<NewOp>(op)) {  // ← Must add this!
  // ...
}
```

**With interfaces:**
```cpp
// No changes needed when adding new operations!
if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
  shapeOp.inferShapes();  // Works for any op with interface
}
```

### 2. Dialect Independence

**Shape inference pass works for:**
- Toy dialect ✅
- TensorFlow dialect ✅
- Your custom dialect ✅
- Any dialect that implements `ShapeInference` ✅

### 3. Separation of Concerns

- **Pass knows:** How to iterate and apply transformations
- **Operation knows:** Its specific transformation logic
- **Interface knows:** The contract between them

### 4. Compile-Time Safety

TableGen generates type-safe code:
```cpp
shapeOp.inferShapes();  // Compile error if signature wrong!
```

### 5. Documentation

Interfaces serve as **living documentation**:
```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];
  // ...
}
```

Anyone reading this knows **exactly** what the interface does and why it exists.

---

## Summary

Let's recap what we've learned:

### Key Concepts

1. **Interfaces Enable Polymorphism**
   - Generic algorithms work across operations
   - Operations provide specific implementations
   - Loose coupling between pass and operation

2. **Defining Interfaces in TableGen**
   - `OpInterface<"Name">` for operation interfaces
   - `InterfaceMethod<description, returnType, name, args, defaultImpl>`
   - Can provide default implementations

3. **Declaring Interface Implementation**
   - Use `DeclareOpInterfaceMethods<Interface>` in operation traits
   - Generates method declarations in C++ header
   - Must provide implementation in `.cpp` file

4. **Using Interfaces**
   - `dyn_cast<Interface>(op)` to query
   - Call methods polymorphically
   - Works across dialect boundaries

5. **Shape Inference Pattern**
   - Build worklist of unknown-shape operations
   - Process operations when operands are ready
   - Invoke `inferShapes()` via interface
   - Iterate until fixed point

6. **MLIR Provides Many Interfaces**
   - `CastOpInterface`, `CallOpInterface`, `InferTypeOpInterface`
   - `MemoryEffectOpInterface`, `LoopLikeOpInterface`
   - All follow the same pattern

### What We Built

- ✅ `ShapeInferenceOpInterface` definition
- ✅ Implementation for `AddOp`, `MulOp`, `TransposeOp`, etc.
- ✅ Generic shape inference pass
- ✅ Complete pipeline: inline → infer → optimize

### Design Philosophy

**"Program to an interface, not an implementation"** - Gang of Four

MLIR's interface system embodies this principle:
- Passes depend on interfaces (contracts)
- Operations provide implementations (concrete behavior)
- New operations integrate seamlessly (extensibility)

---

## What's Next

In **Chapter 6**, we'll do our first **partial lowering**! We'll transform high-level Toy operations into:

- **Affine dialect**: Perfect loop nests with compile-time analysis
- **Loop nests**: For matrix multiplication, transpose, etc.
- **Memory operations**: Actual loads and stores

This is where MLIR's multi-level power really shines - progressive lowering from high-level to low-level!

---

## Exercises

### Exercise 1: Implement a New Interface

Define a `SymmetricOpInterface` for operations that are symmetric (commutative):

```tablegen
def SymmetricOpInterface : OpInterface<"SymmetricOp"> {
  let description = [{
    Interface for operations where operand order doesn't matter.
    E.g., a + b = b + a
  }];
  
  let methods = [
    InterfaceMethod<"Returns true if this operation is symmetric",
                    "bool", "isSymmetric">
  ];
}
```

**Tasks:**
1. Create the `.td` file
2. Declare the interface on `AddOp` and `MulOp`
3. Implement the method (return `true`)
4. Write a pass that finds all symmetric operations

### Exercise 2: Extend Shape Inference

Add a `SimplifyRedundantCast` pattern that removes casts where input and output types match:

```cpp
struct SimplifyRedundantCast : public OpRewritePattern<CastOp> {
  LogicalResult matchAndRewrite(CastOp op, PatternRewriter &rewriter) const {
    // TODO: Check if input type == output type
    // TODO: If yes, replace cast with input value
  }
};
```

**Hint:** Use `getInput().getType()` and `getType()`.

### Exercise 3: Trace Shape Inference

Given this code:
```toy
def foo(a, b) {
  var c = a + b;
  var d = transpose(c);
  return d * d;
}

def main() {
  var x<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var y<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var z = foo(x, y);
  print(z);
}
```

**Tasks:**
1. Draw the initial IR with `tensor<*xf64>` types
2. Show the IR after inlining
3. Trace each iteration of shape inference
4. Show the final IR with concrete types

### Exercise 4: Design a Custom Interface

Design an interface for operations that can be auto-differentiated:

```tablegen
def AutoDiffOpInterface : OpInterface<"AutoDiffOp"> {
  let methods = [
    // TODO: What methods would you need?
    // Hints: gradient computation, chain rule, etc.
  ];
}
```

**Think about:**
- What information does auto-diff need?
- How would operations implement this?
- What would an auto-diff pass look like?

---

## Further Reading

### MLIR Documentation

- **Interfaces**: [https://mlir.llvm.org/docs/Interfaces/](https://mlir.llvm.org/docs/Interfaces/)
- **Operation Definition Specification**: [https://mlir.llvm.org/docs/DefiningDialects/Operations/](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- **Toy Tutorial Chapter 4**: [https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/)

### Design Patterns

- **Interface Segregation Principle** (SOLID principles)
  - Keep interfaces small and focused
  - Better to have many small interfaces than one large one

- **Strategy Pattern**
  - Interfaces are the "strategy" for operations
  - Different operations = different strategies

### Academic Papers

- **Extensible Compilers**
  - How to build compilers that others can extend
  - Interface-based design is key

---

## Reflection Questions

Before moving to Chapter 6, consider:

1. **Interface Granularity**
   - Should each operation have its own interface?
   - Or should we group similar operations?
   - What are the trade-offs?

2. **Interface Stability**
   - What happens if you change an interface method signature?
   - How do you maintain backward compatibility?
   - When is it OK to break interfaces?

3. **Performance**
   - Does polymorphism have runtime cost?
   - Is virtual dispatch expensive in compilers?
   - How can we optimize interface calls?

4. **Alternative Approaches**
   - Could we use templates instead of interfaces?
   - What about static polymorphism?
   - When is dynamic dispatch necessary?

These questions touch on fundamental software engineering trade-offs!

---

**Next up:** Chapter 6, where we lower Toy operations to Affine loops and see MLIR's multi-level IR in action! 🔄
