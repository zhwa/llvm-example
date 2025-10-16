# MLIR for Beginners: A Practical Guide

Welcome to **MLIR for Beginners**, a comprehensive, hands-on guide to understanding and using MLIR (Multi-Level Intermediate Representation) - the compiler infrastructure framework that's revolutionizing how we build compilers, DSLs, and code generators.

## 📚 About This Book

This repository contains:
- **10 comprehensive chapters** covering MLIR from first principles to production deployment
- **Complete code examples** from the Toy tutorial and custom projects
- **90,000+ words** of educational content explaining the "how" and "why" of MLIR
- **Practical examples** including Linalg/GEMM, Kaleidoscope, and custom dialects

### What Makes This Different?

Unlike existing tutorials that jump straight into code:
- ✅ **Beginner-friendly**: Assumes no compiler knowledge
- ✅ **Explains "why"**: Design philosophy and reasoning behind decisions
- ✅ **Progressive complexity**: Each chapter introduces ONE major concept
- ✅ **Complete coverage**: From AST to JIT execution
- ✅ **TableGen mastery**: In-depth coverage of this poorly-documented language
- ✅ **Real examples**: All code drawn from this working repository

---

## 📖 The Book

### Part I: Foundation (Chapters 1-2)
Understanding the "why" of MLIR before diving into code.

#### [Chapter 1: The Compiler Problem Space](book/Chapter1_The_Compiler_Problem_Space.md)
- Why we need intermediate representations
- The multi-level IR philosophy
- MLIR's approach to compiler design
- Overview of the Toy language journey

#### [Chapter 2: Building Intuition - The AST Phase](book/Chapter2_Building_Intuition_The_AST_Phase.md)
- Traditional compiler frontend (lexer, parser)
- Abstract Syntax Trees
- The Toy language specification
- Hands-on: Building a parser

### Part II: Core MLIR Concepts (Chapters 3-5)

#### [Chapter 3: Your First MLIR Dialect](book/Chapter3_Your_First_MLIR_Dialect.md)
- What is a dialect?
- Defining operations with TableGen (ODS)
- Your first MLIR generation (MLIRGen)
- Understanding SSA form

#### [Chapter 4: Transformations and Rewriting](book/Chapter4_Transformations_and_Rewriting.md)
- Pattern matching and rewriting
- C++ patterns vs Declarative Rewrite Rules (DRR)
- Canonicalization and optimization
- Toy dialect optimizations

#### [Chapter 5: Interfaces and Reusability](book/Chapter5_Interfaces_and_Reusability.md)
- Operation interfaces
- Generic algorithms
- Shape inference implementation
- Design for extensibility

### Part III: Lowering and Execution (Chapters 6-7)

#### [Chapter 6: Partial Lowering to Affine](book/Chapter6_Partial_Lowering_to_Affine.md)
- Progressive lowering philosophy
- The Affine dialect
- Lowering Toy to loops
- Polyhedral optimization opportunities

#### [Chapter 7: Complete Lowering to LLVM and JIT](book/Chapter7_Complete_Lowering_to_LLVM_and_JIT.md)
- Lowering to LLVM dialect
- LLVM IR translation
- JIT compilation
- End-to-end execution

### Part IV: Mastery (Chapters 8-10)

#### [Chapter 8: TableGen Mastery](book/Chapter8_TableGen_Mastery.md)
- TableGen design philosophy and history
- Language fundamentals (records, classes, DAGs)
- ODS (Operation Definition Specification)
- Advanced pattern matching
- **NEW**: Comprehensive debugging guide
- **NEW**: C++ vs TableGen lowering comparison

#### [Chapter 9: Designing Custom Dialects](book/Chapter9_Designing_Custom_Dialects.md)
- Dialect design principles
- When to create new dialects
- Architecture patterns
- Complete case study: DSP dialect
- Production best practices

#### [Chapter 10: The MLIR Ecosystem](book/Chapter10_The_MLIR_Ecosystem.md)
- Building MLIR projects
- Complete compilation pipelines
- Multi-target code generation
- Production deployment
- Debugging and profiling
- Community and resources

---

## 🚀 Quick Start

### Prerequisites

- **Windows** (PowerShell)
- **CMake** 3.20+
- **vcpkg** (for LLVM/MLIR dependencies)
- **Visual Studio 2019+** or compatible compiler
- **Git**

### Setup

1. **Install vcpkg** (if not already installed):
   ```powershell
   git clone https://github.com/Microsoft/vcpkg.git D:\tools\vcpkg
   cd D:\tools\vcpkg
   .\bootstrap-vcpkg.bat
   ```

2. **Clone this repository**:
   ```powershell
   git clone https://github.com/zhwa/llvm-example.git
   cd llvm-example
   ```

3. **Install LLVM with MLIR**:
   ```powershell
   vcpkg install llvm[mlir] --triplet=x64-windows
   ```
   ⚠️ **Note**: This takes 1-2 hours on first run!

4. **Configure and build**:
   ```powershell
   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
   ```

### Run Examples

```powershell
# Toy compiler (Chapter 7)
.\build\bin\toyc-ch7.exe .\toy\Ch1\ast.toy -emit=jit

# Linalg GEMM example
.\build\bin\linalg-gemm-example.exe

# Kaleidoscope (LLVM tutorial)
.\build\bin\toyc-ch2.exe
```

---

## 📁 Repository Structure

```
llvm-example/
├── book/                          # 📚 The 10-chapter book
│   ├── Chapter1_The_Compiler_Problem_Space.md
│   ├── Chapter2_Building_Intuition_The_AST_Phase.md
│   ├── Chapter3_Your_First_MLIR_Dialect.md
│   ├── Chapter4_Transformations_and_Rewriting.md
│   ├── Chapter5_Interfaces_and_Reusability.md
│   ├── Chapter6_Partial_Lowering_to_Affine.md
│   ├── Chapter7_Complete_Lowering_to_LLVM_and_JIT.md
│   ├── Chapter8_TableGen_Mastery.md
│   ├── Chapter9_Designing_Custom_Dialects.md
│   └── Chapter10_The_MLIR_Ecosystem.md
│
├── toy/                           # 🎯 Main MLIR tutorial (Ch1-Ch7)
│   ├── Ch1/                       # AST only
│   ├── Ch2/                       # First MLIR dialect
│   ├── Ch3/                       # Transformations
│   ├── Ch4/                       # Interfaces
│   ├── Ch5/                       # Lowering to Affine
│   ├── Ch6/                       # Lowering to LLVM
│   └── Ch7/                       # JIT execution
│
├── Others/
│   ├── LinalgGEMM/                # 🆕 Linalg dialect GEMM example
│   └── llvm-pass-skeleton/        # LLVM pass template
│
├── Kaleidoscope/                  # Traditional LLVM tutorials (Ch2-Ch9)
├── LLVMCoreLibExamples/           # LLVM library examples
│
├── cmake-modules/                 # Build system helpers
├── CMakeLists.txt                 # Root build configuration
├── CMakePresets.json              # Build presets
└── vcpkg.json                     # LLVM/MLIR dependencies
```

### Key Directories

- **`book/`**: All 10 chapters in Markdown format
- **`toy/`**: Progressive MLIR tutorial (the core of the book)
- **`Others/LinalgGEMM/`**: NEW! Practical Linalg dialect example
- **`Kaleidoscope/`**: Traditional LLVM tutorials for comparison
- **`LLVMCoreLibExamples/`**: LLVM API usage examples

---

## 🎓 Learning Path

### For Complete Beginners

1. Start with **Chapter 1** (motivation and context)
2. Follow **Chapters 2-7** in order (core tutorial)
3. Build and run each `toy/ChX/` example as you read
4. Do the exercises at the end of each chapter

### For Experienced Compiler Developers

1. Skim **Chapters 1-2** (you likely know this)
2. Focus on **Chapters 3-5** (MLIR specifics)
3. Deep dive into **Chapter 8** (TableGen mastery)
4. Study **Chapter 9** (dialect design patterns)

### For Specific Topics

- **TableGen**: Chapter 8 (comprehensive guide)
- **Lowering**: Chapters 6-7
- **Optimization**: Chapters 4-5
- **Production Deployment**: Chapter 10
- **Linalg/GEMM**: `Others/LinalgGEMM/` + Chapter 6

---

## 🔧 Build System Details

This project uses **vcpkg** for dependency management and **CMake** for building.

### CMake Configuration

```cmake
# Key CMake variables (from CMakeLists.txt)
CMAKE_CXX_STANDARD=20              # C++20 required
LLVM_DIR                           # Found via vcpkg
MLIR_DIR                           # Found via vcpkg
```

### vcpkg Integration

The project automatically integrates vcpkg through `cmake-modules/AzureVcpkg.cmake` (borrowed from Azure SDK).

**To use a different vcpkg location**, update `CMakePresets.json`:
```json
{
  "cacheVariables": {
    "CMAKE_TOOLCHAIN_FILE": "YOUR_PATH/vcpkg/scripts/buildsystems/vcpkg.cmake"
  }
}
```

### Building Individual Projects

```powershell
# Build specific target
cmake --build build --target toyc-ch7

# Build Linalg GEMM example
cmake --build build --target linalg-gemm-example

# Build everything
cmake --build build
```

---

## ⚠️ Important Notes

### Compiler Flags

- **Do NOT use `/sdl`** (Enable Additional Security Checks) - causes LLVM JIT failures
- **Use `/EHsc`** for exception handling
- **Windows-specific**: Uses PowerShell commands in examples

### Build Time

- **First build with vcpkg**: 1-2 hours (LLVM compilation)
- **Subsequent builds**: Minutes
- **Incremental builds**: Seconds

### Common Issues

1. **vcpkg path**: Update `CMakePresets.json` if vcpkg is not in `D:/tools/vcpkg`
2. **Long paths**: Enable long path support on Windows
3. **Disk space**: LLVM requires ~20GB

---

## 🤝 Contributing

This is an educational resource. Contributions welcome:

- **Typo fixes**: Submit PR
- **Clarifications**: Open issue
- **New examples**: Submit PR with explanation
- **Questions**: Use GitHub Discussions

---

## 📚 Additional Resources

### Official MLIR Resources
- [MLIR Website](https://mlir.llvm.org/)
- [MLIR Documentation](https://mlir.llvm.org/docs/)
- [LLVM Discourse (MLIR Category)](https://discourse.llvm.org/c/mlir/)

---

## 📄 License

This project is licensed under the Apache License 2.0 with LLVM Exceptions - see the [LICENSE](LICENSE) file for details.

---

## 🎯 Goals of This Book

By the end of this book, you will be able to:

- ✅ Understand the **why** behind MLIR's design
- ✅ Build **custom dialects** for your domain
- ✅ Write **transformations** in both C++ and TableGen
- ✅ **Lower** from high-level to executable code
- ✅ **Debug** MLIR programs effectively
- ✅ **Deploy** MLIR compilers to production
- ✅ **Contribute** to the MLIR ecosystem

---

## 🙏 Acknowledgments

This repository contains several categories of examples:

- **`LLVMCoreLibExamples/`**: Most examples in this folder are based on code pieces from *Getting Started with LLVM Core Libraries*.
- **`Kaleidoscope/`**: The official Kaleidoscope examples, except for the MCJIT parts.
- **`toy/`**: The official Toy MLIR example (basis for Chapters 1-7 of this book).
- **`Others/`**: Miscellaneous examples including the new LinalgGEMM demonstration.

---

**Happy learning! Start with [Chapter 1](book/Chapter1_The_Compiler_Problem_Space.md) →**