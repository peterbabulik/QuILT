# QuILT: Quantum-Inspired Learning in Tensor space

**QuILT (Quantum-Inspired Learning in Tensor space)** is a pure Python and PyTorch implementation of a hybrid AI model grounded in the principles of Hilbert space learning and quantum mechanics. This project explores building powerful, data-efficient classifiers from first principles, without reliance on external quantum computing libraries like Qiskit or Quimb.

---

## Core Concepts

This repository is a practical, hands-on exploration inspired by the paper ["Operator-Based Machine Intelligence: A Hilbert Space Framework for Spectral Learning and Symbolic Reasoning"](https://arxiv.org/abs/2507.21189v1). The core idea is to reframe machine learning not as traditional neural network optimization, but as learning an **operator** that acts on data represented as **vectors in a Hilbert space**.

This mathematical language is identical to that of quantum mechanics. QuILT leverages this synergy by building a **fully-differentiable quantum circuit simulator** that acts as a trainable classifier.

The final model achieves **86.67% accuracy** on the non-linear "concentric circles" dataset, demonstrating the power of this approach.

## How It Works

The model, `QuantumCircuitSimulator`, implements the following steps from first principles:

1.  **State Representation:** An N-qubit quantum state `|ψ⟩` is represented as a `2^N` dimensional complex vector using `torch.Tensor`.
2.  **Data Encoding:** Input data features are encoded into the initial state by applying parameterized single-qubit rotation gates (e.g., `RY`).
3.  **Learnable Operator (Variational Circuit):** A sequence of learnable rotation gates (`RY`, `RZ`) and fixed entangling gates (`CNOT`) are applied to the state. These gates form the "operator" that the model learns during training.
4.  **Differentiable Gates:** All quantum gates are constructed using differentiable PyTorch operations (e.g., `torch.stack`, `torch.kron`) to ensure the computational graph is preserved for backpropagation.
5.  **Measurement:** The expectation value of an observable (Pauli-Z on the first qubit) is calculated to produce a single output logit for classification.
6.  **Training:** The entire model is an `nn.Module`, trained end-to-end with a standard PyTorch optimizer (`Adam`) and loss function (`BCEWithLogitsLoss`).

---

## QuILT-NAS: Quantum-Inspired Neural Architecture Search

**QuILT-NAS** extends the core concepts of QuILT to a more complex and powerful application: Neural Architecture Search (NAS). This project uses the same first-principles, quantum-inspired optimizer to discover the optimal architecture for a small Transformer model (`picoTransformer`) designed for mathematical reasoning.

The system successfully executes an end-to-end pipeline where a Variational Quantum Eigensolver (VQE) identifies the best-performing `picoTransformer` architecture from a search space of 16 candidates, guided by a semantic cost function provided by a large language model (LLM) acting as an objective "judge."

### The Technical Pipeline

1.  **The Target Model (`picoTransformer`):** A minimal, character-level Transformer decoder whose key hyperparameters (layers, embedding size, etc.) are configurable.
2.  **The Search Space:** A discrete set of 16 architectures is defined by mapping 4-bit strings (the basis states of a 4-qubit system) to specific `picoTransformer` configurations.
3.  **The "LLM-as-a-Judge" Evaluation:**
    -   Each of the 16 candidate architectures is trained on a subset of the DeepMind Mathematics Dataset.
    -   The trained model is prompted to solve an unseen math problem.
    -   An LLM provides an objective "mathematical correctness" score (1-10) for each answer.
    -   The final `cost` for the architecture is `-correctness_score`.
4.  **The Optimizer (`VQEOptimizer`):**
    -   A `torch.nn.Module` that simulates a Variational Quantum Eigensolver.
    -   The **Problem Hamiltonian `H`** is a `16x16` diagonal matrix where the diagonal entries are the costs derived from the LLM judge.
    -   The VQE is trained to find the quantum state that minimizes the expectation value `⟨ψ(θ)|H|ψ(θ)⟩`, thereby finding the architecture with the best score.

### Final Result: The Winning Architecture

After a full evaluation, `QuILT-NAS` confidently recommended the following architecture as the most capable mathematical reasoner in the search space:

-   **Name:** `Medium-Deep`
-   **Configuration:** 6 layers, 96-dimensional embeddings, 6 attention heads, no dropout, and a learning rate of `1e-3`.

This result demonstrates the success of the `QuILT-NAS` pipeline in performing a sophisticated, semantically-guided optimization, suggesting that for this mathematical reasoning task, **model depth was a more critical factor than width or regularization.**
