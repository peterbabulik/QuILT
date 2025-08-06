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

