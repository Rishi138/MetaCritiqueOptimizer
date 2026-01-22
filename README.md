# MetaCritiqueOptimizer

**MetaCritiqueOptimizer** is a state-isolated orchestration framework that treats LLM behavior as a controllable dynamic system. By utilizing **PID-governed symbolic backpropagation**, the framework automates instruction tuning to optimize agentic policies in real-time.

---

## Core Thesis: Policy over Solution
Unlike standard agents that iterate on a single code block, **MetaCritiqueOptimizer** treats the "ideal solution" as a black-box target. The framework does not optimize for a specific string of code; it optimizes the **instructional latent space** that produces ideal solutions. It bridges the gap between a problem statement and an optimal output by treating the system prompt as a differentiable parameter.

---

## Technical Architecture

### 1. Bidirectional Error Telemetry
The framework tracks performance across four bidirectional dimensions scored from -100 to +100. This captures both **under-behavior** (negative) and **over-behavior** (positive), allowing the system to identify the precise nature of the "drift" from the target policy:

* **Correctness Deviation**: Syntax/logic errors vs. excessive validation.
* **Scope Deviation**: Missing requirements vs. feature bloat/creep.
* **Simplicity Deviation**: Messy code vs. over-engineered abstractions.
* **Optimization Deviation**: Wasteful operations vs. premature/complex optimization.



### 2. PID-Governed Control Logic
To stabilize instruction tuning, the framework implements a formal feedback loop inspired by industrial control theory:

* **Proportional (P)**: Immediate correction based on the current error magnitude.
* **Integral (I)**: Sums error history over a batch of tasks to neutralize systemic biases (e.g., if the model consistently over-engineers, the I-term forces a counter-bias).
* **Derivative (D)**: Measures the rate of improvement to identify the logarithmic "learning curve."



### 3. Symbolic Backpropagation
The framework translates numerical error integrals into natural language constraints. These "symbolic gradients" are backpropagated into the agent’s configuration in two layers:

* **Evaluator Tuning**: Dynamically adjusts the "critic's" harshness to compensate for observed agent drift.
* **Policy Tuning**: Fundamental modification of the agent’s system instructions to shift its problem-solving priors (e.g., "Shift priority from abstraction to minimal execution").

> **Note**: The system uses the Integral term to not only anti-bias the current response via the evaluator, but to recursively tune the very engine being tuned—shifting the underlying policy rather than just correcting its symptoms.

---

## Performance & Convergence

* **System Convergence**: As the error integrals diminish, the controller’s aggression reduces, allowing the black-box optimization to converge on a stable, "ideal" system prompt. The instructions settle as the policy reaches equilibrium with the task environment.
* **Natural Termination**: The system identifies the point where further iteration yields diminishing returns—triggering early stopping to prevent "critique death spirals" and preserve latency.
* **Stateful Memory Loop**: A persistent observation log (`patts.txt`) acts as a long-term cell state, ensuring that learned policy adjustments carry over across different repository tasks.

---

## Final Results (SWE-bench)

| Model | Resolution Rate |
| :--- | :--- |
| **MetaCritiqueOptimizer** | **78.3%** |
| Baseline o4-mini-model | 45.0% |
| **Improvement** | **+74% Relative / +33.3 Pts** |

---

## Key Capabilities

* **Gradient-Free Optimization**: Steers black-box models without weight updates.
* **Test-Time Policy Shift**: Adapts the agent's "thinking" to the specific nuances of a codebase on the fly.
* **Autonomous Remediation**: Automatically detects and corrects "paranoid" or "wasteful" coding patterns through integral feedback.
