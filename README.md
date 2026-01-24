# MetaCritiqueOptimizer

**MetaCritiqueOptimizer** is a self-tuning agent framework that automatically optimizes its own critique system in real-time. By treating instruction prompts as controllable parameters in a feedback loop, the system eliminates manual prompt engineering while achieving high performance on repository-level coding tasks.

---

## Quick Results

| System | Resolution Rate | Percentage Point Improvement | Relative Improvement |
|:-------|:----------------|:----------------------------|:---------------------|
| Baseline o4-mini | 45.0% | — | — |
| V1: Simple Critique | 63.2% | +18.2 pts | +40.4% |
| **V2: MetaCritiqueOptimizer** | **78.3%** | **+33.3 pts** | **+74.0%** |

**SWE-bench-mini bash-only** (Princeton/Lite)

---

## The Problem: Critique Systems Need Manual Tuning

Simple recursive self-critique improves agent performance—but it requires constant manual adjustment:

* **Too harsh** → Over-engineering, "critique death spirals," unnecessary complexity
* **Too lenient** → Incomplete solutions, missed edge cases, failing tests
* **Static prompts** → Can't adapt to different codebases or problem types

The baseline critique system (V1) improved o4-mini from **45% → 63%** on SWE-bench bash-only, but hit a ceiling due to these tuning challenges.

---

## The Solution: Meta-Optimization via Control Theory

**MetaCritiqueOptimizer (V2)** adds a second-order control loop that **tunes the critique system itself**. Instead of manually adjusting prompts, the framework automatically detects systematic errors and rewrites its own instructions to compensate.

### Key Innovation

Traditional approach:
```
Problem → Agent → Solution → Critique → Better Solution
```

The approach:
```
Problem → Agent → Solution → Critique → Better Critique System and Better Agent System → Better Solution
```

By treating system prompts as differentiable parameters (in natural language space), MetaCritiqueOptimizer enables gradient-free optimization of the agent's problem-solving policy itself.

---

## Core Thesis: Optimize the Policy, Not the Solution

Unlike standard agents that iterate on code outputs, **MetaCritiqueOptimizer** treats the "ideal solution" as a black-box target. The framework doesn't optimize for a specific string of code—it optimizes the **instructional latent space** that produces ideal solutions.

This means we're not just making better code—we're making a better code-making system.

---

## Technical Architecture

### 1. Bidirectional Error Telemetry

The framework tracks performance across four bidirectional dimensions scored from **-100 to +100**. This captures both **under-behavior** (negative) and **over-behavior** (positive), allowing the system to identify the precise nature of policy drift:

| Dimension | Negative (-100) | Target (0) | Positive (+100) |
|:----------|:----------------|:-----------|:----------------|
| **Correctness** | Syntax/logic errors | Commands work correctly | Paranoid validation, excessive error handling |
| **Scope** | Missing requirements | Solves exactly the PR | Feature creep, out-of-scope changes |
| **Simplicity** | Messy, convoluted code | Clean, readable commands | Over-engineered, unnecessary complexity |
| **Optimization** | Wasteful operations | Reasonably efficient | Premature optimization, clever but complex |

**Why bidirectional matters**: In V1, it was observed that critique systems often push agents too far in one direction. A critique that successfully eliminates bugs might inadvertently encourage paranoid validation. Bidirectional tracking lets one detect and correct these overcorrections.

---

### 2. PID-Governed Control Logic

To stabilize instruction tuning, the framework implements a formal feedback loop inspired by industrial control theory:

#### Proportional (P)
Immediate correction based on current error magnitude—addresses what's wrong **right now**.

#### Integral (I)
Accumulates error history over a batch of tasks to neutralize systemic biases.

**Example**: If the agent consistently over-engineers (positive abstraction errors across 5 tasks), the I-term builds up and forces a counter-bias in the next prompt rewrite.
```python
# Discrete integral (Riemann sum)
error_accumulations["correctness"] += crit_scores["crit_correctness_score"]
```

#### Derivative (D)
Measures the rate of improvement to identify diminishing returns.

**Note:** This measures change in *distance to target* (`abs(prev) - abs(current)`) rather than a traditional derivative. Discretely, however, it is still measuring the rate of change in the error. Since the target is 0, magnitude reduction is what is being looked for, regardless of direction—moving from +50→+30 or -50→-30 both represent the same 20-point improvement toward the goal.

Used for early stopping to prevent "critique death spirals" and unnecessary API costs.
```python
# Rate of change in distance to target
improvement = abs(prev_score) - abs(current_score)

# Control decision
if avg_improvement < 4:
    stop_iteration()  # Diminishing returns detected
```

**Why PID**: In V1, the critique harshness was manually tuned through trial and error. The PID formulation automates this by treating prompt optimization like a control system—measuring error, accumulating bias signals, and adjusting aggressiveness dynamically.

---

### 3. Passive Aggressiveness Amplification

The observation log (`patts.txt`) serves dual purposes:

* **Knowledge Base**: Records actionable corrections (e.g., "Escape special characters in grep patterns")
* **Frustration Counter**: Duplicate observations indicate persistent failures—the agent repeatedly ignores feedback
* **Implicit Escalation**: The prompt optimizer receives the full observation history, including duplicates

**Mechanism**: 
- Effective corrections disappear from future critiques (behavior fixed)
- Ineffective corrections accumulate as duplicates (behavior persists)
- Observation frequency becomes a natural severity signal

**Example Escalation**:
```
Occurrence 1: "Check file existence before sed operations"
Occurrence 5: "CRITICAL: Always validate file paths before sed/awk operations"
```

Seeing the same rule 3-5 times signals ineffectiveness, automatically increasing aggressiveness in the next prompt rewrite—without explicit duplicate-counting logic.

---

### 4. Symbolic Backpropagation

**Why "backpropagation"?** In neural networks, backpropagation optimizes numeric parameters (weights) by computing gradients that show how to adjust each parameter to reduce error. This framework does something analogous, but in language space:

| Traditional Backprop | Symbolic Backprop (This System) |
|:---------------------|:--------------------------------|
| Parameters: Numeric weights | Parameters: Text prompts |
| Error signal: Loss value | Error signal: Critique scores |
| Gradient: ∂Loss/∂Weight | "Gradient": Error integrations + observations |
| Update: W -= lr * gradient | Update: LLM rewrites prompt based on errors |
| Mechanism: Chain rule + calculus | Mechanism: Pattern recognition + language |

**Why "symbolic"?** One can't compute numeric gradients through text (prompts aren't differentiable). Instead, MetaCritiqueOptimizer uses symbolic reasoning: the system interprets error patterns and generates natural language "corrections" that serve the same purpose as gradients—they tell which direction to adjust the parameters.

**The key difference:** Traditional backprop traces exactly how each weight contributes to the loss (chain rule). It can't be done with prompts. Instead, MetaCritiqueOptimizer accumulates error statistics and uses an LLM to infer what prompt changes would reduce those errors. It's gradient-free optimization guided by linguistic feedback.

---

The framework translates numerical error integrals into natural-language constraints, 
then applies these constraints in two layers:

#### Layer 1: Evaluator Tuning
Dynamically adjusts the critic's harshness to compensate for observed agent drift.
```python
# If scope errors accumulate (agent missing requirements)
new_eval_prompt = "Penalize incomplete solutions more heavily (-40 points for missing PR requirements)"

# If abstraction errors accumulate (agent over-engineering)  
new_eval_prompt = "Reward simplicity. Deduct -30 points for unnecessary complexity"
```

#### Layer 2: Policy Tuning  
Fundamental modification of the agent's system instructions to shift its problem-solving priors.
```python
# Negative scope accumulation (consistently incomplete)
new_sys_prompt = "CRITICAL: Read the FULL PR description. Solve ALL points mentioned."

# Positive abstraction accumulation (consistently over-engineered)
new_sys_prompt = "Use simple grep/sed commands. Avoid complex validation scripts."
```

**The Key Insight**: The system uses the Integral term not just to anti-bias the current response via the evaluator, but to **recursively tune the very engine being tuned**—shifting the underlying policy rather than just correcting its symptoms.

---

## Evolution: From V1 to V2

### V1: Simple Recursive Critique

**Architecture:**
- o4-mini as main agent
- gpt-5-mini as critic  
- Fixed evaluation criteria (Technical Accuracy, Completeness, Repository Understanding, Efficiency)
- Basic improvement tracking via score comparison
- Manual prompt tuning required
- Recursive critique until results diminish per response

**Results:**
- **63.16% resolved** on SWE-bench bash-only
- **+18.16 points** over baseline (45%)

**Limitations Discovered:**

From V1 code comments and testing:
```python
# "Critique authority required tuning"
# "Critique model should be tuned to prevent excessive hypothetical issues"  
# "Causes over-engineering"
# "Critique death spiral"
```

It was observed that:
1. **Over-critical prompts** caused the agent to add unnecessary error handling, breaking working solutions
2. **Under-critical prompts** let incomplete solutions pass, failing tests
3. **Manual tuning was fragile** across different repositories and problem types
4. **No systematic way** to detect when critique was helping vs. hurting

---

### V2: Meta-Optimized Critique (MetaCritiqueOptimizer)

**What Changed:**
- Added bidirectional error tracking (-100 to +100) to detect overcorrections
- Implemented PID control loop for automatic prompt tuning
- Built an observation accumulation system with passive aggressiveness amplification
- Enabled the critique system to **rewrite its own instructions** based on error patterns

**Results:**
- **78.3% resolved** on SWE-bench bash-only
- **+15.14 points** over V1 (63.16%)
- **+33.3 points** over baseline (45%)
- **74% relative improvement** over baseline

**Key Insight:**  

The jump from V1 → V2 shows that **optimizing the critique system** is just as valuable as having critique at all. The meta-optimization layer added as many points (+15) as the initial critique system did (+18).

This validates the core thesis: **second-order optimization matters**.

---

## Performance & Convergence

### System Convergence
As error integrals diminish, the controller's aggression reduces, allowing the optimization to converge on a stable, "ideal" system prompt. The instructions settle as the policy reaches equilibrium with the task environment.

### Natural Termination
The derivative term (improvement rate) identifies the point where further iteration yields diminishing returns—triggering early stopping to prevent "critique death spirals" and preserve latency.

### Stateful Memory Loop
A persistent observation log (`patts.txt`) acts as long-term memory, ensuring that learned policy adjustments carry over across different repository tasks.

### Cost Efficiency
While V2 increases API calls vs. baseline, it **reduces expensive o4-mini retries** by preventing "bad moves" through better critique. The critique investment pays for itself by avoiding wasted solution attempts.

---

## Final Results

### SWE-bench-mini bash-only (Princeton/Lite)

| System | Resolution Rate | vs. Baseline | vs. V1 |
|:-------|:----------------|:-------------|:-------|
| Baseline o4-mini | 45.0% | — | — |
| V1: Simple Critique | 63.2% | +18.2 pts | — |
| **V2: MetaCritiqueOptimizer** | **78.3%** | **+33.3 pts** | **+15.1 pts** |
| **Relative Improvement** | **+74%** | — | **+24%** |

### Relative improvements from initial baseline

| System | Resolution Rate | Percentage Point Improvement | Relative Improvement |
|:-------|:----------------|:----------------------------|:---------------------|
| Baseline o4-mini | 45.0% | — | — |
| V1: Simple Critique | 63.2% | +18.2 pts | +40.4% |
| **V2: MetaCritiqueOptimizer** | **78.3%** | **+33.3 pts** | **+74.0%** |

### Individual Model Performance
- o4-mini alone: **45.0%** resolved
- gpt-5-mini alone: **59.8%** resolved
- **Combined system: 78.3%** resolved (outperforms both)

---

## Cost Analysis

### Development Costs
| Phase | Cost |
|:------|:-----|
| V1 implementation | $0.55 |
| V1 testing and validation | $10.15 | 
| V2 implementation | $0.74 |
| V2 testing and validation | $11.11 |
| **Total** | **$22.55** |

### Production Trade-offs
 **Advantages:**
- Reduced expensive model retries (fewer bad solutions)
- Automatic prompt optimization (no manual engineering)  
- Early stopping prevents runaway costs

 **Considerations:**
- Increased latency per task (iterative critique)
- Higher total API calls (but fewer wasted calls)

### Model Costs (per 1M tokens)
| Model | Input | Output |
|:------|:------|:-------|
| o4-mini-2025-04-16 | $1.10 | $4.40 |
| gpt-5-mini-2025-08-07 | $0.25 | $2.00 |

**Strategic Choice**: Using a cheaper but vertically-specialized model (gpt-5-mini at 59.8% solo performance) to drive a more expensive general model (o4-mini at 45% solo) achieves better results (78.3%) than either alone while optimizing cost.

---

## Key Capabilities

**Gradient-Free Optimization (Black-box optimization)**  
Steers black-box models without weight updates or fine-tuning

**Test-Time Policy Shift**  
Adapts the agent's "thinking" to the specific nuances of a codebase on the fly  

**Autonomous Remediation**  
Automatically detects and corrects "paranoid" or "wasteful" coding patterns through integral feedback

**Zero Manual Tuning**  
Eliminates the prompt engineering trial-and-error cycle that plagued V1

**Cost-Aware**  
Derivative-based early stopping prevents unnecessary critique cycles while maintaining quality

---

## Implementation Notes

### Why o4-mini + gpt-5-mini?

This isn't just about cost—it's about **specialization**:

- **o4-mini** excels at general reasoning, UX, and orchestration (ideal for conversational agents)
- **gpt-5-mini** excels at vertical coding tasks (59.8% solo on bash-only tasks)  
- **Together** they achieve 78.3%, outperforming both individually

This architecture is perfect for production coding assistants, where the head agent handles user interaction and the critique agent provides specialized technical guidance.

### Learning Curve Observations

Both humans and AI exhibit logarithmic improvement curves:
- Initial iterations show large gains
- Marginal improvement decreases over time
- The D-term (derivative) detects this plateau automatically
- Early stopping prevents diminishing-return cycles

### Critical Prompting Lessons

From V1 → V2, it was observed that effective critique requires:

1. **Test-first mindset**: "If tests pass, end cycle immediately"  
2. **Minimum viable fixes**: "Don't gold-plate against theoretical errors"
3. **Critique as guide, not gospel**: "Listen to feedback but use judgment"
4. **Avoid hypotheticals**: "Only address given information, not imagined edge cases"

These principles are now encoded in the meta-optimization system rather than manually tuned.

---

## Conclusion

**MetaCritiqueOptimizer** demonstrates that agent performance can be dramatically improved not just by better models or more compute, but by treating the critique system itself as a learnable, controllable parameter. 

By applying control theory to prompt optimization, MetaCritiqueOptimizer achieved:

- **78.3% resolution rate** on SWE-bench bash-only (vs. 45% baseline)
- **Automated prompt tuning** that eliminates manual trial-and-error
- **Systematic prevention** of critique death spirals and over-engineering
- **Validated progression** from V1 (63%) to V2 (78%) proving the meta-optimization value

The framework shows that **second-order optimization—optimizing the optimizer—can be as valuable as first-order improvements**, opening new directions for autonomous agent development.

---

**Data Source**: [SWE-bench Official Leaderboard](https://www.swebench.com/)

---
