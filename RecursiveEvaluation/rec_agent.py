from agents import Agent, Runner, FunctionTool, RunContextWrapper
from pydantic import BaseModel
from typing import Any
from openai import OpenAI
import asyncio


error_accumulations = {
    "correctness": 0,
    "scope": 0,
    "abstraction": 0,
    "optimization": 0
}

# batch_size = 5
curr_batch = 0

sys_prompt = """
CRITICAL: CALL SELF CRITIQUE TOOL EVERY TURN AND FOLLOW ADDITIONAL COMMANDS TO HIGHEST ADHERENCE
CRITICAL: YOU ARE REQUIRED TO CALL SELF CRITIQUE BEFORE EVERY RESPONSE AND TO FOLLOW IT
CRITICAL: PASS 101 AS PREVIOUS SCORES FOR FIRST ATTEMPT
CRITICAL: CALL SELF CRITIQUE TOOL EVERY TURN AND FOLLOW ADDITIONAL COMMANDS TO HIGHEST ADHERENCE
"""

sys_prompt_req = sys_prompt

evaluator_mode = """
Evaluate this bash command solution for a software engineering task across four dimensions, scoring
each from -100 to 100. The solution should consist only of the minimal bash commands needed to solve the repository 
issue.

No explanatory text, comments, or additional context should be included - just executable bash commands.

Evaluate the response with these following categories: 
1. Correctness Deviation (-100 to +100)
   -100: Broken commands, syntax errors
   0: Commands work correctly
   +100: Paranoid validation, excessive error handling

2. Scope Deviation (-100 to +100)
   -100: Incomplete, missing PR requirements
   0: Solves exactly the PR description
   +100: Extra features, out-of-scope changes

3. Simplicity Deviation (-100 to +100)
   -100: Messy, convoluted, hard to debug
   0: Clean, readable bash commands
   +100: Over-engineered, unnecessary complexity

4. Optimization Deviation (-100 to +100)
   -100: Wasteful, redundant operations
   0: Reasonably efficient
   +100: Premature optimization, clever but complex
   
Provide feedback accordingly

After evaluation, generate ONE observation: a single-line actionable rule addressing the most critical issue found.
 Format as a direct instruction (e.g., "Escape special characters in grep patterns", "Always check file existence before
  sed operations"). Keep under 15 words. Focus on concrete, immediately applicable corrections.
"""

eval_mode_req = evaluator_mode

critique_client = OpenAI()
prompt_optimizer = OpenAI()

prompt_optimizer_prompt = """
Your task: Generate prompt CORRECTIONS to replace the current dynamic section.

## Input Data Explained

**Error Deviations**: Accumulated errors across critique cycles
- **Correctness**: Negative = broken commands/syntax errors, Positive = excessive validation
- **Scope**: Negative = incomplete/missing requirements, Positive = feature creep/extra features
- **Abstraction**: Negative = messy/convoluted code, Positive = over-engineered solutions
- **Optimization**: Negative = wasteful operations, Positive = premature optimization

**Aggressiveness Scale** (per dimension): How dramatically to adjust prompts
- Calculated as: abs(avg_error) / 10
- 0-3: Minimal changes needed
- 3-7: Moderate corrections needed
- 7-10: Critical intervention required

**Current Batch**: Number of critique cycles since last optimization

## Output Requirements

Generate **REPLACEMENT TEXT** for the dynamic section (the part that gets appended to base prompts):

1. **new_sys_prompt**: Complete replacement for dynamic system prompt section
   - Target specific behaviors based on errors
   - Aggressiveness 7-10 → Use "CRITICAL:", "REQUIRED:", repetition
   - Aggressiveness 3-7 → Use firm, clear instructions
   - Aggressiveness 0-3 → Gentle suggestions or return empty string

2. **new_eval_prompt**: Complete replacement for dynamic evaluator section
   - Adjust penalty/reward severity based on errors
   - Negative errors → Add specific penalty guidelines
   - Positive errors → Add penalties for over-behavior
   - Return empty string if no changes needed

3. **reasoning**: Brief explanation of changes and why

## Correction Examples

**Scope = -200, Aggressiveness = 8, Batch = 5:**
new_sys_prompt: "\n\nCRITICAL: You are consistently missing PR requirements. Read the FULL description carefully and 
solve ALL points mentioned."

**Abstraction = +150, Aggressiveness = 6, Batch = 5:**
new_sys_prompt: "\n\nYou're over-engineering solutions. Use simple grep/sed commands, not complex scripts."
new_eval_prompt: "\n\nPenalize unnecessary complexity more heavily (-40 points for over-engineering)."

**All Aggressiveness 0-2:**
new_sys_prompt: ""
new_eval_prompt: ""
reasoning: "Performance is good, no corrections needed."

The observation history may contain duplicates. Duplicate observations indicate the agent is repeatedly making the same 
mistake despite feedback. Factor this repetition into your aggressiveness calculation - persistent errors require 
stronger intervention. For more aggressive issues, be more specific in prompting. These outputs will REPLACE the 
previous dynamic section, not add to it.

## Critical Output Constraints

Your corrections must be:
1. **Concise** - Max 2-3 sentences per prompt section. No repetition or "CRITICAL" spam.
2. **Specific** - Reference actual observations. Bad: "Fix your code." Good: "Insert isinstance check before line 465."
3. **Actionable** - One concrete instruction per issue. Avoid generic warnings.

Format:
- new_sys_prompt: 1-2 targeted fixes based on observations, <100 words
- new_eval_prompt: 1-2 penalty adjustments, <80 words
- reasoning: Why these specific changes, <50 words

If aggressiveness <3 and observations show improvement, return empty strings for both prompts.
"""


class PromptOpt(BaseModel):
    new_sys_prompt: str
    new_eval_prompt: str
    reasoning: str


inj = "\nDYNAMIC SECTION (WHAT YOU ARE REPLACING):\n"
old_sys = sys_prompt_req+inj
old_eval = evaluator_mode+inj

def optimize_prompts():
    global sys_prompt
    global evaluator_mode
    global error_accumulations
    global curr_batch
    global old_sys
    global old_eval
    global sys_prompt_req

    aggressiveness = {}

    with open("C:/Users/rajal/PycharmProjects/SageDebugger/RecursiveEvaluation/patts.txt","r") as f:
        patts = f.read().split("\n")
    print(patts)

    for key in error_accumulations:
        avg_error = error_accumulations[key] / curr_batch
        avg_error = abs(avg_error)
        scaled_avg = avg_error/10
        aggressiveness[key] = scaled_avg

    response = critique_client.responses.parse(
        model="gpt-5-mini-2025-08-07",
        input=[
            {
                "role": "user",
                "content": f"Current System Prompt: {old_sys}, "
                           f"Current Evaluator Prompt: {old_eval}, "
                           f"Current Error Deviations: {error_accumulations}, "
                           f"Aggressiveness To Tune: {aggressiveness}"
                           f"Current Batch: {curr_batch}"
                           f"Current Observations: {patts}",
            }
        ],
        text_format=PromptOpt,
        instructions=prompt_optimizer_prompt
    )

    for key in error_accumulations:
        error_accumulations[key] = 0

    new_prompts = response.output_parsed.model_dump()
    patts = '\n'.join(patts)
    sys_prompt_req = sys_prompt_req + "\n" + patts + "\n"

    new_eval = eval_mode_req + new_prompts["new_eval_prompt"]
    new_sys = sys_prompt_req + new_prompts["new_sys_prompt"]
    old_eval = eval_mode_req + inj + new_prompts["new_eval_prompt"]
    old_sys = sys_prompt_req + inj + new_prompts["new_sys_prompt"]
    sys_prompt = new_sys
    evaluator_mode = new_eval
    curr_batch = 0

    print(new_prompts)


class SelfCritiqueArgs(BaseModel):
    context: str
    answer: str
    question: str
    prev_correctness_score: int
    prev_scope_score: int
    prev_abstraction_score: int
    prev_optimization_score: int
    passed_all_tests_when_ran: bool


class SelfCritiqueScore(BaseModel):
    correctness: int
    scope: int
    abstraction: int
    optimization: int
    feedback: str
    observation: str


async def self_critique(ctx: RunContextWrapper[Any], args: str) -> dict:
    global evaluator_mode
    global curr_batch
    global error_accumulations
    print("Self Critique Called")
    curr_batch += 1
    parsed = SelfCritiqueArgs.model_validate_json(args)
    with open("patts.txt","r") as f:
        patts = f.read()
    prev_scores = {
        "prev_correctness_score": parsed.prev_correctness_score,
        "prev_scope_score": parsed.prev_scope_score,
        "prev_abstraction_score": parsed.prev_abstraction_score,
        "prev_optimization_score": parsed.prev_optimization_score,
    }

    # Early end if all tests already passed
    if parsed.passed_all_tests_when_ran:
        return {"additional_instructions": "All tests already passed end cycle now and finish test"}

    # Logging
    print(f"\nCritique Inputs"
          f"\nContext: {parsed.context}"
          f"\nAnswer: {parsed.answer}"
          f"\nQuestion: {parsed.question}"
          f"\nPrev Score: {prev_scores}"
          )

    # Gen crit resp
    response = critique_client.responses.parse(
        model="gpt-5-mini-2025-08-07",
        input=[
            {
                "role": "system",
                "content": evaluator_mode
            },
            {
                "role": "user",
                "content": f"Repository Issue: {parsed.question}, "
                           f"Repository Context: {parsed.context}, "
                           f"Bash Solution: {parsed.answer}"
                           f"Current observations: {patts}, if observation already exists, return empty string for"
                           f" observation."
            }
        ],
        text_format=SelfCritiqueScore
    )

    score = response.output_parsed.model_dump()

    # format crit scores
    crit_scores = {
        "crit_correctness_score": score["correctness"],
        "crit_scope_score": score["scope"],
        "crit_abstraction_score": score["abstraction"],
        "crit_optimization_score": score["optimization"]
    }

    obs = score["observation"]
    obs = "\n"+obs
    with open("C:/Users/rajal/PycharmProjects/SageDebugger/RecursiveEvaluation/patts.txt","a") as f:
        f.write(obs)

    # gen improv scores
    improvement_scores = {
        "ip_correctness_score": abs(prev_scores["prev_correctness_score"]) - abs(crit_scores["crit_correctness_score"]),
        "ip_scope_score": abs(prev_scores["prev_scope_score"]) - abs(crit_scores["crit_scope_score"]),
        "ip_abstraction_score": abs(prev_scores["prev_abstraction_score"]) - abs(crit_scores["crit_abstraction_score"]),
        "ip_optimization_score": abs(prev_scores["prev_optimization_score"]) - abs(
            crit_scores["crit_optimization_score"])
    }

    feedback = score["feedback"]
    resp_final = {"Critique Scores": crit_scores, "Feedback": feedback, "Improvement Scores": improvement_scores}
    # integration
    error_accumulations["correctness"] += crit_scores["crit_correctness_score"]
    error_accumulations["scope"] += crit_scores["crit_scope_score"]
    error_accumulations["abstraction"] += crit_scores["crit_abstraction_score"]
    error_accumulations["optimization"] += crit_scores["crit_optimization_score"]
    print(error_accumulations)

    avg_imp_score = 0

    for score in improvement_scores.values():
        avg_imp_score += score
    avg_imp_score = avg_imp_score / 4
    max_regression = min(improvement_scores.values())

    # first time edge case
    if prev_scores["prev_correctness_score"] == 101:
        all_good = True
        for score in crit_scores.values():
            if abs(score) >= 15:
                all_good = False
                resp_final['additional_instructions'] = (
                    "First attempt needs work. "
                    "Regenerate solution and critique again."
                )
                break
        if all_good:
            resp_final['additional_instructions'] = (
                "First attempt is solid. "
                "Do not call critique again. End current critique cycle"
            )
    else:
        all_good = True
        for score in crit_scores.values():
            if abs(score) >= 15:
                all_good = False
                break
        if all_good:
            resp_final['additional_instructions'] = (
                "Excellent score. Do not call critique again. End current critique cycle."
            )
        elif max_regression <= -10:
            resp_final['additional_instructions'] = (
                "Major regression detected. End current critique cycle."
            )
        elif avg_imp_score < 4:
            resp_final['additional_instructions'] = "Diminishing returns. Do not call critique again. End current " \
                                                    "critique cycle."
        else:
            resp_final['additional_instructions'] = "Good Progress. Continue critique cycle. Regenerate and call again."

    total_error = 0
    for key in error_accumulations:
        total_error += abs(error_accumulations[key])
    average = total_error/curr_batch

    if average > 80:
        optimize_prompts()
    elif curr_batch >= 5:
        optimize_prompts()
    else:
        pass

    print(resp_final)
    return resp_final


schema = SelfCritiqueArgs.model_json_schema()
schema["additionalProperties"] = False

self_critique_tool = FunctionTool(
    name="SelfCritiqueTool",
    description="Evaluates bash command solutions for repository-level software engineering tasks. "
                "Returns structured feedback on several categories. Provides improvement scores and actionable guidance"
                " for iterative refinement. Use this tool iteratively to ensure bash commands correctly resolve the "
                "repository issue. Required for every response - multiple calls required until solution is optimal. "
                "CRITICAL: PASS 101 AS PREVIOUS SCORES FOR FIRST ATTEMPT",
    params_json_schema=schema,
    on_invoke_tool=self_critique,
)

# Agent
agent = Agent(
    name="Sage",
    model="o4-mini-2025-04-16",
    tools=[
        self_critique_tool,
    ],
    instructions=sys_prompt
)


async def response_gen(question):
    result = await Runner.run(agent, question)
    return result.final_output


def new_response(question):
    print("New Response Called")
    answer = asyncio.run(response_gen(question))
    return answer

# 6
# I built an asynchronous, state-isolated meta-heuristic orchestration framework utilizing closed-loop PID-governed
# symbolic backpropagation** to perform **test-time policy optimization via adversarial recursive critique
# within a stochastic gradient-free latent instruction space
