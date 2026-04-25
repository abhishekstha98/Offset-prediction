# Team Phase Conversation Details

Date: 2026-04-16
Based on: `TEAM_WORKFLOW_GUIDELINE.md`

## 1. Purpose of this document

This document is designed to help the team discuss each phase in a structured way.

For every phase, it gives:

- the purpose of the phase
- the key discussion points
- the decisions that must be made
- the steps to execute
- the deliverables expected at the end
- the exit criteria before moving to the next phase

This can be used directly in weekly meetings, planning calls, or team check-ins.

## 2. How to use this in team meetings

For each phase discussion:

1. Start by restating the goal of the phase.
2. Review what is already available.
3. Discuss the open questions.
4. Assign clear owners for the next actions.
5. Confirm the deliverables and deadline.
6. Do not move to the next phase unless the current phase has a usable output.

## Phase 1. Finalize Scope and Benchmark Definition

### Purpose

The purpose of this phase is to make sure the team is solving one clearly defined problem and not drifting into multiple competing project directions.

### What the conversation should focus on

- What exactly is the paper claiming?
- Are we focusing on `station-level correction` or on general downscaling?
- Are `Tmin` and `Tmax` the final main targets?
- Is `fog/frost proxy` the first application story?
- Are we keeping `air quality` and `evapotranspiration` outside the first round?
- Which benchmark products are mandatory in the first paper?

### Suggested conversation prompts

- What is the one-sentence version of our project?
- If someone asks what the novelty is, what is our answer in one sentence?
- What are we definitely including in the first paper?
- What are we explicitly postponing to later work?
- What would make the scope too broad?

### Decisions to make in the meeting

- final problem statement
- final target variables
- final benchmark list
- final application story
- final split and evaluation policy

### Steps after the meeting

1. Write the final one-sentence project framing.
2. Write a short paragraph describing what is in scope.
3. Write a short paragraph describing what is out of scope.
4. Freeze the benchmark list.
5. Freeze the evaluation protocol.

### Deliverables

- one-sentence problem statement
- short scope statement
- out-of-scope statement
- benchmark list
- evaluation split policy

### Exit criteria

Do not leave this phase unless everyone can explain the project in the same way and the benchmark list is fixed.

## Phase 2. External Baseline Preparation

### Purpose

The purpose of this phase is to compare the current model against strong coarse external baselines, especially `Aurora` and `GraphCast`.

### What the conversation should focus on

- Which baseline should be run first?
- What is the exact evaluation window?
- How will predictions be mapped to station locations?
- How will daily `Tmin/Tmax` be computed from sub-daily outputs if needed?
- What metrics will be reported?
- What counts as a fair comparison?

### Suggested conversation prompts

- Is `Aurora` the fastest first baseline to prepare?
- Should we evaluate on the held-out Netherlands test period first?
- What interpolation method should we use first: `linear` or `nearest`?
- Are there any format or variable naming issues we should expect?
- What result table do we want at the end of this phase?

### Decisions to make in the meeting

- first baseline to run
- test period to use
- interpolation choice
- aggregation method for daily extrema
- output metrics and summary format

### Steps after the meeting

1. Export the station manifest.
2. Run `Aurora` baseline on the held-out test period.
3. Evaluate station-level `Tmin/Tmax`.
4. Save summary metrics and per-station metrics.
5. If feasible, repeat the same process for `GraphCast`.

### Deliverables

- `Aurora` evaluation output
- `GraphCast` evaluation output if completed
- station-level MAE and bias summary
- baseline comparison table

### Exit criteria

Do not leave this phase unless at least one strong external baseline has been evaluated and compared to the current best model.

## Phase 3. Classical Baseline Comparison

### Purpose

The purpose of this phase is to show that the project is stronger than simple alternatives, not only different from large foundation models.

### What the conversation should focus on

- Which classical baselines are necessary for credibility?
- Which baselines are simple enough to implement quickly?
- Which baseline should be considered the strongest classical comparison?
- How should all baseline results be reported together?

### Suggested conversation prompts

- If reviewers ask for simpler baselines, what do we want to already have prepared?
- Which baseline is essential: persistence, linear regression, random forest, quantile mapping?
- Are there any baselines that are too costly for the value they add?
- What is the final benchmark matrix we want to show?

### Decisions to make in the meeting

- final classical baseline list
- implementation order
- common evaluation protocol for all baselines
- result table format

### Steps after the meeting

1. Run raw `ERA5` or `ERA5-Land` station-level baseline.
2. Run persistence baseline.
3. Run per-station linear regression.
4. Run per-station `Random Forest` or `XGBoost`.
5. Run a simple statistical bias-correction baseline.
6. Collect all results into one table.

### Deliverables

- full baseline matrix
- paper-ready benchmark table
- short interpretation of which baselines are strongest

### Exit criteria

Do not leave this phase unless the model has been compared against both external and classical baselines under the same evaluation setup.

## Phase 4. Model Simplification and Ablation

### Purpose

The purpose of this phase is to identify the best lightweight model and avoid committing to a more complex architecture without evidence.

### What the conversation should focus on

- Which model is currently strongest under the strictest protocol?
- Is the multi-channel model actually helping?
- Does `MPT` provide value beyond plain message passing?
- What is the simplest model that still supports the paper claim?

### Suggested conversation prompts

- Are we optimizing for best performance or best performance-to-complexity ratio?
- Does the current evidence support the more complex architecture?
- Which ablation result would change our paper story the most?
- If the simple model is best, are we comfortable making that the main result?

### Decisions to make in the meeting

- main evaluation protocol for ablations
- models to include in the comparison
- criteria for selecting the final main model

### Steps after the meeting

1. Run the baseline graph model under the strict protocol.
2. Run the multi-channel variant under the same protocol.
3. Run `MPT vs plain message passing`.
4. Compare performance, stability, and implementation complexity.
5. Choose the final main model for the paper.

### Deliverables

- ablation result table
- model selection decision
- short justification for the chosen final model

### Exit criteria

Do not leave this phase unless the team has selected one main model and can explain why it is the right model for the paper.

## Phase 5. Sparse-Network Robustness Analysis

### Purpose

The purpose of this phase is to turn sparse-station performance into one of the central scientific contributions of the work.

### What the conversation should focus on

- How should station withholding be reframed as sparse-network robustness?
- Which sparsity settings should be tested?
- Which metrics matter most for seen versus unseen stations?
- How should degradation be visualized?
- What would count as meaningful robustness?

### Suggested conversation prompts

- Why is sparse-network robustness especially important for Nepal?
- How many stations should be withheld at each setting?
- Do we need repeated runs for each sparsity level?
- What supporting diagnostics would strengthen the story?

### Decisions to make in the meeting

- withholding schedule
- reporting metrics
- number of repeats
- plots and diagnostics to produce

### Steps after the meeting

1. Define the final withholding schedule.
2. Run withholding experiments for the selected main model.
3. Report seen-station and unseen-station error.
4. Add diagnostics such as neighbor count and nearest-station distance if feasible.
5. Summarize how performance degrades with increasing sparsity.

### Deliverables

- sparse-network robustness table
- degradation curve
- short interpretation of robustness limits

### Exit criteria

Do not leave this phase unless there is a clear result showing how the model behaves when the station network becomes sparse.

## Phase 6. Regime-Specific Analysis

### Purpose

The purpose of this phase is to show that the model is especially useful in the weather regimes that matter most to the Nepal motivation.

### What the conversation should focus on

- Which regimes are most scientifically meaningful?
- How will these regimes be defined?
- Which regime is most closely tied to inversion-sensitive `Tmin` correction?
- What pattern would support the main narrative?

### Suggested conversation prompts

- Do we expect the biggest gains at night, in winter, or under low-wind conditions?
- What is the cleanest way to define inversion-like conditions?
- Which regime comparisons will be easy to explain in the paper?
- Do we have enough data for stable metrics in each regime?

### Decisions to make in the meeting

- final regime definitions
- subset sizes required for reporting
- plots and tables to produce

### Steps after the meeting

1. Define the regime labels.
2. Split the evaluation data into regime groups.
3. Compute metrics for each regime.
4. Compare gains across groups.
5. Highlight the regimes with the largest improvement.

### Deliverables

- regime-specific metrics table
- one figure showing regime-wise improvement
- interpretation linking results to Nepal motivation

### Exit criteria

Do not leave this phase unless the team can show where the model helps most and why that matters scientifically.

## Phase 7. Nepal Transfer or Fine-Tuning Stage

### Purpose

The purpose of this phase is to connect the source-region benchmark to a real Nepal or Himalayan deployment setting.

### What the conversation should focus on

- What Nepal or Himalayan data is actually available?
- How much target data is enough for a meaningful pilot?
- Will we fine-tune a pretrained model or train from scratch?
- What metadata and covariate alignment issues are expected?
- What counts as success in a low-data setting?

### Suggested conversation prompts

- What target dataset is realistically obtainable first?
- What preprocessing work is needed before any training can happen?
- How will we compare transfer learning against scratch training fairly?
- If Nepal data access is delayed, what is the best backup plan?

### Decisions to make in the meeting

- target dataset choice
- transfer-learning setup
- scratch-training comparison setup
- fallback plan if full Nepal data is delayed

### Steps after the meeting

1. Obtain Nepal or Himalayan station data.
2. Clean and harmonize metadata.
3. Align target data with predictors and covariates.
4. Fine-tune the source-region model.
5. Train a comparison model from scratch.
6. Compare both approaches.

### Deliverables

- target-region pilot dataset
- transfer-learning experiment
- pretrained vs scratch comparison
- short note on feasibility and limitations

### Exit criteria

Do not leave this phase unless there is at least one credible target-region experiment or a clearly documented fallback plan.

## Phase 8. Downstream Application Demo

### Purpose

The purpose of this phase is to show that the correction model helps with a simple real-world decision task and not only with MAE reduction.

### What the conversation should focus on

- What is the simplest valid fog/frost risk formulation?
- Do we have direct visibility or event observations?
- If not, what proxy rule is acceptable?
- How will we compare corrected predictions against raw ERA5 for the application task?
- What result would be convincing enough for the paper?

### Suggested conversation prompts

- Are we building an event detection task or a risk ranking task?
- What variables besides `Tmin` should be included in the proxy?
- How complex can the application demo become before it distracts from the main paper?
- What should remain future work?

### Decisions to make in the meeting

- final fog/frost definition
- direct observation vs proxy approach
- evaluation metric for the demo
- scope limit for the application section

### Steps after the meeting

1. Define the fog/frost event rule.
2. Gather any required humidity, wind, or visibility data.
3. Compute application outputs using raw ERA5.
4. Compute application outputs using corrected predictions.
5. Compare the two approaches and summarize the gain.

### Deliverables

- fog/frost proxy experiment
- event or risk comparison table
- short application interpretation for the paper

### Exit criteria

Do not leave this phase unless there is one simple downstream result that clearly benefits from the correction model.

## 3. Suggested team-facing summary

If you want to explain the meeting flow quickly, use this:

`For each phase, we should discuss the goal, the open technical questions, the decisions we must lock, the concrete steps to execute, and the exact deliverables needed before moving forward.`

## 4. Recommended usage in practice

One practical way to run meetings is:

1. Spend 5 minutes restating the phase goal.
2. Spend 10 to 15 minutes on open technical questions.
3. Spend 5 minutes locking decisions.
4. Spend 5 minutes assigning owners and deadlines.
5. End by reading out the deliverables and exit criteria.

This helps keep the project moving without losing the main paper story.
