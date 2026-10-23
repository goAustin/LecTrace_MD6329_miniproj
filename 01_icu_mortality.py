"""Do numbers survive the trip into the model?

A LecTrace lecture (about 10 minutes). ICU mortality prediction on PhysioNet 2012: a 0.29M-parameter event encoder
trained from scratch (`no_llm`) against Llemr-v1, a 7.1B-parameter EHR language model fine-tuned as in its paper.

Live vs. recorded: everything the small model does is computed here, in numpy, with its trained weights.
Llemr (7.1B parameters) cannot run in a browser, so its numbers are read from the result files of the project.
"""

from __future__ import annotations

import math

import numpy as np
from lectrace import plot, table, text

from _ehr import (SmallModel, about, auroc, event_table, lactate_stats, llemr_event, llemr_lactate_means, load_chart,
                  load_events, load_patient, load_test_set, numeracy_probe, paired_bootstrap_auroc, patient_sweep,
                  recorded_facts, recorded_no_llm, relative_change, share_summary, small_lactate_means,
                  small_shuffle_auroc, standardize_lactate)
from _plots import lactate_figure, shuffle_figure

FACTS = recorded_facts()  # copied from the project's result files; every block names its source
COHORT, NO_LLM, LLEMR = FACTS["cohort"], FACTS["no_llm"], FACTS["llemr"]
LONGER_RUN = FACTS["llemr_continuation_post_hoc"]
PROJECT_RUN = recorded_no_llm()["no_llm"]  # what the project's own export wrote for this patient (CPU, bf16)
SMALL_SHUFFLE = small_shuffle_auroc()  # recorded: no_llm with measured values shuffled between patients
SMALL_MEANS, LLEMR_MEANS = small_lactate_means(), llemr_lactate_means()  # mean risk of 633 patients per lactate target
COSINES = numeracy_probe()  # recorded: frozen BiomedBERT vectors of the texts Llemr writes
LACTATE_COSINE, SAME_COSINE, DIFFERENT_COSINE = COSINES.values()


def main():
    # ---- the problem, the motivation, the approach
    text("# Do numbers survive the trip into the model?")

    text(f"""
        ## The healthcare problem

        An ICU patient is measured all the time, but irregularly: {COHORT['variables']} variables (vital signs, labs,
        GCS, urine output ...) during the first {COHORT['hours']} hours.
        **Task:** from this chart, predict **in-hospital death**, early enough to act.

        - **Data:** PhysioNet 2012, **{COHORT['patients']:,}** ICU stays, **{COHORT['death_rate']:.1%}** died.
        - **Score:** AUROC = the chance that a patient who died gets a higher risk than one who survived.
        - A risk score is only useful at the bedside if it reacts to the **measured values**, not just to
          *which tests were ordered*.
    """)

    text("""
        ## Motivation: can a language model read an ICU chart?

        - EHR language models such as **Llemr** (NeurIPS 2024) turn every event into text, embed it with a frozen text
          encoder, and let a **7B LLM** read one token per event. The promise: one model for every clinical task.
        - A small event encoder trained from scratch has **0.29M** parameters, 25,000 times fewer.
        - **Our pre-registered question:** on P12 split 1, does Llemr, fine-tuned as in its paper, beat the small
          encoder?
    """)

    text(f"""
        ## Approach: same patients, same test set, two ways to read a number

        - Both models see the **same events** of the **same {COHORT['test']:,} test patients** ({COHORT['test_deaths']}
          deaths), scored with the same metric.
        - The protocol was **pre-registered**: its SHA-256 (`{FACTS['preregistration']['sha256'][:8]}...`) was recorded
          before any Llemr result existed.
        - Below, the small model runs **live** on one real patient. Llemr's outputs are **recorded** results of the
          project: it cannot run here.
    """)

    table([
        {"": "an event enters as", "no_llm (small)": "learned vectors for variable, value, time",
         "Llemr-v1 (EHR LLM)": "text via frozen BiomedBERT + 3 raw numbers"},
        {"": "core model", "no_llm (small)": "2-layer Transformer, from scratch",
         "Llemr-v1 (EHR LLM)": "Vicuna-7B + LoRA, fine-tuned"},
        {"": "trained parameters", "no_llm (small)": f"{NO_LLM['parameters'] / 1e6:.2f}M",
         "Llemr-v1 (EHR LLM)": f"{LLEMR['trained_parameters'] / 1e6:.0f}M"},
        {"": "total parameters", "no_llm (small)": f"{NO_LLM['parameters'] / 1e6:.2f}M",
         "Llemr-v1 (EHR LLM)": f"{LLEMR['total_parameters'] / 1e9:.1f}B"},
        {"": "training time", "no_llm (small)": f"{NO_LLM['train_seconds']:.0f} s",
         "Llemr-v1 (EHR LLM)": f"{LLEMR['train_seconds']:,.0f} s"},
        {"": "peak GPU memory", "no_llm (small)": "-",
         "Llemr-v1 (EHR LLM)": f"{LLEMR['peak_cuda_memory_bytes'] / 2 ** 30:.1f} GiB"},
    ], caption=(f"One RTX 4090. Llemr costs about {about(LLEMR['train_seconds'] / NO_LLM['train_seconds'])}x the "
                f"training time, {about(LLEMR['trained_parameters'] / NO_LLM['parameters'])}x the trained and "
                f"{about(LLEMR['total_parameters'] / NO_LLM['parameters'])}x the total parameters of no_llm."))

    # ---- State 1: a chart becomes numbers
    text("""
        ## State 1 · A chart becomes numbers

        **What is happening.** The raw record is a list of lines such as `04:52,Lactate,1.3`.
        Each line becomes an *event*: which variable, when, and how much, rescaled with **training-set statistics**.
        We follow one real test patient (ID 139655) and one event: lactate 1.3 mmol/L at 04:52.
    """)

    patient, events, chart = load_patient(), load_events(), load_chart()  # @inspect patient

    table(event_table(events, chart, 10, 17),
          caption="Events 10-17 as the model receives them: the value as z-score (z), the time as t_abs and dt.")

    e = events[12]  # @inspect e
    lo, hi, mean, std = lactate_stats()  # @inspect lo hi mean std
    z = (min(max(e.raw, lo), hi) - mean) / std  # @inspect z
    t_abs, dt = e.minute / 2880, math.log1p((e.minute - e.previous_minute) / 60)  # @inspect t_abs dt

    text(f"""
        **What changed.** A line of text became three numbers, `(z, t_abs, dt)`, and an integer variable id
        ({e.var_id} = {e.variable}). Names and units are gone. `t_abs` is the time as a fraction of 48 h; `dt` is the
        log of the hours since the previous lactate ({e.previous_minute // 60:02d}:{e.previous_minute % 60:02d}).
        A measurement that was never taken is simply absent: nothing is imputed.

        **Why it matters.** The z-score puts 36 variables of very different scales on one footing, and `log1p`
        squeezes gaps of minutes and of days into one range. From here on, **this triple is all the model knows**
        about this lactate reading.
    """)

    # ---- State 2: an event becomes a token
    text("""
        ## State 2 · One event becomes one token

        **What is happening.** Three *learned* functions turn the three numbers into three 128-d vectors: a lookup
        table for the variable, and two small MLPs, one for the value and one for the time. The vectors are added and
        layer-normalised. All three are trained end to end by the mortality loss.
    """)  # @clear patient lo hi mean std

    model = SmallModel()  # @inspect model
    e_var = model.var_emb(e.var_id)  # @inspect e_var
    e_val = model.value_mlp(z)  # @inspect e_val
    e_time = model.time_mlp(t_abs, dt)  # @inspect e_time
    token = model.norm(e_var + e_val + e_time)  # @inspect token
    llemr = llemr_event(e.variable, e.raw, e.minute)  # @inspect llemr

    text(f"""
        **What changed.** One reading is now a vector in the same 128-d space as every other event, the sum of
        *which variable* (norm {np.linalg.norm(e_var):.1f}), *how much* ({np.linalg.norm(e_val):.1f}) and
        *when* ({np.linalg.norm(e_time):.1f}).

        **Llemr reads the same event differently.** The `llemr` line above is its input: the text
        "{llemr['text']}" goes through a *frozen* BiomedBERT (1,024-d), the raw numbers
        [{llemr['appended'][0]} h, {llemr['appended'][1]}, {llemr['appended'][2]}] are appended, and an MLP projects
        the 1,027-d vector into the LLM. For Llemr, the value is characters in a sentence plus one raw coordinate.

        **Why it matters.** This is where *how numbers enter the model* is decided. `no_llm` has a trained pathway
        for the value; Llemr has a frozen text encoder and one raw scalar.
    """)

    # ---- State 3: 392 tokens become one risk
    text("""
        ## State 3 · 392 tokens become one risk

        **What is happening.** One token per event (391) plus one token for the static facts (age, sex, ICU type ...).
        A 2-layer Transformer lets every event read every other event. The output vectors are **averaged**, and one
        linear layer turns the mean into a single number, the *logit*.
    """)  # @clear e_var e_val e_time llemr e t_abs dt

    tokens = model.tokens(chart)  # @inspect tokens
    hidden = model.transformer(tokens)  # @inspect hidden
    pooled = hidden.mean(axis=0)  # @inspect pooled
    logit = float(pooled @ model.head_w + model.head_b)  # @inspect logit
    risk = 1 / (1 + math.exp(-logit))  # @inspect risk
    shares = share_summary(hidden @ model.head_w / len(hidden), events, model.head_b)  # @inspect shares

    text(f"""
        **What changed.** 392 vectors became one number: logit {logit:.4f}, a risk of {risk:.2%}. The project's own
        run of the same checkpoint (CPU, bf16) recorded {PROJECT_RUN['baseline']['logit']:.4f}; this numpy
        re-implementation agrees to {abs(logit - PROJECT_RUN['baseline']['logit']):.3f}. Because of the mean, the
        logit is exactly *bias + the sum of 392 per-token shares*: about {shares['mean share']:+.4f} each.

        **Why it matters.** Every token gets one vote of weight 1/392, so one lab value has to compete with 391 other
        events. (The model's risks are uncalibrated: compare scenarios, not absolute probabilities.)
    """)

    # ---- State 4: what if the lactate were 4?
    text("""
        ## State 4 · What if the lactate were 4 mmol/L?

        **What is happening.** A clinical counterfactual: replace *only* this lactate reading and run the model again.
        Clinically, a higher lactate (a marker of poor tissue perfusion) goes with higher mortality, so the risk
        should rise, or at least not fall.
    """)  # @clear tokens hidden pooled shares

    z_new = standardize_lactate(4.0)  # @inspect z_new
    token_new = model.token(e.var_id, z_new, t_abs, dt)  # @inspect token_new
    moved = relative_change(token, token_new)  # @inspect moved
    delta = model.logit(chart.with_value(e.index, z_new)) - logit  # @inspect delta

    table(patient_sweep(model, chart, e, patient.record_id),
          caption="Patient 139655, latest lactate edited. no_llm is computed now; Llemr's risks are its recorded run.")

    plot(lactate_figure(SMALL_MEANS, LLEMR_MEANS))

    table([{"cosine similarity of the frozen text vectors": k, "value": v} for k, v in COSINES.items()],
          caption='Texts as Llemr writes them, e.g. "Blood Lactate Blood Gas: 1.5 mmol/L" vs "... 8 mmol/L".')

    text(f"""
        **What changed.** The lactate *token* moved by {moved:.0%} of its length, but this patient's logit moved by
        only {delta:+.4f} (the project's bf16 run: +0.0016): the other 391 events dominate the average. Across the
        633 test patients with a lactate (recorded runs) the direction is clear: `no_llm`'s mean risk **rises** by
        {100 * (SMALL_MEANS[8] - SMALL_MEANS[1]):+.2f} percentage points from 1 to 8 mmol/L, Llemr's is **flat**
        ({100 * (LLEMR_MEANS[8] - LLEMR_MEANS[1]):+.2f}).

        **Why it matters.** A model that does not react to lactate cannot warn about a rising lactate. One reason
        consistent with the table above: the frozen text encoder gives "lactate 1.5" and "lactate 8" almost the same
        vector (cosine {LACTATE_COSINE}), and two readings of *the same variable* look more alike
        (mean {SAME_COSINE}) than two *different variables* do ({DIFFERENT_COSINE}). So the value can reach the LLM
        mainly through one raw scalar.
    """)

    # ---- the key result
    text(f"""
        ## The key result

        Same {COHORT['test']:,} test patients, same pre-registered protocol. We recompute the metrics from the
        saved predictions.
    """)  # @clear model z token logit risk z_new token_new moved delta

    test = load_test_set()  # @inspect test
    auc_small, auc_llemr = auroc(test.y, test.small), auroc(test.y, test.llemr)  # @inspect auc_small auc_llemr
    auc_shuffled = auroc(test.y, test.llemr_shuffled)  # @inspect auc_shuffled
    diff, lo_ci, hi_ci = paired_bootstrap_auroc(test.y, test.llemr, test.small)  # @inspect diff lo_ci hi_ci

    plot(shuffle_figure(SMALL_SHUFFLE, {"original": auc_llemr, "shuffled": auc_shuffled}))

    text(f"""
        **Key result.** On P12 split 1, **Llemr, fine-tuned as in its paper, is worse than the 0.29M-parameter encoder
        trained from scratch.**

        - AUROC: `no_llm` **{auc_small:.3f}**, Llemr **{auc_llemr:.3f}**. Difference (Llemr minus `no_llm`)
          {diff:+.3f}, 95% CI [{lo_ci:+.3f}, {hi_ci:+.3f}] (paired bootstrap, 1,000 resamples).
        - AUPRC (recorded): `no_llm` {NO_LLM['test_auprc']:.3f}, Llemr {LLEMR['test_auprc']:.3f}. `no_llm` trained in
          about 1/{about(LLEMR['train_seconds'] / NO_LLM['train_seconds'])} of the time.
        - A longer Llemr run (post hoc) reached AUROC {LONGER_RUN['test_auroc']:.3f}: no rescue.
        - Measured values shuffled between patients: `no_llm` {SMALL_SHUFFLE['shuffled']:.3f}, Llemr
          {auc_shuffled:.3f}, the same level.
    """)

    text("""
        **What it means for care.**
        - The measured values **are** the signal: scramble them across patients and both models fall to AUROC
          0.66-0.67. What is left comes from which tests were taken and when, and from the static facts.
        - A score that ignores a rising lactate cannot warn about it. Before bedside use, models need simple
          counterfactual checks like State 4, not only an AUROC.
        - Here, the diagnostics point to *how the numbers enter the model*, rather than to model size, as the
          limiting factor.

        **Limits.** One dataset (P12), one split, one seed and one fine-tuning recipe for Llemr; uncalibrated risks;
        retrospective data. This does not show that language models are worse in general.
    """)


if __name__ == "__main__":
    main()
