"""Independent check of every number that the lecture shows.

    python3 tools/verify_numbers.py

The reference values below are typed in from dl_proj/report/BRIEF.md (sections 3 and 5) and from the project's result
files (the `_source` field of each file in ./data names its origin).  The script (1) recomputes metrics from the bundled prediction files with scikit-learn,
(2) builds the lecture into a temporary folder, and (3) checks that the lecture's text, tables and variables show
exactly these numbers.  It exits with a non-zero status if anything differs.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import _ehr as E  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


def close(name: str, got: float, want: float, tol: float) -> None:
    check(name, abs(got - want) <= tol, f"(got {got:.6g}, reference {want:.6g}, tolerance {tol:g})")


# ----------------------------------------------------------------------------------- reference values (BRIEF 3, 5)
BRIEF = {
    "patients": 11988, "deaths": 1707, "death_rate": 0.1424, "test": 1199, "test_deaths": 158,
    "no_llm_params": 288961, "llemr_trained": 340811777, "llemr_total": 7100227586,
    "no_llm_seconds": 30.0, "llemr_seconds": 4539, "llemr_peak_gib": 19.9,
    "auroc_no_llm": 0.8815, "auroc_llemr": 0.7951, "auprc_no_llm": 0.5393, "auprc_llemr": 0.4088,
    "shuffle_auroc_no_llm": 0.6605, "shuffle_auroc_llemr": 0.6730, "shuffle_auprc_llemr": 0.2576,
    "bootstrap": (-0.0864, -0.1186, -0.0568),
    "continuation_auroc": 0.7824,
    "lactate_no_llm": [0.3443, 0.3442, 0.3452, 0.3492, 0.3551],
    "lactate_llemr": [0.2698, 0.2695, 0.2691, 0.2687, 0.2685],
    "cosine_lactate": 0.9958, "cosine_same": 0.9969, "cosine_different": 0.9435,
}

# ----------------------------------------------------------------------------------- 1. data files vs. the brief
facts = E.recorded_facts()
check("cohort", facts["cohort"]["patients"] == BRIEF["patients"] and facts["cohort"]["deaths"] == BRIEF["deaths"])
close("death rate", facts["cohort"]["deaths"] / facts["cohort"]["patients"], BRIEF["death_rate"], 5e-5)
test = E.load_test_set()
check("test set", len(test.y) == BRIEF["test"] and int(test.y.sum()) == BRIEF["test_deaths"],
      f"({len(test.y)} patients, {int(test.y.sum())} deaths)")
model = E.SmallModel()
check("no_llm parameters", model.n_params == BRIEF["no_llm_params"] == facts["no_llm"]["parameters"])
check("Llemr parameters", facts["llemr"]["trained_parameters"] == BRIEF["llemr_trained"]
      and facts["llemr"]["total_parameters"] == BRIEF["llemr_total"])
close("no_llm training seconds", facts["no_llm"]["train_seconds"], BRIEF["no_llm_seconds"], 0.05)
close("Llemr training seconds", facts["llemr"]["train_seconds"], BRIEF["llemr_seconds"], 0.5)
close("Llemr peak GiB", facts["llemr"]["peak_cuda_memory_bytes"] / 2 ** 30, BRIEF["llemr_peak_gib"], 0.05)

# ----------------------------------------------------------------------------------- 2. metrics, two implementations
auc_small, auc_llemr, auc_shuffled = E.auroc(test.y, test.small), E.auroc(test.y, test.llemr), E.auroc(test.y, test.llemr_shuffled)
close("AUROC no_llm", auc_small, BRIEF["auroc_no_llm"], 5e-5)
close("AUROC Llemr", auc_llemr, BRIEF["auroc_llemr"], 5e-5)
close("AUROC Llemr, shuffled values", auc_shuffled, BRIEF["shuffle_auroc_llemr"], 5e-5)
close("AUROC no_llm, shuffled values (recorded)", E.small_shuffle_auroc()["shuffled"], BRIEF["shuffle_auroc_no_llm"], 5e-5)
diff, lo, hi = E.paired_bootstrap_auroc(test.y, test.llemr, test.small)
for name, got, want in zip(("bootstrap difference", "bootstrap lower", "bootstrap upper"), (diff, lo, hi), BRIEF["bootstrap"]):
    close(name, got, want, 5e-5)
close("continuation AUROC (recorded, post hoc)", facts["llemr_continuation_post_hoc"]["test_auroc"], BRIEF["continuation_auroc"], 5e-5)
try:
    from sklearn.metrics import average_precision_score, roc_auc_score

    for name, p in (("no_llm", test.small), ("Llemr", test.llemr), ("Llemr shuffled", test.llemr_shuffled)):
        close(f"AUROC {name} == scikit-learn", E.auroc(test.y, p), float(roc_auc_score(test.y, p)), 1e-12)
    close("AUPRC no_llm (recorded) == scikit-learn on the predictions", facts["no_llm"]["test_auprc"],
          float(average_precision_score(test.y, test.small)), 1e-12)
    close("AUPRC Llemr (recorded) == scikit-learn on the predictions", facts["llemr"]["test_auprc"],
          float(average_precision_score(test.y, test.llemr)), 1e-12)
    close("AUPRC Llemr shuffled", float(average_precision_score(test.y, test.llemr_shuffled)), BRIEF["shuffle_auprc_llemr"], 5e-5)
except ImportError:
    print("skip scikit-learn cross-check (not installed)")
close("AUPRC no_llm vs brief", facts["no_llm"]["test_auprc"], BRIEF["auprc_no_llm"], 5e-5)
close("AUPRC Llemr vs brief", facts["llemr"]["test_auprc"], BRIEF["auprc_llemr"], 5e-5)

# ----------------------------------------------------------------------------------- 3. lactate dose-response, cosines
small_means, llemr_means = E.small_lactate_means(), E.llemr_lactate_means()
for t, want_s, want_l in zip(E.LACTATE_TARGETS, BRIEF["lactate_no_llm"], BRIEF["lactate_llemr"]):
    close(f"mean risk at lactate {t}: no_llm", small_means[t], want_s, 5e-5)
    close(f"mean risk at lactate {t}: Llemr", llemr_means[t], want_l, 5e-5)
c = list(E.numeracy_probe().values())
close("cosine lactate 1.5 vs 8", c[0], BRIEF["cosine_lactate"], 5e-5)
close("cosine same variable (mean)", c[1], BRIEF["cosine_same"], 5e-5)
close("cosine different variables (mean)", c[2], BRIEF["cosine_different"], 5e-5)

# ----------------------------------------------------------------------------------- 4. the patient: live vs. recorded
chart, events, rec = E.load_chart(), E.load_events(), E.recorded_no_llm()
logit = model.logit(chart)
close("patient logit: numpy float32 vs the project's bf16 run", logit, rec["no_llm"]["baseline"]["logit"], 0.01)
close("patient risk vs the project's saved GPU prediction", 1 / (1 + math.exp(-logit)), rec["saved_gpu_prediction_of_e1"], 2e-4)
z_new = E.standardize_lactate(4.0)
close("standardised lactate 4.0", z_new, rec["lactate_edit"]["target_value_std"], 1e-5)
close("standardised lactate 1.3", E.standardize_lactate(1.3), rec["lactate_edit"]["original_value_std"], 1e-5)
delta = model.logit(chart.with_value(12, z_new)) - logit
recorded_delta = rec["no_llm"]["lactate_4"]["logit"] - rec["no_llm"]["baseline"]["logit"]
check("lactate 4 raises the logit, as in the recorded run", delta > 0 and recorded_delta > 0 and abs(delta - recorded_delta) < 0.005,
      f"(now {delta:+.5f}, recorded {recorded_delta:+.5f})")
check("event 12 is the 04:52 lactate of 1.3 mmol/L", (events[12].variable, events[12].clock, events[12].raw) == ("Lactate", "04:52", 1.3))

# ----------------------------------------------------------------------------------- 5. what the lecture actually shows
with tempfile.TemporaryDirectory() as out:
    subprocess.run(["lectrace", "build", "--output", out, "--no-incremental", str(ROOT / "01_icu_mortality.py")],
                   check=True, capture_output=True, cwd=ROOT)
    trace = json.loads((Path(out) / "traces" / "01_icu_mortality.json").read_text(encoding="utf-8"))
check("the lecture ran without an error", trace["metadata"]["error"] is None)
visible, last = [], {}
for step in trace["steps"]:
    for r in step["renderings"]:
        if r["type"] == "markdown":
            visible.append(r["data"])
        elif r["type"] == "table":
            visible += [r["data"].get("caption", ""), *map(str, r["data"]["headers"])]
            visible += [str(cell) for row in r["data"]["rows"] for cell in row]
    for k, v in step["env"].items():
        if isinstance(v, dict) and isinstance(v.get("contents"), (int, float)):
            last[k] = v["contents"]
text = "\n".join(visible)
for shown in ("11,988", "14.2%", "1,199", "158", "341M", "7.1B", "0.29M", "4,539 s", "30 s", "19.9 GiB", "150x", "1,200x", "25,000x",
              "**0.882**", "**0.795**", "-0.086", "[-0.119, -0.057]", "0.539", "0.409", "0.782", "0.660", "0.673", "+1.08", "-0.12",
              "0.9958", "0.9969", "0.9435", "-4.9333", "-4.9307", "0.72%", "30%", "+0.0034", "-0.0126", "norm 12.4", "(2.5)", "(3.4)"):
    check(f"slide shows {shown!r}", shown in text)
for name, want, tol in (("z", rec["lactate_edit"]["original_value_std"], 1e-5), ("z_new", rec["lactate_edit"]["target_value_std"], 1e-5),
                        ("t_abs", 292 / 2880, 1e-9), ("dt", math.log1p(244 / 60), 1e-9), ("logit", logit, 1e-6), ("delta", delta, 1e-6),
                        ("auc_small", BRIEF["auroc_no_llm"], 5e-5), ("auc_llemr", BRIEF["auroc_llemr"], 5e-5),
                        ("auc_shuffled", BRIEF["shuffle_auroc_llemr"], 5e-5), ("diff", BRIEF["bootstrap"][0], 5e-5),
                        ("lo_ci", BRIEF["bootstrap"][1], 5e-5), ("hi_ci", BRIEF["bootstrap"][2], 5e-5)):
    close(f"variable {name} in the trace", last[name], want, tol)
n_steps = len(trace["steps"])
print(f"\n{n_steps} steps; {len(failures)} failure(s)")
sys.exit(1 if failures else 0)
