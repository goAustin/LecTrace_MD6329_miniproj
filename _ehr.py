"""Helpers for 01_icu_mortality.py.

LecTrace does not trace files whose name starts with an underscore, so the bulk numpy code lives here and the lecture
shows only the steps that matter.  Nothing is simulated: the weights are the trained `no_llm` checkpoint of the project
(checkpoints/no_llm_s1.pt, converted to .npz), the patient is test patient 139655 of split 1,
and every "recorded" number is read from a result file in ./data (the `_source` field of each file names its origin).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
PATIENT = DATA / "patient_139655"
LACTATE_TARGETS = (1, 2, 4, 6, 8)  # mmol/L, the targets of the pre-specified dose-response diagnostic


# --------------------------------------------------------------------------------------------- display helpers


class Vec(np.ndarray):
    """A float32 array that LecTrace shows as a short summary instead of hundreds of numbers."""

    def __lectrace__(self):
        a = np.asarray(self, dtype=np.float64)
        if a.ndim == 1:
            return {"shape": list(a.shape), "norm": round(float(np.linalg.norm(a)), 3),
                    "first 4": [round(float(v), 3) for v in a[:4]]}
        return {"shape": list(a.shape), "mean row norm": round(float(np.linalg.norm(a, axis=-1).mean()), 3),
                "row 0, first 4": [round(float(v), 3) for v in a[0, :4]]}


def vec(a) -> Vec:
    return np.ascontiguousarray(a, dtype=np.float32).view(Vec)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------- one real patient


@dataclass
class Patient:
    record_id: int
    age: int
    sex: str
    icu_unit: str
    raw_rows_in_record: int
    events_seen_by_model: int
    died_in_hospital: int  # the model never sees this; it is the label

    def __lectrace__(self):
        return {"who": f"{self.age}-year-old {self.sex}, record {self.record_id}",
                "stay": f"{self.icu_unit}; {self.raw_rows_in_record} raw rows, {self.events_seen_by_model} events used",
                "label": "died in hospital" if self.died_in_hospital else "survived (never shown to the model)"}


@dataclass
class Event:
    """One row of the chart that the model keeps (SPEC 2.1): what was measured, when, and how much."""

    index: int
    clock: str  # hh:mm since ICU admission
    variable: str
    var_id: int
    raw: float
    unit: str
    minute: int
    previous_minute: int  # minute of the previous observation of the same variable (0 = ICU admission if none)

    def __lectrace__(self):
        previous = f"previous {self.variable} at minute {self.previous_minute}" if self.previous_minute else "no earlier one"
        return {"variable": f"{self.variable} (id {self.var_id}), event #{self.index}",
                "value": f"{self.raw:g} {self.unit}".strip(),
                "time": f"{self.clock} = minute {self.minute}; {previous}"}


@dataclass
class Chart:
    """The model input of one patient: one (variable, value, time) triplet per event, plus the static vector."""

    var_id: np.ndarray
    value: np.ndarray  # standardized
    t_abs: np.ndarray  # minutes / 2880
    dt: np.ndarray  # log1p(hours since the previous observation of the same variable)
    static: np.ndarray  # 12 numbers: age, sex, height, weight, ICU type

    def with_value(self, i: int, z: float) -> "Chart":
        value = self.value.copy()
        value[i] = z
        return Chart(self.var_id, value, self.t_abs, self.dt, self.static)


def load_patient() -> Patient:
    p = read_json(PATIENT / "patient.json")
    return Patient(p["record_id"], p["static"]["age"], p["static"]["gender"], p["static"]["icu_type"],
                   p["n_raw_rows"], p["n_events_used_by_the_model"], p["in_hospital_death"])


def load_chart() -> Chart:
    t = np.load(PATIENT / "event_tokens.npz")
    return Chart(t["var_id"].astype(np.int64), t["value_std"], t["t_abs"], t["dt"], t["static_vec"])


def load_events() -> list[Event]:
    t = np.load(PATIENT / "event_tokens.npz")
    names = [str(n) for n in t["var_names"]]
    unit = {r["variable"]: r["unit"] for r in read_csv(DATA / "variable_map.csv")}
    events, last = [], {}
    for i, (v, raw, minute) in enumerate(zip(t["var_id"], t["value_raw"], t["time_min"])):
        v, minute = int(v), int(minute)
        events.append(Event(i, f"{minute // 60:02d}:{minute % 60:02d}", names[v], v, round(float(raw), 4),
                            unit[names[v]], minute, last.get(v, 0)))
        last[v] = minute
    # the time features that were stored for the model must follow from the events
    assert np.allclose([math.log1p((e.minute - e.previous_minute) / 60.0) for e in events], t["dt"], atol=1e-5)
    assert np.allclose([e.minute / 2880.0 for e in events], t["t_abs"], atol=1e-6)
    return events


def event_table(events: list[Event], chart: Chart, first: int, last: int) -> list[dict]:
    """Rows events first..last as the model receives them (for a LecTrace table)."""
    return [{"#": e.index, "time": e.clock, "variable": e.variable, "raw value": e.raw,
             "z (value)": round(float(chart.value[e.index]), 3), "t_abs": round(float(chart.t_abs[e.index]), 4),
             "dt": round(float(chart.dt[e.index]), 3)} for e in events[first:last + 1]]


def lactate_stats() -> tuple[float, float, float, float]:
    """Clip bounds (0.5th and 99.5th percentile) and mean / std of the lactate values of the 9,590 training patients."""
    s = read_json(DATA / "stats_split1_lactate.json")["Lactate"]
    return round(s["lo"], 5), round(s["hi"], 5), round(s["mean"], 5), round(s["std"], 5)


def standardize_lactate(mmol_per_l: float) -> float:
    """Clip to the training 0.5/99.5 percentiles, then z-score with the training mean and std."""
    lo, hi, mean, std = lactate_stats()
    return (min(max(mmol_per_l, lo), hi) - mean) / std


def recorded_facts() -> dict:
    """Numbers copied from the project's result files (cost, AUPRC, cohort sizes); the source of each block is inside."""
    return read_json(DATA / "recorded_numbers.json")


def recorded_no_llm() -> dict:
    """Logit / risk of this patient written by the project's demo export (CPU, bf16 autocast)."""
    return read_json(PATIENT / "recorded_no_llm.json")


# --------------------------------------------------------------------------------------------- the small model


_erf = np.vectorize(math.erf, otypes=[np.float64])


def gelu(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))).astype(np.float32)


def layer_norm(x, weight, bias, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps) * weight + bias).astype(np.float32)


class SmallModel:
    """`no_llm`: variable embedding + value MLP + time MLP -> LayerNorm -> 2-layer pre-LN Transformer (4 heads, no
    positional encoding) -> mean over tokens -> Linear(128, 1).  288,961 trained parameters, float32, numpy only."""

    def __init__(self):
        with np.load(DATA / "no_llm_s1_weights.npz") as f:
            self.w = {k: f[k] for k in f.files}
        self.n_params = int(sum(v.size for v in self.w.values()))
        self.head_w = vec(self.w["head.weight"][0])
        self.head_b = float(self.w["head.bias"][0])

    def __lectrace__(self):
        sha = recorded_no_llm()["checkpoint"]["sha256"]
        return {"parameters": self.n_params, "tensors": len(self.w), "sha-256": f"{sha[:8]}...{sha[-4:]}"}

    # ---- one event -> one token
    def _mlp(self, x, prefix: str) -> np.ndarray:
        w = self.w
        h = gelu(x @ w[prefix + ".0.weight"].T + w[prefix + ".0.bias"])
        return h @ w[prefix + ".2.weight"].T + w[prefix + ".2.bias"]

    def var_emb(self, var_id) -> Vec:
        return vec(self.w["encoder.var_emb.weight"][var_id])

    def value_mlp(self, z) -> Vec:
        return vec(self._mlp(np.asarray(z, dtype=np.float32)[..., None], "encoder.value_mlp"))

    def time_mlp(self, t_abs, dt) -> Vec:
        x = np.stack([np.asarray(t_abs, dtype=np.float32), np.asarray(dt, dtype=np.float32)], axis=-1)
        return vec(self._mlp(x, "encoder.time_mlp"))

    def norm(self, x) -> Vec:
        return vec(layer_norm(x, self.w["encoder.norm.weight"], self.w["encoder.norm.bias"]))

    def token(self, var_id, z, t_abs, dt) -> Vec:
        return self.norm(self.var_emb(var_id) + self.value_mlp(z) + self.time_mlp(t_abs, dt))

    def static_token(self, static) -> Vec:
        w = self.w
        return self.norm(np.asarray(static, dtype=np.float32) @ w["encoder.static_mlp.weight"].T
                         + w["encoder.static_mlp.bias"] + w["encoder.static_type"])

    def tokens(self, chart: Chart) -> Vec:
        """[1 + n_events, 128]: the static token first, then one token per event."""
        events = self.token(chart.var_id, chart.value, chart.t_abs, chart.dt)
        return vec(np.concatenate([np.asarray(self.static_token(chart.static))[None], np.asarray(events)]))

    # ---- all tokens -> hidden states -> one number
    def _attention(self, x, prefix: str, heads: int = 4) -> np.ndarray:
        w = self.w
        n, d = x.shape
        qkv = x @ w[prefix + "in_proj_weight"].T + w[prefix + "in_proj_bias"]
        q, k, v = (qkv[:, i * d:(i + 1) * d].reshape(n, heads, d // heads).transpose(1, 0, 2) for i in range(3))
        scores = q @ k.transpose(0, 2, 1) / math.sqrt(d // heads)
        weights = np.exp(scores - scores.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        out = (weights @ v).transpose(1, 0, 2).reshape(n, d)
        return out @ w[prefix + "out_proj.weight"].T + w[prefix + "out_proj.bias"]

    def transformer(self, x) -> Vec:
        w = self.w
        x = np.asarray(x, dtype=np.float32)
        for layer in range(2):
            p = f"encoder.transformer.layers.{layer}."
            x = x + self._attention(layer_norm(x, w[p + "norm1.weight"], w[p + "norm1.bias"]), p + "self_attn.")
            h = layer_norm(x, w[p + "norm2.weight"], w[p + "norm2.bias"])
            x = x + gelu(h @ w[p + "linear1.weight"].T + w[p + "linear1.bias"]) @ w[p + "linear2.weight"].T \
                + w[p + "linear2.bias"]
        return vec(layer_norm(x, w["encoder.transformer.norm.weight"], w["encoder.transformer.norm.bias"]))

    def logit(self, chart: Chart) -> float:
        hidden = np.asarray(self.transformer(self.tokens(chart)))
        return float(hidden.mean(axis=0) @ np.asarray(self.head_w) + self.head_b)


def lactate_sweep(model: "SmallModel", chart: Chart, event_index: int, baseline_mmol: float) -> list[dict]:
    """Risk of the patient when the latest lactate is replaced by each target (the real value first)."""
    rows = []
    for target in (baseline_mmol, *LACTATE_TARGETS):
        z = float(chart.value[event_index]) if target == baseline_mmol else standardize_lactate(target)
        logit = model.logit(chart.with_value(event_index, z))
        rows.append({"lactate": target, "logit": logit, "risk": 1.0 / (1.0 + math.exp(-logit))})
    return rows


def relative_change(before, after) -> float:
    """Length of the change of a vector relative to the length of the vector before."""
    before, after = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    return float(np.linalg.norm(after - before) / np.linalg.norm(before))


def about(x: float) -> str:
    """Two significant digits, for ratios in the text: 24571.6 -> '25,000'."""
    return f"{float(f'{x:.2g}'):,.0f}"


def patient_sweep(model: "SmallModel", chart: Chart, event: Event, record_id: int) -> list[dict]:
    """Table rows: this patient's risk for each edited lactate, no_llm computed now, Llemr read from its recorded run."""
    live = lactate_sweep(model, chart, event.index, event.raw)
    recorded = llemr_risk_of_patient(record_id)
    rows = []
    for k, r in enumerate(live):
        rows.append({"latest lactate (mmol/L)": f"{r['lactate']:g} (as measured)" if k == 0 else f"{r['lactate']:g}",
                     "no_llm risk (computed now)": f"{100 * r['risk']:.3f}%",
                     "Llemr risk (recorded)": f"{100 * recorded['baseline' if k == 0 else r['lactate']]:.3f}%"})
    return rows


def share_summary(share, events: list[Event], bias: float) -> dict:
    """Bookkeeping of the logit: bias + the sum of the per-token shares (the static token comes first)."""
    share = np.asarray(share, dtype=np.float64)
    return {"tokens": len(share), "bias + sum of shares": round(bias + float(share.sum()), 4),
            "mean share": round(float(share.mean()), 4)}


# --------------------------------------------------------------------------------------------- the same event for Llemr


def llemr_event(variable: str, value: float, minute: int) -> dict:
    """How P12 writes one event for Llemr-v1 (results/llemr/setup.md section 6): the text that the frozen BiomedBERT
    embeds, and the three raw numbers [time (h), value, duration] appended to its 1,024-d [CLS] vector."""
    row = next(r for r in read_csv(DATA / "variable_map.csv") if r["variable"] == variable)
    shown = format(round(value * float(row["factor"]), 2), "g")
    if row["mimic_table"] == "labevents":  # lab format: "{fluid} {label} {category}: {value} {unit}"
        head = " ".join(x for x in (row["fluid"], row["label"], row["category"]) if x)
    else:  # chart format: "{label}: {value} {unit}"
        head = row["label"]
    text = f"{head}: {shown}" + (f" {row['unit']}" if row["unit"] else "")
    return {"text": text, "appended": [round(minute / 60.0, 4), round(value * float(row["factor"]), 4), 0.0]}


# --------------------------------------------------------------------------------------------- recorded results


def _probs(path: Path) -> dict[int, float]:
    return {int(r["RecordID"]): float(r["prob"]) for r in read_csv(path)}


def llemr_risk_of_patient(record_id: int) -> dict:
    """Llemr's recorded risk of one patient: unperturbed and with the latest lactate set to each target."""
    out = {"baseline": _probs(DATA / "preds" / "llemr_ft_s1.csv")[record_id]}
    for t in LACTATE_TARGETS:
        out[t] = _probs(DATA / "llemr_diagnostics" / f"test_lactate_{t}.csv")[record_id]
    return out


def llemr_lactate_means() -> dict[int, float]:
    """Mean predicted risk of Llemr over the 633 test patients with a lactate value (results/llemr/diagnostics)."""
    return {t: float(np.mean(list(_probs(DATA / "llemr_diagnostics" / f"test_lactate_{t}.csv").values())))
            for t in LACTATE_TARGETS}


def small_lactate_means() -> dict[int, float]:
    """The same dose-response for `no_llm` (results/e3/e3_dose_s1.csv, written by the project's E3 run)."""
    rows = [r for r in read_csv(DATA / "results" / "e3_dose_s1.csv") if r["variant"] == "no_llm" and r["cohort"] == "lactate"]
    return {int(r["target"]): float(r["mean_risk"]) for r in rows}


def test_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels and predicted risks of the 1,199 test patients of split 1 (sorted by RecordID): no_llm, Llemr."""
    a = sorted(read_csv(DATA / "preds" / "no_llm_s1.csv"), key=lambda r: int(r["RecordID"]))
    b = sorted(read_csv(DATA / "preds" / "llemr_ft_s1.csv"), key=lambda r: int(r["RecordID"]))
    assert [r["RecordID"] for r in a] == [r["RecordID"] for r in b]
    assert [r["label"] for r in a] == [r["label"] for r in b]
    y = np.array([int(r["label"]) for r in a])
    return y, np.array([float(r["prob"]) for r in a]), np.array([float(r["prob"]) for r in b])


@dataclass
class TestSet:
    """The 1,199 test patients of split 1 with the saved risks of both models (sorted by RecordID)."""

    y: np.ndarray
    small: np.ndarray
    llemr: np.ndarray
    llemr_shuffled: np.ndarray

    def __lectrace__(self):
        return {"patients": int(len(self.y)), "deaths": int(self.y.sum()),
                "saved risks": ["no_llm", "Llemr", "Llemr, shuffled"]}


def load_test_set() -> TestSet:
    y, small, llemr = test_predictions()
    ids = sorted(int(r["RecordID"]) for r in read_csv(DATA / "preds" / "no_llm_s1.csv"))
    assert ids == sorted(_probs(DATA / "llemr_diagnostics" / "test_value_shuffle.csv"))  # same patients everywhere
    return TestSet(y, small, llemr, llemr_shuffled_predictions())


def llemr_shuffled_predictions() -> np.ndarray:
    """Llemr's risks when the measured values were shuffled between the test patients, in RecordID order."""
    shuffled = _probs(DATA / "llemr_diagnostics" / "test_value_shuffle.csv")
    return np.array([shuffled[i] for i in sorted(shuffled)])


def small_shuffle_auroc() -> dict[str, float]:
    """`no_llm` AUROC unperturbed and with shuffled values (results/e2/e2_s1.csv, written by the project's E2 run)."""
    rows = {r["perturbation"]: r for r in read_csv(DATA / "results" / "e2_s1.csv") if r["variant"] == "no_llm"}
    return {"original": float(rows["none"]["auroc"]), "shuffled": float(rows["value_shuffle"]["auroc"])}


def numeracy_probe() -> dict:
    """Cosine similarities of the frozen BiomedBERT [CLS] vectors of low / normal / high values (recorded)."""
    d = read_json(DATA / "numeric_sensitivity_excerpt.json")
    within = [p["cosine"] for p in d["within_variable"]]
    between = [p["cosine"] for p in d["between_variables"]]
    lactate = {(p["a"], p["b"]): p["cosine"] for p in d["within_variable"]}[("Lactate normal", "Lactate high")]
    return {"lactate 1.5 vs lactate 8 mmol/L": round(lactate, 4),
            f"same variable, low / normal / high value (mean of {len(within)} pairs)": round(float(np.mean(within)), 4),
            f"different variables (mean of {len(between)} pairs)": round(float(np.mean(between)), 4)}


# --------------------------------------------------------------------------------------------- metrics


def auroc(y, p) -> float:
    """Area under the ROC curve = P(risk of a patient who died > risk of one who survived), ties count one half."""
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]  # average rank of tied values
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def paired_bootstrap_auroc(y, p_a, p_b, n_boot: int = 1000, seed: int = 1) -> tuple[float, float, float]:
    """AUROC(a) - AUROC(b) with a 95% percentile interval over paired resamples of the patients (project protocol)."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a, dtype=np.float64), np.asarray(p_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if y[idx].min() == y[idx].max():
            continue
        deltas.append(auroc(y[idx], p_a[idx]) - auroc(y[idx], p_b[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return auroc(y, p_a) - auroc(y, p_b), float(lo), float(hi)
