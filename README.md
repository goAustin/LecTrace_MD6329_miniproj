# Do numbers survive the trip into the model?

A [LecTrace](https://praisegee.github.io/lectrace/) lecture (about 10 minutes) on ICU mortality prediction with
PhysioNet 2012 data. A 0.29M-parameter event encoder trained from scratch (`no_llm`) is compared with Llemr-v1, a
7.1B-parameter EHR language model, by following one test patient through the small model and asking what happens
when a lab value (lactate) changes.

The lecture is `01_icu_mortality.py`. The small model runs live in numpy with its trained weights; Llemr's outputs
are recorded in `data/`.

## Run it

Python 3.11 or newer.

```bash
pip install lectrace numpy matplotlib
lectrace serve        # http://localhost:7000
```

On macOS, port 7000 is often taken by AirPlay Receiver ("Address already in use"); run
`lectrace serve --port 7001` and open http://localhost:7001 instead.

To build a static site into `_site/`: `lectrace build --output _site`.

## Data

The patient record and test predictions come from the public
[PhysioNet/Computing in Cardiology Challenge 2012](https://physionet.org/content/challenge-2012/) data.
