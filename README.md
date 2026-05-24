# Quantum Bayesian Inference for ETFs

Applies quantum probability theory to Bayesian updating of ETF return states. Represents return distributions as density matrices, then uses the Petz map (or a fidelity‑based heuristic) to compute a posterior state. The per‑ETF score measures how much the update changes the marginal probability of that ETF – a novel signal for regime shifts.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63, 252, 504, 1008, 2016, 4032, 4536 days)
- Density matrix from covariance (normalised to trace 1)
- Petz quantum Bayes rule or geometric mixing
- Score = absolute diagonal change between prior and posterior
- Best window automatically selected (largest absolute raw signal)
- Two‑tab Streamlit dashboard (auto best + manual window selection)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-quantum-bayes-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Run training: `python train.py`
3. Launch dashboard: `streamlit run streamlit_app.py`
4. GitHub Actions runs daily.

## Interpretation

- The density matrix encodes correlations and marginal probabilities (quantum analog of classical distribution).
- The Petz map is the optimal quantum Bayes rule, generalising classical Bayesian updating.
- Large diagonal change for an ETF means the new data significantly revises its marginal probability – a signal of structural change.
- This is a completely novel application of quantum probability to finance.

## Requirements

See `requirements.txt`.
