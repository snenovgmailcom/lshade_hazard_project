# Algorithms: L-SHADE Implementation

This directory contains a **pure classical implementation of L-SHADE** (*Success-History based Adaptive Differential Evolution with Linear Population Size Reduction*),
following exactly the algorithmic specification of Tanabe & Fukunaga (see [2]).

The implementation is intended as a **baseline reference** for hazard-based, witness-regime, and first-hitting-time analyses.

---

## Logging and Evaluation Support

The implementation records extensive per-generation data in a `history` dictionary, enabling **analysis**, including:

- parameter adaptation dynamics (`M_F`, `M_CR`)
- population and archive evolution
- full population geometry and fitness
- distributions of sampled and successful `(F, CR)`
- fitness improvements used in memory updates
- trial fitness of the generation-best individual (for validation of
  acceptance and sublevel events)

This design supports:
- survival analysis (Kaplan–Meier)
- hazard estimation
- witness regime validation
- cluster geometry and concentration analysis

---

## Intended Use

This implementation is designed for:
- baseline benchmarking (CEC-style)
- controlled experimental studies
- theoretical validation of L-SHADE behavior

**Faithfulness to the original algorithm** is intentionally prioritized over performance optimizations or code simplifications.

---

## AI Assistance Disclosure

Portions of the code in this directory were developed with the assistance of AI tools (Claude, Anthropic).
All AI-assisted contributions were reviewed, tested, and validated by the authors to
ensure correctness, reproducibility, and alignment with the research objectives.

---

## References

[1] R. Tanabe and A. Fukunaga,  
    *Success-History Based Parameter Adaptation for Differential Evolution*,  
    IEEE Congress on Evolutionary Computation (CEC), 2013.

[2] R. Tanabe and A. Fukunaga,  
    *Improving the Search Performance of SHADE Using Linear Population Size Reduction*,  
    IEEE Congress on Evolutionary Computation (CEC), 2014.
