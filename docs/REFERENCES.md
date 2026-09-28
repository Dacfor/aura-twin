# References & Source Documents

> This file replaces the original `document/` folder which contained copyrighted
> PDFs and internal development notes. All references are cited below with links
> to the original publicly accessible sources.

---

## Core Scientific References

### 1. Dinges, D. F. & Metaxas, D. et al. (2008–2012)
**Optical Computer Recognition of Stress, Affect and Fatigue during Performance in Spaceflight**  
National Space Biomedical Research Institute (NSBRI), NASA Taskbook.  
🔗 [NASA Taskbook Entry](https://taskbook.nasaprs.com/tbp/index.cfm?action=public_query_taskbook_content&TASKID=7902)

### 2. Wierwille, W. W. et al. (1994)
**Research on vehicle-based driver status/performance monitoring: development, validation, and refinement of algorithms for detection of driver drowsiness.**  
NHTSA Report DOT-HS-808-247.

### 3. Dinges, D. F. & Grace, R. (1998)
**PERCLOS: A Valid Psychophysiological Measure of Alertness as Assessed by Psychomotor Vigilance.**  
FHWA Report FHWA-MCRT-98-006.

### 4. Ekman, P. & Friesen, W. V. (1978)
**Facial Action Coding System: A Technique for the Measurement of Facial Movement.**  
Consulting Psychologists Press.

### 5. Barrett, L. F. et al. (2019)
**Emotional Expressions Reconsidered: Challenges to Inferring Emotion From Human Facial Movements.**  
*Psychological Science in the Public Interest*, 20(1), 1–68.  
🔗 [DOI: 10.1177/1529100619832930](https://doi.org/10.1177/1529100619832930)

---

## Cardiovascular Digital Twin References

### 6. Heldt, T. et al. (2004)
**Computational Model of Cardiovascular Response to Orthostatic Stress.**  
*Journal of Applied Physiology*, 96(4), 1249–1261.  
🔗 [PubMed: 15943210](https://pubmed.ncbi.nlm.nih.gov/15943210/)

### 7. Limper, U. et al. (2014)
**Interactions of the Human Cardiopulmonary, Hormonal and Body Fluid Systems in Parabolic Flight.**  
🔗 [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0094576506003079)

### 8. Convertino, V. A. (2001)
**Lower Body Negative Pressure as a Tool for Research in Aerospace Physiology and Military Medicine.**  
*Journal of Gravitational Physiology*, 8(2), 1–14.  
🔗 [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S107731420300081X)

---

## ASI Challenge Documentation

### 9. ASI BEX2026 Challenge #3 — Artemis Community
**Scenari BEX2026** (Challenge scenarios document).  
Provided by the Italian Space Agency (ASI) for the Space Hackathon.  
Referenced in: [`docs/BEX2026_CHALLENGE3_DOSSIER.md`](BEX2026_CHALLENGE3_DOSSIER.md)

---

## Internal Design Documents

### 10. AURA-Face Technical Specification v2
Original Italian-language development specification used during the hackathon.  
Key content has been incorporated into:
- [`docs/ARCHITECTURE_AND_PITCH_GUIDE.md`](ARCHITECTURE_AND_PITCH_GUIDE.md)
- [`README.md`](../README.md)

### 11. Cardiovascular 0D Model (MATLAB Reference)
**modello0D_NOBATTITO_LBNP_ACP_StochVarPers_OTT_MINCOST_v3.m**  
Original MATLAB implementation of the lumped-parameter cardiovascular model  
by the AURA cardiovascular sub-team. The Python `cardiovascular.py` module
implements a trajectory-replay visualization derived from this model's outputs.
