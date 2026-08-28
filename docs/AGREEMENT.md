# Label agreement (not three human annotators)

The withdrawn draft reported Cohen’s κ = 0.81 from three human KAP
annotators. That gold set is not in this repository. What *is* here is three
**independent silver codebooks** applied to the same public text:

| Rater | KAP polarity rule |
|-------|-------------------|
| A | Lexicon + subject priors (`kap_polarity`) |
| B | Subject-title taxonomy only (`kap_polarity_b`) |
| C | Body/teaser tokens only, threshold ±2 (`kap_polarity_c`) |

Chart rater A is the 20-bar VisualClaw codebook; chart rater B uses 10-bar
levels. Neither chart rule looks at future returns.

`human_annotation_sample.csv` annotator columns are filled by **codebook B**,
with `annotator_id=codebook_b`. That is a second silver pass, not a person.

## Numbers

From `results/agreement.json`:

**250-row stratified sample**

- KAP Fleiss κ (A,B,C) = 0.32 (po=0.80). B and C are conservative (mostly
  `neutral`), so chance agreement is already high.
- Chart Cohen κ (A vs B) = 0.66

**10k slice (stable)**

- KAP Fleiss κ = 0.50
- KAP Cohen κ A vs B = 0.42, A vs C = 0.63, B vs C = 0.44
- Chart Cohen κ A vs B = 0.52

These figures replace the unverifiable 0.81. They are codebook reliability,
not human reliability. A pretrained multilingual star-rating model (rater D)
on the same public KAP strings yields κ≈0 vs codebook A and is **not** a
person. A three-annotator pass on this sample is still the right next
measurement; until then we do not quote 0.81.
