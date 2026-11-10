# Evaluation: gemini / gemini-3.5-flash-lite

Run on 2026-11-10. 97 model calls in total, of which 0 failed (the model could not be reached or gave no usable reply).

## Against the targets

| Measure | Result | Target | |
|---|---|---|---|
| Parser group accuracy | 100% | ≥ 90% | **PASS** |
| Parser quality recall | 100% | ≥ 80% | **PASS** |
| First-attempt reports without failure | 100% | ≥ 90% | **PASS** |
| Unavailable model calls | 0% | ≤ 10% | **PASS** |

## Requirement parser

66 runs: 66 answered by the model and 0 by the offline keyword rules after a fallback (0%). The figures below use the model-answered runs only; a valid request that was rejected counts as a wrong group and as missing every quality.

- **Group accuracy:** 100% of 54 valid-request runs got an accepted position group.
- **Quality recall:** 78 of 78 required qualities were weighted (100%).
- **Minutes handling:** 100% of 6 runs set min_minutes as expected.
- **Rejections:** 100% of 12 runs that should be rejected were rejected.
- **Valid requests wrongly rejected:** 0 of 54 runs.
- **Consistency:** the same group in every run for 18 of 18 cases, and the same top-weighted metric for 13 of 18. Top metric varied in: P02, P11, P13, P14, P18.
- **Latency** (model-answered, pauses excluded): median 1.5 s, 90th percentile 6.7 s.

## Scouting reports

30 reports (each case's top 3 players).

- **First attempt:** 27 verified, 3 with warnings only, 0 failed the fact check. 100% of model drafts had no failure.
- **Final:** 27 verified, 3 warnings.
- **Needed a regeneration:** 0. **Fell back to the offline template:** 0.
- **Model calls:** 30 for 30 reports.
- **Latency** (pauses excluded): median 2.9 s, 90th percentile 8.4 s.
- **Warnings in final reports:** successful dribbles has low reliability and is mentioned, but the caveats section does not name it. (1); non-penalty expected goals (npxG) has low reliability and is mentioned, but the caveats section does not name it. (2)
