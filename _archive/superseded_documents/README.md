# Excluded from the index (on purpose)

These documents remain in the repository but are NOT ingested. Each one was
measured to cause wrong answers, and the evidence is in
`evaluation/eval_results.json`.

## SSR_NAAC_2018_FINALnew.pdf  (117 chunks)
A 2018 self-study report. Its facts about the institute are now superseded, and
because it is large it dominated retrieval over current sources. Measured
consequences:
  - "What NAAC grade does Thapar hold?" answered "A, as per the Cycle 3
    Reassessment in 2016" instead of the current A++.
  - "NIRF ranking in engineering?" answered "20th ... in NIRF 2018" instead of
    rank 29 in NIRF 2025.
  - "Which student societies?" answered "Over 30 societies" instead of naming them.

## TICC Annual Report.pdf  (231 chunks, 17.6 MB)
An annual report whose subject matter barely overlaps with student questions. It
was the largest single contributor of chunks in the corpus and it pulled
unrelated passages into answers. Measured consequence:
  - "Which companies recruit from Thapar?" answered "M/s Gurbax Singh & sons ...
    DM recyclers" -- e-waste collection vendors mentioned in the report.

## The tradeoff, stated plainly
Excluding these loses some genuine historical and institutional detail. That is
the intended trade: for a system whose whole promise is grounded, current answers
about fees, ranks and deadlines, a large stale source is worse than no source,
because it produces confidently wrong answers rather than an honest refusal.

To re-include one, move it back into `data/` and re-run `python ingest.py`.
