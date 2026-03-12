# Observed failures and what the product does about them

## A visually plausible retrieval miss

On the six-query open-film check, CLIP's first result for “a dark hole in a tree” was
31.875 seconds, outside the annotated interval of 33–44 seconds. None of its top three
fell inside that interval. Overall top-1 and hit-at-three were both 5/6. The raw query,
scores, timestamps and intervals remain in `artifacts/retrieval-evaluation.json`.

This is one film and an illustrative annotation set, not an independent benchmark. The
observed miss is why results are presented as sampled visual candidates with a playable
source, rather than generated assertions about what happened. At most 32 sampled frames
also means short events between samples can be missed entirely.

## A high-confidence object detector error

The actual API check retrieved the cartoon bird at 18.75 seconds. DETR labelled that bird
as **person** with a detector score of **0.9645**. The box was usable for a GrabCut foreground
mask, but the class label was wrong. See `evidence/api-real-models.json` and the contact
sheet. An implementation that reported “96% certain a person appears here” would be
misleading: a detector score is not calibrated correctness, especially under animation
domain shift relative to its training data.

The interface therefore shows the actual frame and marks boxes as predictions. A threshold
of 0.7 reduces displayed boxes, but is not a calibrated safety threshold. No general detection
mAP or segmentation IoU claim is made without an appropriately annotated evaluation set.
Those metrics measure different tasks from the exact-match routing accuracy in the training
report. GrabCut's foreground mask is a classical box-initialized estimate, not semantic class
evidence or an instance segmentation model.

## Saturated SFT rewards during the RL pilot

The supervised router fit the synthetic training requests quickly. A first, unfinished RL
pilot sampled almost exclusively the same correct route, leaving little within-group reward
variation. Before testing any final candidate, the completed SFT checkpoint was retained and
the final RL run used temperature 3 to explore alternative plans. The report includes the
number of nonconstant reward groups and the actual parameter change, rather than assuming
an RL label alone establishes learning or improvement. Both stage checkpoints are published.

In the final run, 8/60 groups had nonconstant rewards and adapter parameters changed by
an L2 norm of 0.8037. SFT validation was 39/40; GRPO validation fell to 37/40. Both later
scored 40/40 on the separate held-out test, but model selection had already chosen SFT.
The service therefore serves SFT. This experiment establishes that the RL update ran and
changed the policy; it does not establish an accuracy gain from RL. The base model scored
8/40 on each split. All these numbers concern a small synthetic routing task.

## Product failures caught by executable tests

- Very short videos initially produced no sampled frame. Selection now uses absolute bin
  boundaries and the fixture checks a real two-second source.
- Loading metadata in a separate process initially triggered restart recovery. Recovery now
  runs only during service startup, protected by an exclusive instance lock.
- Chunked JSON could bypass a header-only size check. The body stream is now bounded too.
- Export links initially disappeared on refresh. Bootstrap now restores the last twelve jobs,
  the latest pending approval and the last eight conversation turns in the visitor session.

These failures are fixed and regression tested. They do not constitute a complete independent
security review, a real-user study, or an externally calibrated video-quality evaluation.
