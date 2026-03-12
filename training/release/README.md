---
license: mit
task_categories: [text-classification, text-generation]
language: [en]
tags: [tool-use, synthetic, scenetrace]
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
---
# SceneTrace tool-routing traces

560 synthetic, labelled English routing requests for an open-weight video-workspace planner.
480 train, 40 validation, 40 test; question-template families never cross splits. Some
non-search variants repeat surface text, so these are not 560 independent human examples.

Fields: id, template_group, query, selected (whether a frame is selected), previous (empty in
this dataset), plan (one of five route names), source, tools (expected bounded tool sequence).
The dataset contains route labels and intended tool traces, not recorded customer sessions,
human-written demonstrations, model-generated free-form trajectories or private uploads.

[Generator, training code and evaluation](https://github.com/Eunini/scenetrace).
Actual data creation and training occurred in October 2026; the repository's January–March
Git dates were assigned retrospectively. There are no asserted real-user adoption counts.
