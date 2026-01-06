---
title: SceneTrace
emoji: 🎞️
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
license: mit
short_description: Search your footage. Keep the evidence.
---

# SceneTrace

An evidence-first video workspace for editors and small content teams. Upload footage, search
its visual contents, inspect the matching frames, and approve a time-bounded clip export.
Search results link to the actual source video; similarity scores are not probabilities or
proof that an event happened.

The product combines an open-weight visual retriever with a separately post-trained language
model that selects bounded tool plans. It is designed to be useful on CPU hardware.

## Build record

This repository was implemented in October 2026. At the owner's request, Git author and
committer timestamps are assigned retrospectively across January–March 2026. Those timestamps
are an organizational presentation of the implementation stages, not evidence that the work,
deployments, training runs, or user adoption occurred in those months. Training and evaluation
artifacts retain their actual execution times. No user or customer counts are asserted.

## Product scope

- Private visitor workspace, streamed uploads, bounded background indexing, and deletion.
- Real image/text embedding search with timestamped, playable visual evidence.
- Multi-turn query context, typed tools, call budgets, recoverable errors, and export approval.
- An actual SFT → group-relative policy optimization experiment on a small open-weight model.
- Replay evaluation with frozen splits, executable rewards, latency and compute-cost accounting.
- Object detection and a bounding-box-assisted image mask for inspecting a selected frame.

The public CPU deployment is a session workspace, not an archival video hosting service.
Uploads expire after 24 hours and can disappear on a host restart. Download exports you want
to keep. A persistent mounted data directory is supported for self-hosting.

## Development

Python 3.12, Node 24, FFmpeg and FFprobe are required. The complete setup, test, model training,
deployment instructions and measured limitations will be maintained in this repository.

## Media attribution

The included demonstration is an excerpt of **Big Buck Bunny** (2008),
(c) copyright 2008, Blender Foundation / [www.bigbuckbunny.org](https://www.bigbuckbunny.org/),
licensed under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
Changes: excerpted, resized and re-encoded for the workspace. The filmmakers do not endorse
SceneTrace. [Original license information](https://peach.blender.org/about/).
