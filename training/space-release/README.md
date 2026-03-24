---
title: SceneTrace
emoji: 🎞️
colorFrom: gray
colorTo: green
sdk: static
app_file: index.html
license: mit
short_description: Search your footage. Keep the evidence.
---

# SceneTrace live workspace

This free static Space embeds the public SceneTrace application. The actual Python backend,
open-weight models and video processing run on the existing CPU server; the frontend is on
Vercel. Docker CPU hosting on Hugging Face required PRO, so this release uses a static
launcher and adds no hosting subscription.

[Open the workspace directly](https://scenetrace-inioluwa.vercel.app) ·
[Source](https://github.com/Eunini/scenetrace) ·
[Router weights and both comparison checkpoints](https://huggingface.co/Ineeoluwa/scenetrace-router-135m) ·
[Training traces](https://huggingface.co/datasets/Ineeoluwa/scenetrace-tool-traces)

Upload footage or try the openly licensed sample, search visual frames, inspect source
timestamps and approve MP4 exports. Use the direct link if your browser blocks iframe cookies.
Uploads are processed on the server in a visitor-scoped workspace and expire after 24 hours.
Keep your originals. Nothing is automatically used for training.

The model experiment is real SFT and finite-action GRPO on SmolLM2-135M. This is a small,
synthetic five-way routing benchmark, not general video understanding. SFT was selected
on validation; both stages scored 40/40 on the separate synthetic test versus the base's 8/40.
See the published report for the RL validation regression and observed vision failures.

Actual implementation and training occurred in October 2026. January–March Git dates in
the source repository were assigned retrospectively and do not establish past development
or adoption. No external user counts are asserted.
