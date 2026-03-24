# Architecture and operational boundaries

The React/Vite workspace is served by FastAPI in one Docker service. A Vercel frontend
can proxy `/api/*` to the same service. All media and model inference stay on the server;
the browser receives only that visitor's metadata, protected media and tool results.

## Data and execution

1. Bootstrap creates a random visitor identifier in a signed, HTTP-only cookie. No password
   or account is required. The cookie is an access capability: possession grants access to
   that workspace. It expires after 24 hours. Public deployments use a separately generated
   signing secret held in the host's secret store.
2. An upload streams to a generated path with a 50 MiB limit. The displayed filename is
   never used as a storage path. Admission checks enforce five videos per visitor, a shared
   disk ceiling and a six-job queue. FFprobe restricts formats, duration and dimensions.
3. A single background worker creates a stripped MP4 proxy, selects at most 32 frames and
   computes pinned CLIP embeddings. Progress and failures remain visible in SQLite.
   FFmpeg gets explicit argument arrays, restricted local protocols and timeouts.
4. The post-trained SmolLM2 language model scores five fixed completions. The selected
   plan dispatches typed tools, never code or free-form SQL. SQLite queries include the
   owner predicate. Context retains eight turns and a bounded prior visual query.
5. Search ranks cosine similarity, filters optional frame sharpness and suppresses nearby
   timestamps. Results are candidate frames, not verified event claims. The source player
   lets the user check the finding at its timestamp.
6. Every request has a four-call tool budget. Planner concurrency is bounded, image inference
   is serialized, and per-workspace rate limits produce clear retry errors.
7. An export creates a pending approval with a validated time range. Only a separate approval
   can queue rendering. Duplicate approvals are rejected. Downloads are owner-scoped.
8. A periodic janitor deletes expired media and metadata; restart recovery marks interrupted
   work failed. Visitor deletion removes video files, embeddings and associated approvals.

## Vision tools

CLIP performs retrieval; it is not fine-tuned here. DETR predicts boxes for its COCO classes
on one selected frame. GrabCut can create a foreground mask initialized from a box. It is
classical image segmentation, not SAM, semantic segmentation or an instance segmentation
model. There is no depth estimator, audio transcription or person identification.

## Deployment

`Dockerfile` pins the Python lockfile, CPU Torch packages and publisher model revisions.
`scripts/download_models.py` caches model assets during the build, keeping first-run network
traffic out of the inference path. PyTorch publisher checkpoints are loaded with
`weights_only=True`; LoRA artifacts use safetensors. No model remote code is trusted.

Set `SCENETRACE_DATA` to a persistent mounted directory and `SCENETRACE_SESSION_SECRET` to a
stable secret for durable self-hosted workspaces. The current backend runs as a restricted systemd service behind Nginx and TLS on the
existing server; the Vercel frontend proxies requests to it. A Hugging Face static Space
launches the public app. Docker CPU Spaces required PRO at publication time, so no Docker
Space or paid subscription was created. The public working directory has a 24-hour TTL. It is a public early release, not an archival hosting system or an
SLA-backed multi-tenant service. A malicious client can create additional visitor sessions;
the queue, disk and concurrent-inference limits bound shared resources, but do not substitute
for authenticated quotas and infrastructure abuse protection at larger scale.

Use one Uvicorn worker per data directory: the in-process queue and media locks are not a
distributed job system. For horizontal scaling, move jobs to a durable external queue,
media to private object storage, and metadata/authorization to Postgres. This release does
not claim to already use Postgres or ClickHouse.

## Privacy and failure handling

The server processes uploads, queries and optional explicit feedback. None is automatically
included in training. Eight conversation turns and feedback remain in the session database
until expiry or deletion. Access logging is disabled to avoid recording user queries or
capability cookies. The app has no analytics, ads, tracking pixels or third-party runtime
scripts. HF/Vercel infrastructure may retain their own operational logs under their policies.

The API rejects cross-origin mutations, uses private/no-store media responses, binds every
media/job/approval/feedback lookup to the visitor, validates identifiers and disallows arbitrary
tool fields. Tests exercise cross-visitor access, malformed media, non-finite JSON numbers,
out-of-range exports, repeated approvals and safe deletion. Host restart, decoding errors,
missing frames, full queues and missing selections produce visible recoverable states.
