# OffloadTranslate · GitHub Actions repository

This directory is the **GitHub repository root**. Copy its contents (including `.github/`) to a new GitHub repository. The Deno server lives separately in the sibling `server/` directory of the ZIP.

## Execution model

1. The Deno server uploads the source document to PersonalStorage in small Base64 chunks and writes `<storage_key>.task.json`.
2. The server uses a PAT to dispatch `glossary.yml` with `storage_key` and `attempt` inputs.
3. The workflow checks out this repository. `worker/llama_runtime.py` downloads the current `ggml-org/llama.cpp` Ubuntu x64 release asset, downloads the GGUF model, starts `llama-server` on `127.0.0.1:18080`, and waits for `/health`.
4. `worker/worker.py` reads the task record and document from PersonalStorage. The glossary workflow stores a JSON glossary and changes task status to `awaiting_confirmation`.
5. After a human confirms the glossary in the browser, Deno writes the approved glossary, increments `attempt`, and dispatches `translate.yml`.
6. The translation worker saves each translated block independently, writes progress/heartbeat into the attempt-specific state JSON, assembles the result, stores output chunks, then marks the task `completed`. The server merges that state into the task response only when its attempt number is current.

No external llama.cpp URL or inference service is used. The llama-server process is local to the Actions runner and listens only on loopback.

## Required GitHub repository secrets

| Secret | Meaning |
|---|---|
| `PERSONALSTORE_TOKEN` | PersonalStorage API token used by the Python worker |
| `PERSONALSTORE_URL` | Optional override; defaults to `https://hc.lxc.nya.je:24687/` |

The Deno server needs its own PAT; it is **not** placed in this worker repository. Do not add the Deno PAT to Actions secrets.

## Optional GitHub repository variables

| Variable | Default | Meaning |
|---|---|---|
| `HF_ENDPOINT` | `https://huggingface.co` | Hugging Face endpoint or mirror |
| `MODEL_REPO` | `Edge-Quant/Hunyuan-7B-Instruct-Q4_K_M-GGUF` | GGUF repository |
| `MODEL_FILE` | `hunyuan-7b-instruct-q4_k_m.gguf` | Q4_K_M model file |
| `MODEL_CONTEXT` | `8192` | llama.cpp context length |
| `MODEL_THREADS` | `4` | CPU inference threads |
| `MODEL_SHA256` | pinned default-model SHA-256 | Optional checksum when overriding the GGUF file |
| `LLAMA_VERSION` | latest release | Pin an exact llama.cpp release tag if needed |

The defaults use the Hunyuan 7B Instruct Q4_K_M GGUF (4.62 GB). Its default file SHA-256 is pinned and verified after download and when restoring the Actions cache. The model can be overridden by setting both `MODEL_REPO` and `MODEL_FILE`; this changes model acquisition, not the execution topology. The workflows cache the GGUF and llama.cpp binary separately. A successful glossary run can therefore cache the model independently of the llama.cpp release version; the translation run reuses each cache when its key matches. Cache eviction is controlled by GitHub; a cache miss simply downloads them again.

## Automated and local tests

The `tests.yml` workflow runs on pushes to `main`, pull requests, and manual dispatch. It checks Python compilation and runs the offline unit tests; it does not download the model or call PersonalStorage.

Run the same checks locally:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m compileall -q worker tests
```

The tests do not download a multi-gigabyte model. A full integration run is performed by dispatching one of the workflows after configuring PersonalStorage secrets.

## Worker behavior and recovery

- `attempt` is a monotonically increasing fencing token. Deno exclusively owns `<key>.task.json`; a worker writes `<key>.attempt.<n>.state.json`. Status reads merge only the state file whose attempt matches the task record, preventing a late old runner from overwriting a newer attempt even in a GET/POST race.
- Each translated block is stored as its own file before the progress record is advanced. A retry skips already stored blocks.
- The Actions job timeout is 350 minutes. Deno's watchdog and the browser recovery path re-dispatch tasks after six hours without a heartbeat.
- Ordinary setup or worker-step failures write a failure state when the PersonalStorage secret is available. A hard job timeout may terminate the runner before its final failure step can write; in that case the task remains stale and the six-hour watchdog can re-dispatch it.
- `concurrency` groups are keyed by the opaque storage key and do not cancel an already running task.

## Supported documents

- `.txt`: translated text output.
- `.md` / `.markdown`: translated Markdown; fenced code blocks, inline code, URLs, and link destinations are protected where possible.
- `.docx`: translated DOCX output. Paragraphs, tables and paragraph styles are retained; complex mixed formatting within a paragraph is simplified to the first run's style.
- `.pdf`: text extraction only (no OCR); output is Markdown text with page separators.

## Important runtime notes

GitHub-hosted standard runners are CPU runners. The model is loaded in 4-bit GGUF form; context and thread counts are deliberately conservative. Model and llama.cpp binaries are downloaded on a fresh runner unless the runner's own cache already contains them. The runner's job timeout is shorter than six hours so the server's retry threshold does not intentionally overlap a still-running job.
