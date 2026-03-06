# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sumi is a macOS desktop app (Tauri 2) that provides system-wide speech-to-text via a global hotkey. It supports both local (Whisper via `whisper-rs` with Metal acceleration, or Qwen3-ASR via `qwen3-asr` crate with Metal acceleration) and cloud STT APIs (Groq/OpenAI/Deepgram/Azure/Custom) for transcription, and pastes the result at the cursor. Optionally uses a local LLM (via `candle`) or cloud API (GitHubModels/Groq/OpenRouter/OpenAI/Gemini/SambaNova/Custom) to polish transcription output. Also supports an "Edit by Voice" mode that applies spoken instructions to selected text via LLM, and a "Meeting Mode" for continuous meeting transcription with file-based transcript storage.

## Commands

```bash
# Run in development mode
cargo tauri dev

# Build for production
cargo tauri build

# Type-check without building
cargo check

# Lint
cargo clippy
```

### Frontend

```bash
# Frontend dev server (started automatically by cargo tauri dev)
# Note: Tauri runs beforeDevCommand from the frontend/ directory
cd frontend && npm run dev

# Frontend type-check
cd frontend && npx svelte-check --tsconfig ./tsconfig.json

# Frontend production build (started automatically by cargo tauri build)
cd frontend && npm run build
```

## Architecture

### Backend

Rust source files (16 modules + platform sub-module):

#### `src/lib.rs` — Core application logic & app setup
- **`AppState`** — shared state managed by Tauri: `is_recording` (AtomicBool), `is_processing` (AtomicBool), `buffer` (Arc<Mutex<Vec<f32>>>), `sample_rate`, `settings`, `mic_available`, `whisper_ctx`, `llm_model`, `captured_context`, `context_override`, `test_mode` (AtomicBool), `voice_rule_mode` (AtomicBool), `last_hotkey_time`, `http_client` (shared reqwest client), `api_key_cache`, `edit_mode` (AtomicBool), `edit_selected_text`, `edit_text_override`, `saved_clipboard`, `vad_ctx` (Silero VAD), `downloading` (AtomicBool), `audio_thread`, `qwen3_asr_ctx`, `model_switching` (AtomicBool), `reconnecting` (AtomicBool), `streaming_active` (AtomicBool), `streaming_cancelled` (AtomicBool), `streaming_result`, `feeder_stop_cv` (Condvar), `feeder_stop_mu`, `meeting_active` (AtomicBool), `meeting_cancelled` (AtomicBool), `meeting_stopping` (AtomicBool), `meeting_session` (AtomicU64), `meeting_start_time`, `active_meeting_note_id`, `streaming_session` (AtomicU64), `whisper_preview_active` (AtomicBool), `whisper_preview_session` (AtomicU64), `registered_edit_shortcut`.
- **Global shortcut handler** — three hotkeys: the main recording toggle (default `Alt+KeyZ`, debug: `Alt+Super+KeyZ`), edit-by-voice (default `Control+Alt+KeyZ`, debug: `Control+Alt+Super+KeyZ`), and meeting mode (configurable, default disabled). Main toggle: first press starts recording + shows overlay; second press stops recording, transcribes, optionally polishes with LLM, copies to clipboard, optionally pastes with Cmd+V, then hides the overlay. Edit-by-voice: copies selected text via Cmd+C, records spoken instruction, applies edit via LLM, pastes result. Meeting mode: toggle continuous transcription with file-based storage. Max recording duration: 120 seconds (auto-stop, normal mode only). Debounce: 300 ms.
- Registers all Tauri commands from `commands.rs` and sets up the tray menu, windows, and global shortcuts.

#### `src/settings.rs` — Settings & data directories
- **`Settings`** — persisted to `~/.sumi/config/settings.json`. Fields: `hotkey`, `auto_paste`, `polish` (PolishConfig), `history_retention_days` (u32, 0 = keep forever), `language` (Option<String>, UI language override), `stt` (SttConfig), `edit_hotkey` (Option<String>, default `"Control+Alt+KeyZ"`), `onboarding_completed` (bool), `mic_device` (Option<String>, preferred mic input device), `meeting_hotkey` (Option<String>, default None = disabled).
- **Data directory layout**: `~/.sumi/` (release) or `~/.sumi-dev/` (debug) with subdirectories: `config/` (settings.json), `models/` (Whisper & LLM GGUF files, Qwen3-ASR model dirs), `history/` (history.db, meeting WAL files), `audio/` (WAV files).

#### `src/commands.rs` — Tauri command handlers
All `#[tauri::command]` functions exposed to the frontend:
- **Recording**: `start_recording`, `stop_recording`, `cancel_recording`
- **Mode control**: `set_test_mode`, `set_voice_rule_mode`, `set_context_override`, `set_edit_text_override`
- **Settings**: `get_settings`, `save_settings`, `update_hotkey`, `update_edit_hotkey`, `update_meeting_hotkey`, `reset_settings`
- **Polish**: `get_default_prompt`, `get_default_prompt_rules`, `test_polish` (async), `generate_rule_from_description` (async)
- **Mic**: `get_mic_status`, `set_mic_device`
- **Whisper models**: `check_model_status`, `download_model`, `list_whisper_models`, `get_system_info`, `get_whisper_model_recommendation`, `switch_whisper_model` (async), `download_whisper_model`
- **LLM models**: `check_llm_model_status`, `download_llm_model`, `list_polish_models`, `switch_polish_model` (async), `download_polish_model`
- **Qwen3-ASR models**: `list_qwen3_asr_models`, `switch_qwen3_asr_model` (async), `download_qwen3_asr_model`
- **VAD**: `check_vad_model_status`, `download_vad_model`
- **Model deletion**: `delete_whisper_model`, `delete_polish_model`, `delete_qwen3_asr_model`, `delete_vad_model`
- **Credentials**: `save_api_key`, `get_api_key`
- **History**: `get_history`, `get_history_page` (async), `get_history_stats` (async), `delete_history_entry` (async), `clear_all_history` (async), `export_history_audio` (async), `get_history_storage_path`
- **Meeting notes**: `list_meeting_notes`, `get_meeting_note`, `rename_meeting_note`, `delete_meeting_note`, `delete_all_meeting_notes`, `get_active_meeting_note_id`, `polish_meeting_note` (async, uses `spawn_blocking`)
- **Permissions**: `check_permissions`, `open_permission_settings`
- **Utilities**: `get_app_icon`, `trigger_undo`, `copy_image_to_clipboard`, `is_dev_mode`, `export_diagnostic_log`

#### `src/stt.rs` — STT configuration
- **`SttConfig`** — fields: `mode` (SttMode: Local or Cloud), `cloud` (SttCloudConfig), `whisper_model` (WhisperModel), `local_engine` (LocalSttEngine: Whisper or Qwen3Asr), `qwen3_asr_model` (Qwen3AsrModel: Qwen3Asr1_7B or Qwen3Asr0_6B), `language` (BCP-47 string, "auto" or specific like "zh-TW").
- **`SttCloudConfig`** — fields: `provider` (SttProvider: Deepgram/Groq/OpenAi/Azure/Custom), `api_key` (#[serde(skip)]), `endpoint`, `model_id`, `language`.
- **`LocalSttEngine`** — enum: `Whisper` (default), `Qwen3Asr`.
- **`Qwen3AsrModel`** — enum: `Qwen3Asr1_7B` (default, ~1.7 GB), `Qwen3Asr0_6B` (~0.6 GB). Model files stored in `~/.sumi/models/qwen3-asr-{1.7b,0.6b}/`.
- **`Qwen3AsrModelInfo`** — serializable model metadata for frontend: `id`, `display_name`, `description`, `size_bytes`, `downloaded`, `file_size_on_disk`, `is_active`.
- **`run_cloud_stt`** — dispatches to cloud STT provider APIs; accepts `prompt` parameter for Whisper-compatible APIs.

#### `src/qwen3_asr.rs` — Qwen3-ASR local STT engine
- **`Qwen3AsrCache`** — cached `AsrInference` instance with loaded model, reused across transcriptions.
- **`warm_model`** / **`transcribe`** / **`invalidate`** — cache management for the Qwen3-ASR model.
- **`transcribe_with_cached_qwen3_asr`** — batch transcription for single audio segments.
- **`run_feeder_loop`** — Qwen3-ASR streaming feeder for normal recording mode. Uses `init_streaming`/`feed_audio`/`finish_streaming` API with 2s tick intervals. Supports `initial_text` for cross-session context continuity. Emits `transcription-partial` events to overlay.
- **`run_meeting_feeder_loop`** — batch-per-segment meeting feeder. Accumulates audio in `chunk_buf`, detects silence via VAD, batch-transcribes segments, appends to WAL file. Uses `MAX_SEGMENT_SAMPLES` (120s) for forced segment flush. Same architecture as Whisper/Cloud meeting feeders.

#### `src/whisper_streaming.rs` — Whisper live preview feeder
- **`run_whisper_preview_loop`** — Whisper streaming feeder for normal recording mode with live preview. Batch-transcribes accumulated audio at 2s intervals, emits `transcription-partial` events. Session-guarded via `whisper_preview_session`.
- **`run_whisper_meeting_feeder_loop`** — Whisper meeting mode feeder. Same batch-per-segment architecture: `chunk_buf` + VAD silence detect + batch transcribe + WAL append.

#### `src/polisher.rs` — AI text polishing
- **`PolishConfig`** — fields: `enabled` (default false), `model` (PolishModel), `custom_prompt` (Option<String>), `mode` (PolishMode: Local or Cloud, default Cloud), `cloud` (CloudConfig), `prompt_rules` (HashMap<String, Vec<PromptRule>>, per-language map), `dictionary` (DictionaryConfig), `reasoning` (bool, default false).
- **`CloudConfig`** — fields: `provider` (CloudProvider: GitHubModels/Groq/OpenRouter/OpenAi/Gemini/SambaNova/Custom), `api_key` (#[serde(skip)]), `endpoint`, `model_id` (default empty, locale-initialized on new install: Chinese locales → "qwen/qwen3-32b", others → "openai/gpt-oss-120b").
- **`PolishModel`** variants: `LlamaTaiwan` (Llama 3 Taiwan 8B, ~4.9 GB), `Qwen25` (Qwen 2.5 7B, ~4.7 GB), `Qwen3` (Qwen 3 8B, ~5.0 GB).
- **`polish_text`** — dispatches to `run_cloud_inference` (OpenAI-compatible HTTP) or `run_llm_inference` (local candle) based on `PolishMode`. Returns `PolishResult { text, reasoning }`.
- **`edit_text_by_instruction`** — "Edit by Voice": takes selected text + spoken instruction, returns edited text via LLM.
- **Prompt rules**: `PromptRule { name, match_type (AppName/BundleId/Url), match_value, prompt, enabled, icon (Option<String>), alt_matches (Vec<MatchCondition>) }`. `MatchCondition { match_type, match_value }` allows multi-match rules. The `icon` field is an optional key for the frontend (e.g. "terminal", "slack"); auto-detected if None. Built-in preset rules for Gmail, Claude Code, Gemini CLI, Codex CLI, Aider, Terminal, VSCode, Cursor, Antigravity, iTerm2, Notion, WhatsApp, Telegram, Slack, Discord, LINE, GitHub, X (Twitter).
- **Dictionary**: `DictionaryConfig { enabled, entries: Vec<DictionaryEntry> }` for proper noun correction, injected into both Whisper initial prompt and LLM system prompt.
- **Reasoning toggle**: When `reasoning` is false, `/no_think` is prepended to suppress model reasoning (e.g. Qwen3 `<think>` blocks).

#### `src/whisper_models.rs` — Multi-model Whisper selection
- **`WhisperModel`** variants: `LargeV3Turbo` (default, 1.62 GB), `LargeV3TurboQ5` (547 MB), `BelleZh` (1.6 GB), `Medium` (1.53 GB), `Small` (488 MB), `Base` (148 MB), `LargeV3TurboZhTw` (1.6 GB). Note: `WhisperModel::all()` returns only 5 managed models (excludes Medium and Small).
- **`WhisperModelInfo`** — serializable model metadata for frontend: `id`, `display_name`, `description`, `size_bytes`, `languages`, `downloaded`, `file_size_on_disk`, `is_active`.
- **`SystemInfo`** — `total_ram_bytes`, `available_disk_bytes`, `is_apple_silicon`, `gpu_vram_bytes`, `has_cuda`, `os`, `arch`.
- **`recommend_model`** — smart model recommendation based on system RAM/VRAM/disk/language preference.

#### `src/transcribe.rs` — Whisper transcription & VAD
- **`WhisperContextCache`** — cached `WhisperContext` with loaded model path, reused across transcriptions.
- **`VadContextCache`** — cached Silero VAD context (`ggml-silero-v6.2.0.bin`).
- **`filter_with_vad`** — Silero VAD speech filtering before Whisper transcription.
- **`has_speech_vad`** — checks if an audio chunk contains speech using Silero VAD, with RMS fallback if VAD unavailable.
- **`transcribe_with_cached_whisper`** — accepts `dictionary_terms` for Whisper initial prompt biasing and `app_name` for context-aware prompting.

#### `src/audio.rs` — Audio recording
- **`spawn_audio_thread`** — creates a persistent always-on cpal input stream at app startup. The callback checks `is_recording` atomically and discards samples when false, giving true zero-latency recording start.
- **`try_reconnect_audio`** — auto-reconnect on mic disconnection.
- **`do_start_recording`** — clears the buffer and flips `is_recording` to true (instant, <5 ms). For Qwen3-ASR local mode, spawns `run_feeder_loop` (normal) or `run_meeting_feeder_loop` (meeting). For Whisper local mode, spawns `run_whisper_preview_loop` (normal) or `run_whisper_meeting_feeder_loop` (meeting). For cloud meeting mode, spawns `run_cloud_meeting_feeder_loop`.
- **`do_stop_recording`** — flips `is_recording` to false, extracts samples, resamples to 16 kHz. Applies VAD filtering (or RMS trimming fallback). Dispatches to local Whisper/Qwen3-ASR or cloud STT based on `SttConfig.mode` and `local_engine`.

#### `src/meeting_notes.rs` — Meeting notes storage (SQLite + WAL files)
- **`MeetingNote`** — fields: `id`, `title`, `transcript`, `created_at`, `updated_at`, `duration_secs`, `stt_model`, `is_recording`, `word_count`, `summary`.
- **`PolishedMeetingNote`** — returned by `polish_meeting_note`: `summary` text.
- SQLite table `meeting_notes` in `history.db`. WAL files for in-progress transcripts stored in `~/.sumi/history/`.
- **WAL file ops**: `append_wal`, `read_wal`, `remove_wal` — append-only files for crash-safe transcript accumulation during recording.
- **CRUD**: `create_note`, `finalize_note`, `get_note` (reads from WAL for recording notes), `list_notes`, `rename_note`, `delete_note`, `delete_all_notes`.
- **`recover_stuck_notes`** — reads WAL files on startup to recover notes stuck in `is_recording=true` state.

#### `src/context_detect.rs` — App context detection
- **`AppContext`** — `app_name`, `bundle_id`, `url`, `terminal_host` (original terminal app name when `app_name` was enriched with a CLI tool name; empty when no enrichment occurred).
- NSWorkspace FFI for frontmost app + osascript for browser URLs. Supports Safari, Chrome, Arc, Brave, Microsoft Edge. Terminal emulators detected: Terminal.app, iTerm2, Ghostty, Warp. Terminal subprocess detection enriches `app_name` with CLI tool names (Claude Code, Gemini CLI, Codex CLI, Aider, Neovim, Vim, Emacs, Helix).
- Cross-platform: Windows uses `GetForegroundWindow`/`QueryFullProcessImageNameW` FFI.
- Captured context fed to LLM prompt for context-aware polishing.

#### `src/history.rs` — Transcription history (SQLite)
- **`HistoryEntry`** — fields: `id`, `timestamp`, `text` (polished), `raw_text`, `reasoning` (Option), `stt_model`, `polish_model`, `duration_secs`, `has_audio`, `stt_elapsed_ms`, `polish_elapsed_ms` (Option), `total_elapsed_ms`, `app_name`, `bundle_id`, `chars_per_sec`, `word_count` (u64, multilingual via UAX#29 word boundaries).
- **`HistoryStats`** — `total_entries`, `total_duration_secs`, `total_chars`, `local_entries`, `local_duration_secs`, `total_words`.
- SQLite database (`history.db`) with WAL mode. Audio files saved as WAV under `~/.sumi/audio/`.
- Functions: `load_history`, `load_history_page` (paginated), `get_stats`, `add_entry`, `delete_entry`, `clear_all`, `migrate_from_json` (legacy migration).
- Retention cleanup: deletes entries older than `history_retention_days` setting.

#### `src/credentials.rs` — API key storage
- Cross-platform credential storage. macOS: `security` CLI (Keychain). Non-macOS: `keyring` crate (Windows Credential Manager).
- Service name format: `sumi-api-key-{provider}` (release) or `sumi-dev-api-key-{provider}` (debug). Functions: `save`, `load`, `delete`.

#### `src/hotkey.rs` — Hotkey parsing
- `parse_key_code`, `parse_hotkey_string`, `hotkey_display_label` — parsing and display of hotkey strings.

#### `src/permissions.rs` — System permissions
- `check_permissions() -> PermissionStatus { microphone, accessibility }` — checks AVFoundation/AXIsProcessTrusted.
- `open_permission_settings(permission_type)` — opens System Settings pane or triggers microphone access prompt.

#### `src/platform/` — Cross-platform abstraction
Replaces the previous `macos_ffi` module. Sub-modules: `macos.rs`, `windows.rs`, `fallback.rs`.
- `set_app_accessory_mode` — LSUIElement equivalent.
- `set_main_window_movable` — enables drag-by-background for the main window (macOS only).
- `setup_overlay_window` — converts NSWindow to `SumiOverlayPanel` (NSPanel subclass), sets window level to `kCGPopUpMenuWindowLevel` (101), disables `hidesOnDeactivate`, joins all Spaces.
- `show_overlay` / `hide_overlay` — `orderFrontRegardless` with alpha=1.0 / alpha=0.0 (keeps window registered across Spaces).
- `simulate_paste` / `simulate_copy` / `simulate_undo` — CGEvent-based HID simulation.

### Frontend (`frontend/`)
Svelte 5 + TypeScript + Vite. Two Vite entry points (`main.html` + `overlay.html`), each mounting a separate Svelte app. Uses `@tauri-apps/api` ESM imports (`withGlobalTauri: false`). Path alias: `$lib → src/lib`.

- **`src/main/`** — Settings window. Pages: StatsPage (landing/default), SettingsPage, PromptRulesPage, DictionaryPage, HistoryPage, MeetingPage, TestWizard, AboutPage. Components: Sidebar, SetupOverlay, ConfirmModal, RuleCard, RuleGridCard, RuleEditorModal, DictEditorModal, HistoryDetailModal, and settings sub-sections (BehaviorSection, LanguageSection, HotkeySection, MicSection, SttSection, PolishSection, DangerZone).
- **`src/overlay/`** — Transparent, always-on-top recording indicator capsule. States: `preparing`, `recording`, `transcribing`, `polishing`, `pasted`, `copied`, `error`, `edited`, `edit_requires_polish`, `processing`, `undo`, `meeting_stopped`. Features 20-bar canvas waveform and elapsed timer with color gradient.
- **`src/lib/`** — Shared code: `types.ts` (TypeScript interfaces), `api.ts` (typed Tauri command wrappers), `constants.ts` (provider metadata, key labels, SVG icons), `utils.ts`, `stores/` (Svelte 5 `$state` rune stores for settings, i18n, UI state, iconCache), `components/` (SettingRow, Toggle, SegmentedControl, Select, Keycaps, Modal, ProgressBar, CloudConfigPanel, InstructionCard, SectionHeader).
- **`src/i18n/`** — 58 locale JSON files (af, ar, az, be, bg, bs, ca, cs, cy, da, de, el, en, es, et, fa, fi, fr, gl, he, hi, hr, hu, hy, id, is, it, ja, kk, kn, ko, lt, lv, mi, mk, mr, ms, ne, nl, no, pl, pt, ro, ru, sk, sl, sr, sv, sw, ta, th, tl, tr, uk, ur, vi, zh-CN, zh-TW), statically imported by the i18n store.

### Two Windows
- **`main`** (settings): 1120x800 px, hidden by default, shown by tray click or "Settings..." menu item; close button hides rather than quits. `titleBarStyle: "Overlay"` with hidden title. Default page is StatsPage.
- **`overlay`**: frameless, transparent, always-on-top, 300x52 px, centered horizontally near the bottom of the screen during recording. Shown/hidden without activating the app via `platform` module.

### Hotkey String Format
Hotkeys are stored as `"Modifier+...+KeyCode"`, e.g. `"Alt+KeyZ"`. Modifiers: `Alt`, `Control`, `Shift`, `Super`. Key codes follow the Web KeyboardEvent `code` property convention (`KeyA`-`KeyZ`, `Digit0`-`Digit9`, `F1`-`F12`, `Space`, `Enter`, etc.).

### Whisper Model
`whisper-rs` (with `metal` feature for GPU acceleration) downloads Whisper models from HuggingFace on first use. 7 model variants available with smart system-based recommendation. The `WhisperContext` is cached in `AppState` and reused across transcriptions. Model download progress is reported to the frontend via Tauri events.

### Qwen3-ASR Model
`qwen3-asr` crate (v0.2.1, with `metal` feature for GPU acceleration) provides an alternative local STT engine. Two model variants: 1.7B (default) and 0.6B. Models downloaded from HuggingFace on first use. The `AsrInference` instance is cached in `AppState` via `Qwen3AsrCache`. Supports both batch transcription and streaming mode (with `initial_text` for cross-session context). Selected via `stt.local_engine = Qwen3Asr`.

### Silero VAD
Optional Silero VAD model (`ggml-silero-v6.2.0.bin`) downloaded separately. Filters out non-speech segments before transcription (both Whisper and Qwen3-ASR). Also used for silence detection in meeting mode feeders and as a gate in the Qwen3-ASR streaming feeder. Falls back to RMS-based silence trimming if not downloaded. Always-on when the model file is present (no user toggle).

### LLM Polish Model
`candle` (HuggingFace's pure Rust ML framework, with `metal` and `cuda` features for GPU acceleration) loads quantized GGUF models. Supported models: Llama 3 Taiwan 8B (Q4_K_M, ~4.9 GB), Qwen 2.5 7B (Q4_K_M, ~4.7 GB), and Qwen 3 8B (Q4_K_M, ~5.0 GB). Model download progress reported via `llm-model-download-progress` Tauri events. Multi-model management with per-model download/switch.

### Meeting Mode
Third independent hotkey (`settings.meeting_hotkey`, default None = disabled). Toggles continuous transcription with file-based transcript storage. Supports all STT engines (Qwen3-ASR, Whisper, Cloud). No polish, no auto-paste. All feeders use batch-per-segment architecture: accumulate audio in `chunk_buf`, detect silence via VAD, batch-transcribe segment, append delta to WAL file. Meeting notes stored in SQLite with WAL file for in-progress transcripts. Frontend: MeetingPage with Apple Notes style split layout (left list + right content). Overlay shows `meeting_stopped` state on stop.

## macOS-Specific Requirements
- `macOSPrivateApi: true` in `tauri.conf.json` is required for `ns_window()` access and transparent windows.
- The app targets macOS primarily; Windows support is implemented via the `platform/` module and `#[cfg(target_os)]` gates. Linux builds compile but overlay and paste features are no-ops.
