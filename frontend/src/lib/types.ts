// ── STT ──

export type SttMode = 'local' | 'cloud';

export type SttProvider = 'deepgram' | 'groq' | 'open_ai' | 'azure' | 'custom';

export interface SttCloudConfig {
  provider: SttProvider;
  api_key: string;
  endpoint: string;
  model_id: string;
  language: string;
}

export type WhisperModelId =
  | 'large_v3_turbo'
  | 'large_v3_turbo_q5'
  | 'medium'
  | 'small'
  | 'base'
  | 'large_v3_turbo_zh_tw';

export type LocalSttEngine = 'whisper' | 'qwen3_asr';

export type Qwen3AsrModelId = 'qwen3_asr1_7_b_full' | 'qwen3_asr1_7_b_q8' | 'qwen3_asr1_7_b' | 'qwen3_asr0_6_b';

export interface Qwen3AsrModelInfo {
  id: Qwen3AsrModelId;
  display_name: string;
  description: string;
  size_bytes: number;
  downloaded: boolean;
  file_size_on_disk: number;
  is_active: boolean;
}

export interface WhisperModelInfo {
  id: WhisperModelId;
  display_name: string;
  description: string;
  size_bytes: number;
  languages: string[];
  downloaded: boolean;
  file_size_on_disk: number;
  is_active: boolean;
}

export interface SystemInfo {
  total_ram_bytes: number;
  available_disk_bytes: number;
  is_apple_silicon: boolean;
  gpu_vram_bytes: number;
  has_cuda: boolean;
  os: string;
  arch: string;
}

export interface SttConfig {
  mode: SttMode;
  cloud: SttCloudConfig;
  whisper_model: WhisperModelId;
  local_engine: LocalSttEngine;
  qwen3_asr_model: Qwen3AsrModelId;
  language: string;
  vad_enabled: boolean;
}

// ── Polish ──

export type PolishMode = 'local' | 'cloud';

export type CloudProvider =
  | 'github_models'
  | 'groq'
  | 'open_router'
  | 'open_ai'
  | 'gemini'
  | 'samba_nova'
  | 'custom';

export type PolishModel = 'llama_taiwan' | 'qwen25' | 'qwen3' | 'qwen3_4b';

export interface PolishModelInfo {
  id: PolishModel;
  display_name: string;
  description: string;
  size_bytes: number;
  downloaded: boolean;
  file_size_on_disk: number;
  is_active: boolean;
}

export type MatchType = 'app_name' | 'bundle_id' | 'url';

export interface MatchCondition {
  match_type: MatchType;
  match_value: string;
}

export interface PromptRule {
  name: string;
  match_type: MatchType;
  match_value: string;
  prompt: string;
  enabled: boolean;
  icon?: string;
  alt_matches?: MatchCondition[];
}

export interface DictionaryEntry {
  term: string;
  enabled: boolean;
}

export interface DictionaryConfig {
  enabled: boolean;
  entries: DictionaryEntry[];
}

export interface CloudConfig {
  provider: CloudProvider;
  api_key: string;
  endpoint: string;
  model_id: string;
}

export interface PolishConfig {
  enabled: boolean;
  model: PolishModel;
  custom_prompt: string | null;
  mode: PolishMode;
  cloud: CloudConfig;
  prompt_rules: Record<string, PromptRule[]>;
  dictionary: DictionaryConfig;
  reasoning: boolean;
}

// ── Settings ──

export interface Settings {
  hotkey: string;
  auto_paste: boolean;
  polish: PolishConfig;
  history_retention_days: number;
  language: string | null;
  stt: SttConfig;
  edit_hotkey: string | null;
  onboarding_completed: boolean;
  mic_device: string | null;
}

// ── History ──

export interface HistoryEntry {
  id: string;
  timestamp: number;
  text: string;
  raw_text: string;
  reasoning: string | null;
  stt_model: string;
  polish_model: string;
  duration_secs: number;
  has_audio: boolean;
  stt_elapsed_ms: number;
  polish_elapsed_ms: number | null;
  total_elapsed_ms: number;
  app_name: string;
  bundle_id: string;
  chars_per_sec: number;
  word_count: number;
}

export interface HistoryPage {
  entries: HistoryEntry[];
  has_more: boolean;
}

export interface HistoryStats {
  total_entries: number;
  total_duration_secs: number;
  total_chars: number;
  local_entries: number;
  local_duration_secs: number;
  total_words: number;
}

// ── API responses ──

export interface MicStatus {
  connected: boolean;
  default_device: string | null;
  devices: string[];
}

export interface ModelStatus {
  engine: string;
  model_exists: boolean;
}

export interface LlmModelStatus {
  model: string;
  model_exists: boolean;
  model_size_bytes: number;
}

export interface PermissionStatus {
  microphone: string;
  accessibility: boolean;
}

export interface TestPolishResult {
  current_result: string;
  edited_result: string;
}

export interface GeneratedRule {
  name: string;
  match_type: string;
  match_value: string;
  prompt: string;
}

export interface DownloadProgress {
  status: 'downloading' | 'complete' | 'error';
  downloaded?: number;
  total?: number;
  message?: string;
}

// ── Pages ──

export type Page =
  | 'stats'
  | 'settings'
  | 'promptRules'
  | 'dictionary'
  | 'history'
  | 'about'
  | 'test';

// ── Overlay ──

export type OverlayStatus =
  | 'preparing'
  | 'recording'
  | 'processing'
  | 'transcribing'
  | 'polishing'
  | 'pasted'
  | 'copied'
  | 'error'
  | 'edited'
  | 'edit_requires_polish';
