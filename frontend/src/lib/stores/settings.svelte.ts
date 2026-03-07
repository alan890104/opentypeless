import type {
  Settings,
  PolishConfig,
  SttConfig,
  CloudConfig,
  SttCloudConfig,
  PromptRule,
  DictionaryConfig,
  PolishModel,
  PolishMode,
  SttMode,
  CloudProvider,
  SttProvider,
  WhisperModelId,
  LocalSttEngine,
  Qwen3AsrModelId,
} from '../types';
import * as api from '../api';
import { DEFAULT_HOTKEY } from '../constants';

// ── Reactive settings state ──

let settings = $state<Settings>({
  hotkey: DEFAULT_HOTKEY,
  auto_paste: true,
  polish: {
    enabled: false,
    model: 'phi4_mini',
    custom_prompt: null,
    mode: 'local',
    cloud: { provider: 'groq', api_key: '', endpoint: '', model_id: 'qwen/qwen3-32b' },
    prompt_rules: {},
    dictionary: { enabled: true, entries: [] },
    reasoning: false,
  },
  history_retention_days: 0,
  language: null,
  stt: {
    mode: 'local',
    cloud: { provider: 'deepgram', api_key: '', endpoint: '', model_id: 'whisper', language: 'auto' },
    whisper_model: 'large_v3_turbo',
    local_engine: 'whisper',
    qwen3_asr_model: 'qwen3_asr1_7_b',
    language: 'auto',
  },
  edit_hotkey: null,
  onboarding_completed: false,
  mic_device: null,
  meeting_hotkey: null,
  idle_mic_timeout_secs: 0,
  record_meeting_audio: false,
});

export function getSettings(): Settings {
  return settings;
}

export function getPolishConfig(): PolishConfig {
  return settings.polish;
}

export function getSttConfig(): SttConfig {
  return settings.stt;
}

export function getHotkey(): string {
  return settings.hotkey;
}

export function getEditHotkey(): string | null {
  return settings.edit_hotkey;
}

export function getOnboardingCompleted(): boolean {
  return settings.onboarding_completed;
}

// ── Load settings from backend ──

export async function load(): Promise<void> {
  const s = await api.getSettings();
  settings = s;

  // Migrate: if legacy localStorage flag exists, carry it over to backend settings
  const legacyOnboarding = localStorage.getItem('sumi-onboarding-completed');
  if (legacyOnboarding === 'true' && !settings.onboarding_completed) {
    settings.onboarding_completed = true;
    try {
      await api.saveSettings(buildPayload());
    } catch {
      // best-effort migration
    }
  }
  localStorage.removeItem('sumi-onboarding-completed');

  // Load API keys from keychain
  try {
    settings.polish.cloud.api_key = await api.getApiKey(settings.polish.cloud.provider);
  } catch {
    settings.polish.cloud.api_key = '';
  }
  try {
    settings.stt.cloud.api_key = await api.getApiKey(`stt_${settings.stt.cloud.provider}`);
  } catch {
    settings.stt.cloud.api_key = '';
  }
}

// ── Save helpers ──

export function buildPayload(): Settings {
  const s = { ...settings };
  // Strip api_keys (stored in keychain, not in settings.json)
  s.polish = { ...s.polish, cloud: { ...s.polish.cloud, api_key: '' } };
  s.stt = { ...s.stt, cloud: { ...s.stt.cloud, api_key: '' } };
  return s;
}

export async function save(): Promise<void> {
  await api.saveSettings(buildPayload());
}

export async function savePolish(): Promise<void> {
  await save();
}

export async function saveStt(): Promise<void> {
  await save();
}

// ── Setters ──

export function setHotkey(hotkey: string) {
  settings.hotkey = hotkey;
}

export function setEditHotkey(hotkey: string | null) {
  settings.edit_hotkey = hotkey;
}

export function getMeetingHotkey(): string | null {
  return settings.meeting_hotkey;
}

export function setMeetingHotkey(hotkey: string | null) {
  settings.meeting_hotkey = hotkey;
}

export function setLanguage(lang: string | null) {
  settings.language = lang;
}

export function setPolishEnabled(enabled: boolean) {
  settings.polish.enabled = enabled;
}

export function setPolishMode(mode: PolishMode) {
  settings.polish.mode = mode;
}

export function setPolishModel(model: PolishModel) {
  settings.polish.model = model;
}

export function setPolishReasoning(reasoning: boolean) {
  settings.polish.reasoning = reasoning;
}

export function setPolishCloudProvider(provider: CloudProvider) {
  settings.polish.cloud.provider = provider;
}

export function setPolishCloudApiKey(key: string) {
  settings.polish.cloud.api_key = key;
}

export function setPolishCloudEndpoint(endpoint: string) {
  settings.polish.cloud.endpoint = endpoint;
}

export function setPolishCloudModelId(modelId: string) {
  settings.polish.cloud.model_id = modelId;
}

export function setSttMode(mode: SttMode) {
  settings.stt.mode = mode;
}

export function setSttCloudProvider(provider: SttProvider) {
  settings.stt.cloud.provider = provider;
}

export function setSttCloudApiKey(key: string) {
  settings.stt.cloud.api_key = key;
}

export function setSttCloudEndpoint(endpoint: string) {
  settings.stt.cloud.endpoint = endpoint;
}

export function setSttCloudModelId(modelId: string) {
  settings.stt.cloud.model_id = modelId;
}

export function setSttLanguage(lang: string) {
  settings.stt.language = lang;
  settings.stt.cloud.language = lang;
}

export function setSttCloudLanguage(lang: string) {
  settings.stt.cloud.language = lang;
  settings.stt.language = lang;
}

export function setSttWhisperModel(model: WhisperModelId) {
  settings.stt.whisper_model = model;
}

export function setSttLocalEngine(engine: LocalSttEngine) {
  settings.stt.local_engine = engine;
}

export function setSttQwen3AsrModel(model: Qwen3AsrModelId) {
  settings.stt.qwen3_asr_model = model;
}


export function setHistoryRetention(days: number) {
  settings.history_retention_days = days;
}

export function setAutoPaste(v: boolean) {
  settings.auto_paste = v;
}

export function setIdleMicTimeout(secs: number) {
  settings.idle_mic_timeout_secs = secs;
}

export function setRecordMeetingAudio(v: boolean) {
  settings.record_meeting_audio = v;
}

// ── Prompt rules ──

export function getCurrentRules(): PromptRule[] {
  // Flatten all language-keyed rules, deduplicating by match_type+match_value
  // (non-"auto" keys take priority since those are user-customized)
  const seen = new Map<string, PromptRule>();
  // Process "auto" first so it can be overwritten by customized rules
  for (const rule of settings.polish.prompt_rules['auto'] ?? []) {
    seen.set(`${rule.match_type}:${rule.match_value}`, rule);
  }
  for (const [key, rules] of Object.entries(settings.polish.prompt_rules)) {
    if (key === 'auto') continue;
    for (const rule of rules) {
      seen.set(`${rule.match_type}:${rule.match_value}`, rule);
    }
  }
  return Array.from(seen.values());
}

export function setCurrentRules(rules: PromptRule[]) {
  // Store all rules under "auto" key, clear others
  settings.polish.prompt_rules = { auto: rules };
}

// ── Dictionary ──

export function getDictionary(): DictionaryConfig {
  return settings.polish.dictionary;
}

export function setDictionaryEnabled(enabled: boolean) {
  settings.polish.dictionary.enabled = enabled;
}

export function setCustomPrompt(prompt: string | null) {
  settings.polish.custom_prompt = prompt;
}

export function markOnboardingComplete() {
  settings.onboarding_completed = true;
}

export function resetOnboarding() {
  settings.onboarding_completed = false;
}
