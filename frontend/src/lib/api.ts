import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import type {
  Settings,
  MicStatus,
  ModelStatus,
  LlmModelStatus,
  PermissionStatus,
  HistoryEntry,
  HistoryPage,
  HistoryStats,
  DownloadProgress,
  TestPolishResult,
  GeneratedRule,
  PromptRule,
  WhisperModelInfo,
  SystemInfo,
  WhisperModelId,
  PolishModelInfo,
  PolishModel,
  Qwen3AsrModelInfo,
  Qwen3AsrModelId,
  TranscriptionPartialPayload,
  MeetingNote,
  PolishedMeetingNote,
} from './types';

// ── Settings ──

export const getSettings = () => invoke<Settings>('get_settings');

export const saveSettings = (newSettings: Settings) =>
  invoke<void>('save_settings', { newSettings });

export const updateHotkey = (hotkey: string) =>
  invoke<void>('update_hotkey', { hotkey });

export const updateEditHotkey = (hotkey: string) =>
  invoke<void>('update_edit_hotkey', { hotkey });

export const updateMeetingHotkey = (hotkey: string | null) =>
  invoke<void>('update_meeting_hotkey', { hotkey });

export const resetSettings = () => invoke<void>('reset_settings');

export const getDefaultPrompt = () =>
  invoke<string>('get_default_prompt');

export const getDefaultPromptRules = (language?: string) =>
  invoke<PromptRule[]>('get_default_prompt_rules', { language: language ?? null });

// ── Recording ──

export const startRecording = () => invoke<void>('start_recording');

export const stopRecording = () => invoke<string>('stop_recording');

export const cancelRecording = () => invoke<void>('cancel_recording');

export const setTestMode = (enabled: boolean) =>
  invoke<void>('set_test_mode', { enabled });

export const setVoiceRuleMode = (enabled: boolean) =>
  invoke<void>('set_voice_rule_mode', { enabled });

export const setContextOverride = (appName: string, bundleId: string, url: string) =>
  invoke<void>('set_context_override', { appName, bundleId, url });

export const triggerUndo = () => invoke<void>('trigger_undo');

export const setEditTextOverride = (text: string) =>
  invoke<void>('set_edit_text_override', { text });

// ── History ──

export const getHistoryStats = () => invoke<HistoryStats>('get_history_stats');

export const getHistory = () => invoke<HistoryEntry[]>('get_history');

export const getHistoryPage = (beforeTimestamp?: number, limit?: number) =>
  invoke<HistoryPage>('get_history_page', { beforeTimestamp: beforeTimestamp ?? null, limit: limit ?? null });

export const deleteHistoryEntry = (id: string) =>
  invoke<void>('delete_history_entry', { id });

export const exportHistoryAudio = (id: string) =>
  invoke<string>('export_history_audio', { id });

export const clearAllHistory = () => invoke<void>('clear_all_history');

export const getHistoryStoragePath = () => invoke<string>('get_history_storage_path');

export const getAppIcon = (bundleId: string) =>
  invoke<string>('get_app_icon', { bundleId });

// ── API Keys ──

export const saveApiKey = (provider: string, key: string) =>
  invoke<void>('save_api_key', { provider, key });

export const getApiKey = (provider: string) =>
  invoke<string>('get_api_key', { provider });

// ── Polish ──

export const testPolish = (testText: string, customPrompt: string) =>
  invoke<TestPolishResult>('test_polish', { testText, customPrompt });

export const generateRuleFromDescription = (description: string) =>
  invoke<GeneratedRule>('generate_rule_from_description', { description });

// ── Models ──

export const checkModelStatus = () => invoke<ModelStatus>('check_model_status');

export const downloadModel = () => invoke<void>('download_model');

export const checkLlmModelStatus = () => invoke<LlmModelStatus>('check_llm_model_status');

export const downloadLlmModel = () => invoke<void>('download_llm_model');

// ── Mic & Permissions ──

export const getMicStatus = () => invoke<MicStatus>('get_mic_status');

export const setMicDevice = (deviceName: string | null) =>
  invoke<void>('set_mic_device', { deviceName });

export const checkPermissions = () => invoke<PermissionStatus>('check_permissions');

export const openPermissionSettings = (permissionType: string) =>
  invoke<void>('open_permission_settings', { permissionType });

// ── Events ──

export const onRecordingStatus = (cb: (status: string) => void): Promise<UnlistenFn> =>
  listen<string>('recording-status', (e) => cb(e.payload));

export const onRecordingMaxDuration = (cb: (secs: number) => void): Promise<UnlistenFn> =>
  listen<number>('recording-max-duration', (e) => cb(e.payload));

export const onAudioLevels = (cb: (levels: number[]) => void): Promise<UnlistenFn> =>
  listen<number[]>('audio-levels', (e) => cb(e.payload));

export const onTranscriptionPartial = (
  cb: (payload: TranscriptionPartialPayload) => void,
): Promise<UnlistenFn> =>
  listen<TranscriptionPartialPayload>('transcription-partial', (e) => cb(e.payload));

export const onTranscriptionResult = (cb: (text: string) => void): Promise<UnlistenFn> =>
  listen<string>('transcription-result', (e) => cb(e.payload));

export const onHotkeyActivated = (cb: (v: boolean) => void): Promise<UnlistenFn> =>
  listen<boolean>('hotkey-activated', (e) => cb(e.payload));

export const onModelDownloadProgress = (
  cb: (p: DownloadProgress) => void,
): Promise<UnlistenFn> =>
  listen<DownloadProgress>('model-download-progress', (e) => cb(e.payload));

export const onLlmModelDownloadProgress = (
  cb: (p: DownloadProgress) => void,
): Promise<UnlistenFn> =>
  listen<DownloadProgress>('llm-model-download-progress', (e) => cb(e.payload));

export const onVoiceRuleStatus = (cb: (status: string) => void): Promise<UnlistenFn> =>
  listen<string>('voice-rule-status', (e) => cb(e.payload));

export const onVoiceRuleLevels = (cb: (levels: number[]) => void): Promise<UnlistenFn> =>
  listen<number[]>('voice-rule-levels', (e) => cb(e.payload));

export const onVoiceRuleTranscript = (cb: (text: string) => void): Promise<UnlistenFn> =>
  listen<string>('voice-rule-transcript', (e) => cb(e.payload));

// ── Whisper Models ──

export const listWhisperModels = () =>
  invoke<WhisperModelInfo[]>('list_whisper_models');

export const getSystemInfo = () =>
  invoke<SystemInfo>('get_system_info');

export const getWhisperModelRecommendation = () =>
  invoke<WhisperModelId>('get_whisper_model_recommendation');

export const switchWhisperModel = (model: WhisperModelId) =>
  invoke<void>('switch_whisper_model', { model });

export const downloadWhisperModel = (model: WhisperModelId) =>
  invoke<void>('download_whisper_model', { model });

export const onWhisperModelDownloadProgress = (
  cb: (p: DownloadProgress) => void,
): Promise<UnlistenFn> =>
  listen<DownloadProgress>('whisper-model-download-progress', (e) => cb(e.payload));

// ── Polish Models ──

export const listPolishModels = () =>
  invoke<PolishModelInfo[]>('list_polish_models');

export const switchPolishModel = (model: PolishModel) =>
  invoke<void>('switch_polish_model', { model });

export const downloadPolishModel = (model: PolishModel) =>
  invoke<void>('download_polish_model', { model });

export const onPolishModelDownloadProgress = (
  cb: (p: DownloadProgress) => void,
): Promise<UnlistenFn> =>
  listen<DownloadProgress>('polish-model-download-progress', (e) => cb(e.payload));

// ── VAD Model ──

export const checkVadModelStatus = () =>
  invoke<{ downloaded: boolean }>('check_vad_model_status');

export const downloadVadModel = () =>
  invoke<void>('download_vad_model');

export const onVadModelDownloadProgress = (
  cb: (p: DownloadProgress) => void,
): Promise<UnlistenFn> =>
  listen<DownloadProgress>('vad-model-download-progress', (e) => cb(e.payload));

// ── Clipboard ──

export const copyImageToClipboard = (pngBytes: number[]) =>
  invoke<void>('copy_image_to_clipboard', { pngBytes });

// ── Dev Mode ──

export const isDevMode = () => invoke<boolean>('is_dev_mode');

// ── Diagnostics ──

export const exportDiagnosticLog = () => invoke<string>('export_diagnostic_log');

// ── Qwen3-ASR Model ──

export const listQwen3AsrModels = () =>
  invoke<Qwen3AsrModelInfo[]>('list_qwen3_asr_models');

export const switchQwen3AsrModel = (model: Qwen3AsrModelId) =>
  invoke<void>('switch_qwen3_asr_model', { model });

export const downloadQwen3AsrModel = (model: string) =>
  invoke<void>('download_qwen3_asr_model', { model });

export const onQwen3AsrDownloadProgress = (
  cb: (p: DownloadProgress & { current_file?: string; file_index?: number; file_count?: number }) => void,
): Promise<UnlistenFn> =>
  listen('qwen3-asr-download-progress', (e) => cb(e.payload as DownloadProgress & { current_file?: string; file_index?: number; file_count?: number }));

// ── Model deletion ──

export const deleteWhisperModel = (model: WhisperModelId) =>
  invoke<number>('delete_whisper_model', { model });

export const deletePolishModel = (model: PolishModel) =>
  invoke<number>('delete_polish_model', { model });

export const deleteQwen3AsrModel = (model: Qwen3AsrModelId) =>
  invoke<number>('delete_qwen3_asr_model', { model });

export const deleteVadModel = () =>
  invoke<number>('delete_vad_model');

export const onModelSwitching = (
  cb: (p: { status: 'start' | 'done' }) => void,
): Promise<UnlistenFn> =>
  listen('model-switching', (e) => cb(e.payload as { status: 'start' | 'done' }));

// ── Meeting Notes ──

export const listMeetingNotes = () =>
  invoke<MeetingNote[]>('list_meeting_notes');

export const getMeetingNote = (id: string) =>
  invoke<MeetingNote>('get_meeting_note', { id });

export const renameMeetingNote = (id: string, title: string) =>
  invoke<void>('rename_meeting_note', { id, title });

export const deleteMeetingNote = (id: string) =>
  invoke<void>('delete_meeting_note', { id });

export const deleteAllMeetingNotes = () =>
  invoke<void>('delete_all_meeting_notes');

export const getActiveMeetingNoteId = () =>
  invoke<string | null>('get_active_meeting_note_id');

export const onMeetingNoteCreated = (
  cb: (p: { id: string; note: MeetingNote }) => void,
): Promise<UnlistenFn> =>
  listen('meeting-note-created', (e) => cb(e.payload as { id: string; note: MeetingNote }));

export const onMeetingNoteUpdated = (
  cb: (p: { id: string; delta: string; duration_secs: number }) => void,
): Promise<UnlistenFn> =>
  listen('meeting-note-updated', (e) =>
    cb(e.payload as { id: string; delta: string; duration_secs: number }),
  );

export const polishMeetingNote = (id: string) =>
  invoke<PolishedMeetingNote>('polish_meeting_note', { id });

export const onMeetingNoteFinalized = (
  cb: (p: { id: string }) => void,
): Promise<UnlistenFn> =>
  listen('meeting-note-finalized', (e) => cb(e.payload as { id: string }));