<script lang="ts">
  import { onDestroy } from 'svelte';
  import { t } from '$lib/stores/i18n.svelte';
  import { getShowSetup, setShowSetup } from '$lib/stores/ui.svelte';
  import { setCurrentPage } from '$lib/stores/ui.svelte';
  import {
    getSttConfig,
    getPolishConfig,
    setSttMode,
    setSttWhisperModel,
    setSttLocalEngine,
    setSttQwen3AsrModel,
    setSttCloudProvider,
    setSttCloudApiKey,
    setSttCloudEndpoint,
    setSttCloudModelId,
    setSttCloudLanguage,
    setPolishMode,
    setPolishEnabled,
    setPolishCloudProvider,
    setPolishCloudApiKey,
    setPolishCloudEndpoint,
    setPolishCloudModelId,
    setPolishModel,
    markOnboardingComplete,
    buildPayload,
  } from '$lib/stores/settings.svelte';
  import {
    checkPermissions,
    openPermissionSettings,
    checkModelStatus,
    checkLlmModelStatus,
    downloadModel,
    downloadLlmModel,
    onModelDownloadProgress,
    onLlmModelDownloadProgress,
    listWhisperModels,
    getWhisperModelRecommendation,
    switchWhisperModel,
    downloadWhisperModel,
    onWhisperModelDownloadProgress,
    checkVadModelStatus,
    downloadVadModel,
    onVadModelDownloadProgress,
    listPolishModels,
    switchPolishModel,
    downloadPolishModel,
    onPolishModelDownloadProgress,
    listQwen3AsrModels,
    switchQwen3AsrModel,
    downloadQwen3AsrModel,
    onQwen3AsrDownloadProgress,
    getSystemInfo,
    saveApiKey,
    getApiKey,
    saveSettings as saveSettingsApi,
  } from '$lib/api';
  import type { DownloadProgress, PermissionStatus, WhisperModelId, WhisperModelInfo, PolishModelInfo, PolishModel, LocalSttEngine, Qwen3AsrModelId, Qwen3AsrModelInfo } from '$lib/types';
  import SegmentedControl from '$lib/components/SegmentedControl.svelte';
  import CloudConfigPanel from '$lib/components/CloudConfigPanel.svelte';
  import ProgressBar from '$lib/components/ProgressBar.svelte';
  import { isMac } from '$lib/constants';
  import { camelCase } from '$lib/utils';

  // ── State machine ──

  type SetupState =
    | 'permissions'
    | 'sttChoice'
    | 'downloading'
    | 'complete'
    | 'polishChoice'
    | 'llmDownloading'
    | 'error';

  let currentState = $state<SetupState>('permissions');
  let fadeOut = $state(false);

  // ── Permissions ──

  let micGranted = $state(false);
  let accGranted = $state(false);
  let permBothGranted = $derived(micGranted && accGranted);
  let permPollTimer: ReturnType<typeof setInterval> | null = null;

  async function pollPermissions() {
    try {
      const perms: PermissionStatus = await checkPermissions();
      micGranted = perms.microphone === 'granted';
      accGranted = perms.accessibility === true;
    } catch {
      // Fallback: assume granted so user can proceed
      micGranted = true;
      accGranted = true;
    }
  }

  function startPermissionPolling() {
    stopPermissionPolling();
    pollPermissions();
    permPollTimer = setInterval(pollPermissions, 1500);
  }

  function stopPermissionPolling() {
    if (permPollTimer) {
      clearInterval(permPollTimer);
      permPollTimer = null;
    }
  }

  async function grantMicrophone() {
    await openPermissionSettings('microphone');
  }

  async function grantAccessibility() {
    await openPermissionSettings('accessibility');
  }

  async function onPermissionsContinue() {
    stopPermissionPolling();
    currentState = 'sttChoice';
    await fetchSttModels();
  }

  // ── STT Choice ──

  let sttMode = $state<string>('local');
  let sttModels = $state<WhisperModelInfo[]>([]);
  let selectedSttModel = $state<WhisperModelId>('large_v3_turbo');
  let recommendedSttModelId = $state<WhisperModelId>('large_v3_turbo');

  let selectedSttModelDownloaded = $derived(
    sttModels.find(m => m.id === selectedSttModel)?.downloaded ?? false
  );

  // Engine selection
  let selectedLocalEngine = $state<LocalSttEngine>('whisper');
  // unified recommended key: 'whisper:<model_id>' or 'qwen3_asr:<model_id>'
  let recommendedKey = $state<string>('');

  // Qwen3-ASR models
  let qwen3AsrModels = $state<Qwen3AsrModelInfo[]>([]);
  let selectedQwen3Model = $state<Qwen3AsrModelId>('qwen3_asr0_6_b');
  let recommendedQwen3ModelId = $state<Qwen3AsrModelId>('qwen3_asr0_6_b');
  let selectedQwen3ModelDownloaded = $derived(
    qwen3AsrModels.find(m => m.id === selectedQwen3Model)?.downloaded ?? false
  );

  let isCurrentModelDownloaded = $derived(
    selectedLocalEngine === 'whisper' ? selectedSttModelDownloaded : selectedQwen3ModelDownloaded
  );

  function recommendLocalEngine(): LocalSttEngine {
    const lang = navigator.language.toLowerCase();
    // zh-TW / zh-Hant: Whisper Turbo TW is purpose-built → stay on Whisper
    if (lang.startsWith('zh-tw') || lang.startsWith('zh-hant')) return 'whisper';
    // Simplified Chinese, Japanese, Korean, Cantonese: Qwen3-ASR wins clearly
    if (lang.startsWith('zh') || lang.startsWith('ja') || lang.startsWith('ko') || lang.startsWith('yue')) {
      return 'qwen3_asr';
    }
    return 'whisper';
  }

  async function recommendQwen3Model(): Promise<Qwen3AsrModelId> {
    try {
      const sys = await getSystemInfo();
      const gb16 = 16 * 1024 * 1024 * 1024;
      const gb6 = 6 * 1024 * 1024 * 1024;
      return (sys.total_ram_bytes >= gb16 && sys.available_disk_bytes >= gb6)
        ? 'qwen3_asr1_7_b'
        : 'qwen3_asr0_6_b';
    } catch {
      return 'qwen3_asr0_6_b';
    }
  }

  async function fetchSttModels() {
    try {
      const [models, rec, q3models, q3rec, engineRec] = await Promise.all([
        listWhisperModels(),
        getWhisperModelRecommendation() as Promise<WhisperModelId>,
        listQwen3AsrModels(),
        recommendQwen3Model(),
        Promise.resolve(recommendLocalEngine()),
      ]);
      sttModels = models;
      recommendedSttModelId = rec;
      selectedSttModel = rec;
      qwen3AsrModels = q3models;
      recommendedQwen3ModelId = q3rec;
      selectedQwen3Model = q3rec;
      selectedLocalEngine = engineRec;
      recommendedKey = engineRec === 'whisper' ? `whisper:${rec}` : `qwen3_asr:${q3rec}`;
    } catch {
      // Fallback to defaults
    }
  }

  // Cloud config bindings for STT
  let sttProvider = $state('deepgram');
  let sttApiKey = $state('');
  let sttEndpoint = $state('');
  let sttModelId = $state('whisper');
  let sttLanguage = $state('');

  const sttModeOptions = [
    { value: 'local', label: 'Local' },
    { value: 'cloud', label: 'Cloud API' },
  ];

  async function onSttModeChange(value: string) {
    sttMode = value;
    if (value === 'cloud') {
      // Pre-populate from existing settings
      const cfg = getSttConfig();
      sttProvider = cfg.cloud.provider;
      sttEndpoint = cfg.cloud.endpoint;
      sttModelId = cfg.cloud.model_id || 'whisper';
      sttLanguage = cfg.cloud.language || detectSttLanguage();
      // Load existing API key from keychain
      await loadSttApiKey();
    }
  }

  async function loadSttApiKey() {
    try {
      const key = await getApiKey('stt_' + sttProvider);
      if (key) sttApiKey = key;
    } catch {
      // No saved key
    }
  }

  function detectSttLanguage(): string {
    const lang = (navigator.language || '').toLowerCase();
    if (lang.startsWith('zh')) {
      if (lang.includes('tw') || lang.includes('hant')) return 'zh-TW';
      return 'zh-CN';
    }
    if (lang.startsWith('ja')) return 'ja';
    if (lang.startsWith('ko')) return 'ko';
    if (lang.startsWith('en')) return 'en';
    return '';
  }

  let prevSttProvider = sttProvider;
  async function onSttCloudChange() {
    // Reload API key from keychain only when provider changes
    if (sttProvider !== prevSttProvider) {
      prevSttProvider = sttProvider;
      await loadSttApiKey();
    }
  }

  let sttCloudValid = $derived.by(() => {
    if (!sttApiKey.trim()) return false;
    if (sttProvider === 'azure' && !sttEndpoint.trim()) return false;
    if (sttProvider === 'custom' && !sttEndpoint.trim()) return false;
    return true;
  });

  async function onSttCloudContinue() {
    // Update store
    setSttMode('cloud');
    setSttCloudProvider(sttProvider as any);
    setSttCloudApiKey(sttApiKey);
    setSttCloudEndpoint(sttEndpoint);
    setSttCloudModelId(sttModelId);
    setSttCloudLanguage(sttLanguage);

    // Save API key to keychain
    if (sttApiKey.trim()) {
      try {
        await saveApiKey('stt_' + sttProvider, sttApiKey.trim());
      } catch (e) {
        console.error('Failed to save STT API key:', e);
      }
    }

    // Save settings
    try {
      await saveSettingsApi(buildPayload());
    } catch (e) {
      console.error('Failed to save STT settings:', e);
    }

    goToPolishChoice();
  }

  async function onSttLocalDownload() {
    setSttMode('local');

    if (selectedLocalEngine === 'whisper') {
      setSttLocalEngine('whisper');
      setSttWhisperModel(selectedSttModel);
      try {
        await switchWhisperModel(selectedSttModel);
      } catch (e) {
        console.error('Failed to switch whisper model:', e);
      }

      if (selectedSttModelDownloaded) {
        try {
          const vadStatus = await checkVadModelStatus();
          if (vadStatus.downloaded) {
            try { await saveSettingsApi(buildPayload()); } catch {}
            goToPolishChoice();
            return;
          }
        } catch {}
      }
    } else {
      // Qwen3-ASR
      setSttLocalEngine('qwen3_asr');
      setSttQwen3AsrModel(selectedQwen3Model);
      try {
        await switchQwen3AsrModel(selectedQwen3Model);
      } catch (e) {
        console.error('Failed to switch Qwen3-ASR model:', e);
      }

      if (selectedQwen3ModelDownloaded) {
        try {
          const vadStatus = await checkVadModelStatus();
          if (vadStatus.downloaded) {
            try { await saveSettingsApi(buildPayload()); } catch {}
            goToPolishChoice();
            return;
          }
        } catch {}
      }
    }
    // Persist local_engine (and model) before starting download
    try { await saveSettingsApi(buildPayload()); } catch {}
    startSttDownload();
  }

  // ── STT Download ──

  let downloadPercent = $state(0);
  let downloadedBytes = $state(0);
  let downloadTotalBytes = $state(0);
  let sttDownloadCurrentFile = $state('');
  let sttDownloadUnlisten: (() => void) | null = null;
  let vadDownloadUnlisten: (() => void) | null = null;
  let qwen3DownloadUnlisten: (() => void) | null = null;
  let whisperDone = $state(false);
  let vadDone = $state(false);

  // Transition to complete only when both Whisper and VAD are ready
  $effect(() => {
    if (whisperDone && vadDone && currentState === 'downloading') {
      downloadPercent = 100;
      currentState = 'complete';
      setTimeout(() => goToPolishChoice(), 1500);
    }
  });

  async function startSttDownload() {
    currentState = 'downloading';
    downloadPercent = 0;
    downloadedBytes = 0;
    downloadTotalBytes = 0;
    sttDownloadCurrentFile = '';
    whisperDone = false;
    vadDone = false;

    // Clean up previous listeners
    if (sttDownloadUnlisten) { sttDownloadUnlisten(); sttDownloadUnlisten = null; }
    if (vadDownloadUnlisten) { vadDownloadUnlisten(); vadDownloadUnlisten = null; }
    if (qwen3DownloadUnlisten) { qwen3DownloadUnlisten(); qwen3DownloadUnlisten = null; }

    // Start VAD model download in parallel (~1.6 MB)
    vadDownloadUnlisten = await onVadModelDownloadProgress((p: DownloadProgress) => {
      if (p.status === 'complete') {
        vadDone = true;
      } else if (p.status === 'error') {
        console.warn('VAD model download failed:', p.message);
        vadDone = true; // treat as done — VAD has RMS fallback
      }
    });
    downloadVadModel().catch(() => { vadDone = true; });

    if (selectedLocalEngine === 'whisper') {
      // Start Whisper model download with progress tracking
      sttDownloadUnlisten = await onWhisperModelDownloadProgress((p: DownloadProgress) => {
        if (p.status === 'downloading') {
          const total = p.total || 1;
          const downloaded = p.downloaded || 0;
          downloadPercent = Math.min((downloaded / total) * 100, 99);
          downloadedBytes = downloaded;
          downloadTotalBytes = total;
        } else if (p.status === 'complete') {
          whisperDone = true;
        } else if (p.status === 'error') {
          errorMessage = p.message || t('setup.errorDefault');
          lastFailedDownload = 'stt';
          currentState = 'error';
        }
      });

      try {
        await downloadWhisperModel(selectedSttModel);
      } catch (e) {
        errorMessage = String(e);
        lastFailedDownload = 'stt';
        currentState = 'error';
      }
    } else {
      // Start Qwen3-ASR model download with progress tracking
      qwen3DownloadUnlisten = await onQwen3AsrDownloadProgress((p) => {
        if (p.status === 'complete') {
          whisperDone = true; // reuse flag — signals STT model is done
        } else if (p.status === 'error') {
          errorMessage = p.message || t('setup.errorDefault');
          lastFailedDownload = 'stt';
          currentState = 'error';
        } else {
          const total = p.total || 1;
          const downloaded = p.downloaded || 0;
          downloadPercent = Math.min((downloaded / total) * 100, 99);
          downloadedBytes = downloaded;
          downloadTotalBytes = total;
          sttDownloadCurrentFile = p.current_file ?? '';
        }
      });

      try {
        await downloadQwen3AsrModel(selectedQwen3Model);
      } catch (e) {
        errorMessage = String(e);
        lastFailedDownload = 'stt';
        currentState = 'error';
      }
    }
  }

  // ── Polish Choice ──

  let polishMode = $state<string>('local');
  let polishModels = $state<PolishModelInfo[]>([]);
  let selectedPolishModel = $state<PolishModel>('phi4_mini');

  let selectedModelDownloaded = $derived(
    polishModels.find(m => m.id === selectedPolishModel)?.downloaded ?? false
  );

  // Cloud config bindings for Polish
  let polishProvider = $state('groq');
  let polishApiKey = $state('');
  let polishEndpoint = $state('');
  let polishModelId = $state('qwen/qwen3-32b');

  const polishModeOptions = [
    { value: 'local', label: 'Local' },
    { value: 'cloud', label: 'Cloud API' },
  ];

  async function fetchPolishModels() {
    try {
      polishModels = await listPolishModels();
      // Pre-select first available model
      if (!polishModels.find(m => m.id === selectedPolishModel)) {
        selectedPolishModel = polishModels.find(m => m.recommended)?.id ?? polishModels[0]?.id ?? 'phi4_mini';
      }
    } catch {
      polishModels = [];
    }
  }

  async function onPolishModeChange(value: string) {
    polishMode = value;
    if (value === 'cloud') {
      // Pre-populate from existing settings
      const cfg = getPolishConfig();
      polishProvider = cfg.cloud.provider;
      polishEndpoint = cfg.cloud.endpoint;
      polishModelId = cfg.cloud.model_id || 'qwen/qwen3-32b';
      // Load existing API key from keychain
      await loadPolishApiKey();
    }
  }

  async function loadPolishApiKey() {
    try {
      const key = await getApiKey(polishProvider);
      if (key) polishApiKey = key;
    } catch {
      // No saved key
    }
  }

  let prevPolishProvider = polishProvider;
  async function onPolishCloudChange() {
    // Reload API key from keychain only when provider changes
    if (polishProvider !== prevPolishProvider) {
      prevPolishProvider = polishProvider;
      await loadPolishApiKey();
    }
  }

  let polishCloudValid = $derived.by(() => {
    if (!polishApiKey.trim()) return false;
    if (polishProvider === 'custom' && !polishEndpoint.trim()) return false;
    return true;
  });

  async function onPolishCloudContinue() {
    // Update store
    setPolishMode('cloud');
    setPolishEnabled(true);
    setPolishCloudProvider(polishProvider as any);
    setPolishCloudApiKey(polishApiKey);
    setPolishCloudEndpoint(polishEndpoint);
    setPolishCloudModelId(polishModelId || 'qwen/qwen3-32b');

    // Save API key to keychain
    if (polishApiKey.trim()) {
      try {
        await saveApiKey(polishProvider, polishApiKey.trim());
      } catch (e) {
        console.error('Failed to save polish API key:', e);
      }
    }

    // Save settings
    try {
      await saveSettingsApi(buildPayload());
    } catch (e) {
      console.error('Failed to save polish settings:', e);
    }

    finishSetup();
  }

  async function onPolishLocalDownload() {
    // Switch to selected model in backend + settings store
    setPolishModel(selectedPolishModel);
    try {
      await switchPolishModel(selectedPolishModel);
    } catch (e) {
      console.error('Failed to switch polish model:', e);
    }

    if (selectedModelDownloaded) {
      setPolishMode('local');
      setPolishEnabled(true);
      finishSetup();
    } else {
      startLlmDownload();
    }
  }

  function onPolishSkip() {
    finishSetup();
  }

  // ── LLM Download ──

  let llmDownloadPercent = $state(0);
  let llmDownloadedBytes = $state(0);
  let llmDownloadTotalBytes = $state(0);
  let llmDownloadUnlisten: (() => void) | null = null;

  async function startLlmDownload() {
    currentState = 'llmDownloading';
    llmDownloadPercent = 0;
    llmDownloadedBytes = 0;
    llmDownloadTotalBytes = 0;

    // Clean up previous listener
    if (llmDownloadUnlisten) {
      llmDownloadUnlisten();
      llmDownloadUnlisten = null;
    }

    llmDownloadUnlisten = await onPolishModelDownloadProgress((p: DownloadProgress) => {
      if (p.status === 'downloading') {
        const total = p.total || 1;
        const downloaded = p.downloaded || 0;
        llmDownloadPercent = Math.min((downloaded / total) * 100, 100);
        llmDownloadedBytes = downloaded;
        llmDownloadTotalBytes = total;
      } else if (p.status === 'complete') {
        llmDownloadPercent = 100;
        // Set polish to local + enabled
        setPolishMode('local');
        setPolishEnabled(true);
        finishSetup();
      } else if (p.status === 'error') {
        console.error('LLM setup download error:', p.message);
        // On error, advance -- user can download from settings later
        finishSetup();
      }
    });

    try {
      await downloadPolishModel(selectedPolishModel);
    } catch (e) {
      console.error('Failed to start LLM setup download:', e);
      finishSetup();
    }
  }

  // ── Error state ──

  let errorMessage = $state('');
  let lastFailedDownload = $state<'stt' | 'llm'>('stt');

  function onRetryDownload() {
    if (lastFailedDownload === 'stt') {
      startSttDownload();
    } else {
      startLlmDownload();
    }
  }

  // ── Navigation helpers ──

  async function goToPolishChoice() {
    currentState = 'polishChoice';
    await fetchPolishModels();
  }

  async function finishSetup() {
    markOnboardingComplete();

    try {
      await saveSettingsApi(buildPayload());
    } catch (e) {
      console.error('Failed to save onboarding completed:', e);
    }

    // Fade out
    fadeOut = true;
    setTimeout(() => {
      setShowSetup(false);
      fadeOut = false;
      setCurrentPage('test');
    }, 300);
  }

  // ── Formatting ──

  function formatBytes(bytes: number): string {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  }

  // ── Lifecycle ──

  // Use $effect to react to getShowSetup() changes, since App.svelte
  // may call setShowSetup(true) after this component's onMount has already fired.
  $effect(() => {
    if (getShowSetup()) {
      currentState = 'permissions';
      startPermissionPolling();
    } else {
      stopPermissionPolling();
    }
  });

  onDestroy(() => {
    stopPermissionPolling();
    if (sttDownloadUnlisten) { sttDownloadUnlisten(); sttDownloadUnlisten = null; }
    if (vadDownloadUnlisten) { vadDownloadUnlisten(); vadDownloadUnlisten = null; }
    if (qwen3DownloadUnlisten) { qwen3DownloadUnlisten(); qwen3DownloadUnlisten = null; }
    if (llmDownloadUnlisten) { llmDownloadUnlisten(); llmDownloadUnlisten = null; }
  });
</script>

{#if getShowSetup()}
  <div class="setup-overlay" class:fade-out={fadeOut}>
    <div class="setup-backdrop"></div>
    <div class="setup-card">

      <!-- ═══ Permissions ═══ -->
      {#if currentState === 'permissions'}
        <div class="setup-state-content" style="animation: setupFadeIn 0.4s ease">
          <div class="setup-icon-shield">
            <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M32 6L8 18V30C8 45.464 18.152 59.68 32 63C45.848 59.68 56 45.464 56 30V18L32 6Z" fill="#007AFF" opacity="0.12"/>
              <path d="M32 8L10 19V30C10 44.36 19.52 57.52 32 60.8C44.48 57.52 54 44.36 54 30V19L32 8Z" stroke="#007AFF" stroke-width="2" fill="none"/>
              <path d="M24 33L30 39L42 27" stroke="#007AFF" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
          <div class="setup-title">{t('setup.permissionsTitle')}</div>
          <div class="setup-desc">{t('setup.permissionsDesc')}</div>

          <div class="setup-permissions-list">
            <!-- Microphone -->
            <div class="setup-permission-row">
              <div class="setup-permission-icon">
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect x="6" y="2" width="6" height="9" rx="3" fill="#007AFF"/>
                  <path d="M4 8.5C4 11.26 6.24 13.5 9 13.5C11.76 13.5 14 11.26 14 8.5" stroke="#007AFF" stroke-width="1.5" stroke-linecap="round"/>
                  <line x1="9" y1="13.5" x2="9" y2="16" stroke="#007AFF" stroke-width="1.5" stroke-linecap="round"/>
                  <line x1="6.5" y1="16" x2="11.5" y2="16" stroke="#007AFF" stroke-width="1.5" stroke-linecap="round"/>
                </svg>
              </div>
              <div class="setup-permission-info">
                <div class="setup-permission-name">{t('setup.permMicName')}</div>
                <div class="setup-permission-desc">{t('setup.permMicDesc')}</div>
              </div>
              <div class="setup-permission-action">
                {#if micGranted}
                  <div class="setup-permission-granted">
                    <svg viewBox="0 0 14 14" fill="none">
                      <path d="M2.5 7.5L5.5 10.5L11.5 4.5" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                  </div>
                {:else}
                  <button class="setup-permission-btn" onclick={grantMicrophone}>{t('setup.permGrant')}</button>
                {/if}
              </div>
            </div>

            <!-- Accessibility -->
            <div class="setup-permission-row">
              <div class="setup-permission-icon">
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="9" cy="5" r="2.5" fill="#007AFF"/>
                  <path d="M9 8.5C6 8.5 3.5 10 3.5 12.5V14.5C3.5 15.05 3.95 15.5 4.5 15.5H13.5C14.05 15.5 14.5 15.05 14.5 14.5V12.5C14.5 10 12 8.5 9 8.5Z" fill="#007AFF"/>
                </svg>
              </div>
              <div class="setup-permission-info">
                <div class="setup-permission-name">{t('setup.permAccName')}</div>
                <div class="setup-permission-desc">{t('setup.permAccDesc', { pasteShortcut: isMac ? 'Cmd+V' : 'Ctrl+V' })}</div>
              </div>
              <div class="setup-permission-action">
                {#if accGranted}
                  <div class="setup-permission-granted">
                    <svg viewBox="0 0 14 14" fill="none">
                      <path d="M2.5 7.5L5.5 10.5L11.5 4.5" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                  </div>
                {:else}
                  <button class="setup-permission-btn" onclick={grantAccessibility}>{t('setup.permGrant')}</button>
                {/if}
              </div>
            </div>
          </div>

          <button
            class="setup-continue-btn"
            disabled={!permBothGranted}
            onclick={onPermissionsContinue}
          >
            {t('setup.permContinue')}
          </button>
        </div>
      {/if}

      <!-- ═══ STT Choice ═══ -->
      {#if currentState === 'sttChoice'}
        <div class="setup-state-content" style="animation: setupFadeIn 0.4s ease">
          <!-- Step indicator -->

          <!-- Mic illustration with wave effects -->
          <div class="setup-mic-wrap">
            <div class="wave"></div>
            <div class="wave"></div>
            <div class="wave"></div>
            <svg class="setup-mic-icon floating" width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="22" y="8" width="20" height="32" rx="10" fill="#007AFF"/>
              <rect x="16" y="28" width="32" height="2" rx="1" fill="none" stroke="#007AFF" stroke-width="2"/>
              <path d="M18 30C18 38.284 24.716 45 33 45H31C22.716 45 16 38.284 16 30" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round"/>
              <path d="M46 30C46 38.284 39.284 45 31 45H33C41.284 45 48 38.284 48 30" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round"/>
              <line x1="32" y1="45" x2="32" y2="53" stroke="#007AFF" stroke-width="2" stroke-linecap="round"/>
              <line x1="25" y1="53" x2="39" y2="53" stroke="#007AFF" stroke-width="2" stroke-linecap="round"/>
            </svg>
          </div>

          <div class="setup-title">{t('setup.sttChoiceTitle')}</div>
          <div class="setup-desc">{t('setup.sttChoiceDesc')}</div>

          <div class="setup-mode-control">
            <SegmentedControl
              options={sttModeOptions}
              value={sttMode}
              onchange={onSttModeChange}
            />
          </div>

          {#if sttMode === 'local'}
            <div class="setup-panel-desc">{t('setup.sttLocalDesc')}</div>

            <!-- Unified model list (Whisper + Qwen3-ASR) -->
            <div class="setup-model-list setup-stt-grid">
              {#each sttModels as model (model.id)}
                <button
                  class="setup-model-row"
                  class:selected={selectedLocalEngine === 'whisper' && selectedSttModel === model.id}
                  onclick={() => { selectedLocalEngine = 'whisper'; selectedSttModel = model.id; }}
                >
                  <div class="setup-model-radio">
                    {#if selectedLocalEngine === 'whisper' && selectedSttModel === model.id}
                      <div class="setup-model-radio-dot"></div>
                    {/if}
                  </div>
                  <div class="setup-model-info">
                    <div class="setup-model-name">
                      {t(`sttModel.${camelCase(model.id)}.name`)}
                      {#if `whisper:${model.id}` === recommendedKey}
                        <span class="setup-model-badge">{t('setup.recommended')}</span>
                      {/if}
                    </div>
                    <div class="setup-model-desc">
                      {t(`sttModel.${camelCase(model.id)}.desc`)} · {formatBytes(model.size_bytes)}
                    </div>
                  </div>
                </button>
              {/each}
              {#each qwen3AsrModels as model (model.id)}
                <button
                  class="setup-model-row"
                  class:selected={selectedLocalEngine === 'qwen3_asr' && selectedQwen3Model === model.id}
                  onclick={() => { selectedLocalEngine = 'qwen3_asr'; selectedQwen3Model = model.id; }}
                >
                  <div class="setup-model-radio">
                    {#if selectedLocalEngine === 'qwen3_asr' && selectedQwen3Model === model.id}
                      <div class="setup-model-radio-dot"></div>
                    {/if}
                  </div>
                  <div class="setup-model-info">
                    <div class="setup-model-name">
                      {t(`sttModel.${camelCase(model.id)}.name`)}
                      {#if `qwen3_asr:${model.id}` === recommendedKey}
                        <span class="setup-model-badge">{t('setup.recommended')}</span>
                      {/if}
                    </div>
                    <div class="setup-model-desc">
                      {t(`sttModel.${camelCase(model.id)}.desc`)} · {formatBytes(model.size_bytes)}
                    </div>
                  </div>
                </button>
              {/each}
            </div>

            <button class="setup-download-btn" onclick={onSttLocalDownload}>
              {isCurrentModelDownloaded ? t('setup.permContinue') : t('setup.sttModelDownloadBtn')}
            </button>
          {:else}
            <div class="setup-panel-desc">{t('setup.sttCloudDesc')}</div>
            <div class="setup-cloud-config">
              <CloudConfigPanel
                type="stt"
                bind:provider={sttProvider}
                bind:apiKey={sttApiKey}
                bind:endpoint={sttEndpoint}
                bind:modelId={sttModelId}
                bind:language={sttLanguage}
                onchange={onSttCloudChange}
              />
            </div>
            <button
              class="setup-download-btn"
              style="margin-top: 18px"
              disabled={!sttCloudValid}
              onclick={onSttCloudContinue}
            >
              {t('setup.sttCloudContinue')}
            </button>
          {/if}
        </div>
      {/if}

      <!-- ═══ Downloading STT Model ═══ -->
      {#if currentState === 'downloading'}
        <div class="setup-state-content" style="animation: setupFadeIn 0.4s ease">

          <div class="setup-mic-wrap">
            <div class="wave"></div>
            <div class="wave"></div>
            <div class="wave"></div>
            <svg class="setup-mic-icon pulsing" width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="22" y="8" width="20" height="32" rx="10" fill="#007AFF"/>
              <rect x="16" y="28" width="32" height="2" rx="1" fill="none" stroke="#007AFF" stroke-width="2"/>
              <path d="M18 30C18 38.284 24.716 45 33 45H31C22.716 45 16 38.284 16 30" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round"/>
              <path d="M46 30C46 38.284 39.284 45 31 45H33C41.284 45 48 38.284 48 30" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round"/>
              <line x1="32" y1="45" x2="32" y2="53" stroke="#007AFF" stroke-width="2" stroke-linecap="round"/>
              <line x1="25" y1="53" x2="39" y2="53" stroke="#007AFF" stroke-width="2" stroke-linecap="round"/>
            </svg>
          </div>

          <div class="setup-title">{t('setup.downloadingTitle')}</div>

          <div class="setup-progress-wrap">
            <ProgressBar
              percent={downloadPercent}
              shimmer={true}
              label="{Math.round(downloadPercent)}%"
              sublabel="{formatBytes(downloadedBytes)} / {formatBytes(downloadTotalBytes)}"
            />
          </div>
        </div>
      {/if}

      <!-- ═══ Complete ═══ -->
      {#if currentState === 'complete'}
        <div class="setup-state-content" style="animation: setupFadeIn 0.4s ease">
          <div class="setup-success-icon">
            <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="32" cy="32" r="28" fill="#34C759"/>
              <path d="M20 33L28 41L44 25" stroke="white" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
          <div class="setup-title">{t('setup.completeTitle')}</div>
          <div class="setup-desc">{t('setup.completeDesc')}</div>
        </div>
      {/if}

      <!-- ═══ Polish Choice ═══ -->
      {#if currentState === 'polishChoice'}
        <div class="setup-state-content" style="animation: setupFadeIn 0.4s ease">

          <!-- Sparkle/AI icon with wave effects -->
          <div class="setup-mic-wrap">
            <div class="wave"></div>
            <div class="wave"></div>
            <div class="wave"></div>
            <svg class="setup-mic-icon floating" width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M32 12L35.5 28.5L52 32L35.5 35.5L32 52L28.5 35.5L12 32L28.5 28.5Z" fill="#007AFF"/>
              <path d="M48 14L49.5 19.5L55 21L49.5 22.5L48 28L46.5 22.5L41 21L46.5 19.5Z" fill="#007AFF" opacity="0.7"/>
              <path d="M50 42L51 46L55 47L51 48L50 52L49 48L45 47L49 46Z" fill="#007AFF" opacity="0.5"/>
            </svg>
          </div>

          <div class="setup-title">{t('setup.polishChoiceTitle')}</div>
          <div class="setup-desc">{t('setup.polishChoiceDesc')}</div>

          <div class="setup-mode-control">
            <SegmentedControl
              options={polishModeOptions}
              value={polishMode}
              onchange={onPolishModeChange}
            />
          </div>

          {#if polishMode === 'local'}
            <div class="setup-panel-desc">{t('setup.polishLocalDesc')}</div>

            <div class="setup-model-list">
              {#each polishModels as model (model.id)}
                <button
                  class="setup-model-row"
                  class:selected={selectedPolishModel === model.id}
                  onclick={() => selectedPolishModel = model.id}
                >
                  <div class="setup-model-radio">
                    {#if selectedPolishModel === model.id}
                      <div class="setup-model-radio-dot"></div>
                    {/if}
                  </div>
                  <div class="setup-model-info">
                    <div class="setup-model-name">
                      {t(`polishModel.${camelCase(model.id)}.name`)}
                      {#if model.recommended}
                        <span class="setup-model-badge">{t('setup.polishRecommended')}</span>
                      {/if}
                    </div>
                    <div class="setup-model-desc">
                      {t(`polishModel.${camelCase(model.id)}.desc`)} · {formatBytes(model.size_bytes)}
                    </div>
                  </div>
                </button>
              {/each}
            </div>

            <button class="setup-download-btn" onclick={onPolishLocalDownload}>
              {selectedModelDownloaded ? t('setup.permContinue') : t('setup.llmDownloadBtn')}
            </button>
          {:else}
            <div class="setup-panel-desc">{t('setup.polishCloudDesc')}</div>
            <div class="setup-cloud-config">
              <CloudConfigPanel
                type="polish"
                bind:provider={polishProvider}
                bind:apiKey={polishApiKey}
                bind:endpoint={polishEndpoint}
                bind:modelId={polishModelId}
                onchange={onPolishCloudChange}
              />
            </div>
            <button
              class="setup-download-btn"
              style="margin-top: 18px"
              disabled={!polishCloudValid}
              onclick={onPolishCloudContinue}
            >
              {t('setup.polishCloudContinue')}
            </button>
          {/if}

          <button class="setup-skip-link" onclick={onPolishSkip}>
            {t('setup.polishSkip')}
          </button>
        </div>
      {/if}

      <!-- ═══ LLM Downloading ═══ -->
      {#if currentState === 'llmDownloading'}
        <div class="setup-state-content" style="animation: setupFadeIn 0.4s ease">

          <div class="setup-mic-wrap">
            <div class="wave"></div>
            <div class="wave"></div>
            <div class="wave"></div>
            <svg class="setup-mic-icon pulsing" width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M32 12L35.5 28.5L52 32L35.5 35.5L32 52L28.5 35.5L12 32L28.5 28.5Z" fill="#007AFF"/>
              <path d="M48 14L49.5 19.5L55 21L49.5 22.5L48 28L46.5 22.5L41 21L46.5 19.5Z" fill="#007AFF" opacity="0.7"/>
              <path d="M50 42L51 46L55 47L51 48L50 52L49 48L45 47L49 46Z" fill="#007AFF" opacity="0.5"/>
            </svg>
          </div>

          <div class="setup-title">{t('setup.llmDownloadingTitle')}</div>

          <div class="setup-progress-wrap">
            <ProgressBar
              percent={llmDownloadPercent}
              shimmer={true}
              label="{Math.round(llmDownloadPercent)}%"
              sublabel="{formatBytes(llmDownloadedBytes)} / {formatBytes(llmDownloadTotalBytes)}"
            />
          </div>
        </div>
      {/if}

      <!-- ═══ Error ═══ -->
      {#if currentState === 'error'}
        <div class="setup-state-content" style="animation: setupFadeIn 0.4s ease">
          <div class="setup-error-icon">
            <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="32" cy="32" r="28" fill="#FF3B30"/>
              <path d="M22 22L42 42M42 22L22 42" stroke="white" stroke-width="3.5" stroke-linecap="round"/>
            </svg>
          </div>
          <div class="setup-title">{t('setup.errorTitle')}</div>
          <div class="setup-error-msg">{errorMessage}</div>
          <button class="setup-retry-btn" onclick={onRetryDownload}>
            {t('setup.retryBtn')}
          </button>
        </div>
      {/if}

    </div>
  </div>
{/if}

<style>
  /* ── Overlay & backdrop ── */
  .setup-overlay {
    position: fixed;
    inset: 0;
    z-index: 1000;
  }

  .setup-overlay.fade-out {
    opacity: 0;
    transition: opacity 0.3s ease;
    pointer-events: none;
  }

  .setup-backdrop {
    position: absolute;
    inset: 0;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
  }

  /* ── Card ── */
  .setup-card {
    position: relative;
    width: 420px;
    margin: 0 auto;
    top: 50%;
    transform: translateY(-50%);
    text-align: center;
  }

  @keyframes setupFadeIn {
    from { opacity: 0; transform: scale(0.96); }
    to { opacity: 1; transform: scale(1); }
  }

  .setup-state-content {
    animation: setupFadeIn 0.4s ease;
  }

  /* ── Title & description ── */
  .setup-title {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text-primary);
    margin-bottom: 8px;
  }

  .setup-desc {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-bottom: 24px;
    padding: 0 20px;
  }

  /* ── Mic illustration ── */
  .setup-mic-wrap {
    display: inline-block;
    position: relative;
    width: 100px;
    height: 100px;
    margin-bottom: 20px;
  }

  .setup-mic-icon {
    position: relative;
    z-index: 2;
  }

  .setup-mic-icon.floating {
    animation: micFloat 3s ease-in-out infinite;
  }

  .setup-mic-icon.pulsing {
    animation: micPulse 1.5s ease-in-out infinite;
  }

  .setup-mic-wrap .wave {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 64px;
    height: 64px;
    margin: -32px 0 0 -32px;
    border-radius: 50%;
    border: 2px solid var(--accent-blue);
    opacity: 0;
    animation: waveExpand 3s ease-out infinite;
  }

  .setup-mic-wrap .wave:nth-child(2) { animation-delay: 1s; }
  .setup-mic-wrap .wave:nth-child(3) { animation-delay: 2s; }

  @keyframes micFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
  }

  @keyframes micPulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.06); opacity: 0.85; }
  }

  @keyframes waveExpand {
    0% { transform: scale(0.8); opacity: 0.6; }
    100% { transform: scale(1.8); opacity: 0; }
  }

  /* ── Permissions list ── */
  .setup-permissions-list {
    text-align: left;
    margin: 0 auto 24px;
    max-width: 340px;
  }

  .setup-permission-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    border-radius: var(--radius-md);
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    margin-bottom: 8px;
  }

  .setup-permission-row:last-child {
    margin-bottom: 0;
  }

  .setup-permission-icon {
    flex-shrink: 0;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
  }

  .setup-permission-info {
    flex: 1;
    min-width: 0;
  }

  .setup-permission-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1px;
  }

  .setup-permission-desc {
    font-size: 11px;
    color: var(--text-tertiary);
  }

  .setup-permission-action {
    flex-shrink: 0;
  }

  .setup-permission-btn {
    flex-shrink: 0;
    padding: 5px 14px;
    background: var(--accent-blue);
    color: #ffffff;
    border: none;
    border-radius: var(--radius-sm);
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s ease;
  }

  .setup-permission-btn:hover {
    background: #0066d6;
  }

  .setup-permission-granted {
    flex-shrink: 0;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: #34C759;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .setup-permission-granted svg {
    width: 14px;
    height: 14px;
  }

  /* ── Buttons ── */
  .setup-continue-btn {
    display: inline-block;
    padding: 10px 28px;
    background: var(--accent-blue);
    color: #ffffff;
    border: none;
    border-radius: var(--radius-md);
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s ease, opacity 0.15s ease;
  }

  .setup-continue-btn:hover:not(:disabled) {
    background: #0066d6;
  }

  .setup-continue-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .setup-download-btn {
    display: inline-block;
    padding: 10px 28px;
    background: var(--accent-blue);
    color: #ffffff;
    border: none;
    border-radius: var(--radius-md);
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s ease;
  }

  .setup-download-btn:hover:not(:disabled) {
    background: #0066d6;
  }

  .setup-download-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .setup-retry-btn {
    display: inline-block;
    padding: 10px 28px;
    background: var(--accent-blue);
    color: #ffffff;
    border: none;
    border-radius: var(--radius-md);
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s ease;
  }

  .setup-retry-btn:hover {
    background: #0066d6;
  }

  .setup-skip-link {
    display: block;
    margin: 14px auto 0;
    font-size: 13px;
    color: var(--text-tertiary);
    text-decoration: underline;
    cursor: pointer;
    background: none;
    border: none;
    font-family: 'Inter', sans-serif;
  }

  .setup-skip-link:hover {
    color: var(--text-secondary);
  }

  /* ── Mode control ── */
  .setup-mode-control {
    margin: 12px auto 16px;
    width: fit-content;
  }



  /* ── Panel description ── */
  .setup-panel-desc {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-bottom: 14px;
    white-space: pre-line;
  }

  /* ── Cloud config in setup ── */
  .setup-cloud-config {
    text-align: left;
    margin: 0 auto 16px;
    max-width: 360px;
  }

  .setup-cloud-config :global(.cloud-row) {
    margin-bottom: 10px;
  }

  .setup-cloud-config :global(.cloud-row:last-child) {
    margin-bottom: 0;
  }

  /* ── Progress wrap ── */
  .setup-progress-wrap {
    margin: 0 20px 8px;
  }

  /* ── Success icon ── */
  .setup-success-icon {
    display: inline-block;
    margin-bottom: 16px;
    animation: iconPop 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  }

  /* ── Error icon & message ── */
  .setup-error-icon {
    display: inline-block;
    margin-bottom: 16px;
    animation: iconPop 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  }

  @keyframes iconPop {
    0% { transform: scale(0); }
    100% { transform: scale(1); }
  }

  .setup-error-msg {
    font-size: 13px;
    color: #ff3b30;
    margin-bottom: 20px;
    padding: 0 20px;
    word-break: break-word;
  }

  /* ── Shield icon ── */
  .setup-icon-shield {
    margin-bottom: 20px;
  }

  /* ── Model list (polish choice) ── */
  .setup-model-list {
    text-align: left;
    margin: 0 auto 18px;
    max-width: 360px;
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    overflow: hidden;
  }

  .setup-model-row {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    padding: 12px 14px;
    background: var(--bg-secondary);
    border: none;
    border-bottom: 1px solid var(--border-color);
    cursor: pointer;
    transition: background 0.15s ease;
    font-family: 'Inter', sans-serif;
    text-align: left;
  }

  .setup-model-row:last-child {
    border-bottom: none;
  }

  .setup-model-row:hover {
    background: var(--bg-tertiary, var(--bg-secondary));
  }

  .setup-model-row.selected {
    background: rgba(0, 122, 255, 0.06);
  }

  .setup-model-radio {
    flex-shrink: 0;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    border: 2px solid var(--text-tertiary);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: border-color 0.15s ease;
  }

  .setup-model-row.selected .setup-model-radio {
    border-color: var(--accent-blue);
  }

  .setup-model-radio-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--accent-blue);
  }

  .setup-model-info {
    flex: 1;
    min-width: 0;
  }

  .setup-model-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .setup-model-badge {
    font-size: 10px;
    font-weight: 600;
    color: var(--accent-blue);
    background: rgba(0, 122, 255, 0.1);
    padding: 1px 6px;
    border-radius: 4px;
    white-space: nowrap;
  }

  .setup-model-desc {
    font-size: 11px;
    color: var(--text-tertiary);
    margin-top: 2px;
  }

  /* ── STT model grid (2-column) ── */
  .setup-stt-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    border: none;
    border-radius: 0;
    overflow: visible;
    max-width: 400px;
  }

  .setup-stt-grid .setup-model-row {
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    align-items: flex-start;
  }

  /* restore bottom border removed by :last-child rule */
  .setup-stt-grid .setup-model-row:last-child {
    border-bottom: 1px solid var(--border-color);
  }

  .setup-stt-grid .setup-model-desc {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

</style>
