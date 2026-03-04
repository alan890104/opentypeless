<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { t } from '$lib/stores/i18n.svelte';
  import {
    getSttConfig,
    setSttMode,
    setSttWhisperModel,
    setSttLocalEngine,
    setSttQwen3AsrModel,
    setSttLanguage,
    setSttCloudProvider,
    setSttCloudApiKey,
    setSttCloudEndpoint,
    setSttCloudModelId,
    setSttCloudLanguage,
    setVadEnabled,
    saveStt,
  } from '$lib/stores/settings.svelte';
  import { STT_LANGUAGES } from '$lib/constants';
  import {
    listWhisperModels,
    switchWhisperModel,
    downloadWhisperModel,
    onWhisperModelDownloadProgress,
    getWhisperModelRecommendation,
    checkVadModelStatus,
    downloadVadModel,
    onVadModelDownloadProgress,
    listQwen3AsrModels,
    switchQwen3AsrModel,
    downloadQwen3AsrModel,
    onQwen3AsrDownloadProgress,
    saveApiKey,
  } from '$lib/api';
  import type {
    SttMode,
    WhisperModelId,
    WhisperModelInfo,
    DownloadProgress,
    LocalSttEngine,
    Qwen3AsrModelId,
    Qwen3AsrModelInfo,
  } from '$lib/types';
  import type { UnlistenFn } from '@tauri-apps/api/event';
  import SettingRow from '$lib/components/SettingRow.svelte';
  import Toggle from '$lib/components/Toggle.svelte';
  import SegmentedControl from '$lib/components/SegmentedControl.svelte';
  import ProgressBar from '$lib/components/ProgressBar.svelte';
  import CloudConfigPanel from '$lib/components/CloudConfigPanel.svelte';
  import { formatSize, camelCase } from '$lib/utils';
  import SectionHeader from '$lib/components/SectionHeader.svelte';

  // ── Whisper model state ──

  let models = $state<WhisperModelInfo[]>([]);
  let recommendedModel = $state<WhisperModelId | null>(null);
  let whisperSwitching = $state(false);

  // ── Qwen3-ASR model state ──

  let qwen3Models = $state<Qwen3AsrModelInfo[]>([]);
  let qwen3Switching = $state(false);

  // ── Local model download state (shared by Whisper and Qwen3-ASR) ──

  let downloadingModelId = $state<string | null>(null);
  let downloadPercent = $state(0);
  let downloadedBytes = $state(0);
  let totalBytes = $state(0);
  let downloadErrorModelId = $state<string | null>(null);
  let unlisten: UnlistenFn | null = null;

  // ── VAD state ──

  let vadDownloading = $state(false);
  let vadUnlisten: UnlistenFn | null = null;

  async function onVadToggle(checked: boolean) {
    if (checked) {
      try {
        const status = await checkVadModelStatus();
        if (!status.downloaded) {
          vadDownloading = true;

          if (vadUnlisten) { vadUnlisten(); vadUnlisten = null; }
          vadUnlisten = await onVadModelDownloadProgress((d) => {
            if (d.status === 'complete') {
              vadDownloading = false;
              if (vadUnlisten) { vadUnlisten(); vadUnlisten = null; }
            } else if (d.status === 'error') {
              vadDownloading = false;
              console.error('VAD model download error:', d.message);
              setVadEnabled(false);
              saveStt();
              if (vadUnlisten) { vadUnlisten(); vadUnlisten = null; }
            }
          });

          try {
            await downloadVadModel();
          } catch (e) {
            vadDownloading = false;
            setVadEnabled(false);
            saveStt();
            console.error('Failed to start VAD model download:', e);
            return;
          }
        }
      } catch (e) {
        console.error('Failed to check VAD model status:', e);
      }
    }
    setVadEnabled(checked);
    saveStt();
  }

  let sttModeOptions = $derived([
    { value: 'local', label: t('settings.stt.modeLocal') },
    { value: 'cloud', label: t('settings.stt.modeCloud') },
  ]);

  let sttConfig = $derived(getSttConfig());

  // ── Whisper model loading ──

  async function loadModels() {
    try {
      models = await listWhisperModels();
    } catch (e) {
      console.error('Failed to list whisper models:', e);
    }
    try {
      recommendedModel = await getWhisperModelRecommendation();
    } catch (e) {
      console.error('Failed to get recommendation:', e);
    }
  }

  async function onSelectModel(modelId: WhisperModelId) {
    const prevModelId = sttConfig.whisper_model;
    const prevEngine = sttConfig.local_engine ?? 'whisper';
    const engineChanged = prevEngine !== 'whisper';
    if (modelId === prevModelId && !engineChanged) return;
    setSttWhisperModel(modelId);
    setSttLocalEngine('whisper');
    await saveStt();
    whisperSwitching = true;
    try {
      await switchWhisperModel(modelId);
    } catch (e) {
      console.error('Failed to switch whisper model:', e);
      if (!destroyed) {
        setSttWhisperModel(prevModelId);
        setSttLocalEngine(prevEngine as LocalSttEngine);
        await saveStt();
      }
    }
    if (destroyed) return;
    whisperSwitching = false;
    await loadModels();
  }

  async function startWhisperDownload(modelId: WhisperModelId) {
    downloadingModelId = modelId;
    downloadErrorModelId = null;
    downloadPercent = 0;
    downloadedBytes = 0;
    totalBytes = 0;

    if (unlisten) { unlisten(); unlisten = null; }

    unlisten = await onWhisperModelDownloadProgress((d: DownloadProgress) => {
      if (d.status === 'downloading') {
        const pct = d.downloaded && d.total ? (d.downloaded / d.total) * 100 : 0;
        downloadPercent = Math.min(pct, 100);
        downloadedBytes = d.downloaded ?? 0;
        totalBytes = d.total ?? 0;
      } else if (d.status === 'complete') {
        downloadingModelId = null;
        if (unlisten) { unlisten(); unlisten = null; }
        loadModels();
      } else if (d.status === 'error') {
        downloadingModelId = null;
        downloadErrorModelId = modelId;
        console.error('Whisper model download error:', d.message);
        if (unlisten) { unlisten(); unlisten = null; }
      }
    });

    try {
      await downloadWhisperModel(modelId);
    } catch (e) {
      downloadingModelId = null;
      downloadErrorModelId = modelId;
      console.error('Failed to start whisper model download:', e);
    }
  }

  // ── Qwen3-ASR model loading ──

  async function loadQwen3Models() {
    try {
      qwen3Models = await listQwen3AsrModels();
    } catch (e) {
      console.error('Failed to list Qwen3-ASR models:', e);
    }
  }

  async function onSelectQwen3Model(modelId: Qwen3AsrModelId) {
    const prevModelId = sttConfig.qwen3_asr_model;
    const prevEngine = sttConfig.local_engine ?? 'whisper';
    const engineChanged = prevEngine !== 'qwen3_asr';
    if (modelId === prevModelId && !engineChanged) return;
    setSttQwen3AsrModel(modelId);
    setSttLocalEngine('qwen3_asr');
    await saveStt();
    qwen3Switching = true;
    try {
      await switchQwen3AsrModel(modelId);
    } catch (e) {
      console.error('Failed to switch Qwen3-ASR model:', e);
      if (!destroyed) {
        setSttQwen3AsrModel(prevModelId ?? 'qwen3_asr1_7_b');
        setSttLocalEngine(prevEngine as LocalSttEngine);
        await saveStt();
      }
    }
    if (destroyed) return;
    qwen3Switching = false;
    await loadQwen3Models();
  }

  async function startQwen3Download(modelId: Qwen3AsrModelId) {
    downloadingModelId = modelId;
    downloadErrorModelId = null;
    downloadPercent = 0;
    downloadedBytes = 0;
    totalBytes = 0;

    if (unlisten) { unlisten(); unlisten = null; }

    unlisten = await onQwen3AsrDownloadProgress((d) => {
      if (d.status === 'complete') {
        downloadingModelId = null;
        if (unlisten) { unlisten(); unlisten = null; }
        loadQwen3Models();
      } else if (d.status === 'error') {
        downloadingModelId = null;
        downloadErrorModelId = modelId;
        console.error('Qwen3-ASR download error:', d.message);
        if (unlisten) { unlisten(); unlisten = null; }
      } else {
        const pct = d.downloaded && d.total ? (d.downloaded / d.total) * 100 : 0;
        downloadPercent = Math.min(pct, 100);
        downloadedBytes = d.downloaded ?? 0;
        totalBytes = d.total ?? 0;
      }
    });

    try {
      await downloadQwen3AsrModel(modelId);
    } catch (e) {
      downloadingModelId = null;
      downloadErrorModelId = modelId;
      console.error('Failed to start Qwen3-ASR download:', e);
    }
  }

  // ── Mode change ──

  function onModeChange(value: string) {
    setSttMode(value as SttMode);
    saveStt();
  }

  // ── Cloud config change ──

  async function onCloudChange() {
    await saveStt();
  }

  async function onSttApiKeyChange() {
    try {
      await saveApiKey('stt_' + cloudProvider, cloudApiKey);
    } catch (e) {
      console.error('Failed to save STT API key to keychain:', e);
    }
  }

  // ── Cloud config bindings ──
  let cloudProvider = $state(getSttConfig().cloud.provider);
  let cloudApiKey = $state(getSttConfig().cloud.api_key);
  let cloudEndpoint = $state(getSttConfig().cloud.endpoint);
  let cloudModelId = $state(getSttConfig().cloud.model_id);
  let cloudLanguage = $state(getSttConfig().language);

  $effect(() => {
    const cfg = getSttConfig();
    cloudProvider = cfg.cloud.provider;
    cloudApiKey = cfg.cloud.api_key;
    cloudEndpoint = cfg.cloud.endpoint;
    cloudModelId = cfg.cloud.model_id;
    cloudLanguage = cfg.language;
  });

  // Re-fetch model recommendation when STT language changes
  let prevSttLang = $state(getSttConfig().language);
  $effect(() => {
    const lang = getSttConfig().language;
    if (lang !== prevSttLang) {
      prevSttLang = lang;
      getWhisperModelRecommendation().then(rec => {
        recommendedModel = rec;
      }).catch(() => {});
    }
  });

  let destroyed = false;

  onMount(() => {
    loadModels();
    loadQwen3Models();
  });

  onDestroy(() => {
    destroyed = true;
    if (unlisten) { unlisten(); unlisten = null; }
    if (vadUnlisten) { vadUnlisten(); vadUnlisten = null; }
  });
</script>

<div class="section">
  <SectionHeader title={t('settings.stt')}>
    {#snippet icon()}
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
        <line x1="12" x2="12" y1="19" y2="22"/>
      </svg>
    {/snippet}
  </SectionHeader>

  <!-- Mode selector -->
  <SettingRow name={t('settings.stt.mode')} desc={t('settings.stt.modeDesc')}>
    <SegmentedControl
      options={sttModeOptions}
      value={sttConfig.mode}
      onchange={onModeChange}
    />
  </SettingRow>

  <!-- Local panel -->
  {#if sttConfig.mode === 'local'}
    <div class="sub-settings">
      <!-- Language selector (shared) -->
      <SettingRow name={t('settings.stt.language')} desc={t('settings.stt.languageDesc')}>
        <select
          class="language-select"
          value={sttConfig.language}
          onchange={(e) => {
            const val = (e.target as HTMLSelectElement).value;
            setSttLanguage(val);
            saveStt();
          }}
        >
          {#each STT_LANGUAGES as lang}
            <option value={lang.value}>{lang.label}</option>
          {/each}
        </select>
      </SettingRow>

      <!-- VAD toggle -->
      <SettingRow name={t('settings.stt.vad')} desc={t('settings.stt.vadDesc')}>
        {#if vadDownloading}
          <span class="vad-downloading">{t('settings.stt.downloading')}</span>
        {:else}
          <Toggle checked={sttConfig.vad_enabled} onchange={onVadToggle} />
        {/if}
      </SettingRow>

      <!-- Unified local model list -->
      <div class="model-list">
        <!-- Whisper section -->
        <div class="model-section-header">Whisper</div>
        <div class="model-section-group" class:switching={whisperSwitching}>
          {#each models as model (model.id)}
            {@const isActive = (sttConfig.local_engine ?? 'whisper') === 'whisper' && model.id === sttConfig.whisper_model}
            {@const isDownloading = downloadingModelId === model.id}
            {@const isRecommended = model.id === recommendedModel}
            {@const isSwitchingThis = whisperSwitching && isActive}
            {@const isError = model.id === downloadErrorModelId}
            <!-- svelte-ignore a11y_no_static_element_interactions -->
            <div
              class="model-row"
              class:active={isActive}
              class:disabled={(!model.downloaded && !isDownloading && !isError) || whisperSwitching}
              onclick={() => !whisperSwitching && model.downloaded && onSelectModel(model.id)}
              onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); !whisperSwitching && model.downloaded && onSelectModel(model.id); } }}
              role="radio"
              aria-checked={isActive}
              tabindex="0"
            >
              <div class="model-radio" class:checked={isActive}>
                {#if isSwitchingThis}
                  <div class="model-inline-spinner"></div>
                {:else if isActive}
                  <div class="model-radio-dot"></div>
                {/if}
              </div>
              <div class="model-info">
                <div class="model-name-row">
                  <span class="model-name">{t(`sttModel.${camelCase(model.id)}.name`)}</span>
                  {#if isRecommended}
                    <span class="model-badge">{t('settings.stt.recommended')}</span>
                  {/if}
                </div>
                <div class="model-desc">{t(`sttModel.${camelCase(model.id)}.desc`)}</div>
                <div class="model-size">{formatSize(model.size_bytes)}</div>
              </div>
              <div class="model-action">
                {#if model.downloaded}
                  <span class="model-downloaded-check">
                    <svg viewBox="0 0 14 14" fill="none">
                      <path d="M2.5 7.5L5.5 10.5L11.5 4.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                  </span>
                {:else if isDownloading}
                  <span class="model-downloading-label">{Math.round(downloadPercent)}%</span>
                {:else if isError}
                  <button
                    class="model-retry-btn"
                    onclick={(e) => { e.stopPropagation(); startWhisperDownload(model.id); }}
                  >{t('settings.stt.retry')}</button>
                {:else}
                  <button
                    class="model-download-btn"
                    onclick={(e) => { e.stopPropagation(); startWhisperDownload(model.id); }}
                  >{t('settings.stt.download')}</button>
                {/if}
              </div>
            </div>

            {#if isDownloading}
              <div class="model-progress-wrap">
                <ProgressBar
                  percent={downloadPercent}
                  label="{Math.round(downloadPercent)}%"
                  sublabel="{formatSize(downloadedBytes)} / {formatSize(totalBytes)}"
                  shimmer
                />
              </div>
            {/if}
          {/each}
        </div>

        <!-- Qwen3-ASR section -->
        <div class="model-section-header">Qwen3-ASR</div>
        <div class="model-section-group" class:switching={qwen3Switching}>
          {#each qwen3Models as model (model.id)}
            {@const isActive = (sttConfig.local_engine ?? 'whisper') === 'qwen3_asr' && model.id === (sttConfig.qwen3_asr_model ?? 'qwen3_asr1_7_b')}
            {@const isThisDownloading = model.id === downloadingModelId}
            {@const isSwitchingThis = qwen3Switching && isActive}
            {@const isError = model.id === downloadErrorModelId}
            <!-- svelte-ignore a11y_no_static_element_interactions -->
            <div
              class="model-row"
              class:active={isActive}
              class:disabled={(!model.downloaded && !isThisDownloading && !isError) || qwen3Switching}
              onclick={() => !qwen3Switching && model.downloaded && onSelectQwen3Model(model.id)}
              onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); !qwen3Switching && model.downloaded && onSelectQwen3Model(model.id); } }}
              role="radio"
              aria-checked={isActive}
              tabindex="0"
            >
              <div class="model-radio" class:checked={isActive}>
                {#if isSwitchingThis}
                  <div class="model-inline-spinner"></div>
                {:else if isActive}
                  <div class="model-radio-dot"></div>
                {/if}
              </div>
              <div class="model-info">
                <div class="model-name-row">
                  <span class="model-name">{t(`sttModel.${camelCase(model.id)}.name`)}</span>
                </div>
                <div class="model-desc">{t(`sttModel.${camelCase(model.id)}.desc`)}</div>
                <div class="model-size">{formatSize(model.size_bytes)}</div>
              </div>
              <div class="model-action">
                {#if model.downloaded}
                  <span class="model-downloaded-check">
                    <svg viewBox="0 0 14 14" fill="none">
                      <path d="M2.5 7.5L5.5 10.5L11.5 4.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                  </span>
                {:else if isThisDownloading}
                  <span class="model-downloading-label">{Math.round(downloadPercent)}%</span>
                {:else if isError}
                  <button
                    class="model-retry-btn"
                    onclick={(e) => { e.stopPropagation(); startQwen3Download(model.id); }}
                  >{t('settings.stt.retry')}</button>
                {:else}
                  <button
                    class="model-download-btn"
                    onclick={(e) => { e.stopPropagation(); startQwen3Download(model.id); }}
                  >{t('settings.stt.download')}</button>
                {/if}
              </div>
            </div>

            {#if isThisDownloading}
              <div class="model-progress-wrap">
                <ProgressBar
                  percent={downloadPercent}
                  label="{Math.round(downloadPercent)}%"
                  sublabel="{formatSize(downloadedBytes)} / {formatSize(totalBytes)}"
                  shimmer
                />
              </div>
            {/if}
          {/each}
        </div>
      </div>
    </div>
  {/if}

  <!-- Cloud panel -->
  {#if sttConfig.mode === 'cloud'}
    <div class="sub-settings">
      <CloudConfigPanel
        type="stt"
        bind:provider={cloudProvider}
        bind:apiKey={cloudApiKey}
        bind:endpoint={cloudEndpoint}
        bind:modelId={cloudModelId}
        bind:language={cloudLanguage}
        onchange={async () => {
          setSttCloudProvider(cloudProvider as any);
          setSttCloudApiKey(cloudApiKey);
          setSttCloudEndpoint(cloudEndpoint);
          setSttCloudModelId(cloudModelId);
          setSttCloudLanguage(cloudLanguage);
          await onCloudChange();
        }}
        onapiKeyChange={onSttApiKeyChange}
      />
    </div>
  {/if}
</div>


<style>
  .section {
    margin-bottom: 32px;
  }


  .sub-settings {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 12px;
  }

  .model-list {
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  .model-section-header {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    padding: 10px 2px 4px;
  }

  .model-section-header:first-child {
    padding-top: 2px;
  }

  .model-section-group {
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin-bottom: 4px;
  }

  .model-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: var(--radius-md);
    border: 1px solid var(--border-color);
    background: var(--bg-secondary);
    cursor: pointer;
    transition: all 0.15s ease;
    text-align: left;
    font-family: 'Inter', sans-serif;
    -webkit-app-region: no-drag;
    app-region: no-drag;
  }

  .model-row:hover:not(.disabled) {
    border-color: var(--accent-blue);
  }

  .model-row.active {
    border-color: var(--accent-blue);
    background: color-mix(in srgb, var(--accent-blue) 6%, var(--bg-secondary));
  }

  .model-row.disabled {
    opacity: 0.7;
    cursor: default;
  }

  .model-radio {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    border: 2px solid var(--text-tertiary);
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: border-color 0.15s ease;
  }

  .model-radio.checked {
    border-color: var(--accent-blue);
  }

  .model-radio-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-blue);
  }

  .model-info {
    flex: 1;
    min-width: 0;
  }

  .model-name-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .model-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .model-badge {
    font-size: 10px;
    font-weight: 600;
    color: var(--accent-blue);
    background: color-mix(in srgb, var(--accent-blue) 12%, transparent);
    padding: 1px 6px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }

  .model-desc {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 1px;
  }

  .model-size {
    font-size: 11px;
    color: var(--text-tertiary);
    margin-top: 1px;
  }

  .model-action {
    flex-shrink: 0;
  }

  .model-downloaded-check {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    color: var(--accent-green);
  }

  .model-downloaded-check svg {
    width: 14px;
    height: 14px;
  }

  .model-downloading-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--accent-blue);
  }

  .model-download-btn {
    -webkit-app-region: no-drag;
    app-region: no-drag;
    padding: 4px 12px;
    border: none;
    border-radius: var(--radius-sm);
    background: var(--accent-blue);
    color: #ffffff;
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    white-space: nowrap;
  }

  .model-download-btn:hover {
    background: #0066d6;
  }

  .model-retry-btn {
    -webkit-app-region: no-drag;
    app-region: no-drag;
    padding: 4px 12px;
    border: none;
    border-radius: var(--radius-sm);
    background: #ff3b30;
    color: #ffffff;
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    white-space: nowrap;
  }

  .model-retry-btn:hover {
    background: #d63027;
  }

  .model-progress-wrap {
    padding: 0 12px 8px;
  }

  .model-section-group.switching {
    opacity: 0.6;
    pointer-events: none;
  }

  .model-inline-spinner {
    width: 8px;
    height: 8px;
    border: 1.5px solid rgba(0, 0, 0, 0.1);
    border-top-color: var(--accent-blue);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .vad-downloading {
    font-size: 12px;
    font-weight: 600;
    color: var(--accent-blue);
    animation: vad-pulse 1.2s ease-in-out infinite;
  }

  @keyframes vad-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .language-select {
    padding: 7px 28px 7px 12px;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
    background: var(--bg-primary);
    cursor: pointer;
    outline: none;
    appearance: none;
    -webkit-appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg width='10' height='6' viewBox='0 0 10 6' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 1L5 5L9 1' stroke='%236e6e73' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
    min-width: 160px;
    transition: border-color 0.15s ease, background-color 0.15s ease;
  }

  .language-select:hover {
    border-color: rgba(0, 0, 0, 0.15);
    background-color: var(--bg-sidebar);
  }

  .language-select:focus {
    outline: none;
    border-color: var(--accent-blue);
  }
</style>
