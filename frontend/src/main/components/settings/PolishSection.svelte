<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { t } from '$lib/stores/i18n.svelte';
  import {
    getPolishConfig,
    setPolishEnabled,
    setPolishMode,
    setPolishModel,
    setPolishReasoning,
    setPolishCloudProvider,
    setPolishCloudApiKey,
    setPolishCloudEndpoint,
    setPolishCloudModelId,
    savePolish,
  } from '$lib/stores/settings.svelte';
  import {
    listPolishModels,
    switchPolishModel,
    downloadPolishModel,
    onPolishModelDownloadProgress,
    deletePolishModel,
    saveApiKey,
  } from '$lib/api';
  import { showConfirm } from '$lib/stores/ui.svelte';
  import type { PolishMode, PolishModel, PolishModelInfo, DownloadProgress } from '$lib/types';
  import type { UnlistenFn } from '@tauri-apps/api/event';
  import SettingRow from '$lib/components/SettingRow.svelte';
  import Toggle from '$lib/components/Toggle.svelte';
  import SegmentedControl from '$lib/components/SegmentedControl.svelte';
  import ProgressBar from '$lib/components/ProgressBar.svelte';
  import CloudConfigPanel from '$lib/components/CloudConfigPanel.svelte';
  import { formatSize, camelCase } from '$lib/utils';
  import SectionHeader from '$lib/components/SectionHeader.svelte';

  // ── Model list from backend ──

  let models = $state<PolishModelInfo[]>([]);
  let downloadingModelId = $state<PolishModel | null>(null);
  let downloadPercent = $state(0);
  let downloadedBytes = $state(0);
  let totalBytes = $state(0);
  let downloadError = $state(false);
  let unlisten: UnlistenFn | null = null;

  let polishConfig = $derived(getPolishConfig());

  let modeOptions = $derived([
    { value: 'local', label: t('settings.polish.modeLocal') },
    { value: 'cloud', label: t('settings.polish.modeCloud') },
  ]);


  async function loadModels() {
    try {
      models = await listPolishModels();
    } catch (e) {
      console.error('Failed to list polish models:', e);
    }
  }

  async function onSelectModel(modelId: PolishModel) {
    if (modelId === polishConfig.model) return;
    setPolishModel(modelId);
    try {
      await switchPolishModel(modelId);
    } catch (e) {
      console.error('Failed to switch polish model:', e);
    }
    await loadModels();
  }

  async function startDownload(modelId: PolishModel) {
    downloadingModelId = modelId;
    downloadError = false;
    downloadPercent = 0;
    downloadedBytes = 0;
    totalBytes = 0;

    if (unlisten) {
      unlisten();
      unlisten = null;
    }

    unlisten = await onPolishModelDownloadProgress((d: DownloadProgress) => {
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
        downloadError = true;
        console.error('Polish model download error:', d.message);
        if (unlisten) { unlisten(); unlisten = null; }
      }
    });

    try {
      await downloadPolishModel(modelId);
    } catch (e) {
      downloadingModelId = null;
      downloadError = true;
      console.error('Failed to start polish model download:', e);
    }
  }

  // ── Model deletion ──

  function confirmDeletePolish(model: PolishModelInfo) {
    const isActive = model.id === polishConfig.model;
    const msg = t('model.deleteConfirm')
      .replace('{name}', t(`polishModel.${camelCase(model.id)}.name`))
      .replace('{size}', formatSize(model.file_size_on_disk || model.size_bytes))
      + (isActive ? t('model.deleteActiveWarning') : '');
    showConfirm(t('model.delete'), msg, t('model.delete'), async () => {
      try {
        await deletePolishModel(model.id);
        await loadModels();
      } catch (e) {
        console.error('Failed to delete polish model:', e);
      }
    });
  }

  // ── Event handlers ──

  function onTogglePolish(checked: boolean) {
    setPolishEnabled(checked);
    savePolish();
  }

  function onToggleReasoning(checked: boolean) {
    setPolishReasoning(checked);
    savePolish();
  }

  function onModeChange(value: string) {
    setPolishMode(value as PolishMode);
    savePolish();
  }

  // ── Cloud config ──
  let cloudProvider = $state(getPolishConfig().cloud.provider);
  let cloudApiKey = $state(getPolishConfig().cloud.api_key);
  let cloudEndpoint = $state(getPolishConfig().cloud.endpoint);
  let cloudModelId = $state(getPolishConfig().cloud.model_id);

  // Sync from store when settings are reloaded externally
  $effect(() => {
    const cfg = getPolishConfig();
    cloudProvider = cfg.cloud.provider;
    cloudApiKey = cfg.cloud.api_key;
    cloudEndpoint = cfg.cloud.endpoint;
    cloudModelId = cfg.cloud.model_id;
  });

  async function onCloudChange() {
    setPolishCloudProvider(cloudProvider as any);
    setPolishCloudApiKey(cloudApiKey);
    setPolishCloudEndpoint(cloudEndpoint);
    setPolishCloudModelId(cloudModelId);
    await savePolish();
  }

  async function onApiKeyChange() {
    try {
      await saveApiKey(cloudProvider, cloudApiKey);
    } catch (e) {
      console.error('Failed to save polish API key to keychain:', e);
    }
  }

  onMount(() => {
    loadModels();
  });

  onDestroy(() => {
    if (unlisten) { unlisten(); unlisten = null; }
  });
</script>

<div class="section">
  <SectionHeader title={t('settings.polish')}>
    {#snippet icon()}
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>
        <path d="M20 3v4"/>
        <path d="M22 5h-4"/>
      </svg>
    {/snippet}
  </SectionHeader>

  <!-- Polish toggle -->
  <SettingRow name={t('settings.polish.toggle')} desc={t('settings.polish.toggleDesc')}>
    <Toggle checked={polishConfig.enabled} onchange={onTogglePolish} />
  </SettingRow>

  <!-- Sub-settings (visible when enabled) -->
  {#if polishConfig.enabled}
    <div class="sub-settings">
      <!-- Mode selector -->
      <SettingRow name={t('settings.polish.mode')}>
        <SegmentedControl
          options={modeOptions}
          value={polishConfig.mode}
          onchange={onModeChange}
        />
      </SettingRow>

      <!-- Reasoning toggle -->
      <SettingRow
        name={t('settings.polish.reasoning')}
        desc={t('settings.polish.reasoningDesc')}
      >
        <Toggle checked={polishConfig.reasoning} onchange={onToggleReasoning} />
      </SettingRow>

      <!-- Local panel: multi-model selector -->
      {#if polishConfig.mode === 'local'}
        <div class="local-panel">
          <div class="model-list-label">{t('settings.polish.localModel')}</div>
          <div class="model-list">
            {#each models as model (model.id)}
              {@const isActive = model.id === polishConfig.model}
              {@const isDownloading = downloadingModelId === model.id}
              <!-- svelte-ignore a11y_no_static_element_interactions -->
              <div
                class="model-row"
                class:active={isActive}
                class:disabled={!model.downloaded && !isDownloading}
                onclick={() => model.downloaded && onSelectModel(model.id)}
                onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); model.downloaded && onSelectModel(model.id); } }}
                role="radio"
                aria-checked={isActive}
                tabindex="0"
              >
                <!-- Radio indicator -->
                <div class="model-radio" class:checked={isActive}>
                  {#if isActive}
                    <div class="model-radio-dot"></div>
                  {/if}
                </div>

                <!-- Info -->
                <div class="model-info">
                  <div class="model-name-row">
                    <span class="model-name">{t(`polishModel.${camelCase(model.id)}.name`)}</span>
                  </div>
                  <div class="model-desc">{t(`polishModel.${camelCase(model.id)}.desc`)}</div>
                  <div class="model-size">{formatSize(model.size_bytes)}</div>
                </div>

                <!-- Action -->
                <div class="model-action">
                  {#if model.downloaded}
                    <span class="model-downloaded-check">
                      <svg viewBox="0 0 14 14" fill="none">
                        <path d="M2.5 7.5L5.5 10.5L11.5 4.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                      </svg>
                    </span>
                    <button
                      class="model-delete-btn"
                      title={t('model.delete')}
                      onclick={(e) => { e.stopPropagation(); confirmDeletePolish(model); }}
                    >
                      <svg viewBox="0 0 14 14" fill="none">
                        <path d="M2 3.5h10M5 3.5V2.5a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v1M11 3.5l-.5 8a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1l-.5-8" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>
                      </svg>
                    </button>
                  {:else if isDownloading}
                    <span class="model-downloading-label">{Math.round(downloadPercent)}%</span>
                  {:else}
                    <button
                      class="model-download-btn"
                      onclick={(e) => { e.stopPropagation(); startDownload(model.id); }}
                    >
                      {t('settings.polish.download')}
                    </button>
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
        </div>
      {/if}

      <!-- Cloud panel -->
      {#if polishConfig.mode === 'cloud'}
        <div class="cloud-panel">
          <CloudConfigPanel
            type="polish"
            bind:provider={cloudProvider}
            bind:apiKey={cloudApiKey}
            bind:endpoint={cloudEndpoint}
            bind:modelId={cloudModelId}
            onchange={onCloudChange}
            onapiKeyChange={onApiKeyChange}
          />
        </div>
      {/if}

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

  .local-panel,
  .cloud-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .model-list-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 8px;
  }

  .model-list {
    display: flex;
    flex-direction: column;
    gap: 2px;
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
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .model-delete-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    padding: 0;
    border: none;
    border-radius: 4px;
    background: transparent;
    color: var(--text-tertiary);
    cursor: pointer;
    transition: all 0.15s ease;
    -webkit-app-region: no-drag;
    app-region: no-drag;
  }

  .model-delete-btn:hover {
    color: #ff3b30;
    background: rgba(255, 59, 48, 0.1);
  }

  .model-delete-btn svg {
    width: 13px;
    height: 13px;
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

  .model-progress-wrap {
    padding: 0 12px 8px;
  }
</style>
