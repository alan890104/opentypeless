<script lang="ts">
  import { open } from '@tauri-apps/plugin-dialog';
  import { t } from '$lib/stores/i18n.svelte';
  import {
    getDataRoot,
    checkDataRootTarget,
    migrateDataRoot,
    onDataRootMigrationProgress,
  } from '$lib/api';
  import type { DataRootMigrationProgress } from '$lib/types';
  import SettingRow from '$lib/components/SettingRow.svelte';
  import SectionHeader from '$lib/components/SectionHeader.svelte';
  import { onMount, onDestroy } from 'svelte';
  import type { UnlistenFn } from '@tauri-apps/api/event';

  // ── State ─────────────────────────────────────────────────────────────────

  let currentPath = $state<string | null>(null);

  // Migration dialog
  let showMigrateDialog = $state(false);
  let pendingPath = $state('');
  let alreadyHasData = $state(false);
  let noSpaceError = $state(false);

  // Progress modal
  let showProgress = $state(false);
  let progressPhase = $state('');
  let progressBytes = $state(0);
  let progressTotal = $state(0);

  let errorMsg = $state('');
  let unlisten: UnlistenFn | null = null;

  onMount(async () => {
    currentPath = await getDataRoot();
    unlisten = await onDataRootMigrationProgress(onProgress);
  });

  onDestroy(() => {
    unlisten?.();
  });

  function onProgress(p: DataRootMigrationProgress) {
    progressPhase = p.phase;
    progressBytes = p.bytes_done;
    progressTotal = p.bytes_total;
    if (p.phase === 'done') {
      showProgress = false;
    }
  }

  // ── Choose folder ─────────────────────────────────────────────────────────

  async function handleChooseFolder() {
    errorMsg = '';
    const selected = await open({ directory: true, multiple: false });
    if (!selected || typeof selected !== 'string') return;

    const check = await checkDataRootTarget(selected).catch(() => null);
    if (!check) return;

    if (!check.has_enough_space) {
      noSpaceError = true;
      pendingPath = selected;
      alreadyHasData = check.already_has_data;
      showMigrateDialog = true;
      return;
    }

    noSpaceError = false;
    pendingPath = selected;
    alreadyHasData = check.already_has_data;
    showMigrateDialog = true;
  }

  // ── Migration actions ─────────────────────────────────────────────────────

  async function doMigrate(strategy: 'move' | 'change_only') {
    showMigrateDialog = false;
    if (strategy === 'move') showProgress = true;
    try {
      await migrateDataRoot(pendingPath, strategy);
      currentPath = pendingPath;
    } catch (e: unknown) {
      showProgress = false;
      const raw = String(e);
      if (raw === 'recording_active') errorMsg = t('settings.storage.err.recording');
      else if (raw === 'meeting_active') errorMsg = t('settings.storage.err.meeting');
      else errorMsg = t('settings.storage.err.generic').replace('{msg}', raw);
    }
  }

  async function handleReset() {
    errorMsg = '';
    try {
      await migrateDataRoot(null, 'reset');
      currentPath = null;
    } catch (e: unknown) {
      errorMsg = String(e);
    }
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  function formatBytes(b: number): string {
    if (b >= 1024 ** 3) return `${(b / 1024 ** 3).toFixed(1)} GB`;
    if (b >= 1024 ** 2) return `${(b / 1024 ** 2).toFixed(0)} MB`;
    return `${(b / 1024).toFixed(0)} KB`;
  }

  const displayPath = $derived(currentPath ?? t('settings.storage.default'));
</script>

<!-- ── Main section ────────────────────────────────────────────────────────── -->
<div class="section">
  <SectionHeader title={t('settings.storage')}>
    {#snippet icon()}
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 12H2"/>
        <path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>
        <line x1="6" y1="16" x2="6.01" y2="16"/>
        <line x1="10" y1="16" x2="10.01" y2="16"/>
      </svg>
    {/snippet}
  </SectionHeader>

  <SettingRow name={t('settings.storage.dataRoot')} desc={t('settings.storage.dataRootDesc')}>
    <div class="path-row">
      <span class="path-label" title={displayPath}>{displayPath}</span>
      <div class="btn-group">
        <button class="action-btn" onclick={handleChooseFolder}>
          {t('settings.storage.choose')}
        </button>
        {#if currentPath}
          <button class="action-btn secondary" onclick={handleReset}>
            {t('settings.storage.reset')}
          </button>
        {/if}
      </div>
    </div>
  </SettingRow>

  {#if errorMsg}
    <p class="error-msg">{errorMsg}</p>
  {/if}
</div>

<!-- ── Migration dialog ───────────────────────────────────────────────────── -->
{#if showMigrateDialog}
  <div class="modal-overlay">
    <div class="modal-backdrop" role="presentation" onclick={() => (showMigrateDialog = false)}></div>
    <div class="modal">
      <h2 class="modal-title">{t('settings.storage.migrate.title')}</h2>
      <p class="modal-msg">{t('settings.storage.migrate.message')}</p>

      {#if noSpaceError}
        <p class="warn-msg">{t('settings.storage.migrate.noSpace')}</p>
      {:else if alreadyHasData}
        <p class="warn-msg">{t('settings.storage.migrate.alreadyHasData')}</p>
      {/if}

      <div class="modal-options">
        <button class="option-btn" disabled={noSpaceError} onclick={() => doMigrate('move')}>
          <span class="opt-label">{t('settings.storage.migrate.move')}</span>
          <span class="opt-desc">{t('settings.storage.migrate.moveDesc')}</span>
        </button>
        <button class="option-btn" onclick={() => doMigrate('change_only')}>
          <span class="opt-label">{t('settings.storage.migrate.changeOnly')}</span>
          <span class="opt-desc">{t('settings.storage.migrate.changeOnlyDesc')}</span>
        </button>
      </div>

      <button class="cancel-btn" onclick={() => (showMigrateDialog = false)}>
        {t('settings.storage.migrate.cancel')}
      </button>
    </div>
  </div>
{/if}

<!-- ── Progress modal ─────────────────────────────────────────────────────── -->
{#if showProgress}
  <div class="modal-overlay">
    <div class="modal-backdrop" role="presentation"></div>
    <div class="modal">
      <h2 class="modal-title">{t('settings.storage.progress.title')}</h2>
      <p class="modal-msg">{t('settings.storage.progress.desc')}</p>
      <div class="progress-bar-track">
        <div
          class="progress-bar-fill"
          style="width: {progressTotal > 0 ? Math.round((progressBytes / progressTotal) * 100) : 0}%"
        ></div>
      </div>
      {#if progressTotal > 0}
        <p class="progress-bytes">{formatBytes(progressBytes)} / {formatBytes(progressTotal)}</p>
      {/if}
    </div>
  </div>
{/if}

<style>
  .section {
    margin-bottom: 32px;
  }

  .path-row {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
  }

  .path-label {
    font-size: 12px;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 240px;
    direction: rtl;
    text-align: left;
  }

  .btn-group {
    display: flex;
    gap: 6px;
    flex-shrink: 0;
  }

  .action-btn {
    padding: 6px 14px;
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    background: var(--bg-secondary);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s;
    white-space: nowrap;
  }

  .action-btn:hover {
    background: var(--bg-hover);
  }

  .action-btn.secondary {
    color: var(--text-secondary);
  }

  .error-msg {
    font-size: 12px;
    color: #ff3b30;
    margin: 4px 0 0 0;
    padding: 0 4px;
  }

  /* ── Modals ── */

  .modal-overlay {
    position: fixed;
    inset: 0;
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .modal-backdrop {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.45);
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
  }

  .modal {
    position: relative;
    background: var(--bg-primary);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    padding: 28px 28px 22px;
    width: 380px;
    max-width: 90vw;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
  }

  .modal-title {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 8px 0;
  }

  .modal-msg {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 0 0 14px 0;
    line-height: 1.5;
  }

  .warn-msg {
    font-size: 12px;
    color: #ff9f0a;
    background: rgba(255, 159, 10, 0.08);
    border: 1px solid rgba(255, 159, 10, 0.2);
    border-radius: var(--radius-sm);
    padding: 8px 12px;
    margin: 0 0 12px 0;
    line-height: 1.4;
  }

  .modal-options {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 14px;
  }

  .option-btn {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 12px 14px;
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    background: var(--bg-secondary);
    cursor: pointer;
    text-align: left;
    transition: background 0.15s, border-color 0.15s;
    gap: 3px;
    width: 100%;
  }

  .option-btn:hover:not(:disabled) {
    background: var(--bg-hover);
    border-color: var(--accent-blue);
  }

  .option-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .opt-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .opt-desc {
    font-size: 11px;
    color: var(--text-secondary);
  }

  .cancel-btn {
    width: 100%;
    padding: 8px;
    border: none;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    transition: color 0.15s;
  }

  .cancel-btn:hover {
    color: var(--text-primary);
  }

  /* ── Progress ── */

  .progress-bar-track {
    height: 6px;
    background: var(--bg-tertiary);
    border-radius: 3px;
    overflow: hidden;
    margin: 16px 0 8px;
  }

  .progress-bar-fill {
    height: 100%;
    background: var(--accent-blue);
    border-radius: 3px;
    transition: width 0.2s ease;
  }

  .progress-bytes {
    font-size: 12px;
    color: var(--text-secondary);
    text-align: center;
    margin: 0;
  }
</style>
