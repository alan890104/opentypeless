<script lang="ts">
  import { onDestroy } from 'svelte';
  import { t } from '$lib/stores/i18n.svelte';
  import SectionHeader from '$lib/components/SectionHeader.svelte';
  import { getHotkey, getEditHotkey, setHotkey, setEditHotkey, getPolishConfig, getMeetingHotkey, setMeetingHotkey } from '$lib/stores/settings.svelte';
  import { updateHotkey, updateEditHotkey, updateMeetingHotkey } from '$lib/api';
  import Keycaps from '$lib/components/Keycaps.svelte';
  import { MODIFIER_SYMBOLS } from '$lib/constants';

  const modifierHint = Object.values(MODIFIER_SYMBOLS).join(' ');

  // ── Primary hotkey capture ──

  let isCapturing = $state(false);
  let capturedModifiers = $state(new Set<string>());
  let capturedCode = $state('');

  function startCapture() {
    isCapturing = true;
    capturedModifiers = new Set();
    capturedCode = '';
    document.addEventListener('keydown', onCaptureKeydown);
    document.addEventListener('keyup', onCaptureKeyup);
  }

  function cancelCapture() {
    isCapturing = false;
    capturedModifiers = new Set();
    capturedCode = '';
    document.removeEventListener('keydown', onCaptureKeydown);
    document.removeEventListener('keyup', onCaptureKeyup);
  }

  function onCaptureKeydown(e: KeyboardEvent) {
    e.preventDefault();
    e.stopPropagation();

    if (e.key === 'Escape') {
      cancelCapture();
      return;
    }

    const mods = new Set<string>();
    if (e.altKey) mods.add('Alt');
    if (e.ctrlKey) mods.add('Control');
    if (e.shiftKey) mods.add('Shift');
    if (e.metaKey) mods.add('Super');
    capturedModifiers = mods;

    const nonModifiers = ['Alt', 'Control', 'Shift', 'Meta'];
    if (!nonModifiers.includes(e.key)) {
      capturedCode = e.code;
    }

    if (capturedCode) {
      confirmCapture();
    }
  }

  function onCaptureKeyup(_e: KeyboardEvent) {
    // Wait for a non-modifier key
  }

  async function confirmCapture() {
    const parts: string[] = [];
    for (const mod of ['Control', 'Alt', 'Shift', 'Super']) {
      if (capturedModifiers.has(mod)) parts.push(mod);
    }
    parts.push(capturedCode);
    const newHotkey = parts.join('+');

    try {
      await updateHotkey(newHotkey);
      setHotkey(newHotkey);
    } catch (e) {
      console.error('Failed to update hotkey:', e);
    }

    cancelCapture();
  }

  // Build a temporary hotkey string for capture preview
  let capturePreviewHotkey = $derived.by(() => {
    const parts: string[] = [];
    for (const mod of ['Control', 'Alt', 'Shift', 'Super']) {
      if (capturedModifiers.has(mod)) parts.push(mod);
    }
    if (capturedCode) parts.push(capturedCode);
    return parts.join('+');
  });

  // ── Edit hotkey capture ──

  let isEditCapturing = $state(false);
  let editCapturedModifiers = $state(new Set<string>());
  let editCapturedCode = $state('');

  function startEditCapture() {
    isEditCapturing = true;
    editCapturedModifiers = new Set();
    editCapturedCode = '';
    document.addEventListener('keydown', onEditCaptureKeydown);
    document.addEventListener('keyup', onEditCaptureKeyup);
  }

  function cancelEditCapture() {
    isEditCapturing = false;
    editCapturedModifiers = new Set();
    editCapturedCode = '';
    document.removeEventListener('keydown', onEditCaptureKeydown);
    document.removeEventListener('keyup', onEditCaptureKeyup);
  }

  function onEditCaptureKeydown(e: KeyboardEvent) {
    e.preventDefault();
    e.stopPropagation();

    if (e.key === 'Escape') {
      cancelEditCapture();
      return;
    }

    const mods = new Set<string>();
    if (e.altKey) mods.add('Alt');
    if (e.ctrlKey) mods.add('Control');
    if (e.shiftKey) mods.add('Shift');
    if (e.metaKey) mods.add('Super');
    editCapturedModifiers = mods;

    const nonModifiers = ['Alt', 'Control', 'Shift', 'Meta'];
    if (!nonModifiers.includes(e.key)) {
      editCapturedCode = e.code;
    }

    if (editCapturedCode) {
      confirmEditCapture();
    }
  }

  function onEditCaptureKeyup(_e: KeyboardEvent) {
    // Wait for a non-modifier key
  }

  async function confirmEditCapture() {
    const parts: string[] = [];
    for (const mod of ['Control', 'Alt', 'Shift', 'Super']) {
      if (editCapturedModifiers.has(mod)) parts.push(mod);
    }
    parts.push(editCapturedCode);
    const newEditHotkey = parts.join('+');

    // Must differ from primary hotkey
    if (newEditHotkey === getHotkey()) {
      console.warn('Edit hotkey must differ from primary hotkey');
      cancelEditCapture();
      return;
    }

    try {
      await updateEditHotkey(newEditHotkey);
      setEditHotkey(newEditHotkey);
    } catch (e) {
      console.error('Failed to update edit hotkey:', e);
    }

    cancelEditCapture();
  }

  let editCapturePreviewHotkey = $derived.by(() => {
    const parts: string[] = [];
    for (const mod of ['Control', 'Alt', 'Shift', 'Super']) {
      if (editCapturedModifiers.has(mod)) parts.push(mod);
    }
    if (editCapturedCode) parts.push(editCapturedCode);
    return parts.join('+');
  });

  async function clearEditHotkey() {
    try {
      await updateEditHotkey('');
      setEditHotkey(null);
    } catch (e) {
      console.error('Failed to clear edit hotkey:', e);
    }
  }

  // ── Meeting hotkey capture ──

  let isMeetingCapturing = $state(false);
  let meetingCapturedModifiers = $state(new Set<string>());
  let meetingCapturedCode = $state('');

  function startMeetingCapture() {
    isMeetingCapturing = true;
    meetingCapturedModifiers = new Set();
    meetingCapturedCode = '';
    document.addEventListener('keydown', onMeetingCaptureKeydown);
    document.addEventListener('keyup', onMeetingCaptureKeyup);
  }

  function cancelMeetingCapture() {
    isMeetingCapturing = false;
    meetingCapturedModifiers = new Set();
    meetingCapturedCode = '';
    document.removeEventListener('keydown', onMeetingCaptureKeydown);
    document.removeEventListener('keyup', onMeetingCaptureKeyup);
  }

  function onMeetingCaptureKeydown(e: KeyboardEvent) {
    e.preventDefault();
    e.stopPropagation();

    if (e.key === 'Escape') {
      cancelMeetingCapture();
      return;
    }

    const mods = new Set<string>();
    if (e.altKey) mods.add('Alt');
    if (e.ctrlKey) mods.add('Control');
    if (e.shiftKey) mods.add('Shift');
    if (e.metaKey) mods.add('Super');
    meetingCapturedModifiers = mods;

    const nonModifiers = ['Alt', 'Control', 'Shift', 'Meta'];
    if (!nonModifiers.includes(e.key)) {
      meetingCapturedCode = e.code;
    }

    if (meetingCapturedCode) {
      confirmMeetingCapture();
    }
  }

  function onMeetingCaptureKeyup(_e: KeyboardEvent) {
    // Wait for a non-modifier key
  }

  async function confirmMeetingCapture() {
    const parts: string[] = [];
    for (const mod of ['Control', 'Alt', 'Shift', 'Super']) {
      if (meetingCapturedModifiers.has(mod)) parts.push(mod);
    }
    parts.push(meetingCapturedCode);
    const newMeetingHotkey = parts.join('+');

    // Require at least one modifier to avoid swallowing bare keypresses globally.
    if (meetingCapturedModifiers.size === 0) {
      console.warn('Meeting hotkey must include at least one modifier key');
      cancelMeetingCapture();
      return;
    }

    // Must differ from primary and edit hotkeys
    if (newMeetingHotkey === getHotkey() || newMeetingHotkey === getEditHotkey()) {
      console.warn('Meeting hotkey must differ from primary and edit hotkeys');
      cancelMeetingCapture();
      return;
    }

    try {
      await updateMeetingHotkey(newMeetingHotkey);
      setMeetingHotkey(newMeetingHotkey);
    } catch (e) {
      console.error('Failed to update meeting hotkey:', e);
    }

    cancelMeetingCapture();
  }

  let meetingCapturePreviewHotkey = $derived.by(() => {
    const parts: string[] = [];
    for (const mod of ['Control', 'Alt', 'Shift', 'Super']) {
      if (meetingCapturedModifiers.has(mod)) parts.push(mod);
    }
    if (meetingCapturedCode) parts.push(meetingCapturedCode);
    return parts.join('+');
  });

  async function clearMeetingHotkey() {
    try {
      await updateMeetingHotkey(null);
      setMeetingHotkey(null);
    } catch (e) {
      console.error('Failed to clear meeting hotkey:', e);
    }
  }

  // Cleanup on destroy
  onDestroy(() => {
    if (isCapturing) cancelCapture();
    if (isEditCapturing) cancelEditCapture();
    if (isMeetingCapturing) cancelMeetingCapture();
  });
</script>

<div class="section">
  <SectionHeader title={t('settings.shortcuts')}>
    {#snippet icon()}
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <rect x="2" y="4" width="20" height="16" rx="2"/>
        <path d="M6 8h.01M10 8h.01M14 8h.01M18 8h.01M6 12h.01M10 12h.01M14 12h.01M18 12h.01M8 16h8"/>
      </svg>
    {/snippet}
  </SectionHeader>

  <!-- Primary hotkey -->
  <div class="edit-hotkey-info">
    <div class="edit-hotkey-name">{t('settings.shortcuts.hotkey')}</div>
    <div class="edit-hotkey-desc">{t('settings.shortcuts.hotkeyDesc')}</div>
  </div>

  {#if !isCapturing}
    <div class="hotkey-row">
      <Keycaps hotkey={getHotkey()} />
      <button class="hotkey-btn" onclick={startCapture}>{t('settings.shortcuts.change')}</button>
    </div>
  {:else}
    <div class="hotkey-capture active">
      <div class="capture-label">{t('settings.shortcuts.captureLabel')}</div>
      <div class="capture-preview">
        {#if capturePreviewHotkey}
          <Keycaps hotkey={capturePreviewHotkey} />
        {/if}
      </div>
      <div class="capture-hint">{t('settings.shortcuts.captureHint', { modifiers: modifierHint })}</div>
      <div class="capture-actions">
        <button class="btn-cancel" onclick={cancelCapture}>{t('settings.shortcuts.cancel')}</button>
      </div>
    </div>
  {/if}

  <!-- Edit by voice hotkey -->
  <div class="edit-hotkey-section">
    <div class="edit-hotkey-info">
      <div class="edit-hotkey-name">{t('settings.shortcuts.editHotkey')}</div>
      <div class="edit-hotkey-desc">{t('settings.shortcuts.editHotkeyDesc')}</div>
      {#if !getPolishConfig().enabled}
        <div class="edit-hotkey-hint">{t('settings.shortcuts.editRequiresPolish')}</div>
      {/if}
    </div>

    {#if !isEditCapturing}
      <div class="hotkey-row">
        {#if getEditHotkey()}
          <Keycaps hotkey={getEditHotkey()!} />
        {:else}
          <span class="not-set">{t('settings.shortcuts.notSet')}</span>
        {/if}
        <button class="hotkey-btn" onclick={startEditCapture}>{t('settings.shortcuts.change')}</button>
      </div>
    {:else}
      <div class="hotkey-capture active">
        <div class="capture-label">{t('settings.shortcuts.captureLabel')}</div>
        <div class="capture-preview">
          {#if editCapturePreviewHotkey}
            <Keycaps hotkey={editCapturePreviewHotkey} />
          {/if}
        </div>
        <div class="capture-hint">{t('settings.shortcuts.captureHint', { modifiers: modifierHint })}</div>
        <div class="capture-actions">
          <button class="btn-cancel" onclick={cancelEditCapture}>{t('settings.shortcuts.cancel')}</button>
        </div>
      </div>
    {/if}
  </div>

  <!-- Meeting mode hotkey -->
  <div class="edit-hotkey-section">
    <div class="edit-hotkey-info">
      <div class="edit-hotkey-name">{t('settings.shortcuts.meetingHotkey')}</div>
      <div class="edit-hotkey-desc">{t('settings.shortcuts.meetingHotkeyDesc')}</div>
    </div>

    {#if !isMeetingCapturing}
      <div class="hotkey-row">
        {#if getMeetingHotkey()}
          <Keycaps hotkey={getMeetingHotkey()!} />
        {:else}
          <span class="not-set">{t('settings.shortcuts.meetingNotSet')}</span>
        {/if}
        <div style="display: flex; gap: 8px; align-items: center;">
          {#if getMeetingHotkey()}
            <button class="hotkey-btn" onclick={clearMeetingHotkey}>{t('settings.shortcuts.clear')}</button>
          {/if}
          <button class="hotkey-btn" onclick={startMeetingCapture}>{t('settings.shortcuts.change')}</button>
        </div>
      </div>
    {:else}
      <div class="hotkey-capture active">
        <div class="capture-label">{t('settings.shortcuts.captureLabel')}</div>
        <div class="capture-preview">
          {#if meetingCapturePreviewHotkey}
            <Keycaps hotkey={meetingCapturePreviewHotkey} />
          {/if}
        </div>
        <div class="capture-hint">{t('settings.shortcuts.captureHint', { modifiers: modifierHint })}</div>
        <div class="capture-actions">
          <button class="btn-cancel" onclick={cancelMeetingCapture}>{t('settings.shortcuts.cancel')}</button>
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .section {
    margin-bottom: 32px;
  }


  .hotkey-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .hotkey-btn {
    -webkit-app-region: no-drag;
    app-region: no-drag;
    padding: 7px 16px;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    background: var(--bg-primary);
    color: var(--text-secondary);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
    white-space: nowrap;
  }

  .hotkey-btn:hover {
    background: var(--bg-sidebar);
    color: var(--text-primary);
    border-color: rgba(0, 0, 0, 0.15);
  }

  .hotkey-capture {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 14px;
    padding: 16px 0 4px;
  }

  .capture-label {
    font-size: 14px;
    font-weight: 500;
    color: var(--accent-blue);
    animation: captureGlow 2s ease-in-out infinite;
  }

  @keyframes captureGlow {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .capture-preview {
    display: flex;
    align-items: center;
    gap: 6px;
    min-height: 34px;
  }

  .capture-hint {
    font-size: 12px;
    color: var(--text-tertiary);
  }

  .capture-actions {
    display: flex;
    gap: 8px;
  }

  .capture-actions button {
    -webkit-app-region: no-drag;
    app-region: no-drag;
    padding: 6px 14px;
    border-radius: var(--radius-sm);
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
    border: none;
  }

  .btn-cancel {
    background: var(--bg-hover);
    color: var(--text-secondary);
  }

  .btn-cancel:hover {
    background: var(--bg-active);
    color: var(--text-primary);
  }

  .edit-hotkey-section {
    margin-top: 16px;
  }

  .edit-hotkey-info {
    margin-bottom: 8px;
  }

  .edit-hotkey-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .edit-hotkey-desc {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 2px;
  }

  .edit-hotkey-hint {
    font-size: 11px;
    color: #c87800;
    margin-top: 4px;
  }

  .not-set {
    font-size: 13px;
    color: var(--text-tertiary);
  }
</style>
