<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import type { PromptRule, MatchType, MatchCondition } from '$lib/types';
  import { t } from '$lib/stores/i18n.svelte';
  import { getHotkey } from '$lib/stores/settings.svelte';
  import { DEFAULT_HOTKEY, formatHotkeyDisplay, RULE_ICON_SVG, ICON_PICKER_LIST, detectRuleIconKey } from '$lib/constants';
  import {
    setVoiceRuleMode,
    generateRuleFromDescription,
    getDefaultPromptRules,
    onVoiceRuleStatus,
    onVoiceRuleLevels,
    onVoiceRuleTranscript,
  } from '$lib/api';
  import type { UnlistenFn } from '@tauri-apps/api/event';
  import { getCurrentRules } from '$lib/stores/settings.svelte';
  import { getLocale } from '$lib/stores/i18n.svelte';

  let {
    visible,
    editIndex,
    onclose,
    onsave,
  }: {
    visible: boolean;
    editIndex: number;
    onclose: () => void;
    onsave: (rule: PromptRule) => void;
  } = $props();

  // Form fields
  let name = $state('');
  let matchType = $state<MatchType>('app_name');
  let matchValue = $state('');
  let prompt = $state('');
  let iconKey = $state<string | undefined>(undefined);
  let showIconPicker = $state(false);
  let altMatches = $state<MatchCondition[]>([]);

  // Voice rule state
  type VoiceState = 'idle' | 'recording' | 'processing';
  let voiceState = $state<VoiceState>('idle');
  let processingText = $state('');
  let voiceModeActive = $state(false);
  let unlisteners: UnlistenFn[] = [];
  let voiceLevels = $state(new Float32Array(20));
  let canvasEl: HTMLCanvasElement | undefined = $state();
  let waveAnimId: number | null = null;

  // Default rule matching current profile (for reset)
  let matchingDefault = $state<PromptRule | null>(null);

  // Refs for validation focus
  let matchValueInput: HTMLInputElement | undefined = $state();
  let promptTextarea: HTMLTextAreaElement | undefined = $state();

  const title = $derived(
    editIndex >= 0 ? t('settings.polish.editRule') : t('settings.polish.addRule')
  );

  const hotkey = $derived(getHotkey() || DEFAULT_HOTKEY);
  const hotkeyDisplay = $derived(formatHotkeyDisplay(hotkey));

  // Populate form when modal opens
  $effect(() => {
    if (visible) {
      matchingDefault = null;
      if (editIndex >= 0) {
        const rules = getCurrentRules();
        const rule = rules[editIndex];
        if (rule) {
          name = rule.name || '';
          matchType = rule.match_type || 'app_name';
          matchValue = rule.match_value || '';
          prompt = rule.prompt || '';
          iconKey = rule.icon || undefined;
          altMatches = (rule.alt_matches || []).map((a) => ({ ...a }));
          findMatchingDefault(rule);
        }
      } else {
        name = '';
        matchType = 'app_name';
        matchValue = '';
        prompt = '';
        iconKey = undefined;
        altMatches = [];
      }
      showIconPicker = false;
      voiceState = 'idle';
      enableVoiceMode();
    } else {
      disableVoiceMode();
    }
  });

  async function findMatchingDefault(rule: PromptRule) {
    try {
      const defaults = await getDefaultPromptRules(getLocale());
      const keys = new Set<string>();
      keys.add(`${rule.match_type}::${rule.match_value}`);
      for (const alt of rule.alt_matches || []) {
        keys.add(`${alt.match_type}::${alt.match_value}`);
      }
      matchingDefault = defaults.find((d) => {
        if (keys.has(`${d.match_type}::${d.match_value}`)) return true;
        for (const alt of d.alt_matches || []) {
          if (keys.has(`${alt.match_type}::${alt.match_value}`)) return true;
        }
        return false;
      }) ?? null;
    } catch {
      // ignore
    }
  }

  function resetToDefault() {
    if (!matchingDefault) return;
    name = matchingDefault.name || '';
    matchType = matchingDefault.match_type || 'app_name';
    matchValue = matchingDefault.match_value || '';
    prompt = matchingDefault.prompt || '';
    iconKey = matchingDefault.icon || undefined;
    altMatches = (matchingDefault.alt_matches || []).map((a) => ({ ...a }));
  }

  async function enableVoiceMode() {
    if (voiceModeActive) return;
    voiceModeActive = true;

    try {
      await setVoiceRuleMode(true);
    } catch (e) {
      console.error('Failed to enable voice rule mode:', e);
    }

    unlisteners.push(
      await onVoiceRuleStatus((status) => {
        if (!voiceModeActive) return;
        if (status === 'recording') {
          voiceState = 'recording';
        } else if (status === 'transcribing') {
          voiceState = 'processing';
          processingText = t('promptRules.voiceTranscribing');
        }
      })
    );

    unlisteners.push(
      await onVoiceRuleLevels((levels) => {
        if (voiceModeActive && levels) {
          voiceLevels = new Float32Array(levels);
        }
      })
    );

    unlisteners.push(
      await onVoiceRuleTranscript(async (transcript) => {
        if (!transcript || !transcript.trim()) {
          if (voiceModeActive) voiceState = 'idle';
          return;
        }

        voiceState = 'processing';
        processingText = t('promptRules.voiceGenerating');

        try {
          const rule = await generateRuleFromDescription(transcript.trim());
          if (editIndex >= 0) {
            // Editing: only update the prompt, keep name/match/altMatches
            prompt = rule.prompt || '';
          } else {
            fillFields(rule);
          }
        } catch (e) {
          console.error('[Sumi] Voice rule generation failed:', e);
        }

        if (voiceModeActive) voiceState = 'idle';
      })
    );
  }

  async function disableVoiceMode() {
    if (!voiceModeActive) return;
    voiceModeActive = false;
    stopWaveform();

    try {
      await setVoiceRuleMode(false);
    } catch {
      // ignore
    }

    for (const unlisten of unlisteners) unlisten();
    unlisteners = [];
  }

  function fillFields(data: { name: string; match_type: string; match_value: string; prompt: string }) {
    name = data.name || '';
    matchType = (data.match_type as MatchType) || 'app_name';
    matchValue = data.match_value || '';
    prompt = data.prompt || '';
    altMatches = [];
  }

  // Waveform drawing
  $effect(() => {
    if (voiceState === 'recording') {
      startWaveform();
    } else {
      stopWaveform();
    }
  });

  function startWaveform() {
    if (!canvasEl) return;
    const wrap = canvasEl.parentElement;
    if (!wrap) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = wrap.getBoundingClientRect();
    canvasEl.width = Math.round(rect.width) * dpr;
    canvasEl.height = Math.round(rect.height) * dpr;
    canvasEl.style.width = rect.width + 'px';
    canvasEl.style.height = rect.height + 'px';
    const ctx = canvasEl.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const barWidth = 3;
    const barGap = 2;
    const numBars = Math.floor((w + barGap) / (barWidth + barGap));
    const maxBarH = h * 0.8;

    function draw() {
      waveAnimId = requestAnimationFrame(draw);
      ctx!.clearRect(0, 0, w, h);
      const totalW = numBars * barWidth + (numBars - 1) * barGap;
      const startX = (w - totalW) / 2;
      const cy = h / 2;

      for (let i = 0; i < numBars; i++) {
        const srcIdx = Math.floor((i * voiceLevels.length) / numBars);
        const level = voiceLevels[srcIdx] || 0;
        const bh = Math.max(3, level * maxBarH);
        const x = startX + i * (barWidth + barGap);
        const y = cy - bh / 2;
        ctx!.fillStyle = level > 0.05 ? '#007AFF' : '#e0e0e0';
        ctx!.beginPath();
        ctx!.roundRect(x, y, barWidth, bh, barWidth / 2);
        ctx!.fill();
      }
    }
    draw();
  }

  function stopWaveform() {
    if (waveAnimId !== null) {
      cancelAnimationFrame(waveAnimId);
      waveAnimId = null;
    }
    voiceLevels = new Float32Array(20);
  }

  function addAltMatch() {
    altMatches = [...altMatches, { match_type: 'app_name', match_value: '' }];
  }

  function removeAltMatch(index: number) {
    altMatches = altMatches.filter((_, i) => i !== index);
  }

  function handleSave() {
    if (!matchValue.trim()) {
      matchValueInput?.focus();
      return;
    }
    if (!prompt.trim()) {
      promptTextarea?.focus();
      return;
    }

    // Filter out alt matches with empty values
    const filteredAltMatches = altMatches.filter((a) => a.match_value.trim());

    const rule: PromptRule = {
      name: name.trim() || t('settings.polish.untitledRule'),
      match_type: matchType,
      match_value: matchValue.trim(),
      prompt: prompt.trim(),
      enabled: true,
      icon: iconKey,
      alt_matches: filteredAltMatches.length > 0 ? filteredAltMatches : undefined,
    };

    // Preserve enabled state when editing
    if (editIndex >= 0) {
      const existingRules = getCurrentRules();
      if (existingRules[editIndex]) {
        rule.enabled = existingRules[editIndex].enabled;
      }
    }

    onsave(rule);
  }

  function handleBackdropClick() {
    onclose();
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      onclose();
    }
  }

  onDestroy(() => {
    disableVoiceMode();
  });
</script>

<svelte:window onkeydown={visible ? handleKeydown : undefined} />

{#if visible}
  <div class="rule-editor-overlay">
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div class="rule-editor-backdrop" onclick={handleBackdropClick}></div>
    <div class="rule-editor-card">
      <div class="rule-editor-title">{title}</div>

      <!-- Icon Picker -->
      <!-- svelte-ignore a11y_click_events_have_key_events -->
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div class="icon-picker-section">
        <div class="icon-picker-current" onclick={() => showIconPicker = !showIconPicker}>
          <span class="icon-picker-preview">
            {@html iconKey && RULE_ICON_SVG[iconKey]
              ? RULE_ICON_SVG[iconKey]
              : (detectRuleIconKey({ name, match_value: matchValue })
                  ? RULE_ICON_SVG[detectRuleIconKey({ name, match_value: matchValue })!]
                  : '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M4 4h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zm0 2v12h16V6H4zm1 1h5v2H5V7z"/></svg>')}
          </span>
          <span class="icon-picker-label">{t('promptRules.changeIcon')}</span>
          <svg class="icon-picker-chevron" class:open={showIconPicker} width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5l10 7-10 7z"/></svg>
        </div>
        {#if showIconPicker}
          <div class="icon-picker-grid">
            <button
              class="icon-picker-item"
              class:selected={!iconKey}
              onclick={() => { iconKey = undefined; showIconPicker = false; }}
              title={t('promptRules.autoIcon')}
            >
              <span class="icon-picker-item-icon">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
              </span>
              <span class="icon-picker-item-label">{t('promptRules.autoIcon')}</span>
            </button>
            {#each ICON_PICKER_LIST as item}
              <button
                class="icon-picker-item"
                class:selected={iconKey === item.key}
                onclick={() => { iconKey = item.key; showIconPicker = false; }}
                title={item.labelKey ? t(item.labelKey) : item.label}
              >
                <span class="icon-picker-item-icon">{@html RULE_ICON_SVG[item.key]}</span>
                <span class="icon-picker-item-label">{item.labelKey ? t(item.labelKey) : item.label}</span>
              </button>
            {/each}
          </div>
        {/if}
      </div>

      <!-- Voice Rule Panel -->
      <div class="voice-rule-panel">
        {#if voiceState === 'idle'}
          <div class="voice-rule-idle">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="11" rx="3"/><path d="M5 10a7 7 0 0 0 14 0"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
            <div class="voice-rule-idle-text">
              <span>{@html t('promptRules.voiceTipMain').replace('{hotkey}', `<kbd>${hotkeyDisplay}</kbd>`)}</span>
              <span class="voice-rule-example">{t('promptRules.voiceTipExample')}</span>
            </div>
          </div>
        {/if}

        {#if voiceState === 'recording'}
          <div class="voice-rule-recording active">
            <div class="voice-rule-canvas-wrap">
              <canvas bind:this={canvasEl}></canvas>
            </div>
            <div class="voice-rule-rec-hint">
              {@html t('promptRules.voiceRecHint').replace('{hotkey}', `<kbd>${hotkeyDisplay}</kbd>`)}
            </div>
            <span class="voice-rule-example">{t('promptRules.voiceTipExample')}</span>
          </div>
        {/if}

        {#if voiceState === 'processing'}
          <div class="voice-rule-processing active">
            <div class="voice-rule-spinner"></div>
            <span class="voice-rule-processing-text">{processingText}</span>
          </div>
        {/if}
      </div>

      <!-- Form Fields -->
      <div class="rule-editor-field">
        <div class="rule-editor-label">{t('settings.polish.ruleName')}</div>
        <input
          type="text"
          class="rule-editor-input"
          bind:value={name}
          placeholder={t('settings.polish.ruleNamePlaceholder')}
        />
      </div>

      <!-- Match Conditions -->
      <div class="rule-editor-field">
        <div class="rule-editor-label">{t('promptRules.matchConditions')}</div>
        <div class="rule-editor-hint" style="margin-bottom: 8px; margin-top: 0;">{t('promptRules.altMatchHint')}</div>

        <!-- Primary condition -->
        <div class="match-condition-row">
          <select class="match-condition-select" bind:value={matchType}>
            <option value="app_name">{t('settings.polish.matchAppName')}</option>
            <option value="bundle_id">{t('settings.polish.matchBundleId')}</option>
            <option value="url">{t('settings.polish.matchUrl')}</option>
          </select>
          <input
            type="text"
            class="match-condition-input"
            bind:this={matchValueInput}
            bind:value={matchValue}
            placeholder={t('settings.polish.ruleMatchValuePlaceholder')}
          />
        </div>

        <!-- Alt conditions -->
        {#each altMatches as alt, i}
          <div class="match-condition-or">{t('promptRules.or')}</div>
          <div class="match-condition-row">
            <select class="match-condition-select" bind:value={alt.match_type}>
              <option value="app_name">{t('settings.polish.matchAppName')}</option>
              <option value="bundle_id">{t('settings.polish.matchBundleId')}</option>
              <option value="url">{t('settings.polish.matchUrl')}</option>
            </select>
            <input
              type="text"
              class="match-condition-input"
              bind:value={alt.match_value}
              placeholder={t('settings.polish.ruleMatchValuePlaceholder')}
            />
            <button class="match-condition-remove" onclick={() => removeAltMatch(i)} title="Remove">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          </div>
        {/each}

        <button class="match-condition-add" onclick={addAltMatch}>
          {t('promptRules.addAltMatch')}
        </button>
      </div>

      <div class="rule-editor-field">
        <div class="rule-editor-label">{t('settings.polish.rulePrompt')}</div>
        <textarea
          class="rule-editor-textarea"
          bind:this={promptTextarea}
          bind:value={prompt}
          placeholder={t('settings.polish.rulePromptPlaceholder')}
        ></textarea>
        <div class="rule-editor-hint">{t('settings.polish.rulePromptHint')}</div>
      </div>

      <div class="rule-editor-actions">
        {#if matchingDefault}
          <button class="rule-editor-reset" onclick={resetToDefault}>{t('promptRules.resetOne')}</button>
        {/if}
        <div class="rule-editor-actions-right">
          <button class="rule-editor-cancel" onclick={onclose}>{t('settings.polish.ruleCancel')}</button>
          <button class="rule-editor-save" onclick={handleSave}>{t('settings.polish.ruleSave')}</button>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .rule-editor-overlay {
    position: fixed;
    inset: 0;
    z-index: 2000;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .rule-editor-backdrop {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
  }

  .rule-editor-card {
    position: relative;
    width: 420px;
    max-height: 80vh;
    overflow-y: auto;
    background: var(--bg-primary);
    border-radius: var(--radius-lg);
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12), 0 0 0 1px var(--border-subtle);
    animation: ruleEditorFadeIn 0.15s ease;
  }

  @keyframes ruleEditorFadeIn {
    from {
      opacity: 0;
      transform: scale(0.96);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }

  .rule-editor-title {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 16px;
  }

  .rule-editor-field {
    margin-bottom: 14px;
  }

  .rule-editor-label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 6px;
  }

  .rule-editor-input,
  .match-condition-select,
  .match-condition-input {
    height: 34px;
    padding: 0 10px;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    line-height: 34px;
    outline: none;
    transition: border-color 0.15s ease;
    box-sizing: border-box;
    -webkit-appearance: none;
    appearance: none;
  }

  .rule-editor-input {
    width: 100%;
  }

  .rule-editor-input:focus,
  .match-condition-select:focus,
  .match-condition-input:focus {
    border-color: var(--accent-blue);
  }

  .rule-editor-textarea {
    width: 100%;
    min-height: 80px;
    padding: 8px 10px;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    outline: none;
    resize: vertical;
    transition: border-color 0.15s ease;
    box-sizing: border-box;
  }

  .rule-editor-textarea:focus {
    border-color: var(--accent-blue);
  }

  .rule-editor-hint {
    font-size: 11px;
    color: var(--text-tertiary);
    margin-top: 4px;
  }

  .rule-editor-actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 18px;
  }

  .rule-editor-actions-right {
    display: flex;
    gap: 8px;
    margin-left: auto;
  }

  .rule-editor-actions button,
  .rule-editor-actions-right button {
    padding: 7px 18px;
    border-radius: var(--radius-sm);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
    border: none;
  }

  .rule-editor-reset {
    padding: 7px 14px;
    background: transparent;
    color: var(--text-tertiary);
    font-size: 12px;
  }

  .rule-editor-reset:hover {
    color: var(--accent-blue);
  }

  .rule-editor-cancel {
    background: var(--bg-hover);
    color: var(--text-secondary);
  }

  .rule-editor-cancel:hover {
    background: var(--bg-active);
    color: var(--text-primary);
  }

  .rule-editor-save {
    background: var(--accent-blue);
    color: #fff;
  }

  .rule-editor-save:hover {
    filter: brightness(1.1);
  }

  /* ── Voice Rule Panel ── */
  .voice-rule-panel {
    margin-bottom: 16px;
    overflow: hidden;
    transition: all 0.3s ease;
  }

  .voice-rule-idle {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 18px;
    background: #f0f7ff;
    border-radius: var(--radius-md);
  }

  .voice-rule-idle svg {
    flex-shrink: 0;
  }

  .voice-rule-idle-text {
    font-size: 13px;
    color: var(--text-primary);
    line-height: 1.5;
  }

  .voice-rule-idle-text :global(kbd) {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 4px;
    background: #fff;
    border: 1px solid var(--border-subtle);
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
    box-shadow: 0 1px 1px rgba(0, 0, 0, 0.06);
  }

  .voice-rule-example {
    display: block;
    margin-top: 6px;
    font-size: 11px;
    color: var(--text-tertiary);
    border-left: 2px solid var(--accent-blue);
    padding-left: 10px;
    line-height: 1.5;
  }

  .voice-rule-recording {
    display: none;
    flex-direction: column;
    align-items: stretch;
    padding: 16px;
    background: #f0f7ff;
    border-radius: var(--radius-md);
    gap: 12px;
  }

  .voice-rule-recording.active {
    display: flex;
  }

  .voice-rule-canvas-wrap {
    width: 100%;
    height: 64px;
    border-radius: var(--radius-sm);
    overflow: hidden;
  }

  .voice-rule-canvas-wrap canvas {
    display: block;
    width: 100%;
    height: 100%;
  }

  .voice-rule-rec-hint {
    font-size: 13px;
    color: var(--text-secondary);
    text-align: center;
    line-height: 1.4;
  }

  .voice-rule-rec-hint :global(kbd) {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 4px;
    background: #fff;
    border: 1px solid var(--border-subtle);
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
    box-shadow: 0 1px 1px rgba(0, 0, 0, 0.06);
  }

  .voice-rule-processing {
    display: none;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 18px;
    background: #f0f7ff;
    border-radius: var(--radius-md);
  }

  .voice-rule-processing.active {
    display: flex;
  }

  .voice-rule-spinner {
    width: 16px;
    height: 16px;
    border: 2px solid rgba(0, 122, 255, 0.2);
    border-top-color: var(--accent-blue);
    border-radius: 50%;
    animation: voiceRuleSpin 0.6s linear infinite;
  }

  @keyframes voiceRuleSpin {
    to {
      transform: rotate(360deg);
    }
  }

  .voice-rule-processing-text {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
  }

  /* ── Match Conditions ── */
  .match-condition-row {
    display: flex;
    gap: 6px;
    margin-bottom: 4px;
    align-items: center;
  }

  .match-condition-select {
    width: 120px;
    flex-shrink: 0;
  }

  .match-condition-input {
    flex: 1;
    min-width: 0;
  }

  .match-condition-remove {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border: none;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--text-tertiary);
    cursor: pointer;
    transition: all 0.12s ease;
    padding: 0;

  }

  .match-condition-remove:hover {
    background: rgba(239, 68, 68, 0.1);
    color: rgb(239, 68, 68);
  }

  .match-condition-or {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-tertiary);
    text-transform: uppercase;
    padding: 2px 0;
    margin-left: 4px;
  }

  .match-condition-add {
    display: inline-block;
    margin-top: 4px;
    padding: 4px 0;
    border: none;
    background: transparent;
    color: var(--accent-blue);
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.12s ease;

  }

  .match-condition-add:hover {
    opacity: 0.7;
  }

  /* ── Icon Picker ── */
  .icon-picker-section {
    margin-bottom: 14px;
  }

  .icon-picker-current {
    display: flex;
    align-items: center;
    gap: 8px;
    height: 34px;
    padding: 0 10px;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .icon-picker-current:hover {
    border-color: var(--accent-blue);
  }

  .icon-picker-preview {
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .icon-picker-preview :global(svg) {
    width: 20px;
    height: 20px;
  }

  .icon-picker-label {
    flex: 1;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
  }

  .icon-picker-chevron {
    color: var(--text-tertiary);
    transition: transform 0.15s ease;
    flex-shrink: 0;
  }

  .icon-picker-chevron.open {
    transform: rotate(90deg);
  }

  .icon-picker-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(64px, 1fr));
    gap: 4px;
    margin-top: 8px;
    padding: 8px;
    background: var(--bg-sidebar);
    border-radius: var(--radius-sm);
    max-height: 180px;
    overflow-y: auto;
  }

  .icon-picker-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 8px 4px;
    border: 2px solid transparent;
    border-radius: var(--radius-sm);
    background: transparent;
    cursor: pointer;
    transition: all 0.12s ease;
    font-family: 'Inter', sans-serif;
  }

  .icon-picker-item:hover {
    background: var(--bg-hover);
  }

  .icon-picker-item.selected {
    border-color: var(--accent-blue);
    background: rgba(0, 122, 255, 0.06);
  }

  .icon-picker-item-icon {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .icon-picker-item-icon :global(svg) {
    width: 24px;
    height: 24px;
  }

  .icon-picker-item-label {
    font-size: 9px;
    color: var(--text-tertiary);
    text-align: center;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
  }
</style>
