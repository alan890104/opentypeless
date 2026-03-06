<script lang="ts">
  import { onMount } from 'svelte';
  import type { PromptRule } from '$lib/types';
  import { t, getLocale } from '$lib/stores/i18n.svelte';
  import {
    getCurrentRules,
    setCurrentRules,
    savePolish,
    getPolishConfig,
  } from '$lib/stores/settings.svelte';
  import { setCurrentPage, setHighlightSection, showConfirm } from '$lib/stores/ui.svelte';
  import { getDefaultPromptRules } from '$lib/api';
  import RuleGridCard from '../components/RuleGridCard.svelte';
  import RuleEditorModal from '../components/RuleEditorModal.svelte';

  const polishEnabled = $derived(getPolishConfig().enabled);

  function goToPolishSettings() {
    setHighlightSection('polish');
    setCurrentPage('settings');
  }

  let editorVisible = $state(false);
  let editingIndex = $state(-1);

  const rules = $derived(getCurrentRules());

  // Build sorted rules with original indices
  const sortedRules = $derived.by(() => {
    const r = rules;
    if (!r || r.length === 0) return [];
    return r
      .map((rule, i) => ({ rule, origIndex: i }))
      .sort((a, b) => {
        const nameA = (a.rule.name || '').toLowerCase();
        const nameB = (b.rule.name || '').toLowerCase();
        return nameA.localeCompare(nameB);
      });
  });

  onMount(async () => {
    await initDefaultRules();
  });

  async function initDefaultRules() {
    try {
      const defaults = await getDefaultPromptRules(getLocale());
      if (!defaults || defaults.length === 0) return;

      const currentRules = getCurrentRules();
      if (!currentRules || currentRules.length === 0) {
        setCurrentRules(defaults);
        await savePolish();
        return;
      }

      // Merge missing defaults by match_type + match_value (including alt_matches)
      const existingKeys = new Set<string>();
      for (const r of currentRules) {
        existingKeys.add(`${r.match_type}::${r.match_value}`);
        for (const alt of r.alt_matches || []) {
          existingKeys.add(`${alt.match_type}::${alt.match_value}`);
        }
      }
      const missing = defaults.filter((d) => {
        // Check if primary or any alt match already exists
        if (existingKeys.has(`${d.match_type}::${d.match_value}`)) return false;
        for (const alt of d.alt_matches || []) {
          if (existingKeys.has(`${alt.match_type}::${alt.match_value}`)) return false;
        }
        return true;
      });
      if (missing.length > 0) {
        setCurrentRules([...currentRules, ...missing]);
        await savePolish();
      }
    } catch (e) {
      console.error('Failed to load default prompt rules:', e);
    }
  }

  function openEditor(index: number) {
    editingIndex = index;
    editorVisible = true;
  }

  function closeEditor() {
    editorVisible = false;
    editingIndex = -1;
  }

  async function handleSave(rule: PromptRule) {
    const currentRules = getCurrentRules().slice();
    if (editingIndex >= 0) {
      currentRules[editingIndex] = rule;
    } else {
      currentRules.push(rule);
    }
    setCurrentRules(currentRules);
    closeEditor();
    await savePolish();
  }

  async function handleDelete(index: number) {
    const currentRules = getCurrentRules().slice();
    currentRules.splice(index, 1);
    setCurrentRules(currentRules);
    await savePolish();
  }

  async function handleToggle(index: number) {
    const currentRules = getCurrentRules().slice();
    currentRules[index] = { ...currentRules[index], enabled: !currentRules[index].enabled };
    setCurrentRules(currentRules);
    await savePolish();
  }

  function handleResetToDefaults() {
    showConfirm(
      t('promptRules.resetDefaultsTitle'),
      t('promptRules.resetDefaultsMessage'),
      t('confirm.reset'),
      async () => {
        try {
          const defaults = await getDefaultPromptRules(getLocale());
          setCurrentRules(defaults);
          await savePolish();
        } catch (e) {
          console.error('Failed to reset prompt rules:', e);
        }
      },
    );
  }
</script>

<div class="page">
  {#if !polishEnabled}
    <div class="polish-disabled-banner">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      <span class="polish-disabled-text">{t('promptRules.polishDisabled')}</span>
      <button class="polish-disabled-btn" onclick={goToPolishSettings}>{t('promptRules.enablePolish')}</button>
    </div>
  {/if}

  <div class:page-disabled={!polishEnabled}>
    <div class="page-header">
      <h1 class="page-title">{t('promptRules.title')}</h1>
      <button class="reset-defaults-btn" onclick={handleResetToDefaults}>{t('promptRules.resetDefaults')}</button>
    </div>
    <div class="prompt-rules-desc">{t('promptRules.desc')}</div>

    <!-- Rule Grid -->
    <div class="prompt-rules-grid">
      {#if sortedRules.length === 0}
        <div class="prompt-rules-empty">{t('settings.polish.noRules')}</div>
      {:else}
        {#each sortedRules as { rule, origIndex } (origIndex)}
          <RuleGridCard
            {rule}
            index={origIndex}
            onEdit={() => openEditor(origIndex)}
            onDelete={() => handleDelete(origIndex)}
            onToggle={() => handleToggle(origIndex)}
          />
        {/each}
      {/if}
      <!-- Add rule tile -->
      <!-- svelte-ignore a11y_click_events_have_key_events -->
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div class="grid-tile-add" onclick={() => openEditor(-1)}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
        </svg>
        <span>{t('settings.polish.addRule')}</span>
      </div>
    </div>
  </div>
</div>

<!-- Rule Editor Modal -->
<RuleEditorModal
  visible={editorVisible}
  editIndex={editingIndex}
  onclose={closeEditor}
  onsave={handleSave}
/>

<style>
  .polish-disabled-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    margin-bottom: 16px;
    border-radius: var(--radius-md);
    background: rgba(251, 191, 36, 0.08);
    border: 1px solid rgba(251, 191, 36, 0.25);
    color: var(--text-secondary);
  }

  .polish-disabled-banner svg {
    flex-shrink: 0;
    color: rgb(217, 164, 6);
  }

  .polish-disabled-text {
    flex: 1;
    font-size: 12.5px;
    line-height: 1.4;
  }

  .polish-disabled-btn {
    flex-shrink: 0;
    padding: 5px 12px;
    border-radius: 6px;
    border: none;
    background: var(--accent-blue);
    color: #fff;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s ease;

  }

  .polish-disabled-btn:hover {
    opacity: 0.85;
  }

  .page-disabled {
    opacity: 0.35;
    pointer-events: none;
  }

  .page-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 28px;
  }

  .page-title {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text-primary);
    margin-bottom: 0;
  }

  .reset-defaults-btn {
    padding: 5px 12px;
    border-radius: 6px;
    border: 1px solid var(--border-subtle);
    background: transparent;
    color: var(--text-secondary);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;

  }

  .reset-defaults-btn:hover {
    background: rgba(239, 68, 68, 0.08);
    border-color: rgba(239, 68, 68, 0.3);
    color: rgb(239, 68, 68);
  }

  .prompt-rules-desc {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 16px;
  }

  .prompt-rules-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
    gap: 10px;
    align-items: start;
  }

  .prompt-rules-empty {
    grid-column: 1 / -1;
    font-size: 12px;
    color: var(--text-tertiary);
    text-align: center;
    padding: 20px 12px;
    background: var(--bg-sidebar);
    border-radius: var(--radius-md);
  }

  .grid-tile-add {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 14px 8px;
    border-radius: var(--radius-md);
    background: var(--bg-sidebar);
    cursor: pointer;
    transition: all 0.15s ease;
    border: 2px dashed var(--border-subtle);
    min-height: 90px;
    color: var(--text-tertiary);

  }

  .grid-tile-add:hover {
    background: var(--bg-hover);
    border-color: var(--accent-blue);
    color: var(--accent-blue);
  }

  .grid-tile-add span {
    font-size: 11px;
    font-weight: 500;
  }
</style>
