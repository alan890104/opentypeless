<script lang="ts">
  import { onDestroy } from 'svelte';
  import { t } from '$lib/stores/i18n.svelte';
  import { getHighlightSection, setHighlightSection } from '$lib/stores/ui.svelte';
  import LanguageSection from '../components/settings/LanguageSection.svelte';
  import HotkeySection from '../components/settings/HotkeySection.svelte';
  import BehaviorSection from '../components/settings/BehaviorSection.svelte';
  import MicSection from '../components/settings/MicSection.svelte';
  import SttSection from '../components/settings/SttSection.svelte';
  import PolishSection from '../components/settings/PolishSection.svelte';
  import StorageSection from '../components/settings/StorageSection.svelte';
  import DangerZone from '../components/settings/DangerZone.svelte';

  let polishSectionEl = $state<HTMLElement | null>(null);
  let highlightTimer: ReturnType<typeof setTimeout> | null = null;

  $effect(() => {
    if (getHighlightSection() === 'polish' && polishSectionEl) {
      // Delay scroll to let the page fully render and layout after navigation
      highlightTimer = setTimeout(() => {
        polishSectionEl?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        setTimeout(() => {
          setHighlightSection(null);
        }, 2000);
      }, 150);
    }
  });

  onDestroy(() => {
    if (highlightTimer) clearTimeout(highlightTimer);
    if (getHighlightSection()) setHighlightSection(null);
  });
</script>

<div class="page">
  <h1 class="page-title">{t('settings.title')}</h1>

  <LanguageSection />
  <div class="section-divider"></div>

  <HotkeySection />
  <div class="section-divider"></div>

  <BehaviorSection />
  <div class="section-divider"></div>

  <MicSection />
  <div class="section-divider"></div>

  <SttSection />
  <div class="section-divider"></div>

  <div
    id="polish-section"
    bind:this={polishSectionEl}
    class:highlight={getHighlightSection() === 'polish'}
  >
    <PolishSection />
  </div>
  <div class="section-divider"></div>

  <StorageSection />
  <div class="section-divider"></div>

  <DangerZone />
</div>

<style>
  .page-title {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text-primary);
    margin-bottom: 28px;
  }

  .section-divider {
    height: 1px;
    background: var(--border-divider);
    margin-bottom: 32px;
  }

  #polish-section {
    border-radius: var(--radius-lg);
  }

  #polish-section.highlight {
    animation: polish-fade 2s ease-out forwards;
  }

  @keyframes polish-fade {
    0% {
      background: color-mix(in srgb, var(--accent-blue) 8%, transparent);
    }
    100% {
      background: transparent;
    }
  }
</style>
