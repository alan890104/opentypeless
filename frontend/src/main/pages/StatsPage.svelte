<script lang="ts">
  import { onMount } from 'svelte';
  import { t } from '$lib/stores/i18n.svelte';
  import { getHotkey } from '$lib/stores/settings.svelte';
  import { getHistoryStats } from '$lib/api';
  import Keycaps from '$lib/components/Keycaps.svelte';
  import type { HistoryStats } from '$lib/types';

  let stats = $state<HistoryStats | null>(null);
  let showBars = $state(false);

  // ── Benchmarks ──

  interface Benchmark {
    key: string;
    emoji: string;
    speed: number;
  }

  const BENCHMARKS: Benchmark[] = [
    { key: 'handwriting', emoji: '\u270D\uFE0F', speed: 15 },
    { key: 'typing',      emoji: '\u2328\uFE0F', speed: 40 },
    { key: 'speech',      emoji: '\uD83D\uDDE3\uFE0F', speed: 220 },
    { key: 'audiobook',   emoji: '\uD83C\uDFA7', speed: 250 },
    { key: 'auctioneer',  emoji: '\uD83D\uDD28', speed: 350 },
    { key: 'eminem',      emoji: '\uD83C\uDFA4', speed: 450 },
  ];

  interface BarEntry {
    label: string;
    emoji: string;
    speed: number;
    isUser: boolean;
  }

  function getVisibleBenchmarks(userSpeed: number): BarEntry[] {
    // All benchmarks below user + first one above
    const below = BENCHMARKS.filter(b => b.speed <= userSpeed);
    const above = BENCHMARKS.filter(b => b.speed > userSpeed);
    const visible = [...below];
    if (above.length > 0) visible.push(above[0]);

    // Insert user row in sorted position
    const entries: BarEntry[] = visible.map(b => ({
      label: t(`stats.bench.${b.key}`),
      emoji: b.emoji,
      speed: b.speed,
      isUser: false,
    }));

    const userEntry: BarEntry = {
      label: t('stats.bench.you'),
      emoji: '\uD83C\uDF99\uFE0F',
      speed: userSpeed,
      isUser: true,
    };

    // Find insertion index (after all entries with speed <= userSpeed)
    let idx = entries.findIndex(e => e.speed > userSpeed);
    if (idx === -1) idx = entries.length;
    entries.splice(idx, 0, userEntry);

    return entries;
  }

  // ── Fun facts ──

  function pickFunFact(): string {
    if (!stats || stats.total_entries === 0) return t('stats.fact.keepGoing');

    const speed = stats.total_words / stats.total_duration_secs * 60;
    const typingTime = stats.total_words / 40 * 60;
    const saved = Math.max(0, typingTime - stats.total_duration_secs);

    interface Fact { text: string; eligible: boolean }
    const candidates: Fact[] = [
      {
        text: t('stats.fact.fasterThanTyping', { multiplier: (speed / 40).toFixed(1) }),
        eligible: speed > 40,
      },
      {
        text: t('stats.fact.fasterThanSpeech'),
        eligible: speed > 220,
      },
      {
        text: t('stats.fact.tweetCount', { count: Math.round(stats.total_words / 50) }),
        eligible: stats.total_words >= 50,
      },
      {
        text: t('stats.fact.novelProgress', { percent: (stats.total_words / 80000 * 100).toFixed(1) }),
        eligible: stats.total_words >= 100,
      },
      {
        text: t('stats.fact.coffeeBrews', { count: Math.round(saved / 240) }),
        eligible: saved >= 240,
      },
      {
        text: t('stats.fact.animeEpisodes', { count: Math.round(saved / 1320) }),
        eligible: saved >= 1320,
      },
      {
        text: t('stats.fact.streak', { count: stats.total_entries }),
        eligible: stats.total_entries >= 10,
      },
    ];

    const eligible = candidates.filter(f => f.eligible);
    if (eligible.length === 0) return t('stats.fact.keepGoing');

    // Day-of-epoch seed for daily rotation
    const dayIndex = Math.floor(Date.now() / 86400000);
    return eligible[dayIndex % eligible.length].text;
  }

  // ── Stats helpers ──

  onMount(async () => {
    try {
      stats = await getHistoryStats();
    } catch (e) {
      console.error('Failed to load stats:', e);
      stats = { total_entries: 0, total_duration_secs: 0, total_chars: 0, local_entries: 0, local_duration_secs: 0, total_words: 0, local_polish_entries: 0, local_polish_input_chars: 0, local_polish_output_chars: 0 };
    }
    // Trigger bar animation after mount + paint
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        showBars = true;
      });
    });
  });

  interface DurationParts {
    segments: { value: string; unit: string }[];
  }

  function durationParts(secs: number): DurationParts {
    if (secs < 60) {
      return { segments: [{ value: String(Math.round(secs)), unit: t('stats.unitSec') }] };
    }
    if (secs < 3600) {
      return { segments: [{ value: String(Math.round(secs / 60)), unit: t('stats.unitMin') }] };
    }
    const h = Math.floor(secs / 3600);
    const m = Math.round((secs % 3600) / 60);
    return {
      segments: [
        { value: String(h), unit: t('stats.unitHr') },
        { value: String(m), unit: t('stats.unitMin') },
      ],
    };
  }

  function formatWords(n: number): { value: string; unit: string } {
    if (n >= 1_000_000) return { value: (n / 1_000_000).toFixed(1), unit: `M ${t('stats.unitChars')}` };
    if (n >= 1_000) return { value: (n / 1_000).toFixed(1), unit: `K ${t('stats.unitChars')}` };
    return { value: String(n), unit: t('stats.unitChars') };
  }

  let avgSpeed = $derived.by(() => {
    if (!stats || stats.total_duration_secs === 0) return 0;
    return Math.round(stats.total_words / stats.total_duration_secs * 60);
  });

  let timeSaved = $derived.by(() => {
    if (!stats || stats.total_words === 0) return 0;
    const typingTime = stats.total_words / 40 * 60;
    return Math.max(0, typingTime - stats.total_duration_secs);
  });

  let moneySaved = $derived.by(() => {
    if (!stats) return '$0.00';
    // STT: Groq Whisper API rate $0.006/min
    const sttDollars = stats.local_duration_secs / 60 * 0.006;
    // LLM cost approximation: GPT-4o rates ($2.50/1M input + $10/1M output) used as
    // a conservative reference regardless of the actual provider/model configured.
    // ~4 chars/token; +200 tokens per call for system prompt overhead.
    const inputTokens = stats.local_polish_input_chars / 4 + stats.local_polish_entries * 200;
    const outputTokens = stats.local_polish_output_chars / 4;
    const llmDollars = inputTokens * 2.5 / 1_000_000 + outputTokens * 10 / 1_000_000;
    const dollars = sttDollars + llmDollars;
    if (dollars < 0.01) return '$0.00';
    return `$${dollars.toFixed(2)}`;
  });

  let durationDisplay = $derived(stats ? durationParts(stats.total_duration_secs) : null);
  let charsDisplay = $derived(stats ? formatWords(stats.total_words) : null);
  let timeSavedDisplay = $derived(durationParts(timeSaved));

  let visibleBars = $derived(getVisibleBenchmarks(avgSpeed));
  let funFact = $derived(pickFunFact());
  let maxBarSpeed = $derived(Math.max(...visibleBars.map(b => b.speed), 1));
</script>

<div class="page">
  <!-- Hero -->
  <div class="hero">
    <h1 class="hero-title">{t('stats.heroTitle')}</h1>
    <p class="hero-subtitle">
      {t('stats.heroSubtitle')}
      <span class="hero-keycaps"><Keycaps hotkey={getHotkey()} /></span>
      {t('stats.heroSubtitleAfter')}
    </p>
  </div>

  {#if stats === null}
    <div class="loading"></div>
  {:else if stats.total_entries === 0}
    <div class="empty-state">
      <img class="empty-img" src="girl-suming.png" alt="" />
      <div class="empty-title">{t('stats.emptyTitle')}</div>
      <div class="empty-hint">{t('stats.emptyHint')}</div>
    </div>
  {:else}
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-row">
          <span class="stat-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
          </span>
          <span class="stat-number">
            <span class="num">{moneySaved}</span>
          </span>
        </div>
        <div class="stat-label">{t('stats.moneySaved')}</div>
      </div>

      <div class="stat-card">
        <div class="stat-row">
          <span class="stat-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
          </span>
          <span class="stat-number">
            {#if durationDisplay}
              {#each durationDisplay.segments as seg}
                <span class="num">{seg.value}</span> <span class="unit">{seg.unit}</span>{' '}
              {/each}
            {/if}
          </span>
        </div>
        <div class="stat-label">{t('stats.totalDuration')}</div>
      </div>

      <div class="stat-card">
        <div class="stat-row">
          <span class="stat-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
          </span>
          <span class="stat-number">
            {#if charsDisplay}
              <span class="num">{charsDisplay.value}</span><span class="unit">{charsDisplay.unit}</span>
            {/if}
          </span>
        </div>
        <div class="stat-label">{t('stats.totalChars')}</div>
      </div>

      <div class="stat-card">
        <div class="stat-row">
          <span class="stat-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
          </span>
          <span class="stat-number">
            {#each timeSavedDisplay.segments as seg}
              <span class="num">{seg.value}</span> <span class="unit">{seg.unit}</span>{' '}
            {/each}
          </span>
        </div>
        <div class="stat-label">{t('stats.timeSaved')}</div>
      </div>

      <div class="stat-card">
        <div class="stat-row">
          <span class="stat-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
          </span>
          <span class="stat-number">
            <span class="num">{avgSpeed}</span> <span class="unit">{t('stats.speedUnit')}</span>
          </span>
        </div>
        <div class="stat-label">{t('stats.avgSpeed')}</div>
      </div>

      <div class="stat-card">
        <div class="stat-row">
          <span class="stat-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M12 6v6l4 2"/></svg>
          </span>
          <span class="stat-number">
            <span class="num">{stats.total_entries}</span> <span class="unit">{t('stats.unitTimes')}</span>
          </span>
        </div>
        <div class="stat-label">{t('stats.totalEntries')}</div>
      </div>
    </div>

    <!-- Speed Chart -->
    <div class="speed-chart">
      <div class="speed-chart-title">{t('stats.speedChart')}</div>
      <div class="bars-container">
        {#each visibleBars as bar, i}
          <div class="bar-row">
            <div class="bar-label" class:bar-label-user={bar.isUser}>
              <span class="bar-emoji">{bar.emoji}</span>
              <span class="bar-name">{bar.label}</span>
            </div>
            <div class="bar-track">
              <div
                class="bar-fill"
                class:bar-fill-user={bar.isUser}
                style="width: {showBars ? (bar.speed / maxBarSpeed * 100) : 0}%; transition-delay: {i * 80}ms"
              ></div>
            </div>
            <div class="bar-speed" class:bar-speed-user={bar.isUser}>{bar.speed}</div>
          </div>
        {/each}
      </div>

      <div class="chart-divider"></div>
      <div class="fun-fact">{funFact}</div>
    </div>
  {/if}
</div>

<style>
  .page {
    display: flex;
    flex-direction: column;
    gap: 28px;
    min-height: calc(100vh - 56px - 44px);
  }

  /* -- Hero -- */
  .hero-title {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--text-primary);
    line-height: 1.3;
    margin-bottom: 8px;
  }

  .hero-subtitle {
    font-size: 14px;
    color: var(--text-tertiary);
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    line-height: 1.6;
  }

  .hero-keycaps {
    display: inline-flex;
  }

  /* -- Stats grid -- */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
  }

  .stat-card {
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg, 12px);
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 4px;
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
  }

  .stat-card:hover {
    border-color: rgba(0, 0, 0, 0.12);
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
  }

  .stat-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .stat-icon {
    color: var(--text-tertiary);
    width: 20px;
    height: 20px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .stat-icon :global(svg) {
    width: 20px;
    height: 20px;
  }

  .stat-number {
    display: flex;
    align-items: baseline;
    gap: 3px;
    flex-wrap: wrap;
  }

  .stat-number .num {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--text-primary);
    line-height: 1;
  }

  .stat-number .unit {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    margin-right: 2px;
  }

  .stat-label {
    font-size: 12px;
    color: var(--text-tertiary);
    font-weight: 500;
    padding-left: 28px;
  }

  /* -- Speed Chart -- */
  .speed-chart {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    padding: 20px 24px;
    background: var(--bg-sidebar);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg, 12px);
  }

  .speed-chart-title {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-tertiary);
    margin-bottom: 16px;
  }

  .bars-container {
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex: 1;
    justify-content: center;
  }

  .bar-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .bar-label {
    width: 120px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: var(--text-secondary);
    text-align: right;
    justify-content: flex-end;
  }

  .bar-label-user {
    color: #5b9cf5;
    font-weight: 700;
  }

  .bar-emoji {
    font-size: 15px;
    line-height: 1;
  }

  .bar-name {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .bar-track {
    flex: 1;
    height: 22px;
    background: var(--bg-primary);
    border-radius: 6px;
    overflow: hidden;
  }

  .bar-fill {
    height: 100%;
    border-radius: 6px;
    background: var(--text-tertiary);
    opacity: 0.3;
    width: 0%;
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
  }

  .bar-fill-user {
    background: #5b9cf5;
    opacity: 1;
  }

  .bar-speed {
    width: 40px;
    flex-shrink: 0;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-tertiary);
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  .bar-speed-user {
    color: #5b9cf5;
  }

  .chart-divider {
    height: 1px;
    background: var(--border-subtle);
    margin: 16px 0;
  }

  .fun-fact {
    font-size: 14px;
    color: var(--text-secondary);
    text-align: center;
    line-height: 1.5;
    padding: 0 12px;
  }

  /* -- Empty state -- */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    flex: 1;
    gap: 6px;
  }

  .empty-img {
    width: 220px;
    height: 220px;
    object-fit: contain;
    margin-bottom: 8px;
  }

  .empty-title {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-secondary);
  }

  .empty-hint {
    font-size: 13px;
    color: var(--text-tertiary);
  }

  .loading {
    display: flex;
    justify-content: center;
    flex: 1;
    align-items: center;
  }

  @media (max-width: 900px) {
    .stats-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>
