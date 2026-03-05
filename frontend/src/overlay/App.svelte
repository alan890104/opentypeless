<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { t, initLocale } from '$lib/stores/i18n.svelte';
  import {
    onRecordingStatus,
    onRecordingMaxDuration,
    onAudioLevels,
    onModelSwitching,
    onTranscriptionPartial,
    triggerUndo,
    getSettings,
  } from '$lib/api';
  import { getCurrentWindow } from '@tauri-apps/api/window';
  import type { UnlistenFn } from '@tauri-apps/api/event';
  import type { OverlayStatus } from '$lib/types';

  // ── Constants ──
  const NUM_BARS = 12;
  const BAR_W = 2;
  const BAR_GAP = 2;
  const CW = NUM_BARS * (BAR_W + BAR_GAP) - BAR_GAP; // 46
  const CH = 32;
  const UNDO_DURATION = 5000;
  const TIMER_INTERVAL = 200;
  const INTERPOLATION_FACTOR = 0.25;

  // ── State ──
  type Phase =
    | 'preparing'
    | 'recording'
    | 'edit_recording'
    | 'meeting_recording'
    | 'meeting_stopped'
    | 'processing'
    | 'transcribing'
    | 'polishing'
    | 'pasted'
    | 'copied'
    | 'error'
    | 'edited'
    | 'edit_requires_polish'
    | 'undo'
    | 'switching';

  let phase: Phase = $state('preparing');
  let timerText: string = $state('0:00');
  let recProgress: number = $state(0);
  let maxDuration: number = $state(30);
  let undoAnimating: boolean = $state(false);
  let partialText: string = $state('');

  // ── Canvas & waveform ──
  let canvasEl: HTMLCanvasElement | undefined = $state();
  let waveCtx: CanvasRenderingContext2D | null = null;
  let currentLevels = new Array(NUM_BARS).fill(0);
  let targetLevels = new Array(NUM_BARS).fill(0);
  let waveAnimId: number | null = null;

  // ── Timers ──
  let startTime = 0;
  let timerInterval: ReturnType<typeof setInterval> | null = null;
  let undoTimeout: ReturnType<typeof setTimeout> | null = null;
  let editedTimeout: ReturnType<typeof setTimeout> | null = null;
  let switchingTimeout: ReturnType<typeof setTimeout> | null = null;

  // ── Undo bar element for reflow trick ──
  let undoBarEl: HTMLDivElement | undefined = $state();

  // ── Event unlisteners ──
  let unlisteners: UnlistenFn[] = [];

  // ── Capsule class computation ──
  let capsuleClass: string = $derived.by(() => {
    switch (phase) {
      case 'preparing':
        return 'capsule preparing';
      case 'recording':
        return partialText.length > 0 ? 'capsule recording has-partial' : 'capsule recording';
      case 'edit_recording':
        return 'capsule edit-recording';
      case 'meeting_recording':
        return 'capsule meeting-recording';
      case 'meeting_stopped':
        return 'capsule result success';
      case 'processing':
        return 'capsule processing';
      case 'transcribing':
        return 'capsule transcribing';
      case 'polishing':
        return 'capsule polishing';
      case 'pasted':
      case 'copied':
      case 'edited':
        return 'capsule result success';
      case 'error':
      case 'edit_requires_polish':
        return 'capsule result error-state';
      case 'undo':
        return 'capsule undo-state';
      case 'switching':
        return 'capsule switching';
      default:
        return 'capsule';
    }
  });

  // ── Label text computation ──
  let labelText: string = $derived.by(() => {
    switch (phase) {
      case 'preparing':
        return t('overlay.preparing');
      case 'recording':
        return t('overlay.recording');
      case 'edit_recording':
        return t('overlay.editRecording');
      case 'meeting_recording':
        return t('overlay.meetingRecording');
      case 'meeting_stopped':
        return t('overlay.meetingStopped');
      case 'processing':
      case 'transcribing':
        return t('overlay.transcribing');
      case 'polishing':
        return t('overlay.polishing');
      case 'pasted':
        return t('overlay.pasted');
      case 'copied':
        return t('overlay.copied');
      case 'error':
        return t('overlay.failed');
      case 'edit_requires_polish':
        return t('overlay.editRequiresPolish');
      case 'edited':
        return t('overlay.edited');
      case 'undo':
        return t('overlay.undo');
      case 'switching':
        return t('overlay.modelSwitching');
      default:
        return '';
    }
  });

  // ── Icon state derivations ──
  // Use helper to avoid TS narrowing issues with union types in $derived
  function is(...phases: Phase[]): boolean {
    return phases.includes(phase);
  }

  let showDot: boolean = $derived.by(() => false); // dot is never shown in practice (CSS handles it on .recording)
  let showSpinner: boolean = $derived.by(() => is('preparing', 'processing', 'transcribing', 'polishing', 'switching'));
  let showWaveform: boolean = $derived.by(() => is('recording', 'edit_recording', 'meeting_recording'));
  let showIconResult: boolean = $derived.by(() => is('pasted', 'copied', 'error', 'edit_requires_polish', 'edited', 'meeting_stopped'));
  let showTimer: boolean = $derived.by(() => is('recording', 'edit_recording', 'meeting_recording'));
  let showUndoIcon: boolean = $derived.by(() => is('undo'));
  let showUndoBar: boolean = $derived.by(() => is('undo'));
  let isCheckIcon: boolean = $derived.by(() => is('pasted', 'copied', 'edited', 'meeting_stopped'));
  let isErrorIcon: boolean = $derived.by(() => is('error', 'edit_requires_polish'));
  let isPolishSpinner: boolean = $derived.by(() => is('polishing'));
  let isSwitchingSpinner: boolean = $derived.by(() => is('switching'));

  // ── Partial text display (live preview during Qwen3-ASR recording) ──
  let showingPartial: boolean = $derived.by(() => is('recording') && partialText.length > 0);
  let displayLabelText: string = $derived(showingPartial ? partialText : labelText);

  // ── Waveform animation ──
  function animateWaveform() {
    if (!waveCtx) return;
    waveCtx.clearRect(0, 0, CW, CH);
    waveCtx.fillStyle = 'rgba(255, 255, 255, 0.85)';
    for (let i = 0; i < NUM_BARS; i++) {
      currentLevels[i] += (targetLevels[i] - currentLevels[i]) * INTERPOLATION_FACTOR;
      const h = Math.max(3, currentLevels[i] * CH);
      const x = i * (BAR_W + BAR_GAP);
      const y = (CH - h) / 2;
      waveCtx.fillRect(x, y, BAR_W, h);
    }
    waveAnimId = requestAnimationFrame(animateWaveform);
  }

  function startWaveform() {
    currentLevels.fill(0);
    targetLevels.fill(0);
    if (!waveAnimId) animateWaveform();
  }

  function stopWaveform() {
    if (waveAnimId) {
      cancelAnimationFrame(waveAnimId);
      waveAnimId = null;
    }
    // Reset context so it gets re-initialized when canvas is re-rendered
    waveCtx = null;
  }

  // ── Timer ──
  function formatElapsed(elapsed: number): string {
    if (elapsed >= 3600) {
      const h = Math.floor(elapsed / 3600);
      const m = Math.floor((elapsed % 3600) / 60);
      const s = elapsed % 60;
      return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    const mins = Math.floor(elapsed / 60);
    const secs = elapsed % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  function startTimer() {
    startTime = Date.now();
    timerText = '0:00';
    recProgress = 0;
    timerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      timerText = formatElapsed(elapsed);
      // maxDuration=0 means unlimited — keep progress at 0 to avoid the color gradient shifting
      recProgress = maxDuration > 0 ? Math.min(elapsed / maxDuration, 1) : 0;
    }, TIMER_INTERVAL);
  }

  function stopTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
  }

  // ── Undo ──
  function clearUndoTimeout() {
    if (undoTimeout) {
      clearTimeout(undoTimeout);
      undoTimeout = null;
    }
  }

  function clearEditedTimeout() {
    if (editedTimeout) {
      clearTimeout(editedTimeout);
      editedTimeout = null;
    }
  }

  function clearSwitchingTimeout() {
    if (switchingTimeout) {
      clearTimeout(switchingTimeout);
      switchingTimeout = null;
    }
  }

  // ── State transitions ──
  function clearCommon() {
    clearUndoTimeout();
    clearEditedTimeout();
    clearSwitchingTimeout();
    stopTimer();
    stopWaveform();
    undoAnimating = false;
    partialText = '';
  }

  function setPreparing() {
    clearCommon();
    phase = 'preparing';
  }

  function setRecording() {
    clearCommon();
    phase = 'recording';
    startTimer();
    startWaveform();
  }

  function setEditRecording() {
    clearCommon();
    phase = 'edit_recording';
    startTimer();
    startWaveform();
  }

  function setMeetingRecording() {
    clearCommon();
    phase = 'meeting_recording';
    startTimer();
    startWaveform();
  }

  function setMeetingStopped() {
    clearCommon();
    phase = 'meeting_stopped';
  }

  function setProcessing() {
    clearCommon();
    phase = 'processing';
  }

  function setTranscribing() {
    clearCommon();
    phase = 'transcribing';
  }

  function setPolishing() {
    clearCommon();
    phase = 'polishing';
  }

  function setPasted() {
    clearCommon();
    phase = 'pasted';
  }

  function setCopied() {
    clearCommon();
    phase = 'copied';
  }

  function setError() {
    clearCommon();
    phase = 'error';
  }

  function setEditRequiresPolish() {
    clearCommon();
    phase = 'edit_requires_polish';
  }

  function setEdited() {
    clearCommon();
    phase = 'edited';
    // After 500ms, transition to undo state
    editedTimeout = setTimeout(() => setUndo(), 500);
  }

  function setUndo() {
    phase = 'undo';

    // Trigger undo bar animation after a tick so the element is rendered
    requestAnimationFrame(() => {
      if (undoBarEl) {
        undoBarEl.style.animation = 'none';
        // Force reflow
        void undoBarEl.offsetWidth;
        undoBarEl.style.animation = `undoCountdown ${UNDO_DURATION}ms linear forwards`;
      }
      undoAnimating = true;
    });

    // Auto-clear after 5s
    undoTimeout = setTimeout(() => {
      undoAnimating = false;
    }, UNDO_DURATION);
  }

  // ── Undo ──
  async function handleUndoClick() {
    if (phase !== 'undo') return;
    clearUndoTimeout();
    undoAnimating = false;
    phase = 'preparing'; // Reset visual while undo processes

    try {
      await triggerUndo();
    } catch (e) {
      console.error('Undo failed:', e);
    }
  }

  function handleCapsuleClick() {
    if (phase === 'undo') {
      handleUndoClick();
    }
  }

  // ── Handle recording-status event ──
  function handleStatus(status: string) {
    switch (status as OverlayStatus) {
      case 'preparing':
        setPreparing();
        break;
      case 'recording':
        setRecording();
        break;
      case 'edit_recording':
        setEditRecording();
        break;
      case 'meeting_recording':
        setMeetingRecording();
        break;
      case 'meeting_stopped':
        setMeetingStopped();
        break;
      case 'processing':
        setProcessing();
        break;
      case 'transcribing':
        setTranscribing();
        break;
      case 'polishing':
        setPolishing();
        break;
      case 'pasted':
        setPasted();
        break;
      case 'copied':
        setCopied();
        break;
      case 'error':
        setError();
        break;
      case 'edited':
        setEdited();
        break;
      case 'edit_requires_polish':
        setEditRequiresPolish();
        break;
    }
  }

  // ── Canvas setup (runs when canvasEl is bound) ──
  $effect(() => {
    if (canvasEl && !waveCtx) {
      const dpr = window.devicePixelRatio || 1;
      canvasEl.width = CW * dpr;
      canvasEl.height = CH * dpr;
      canvasEl.style.width = CW + 'px';
      canvasEl.style.height = CH + 'px';
      const ctx = canvasEl.getContext('2d');
      if (ctx) {
        ctx.scale(dpr, dpr);
        waveCtx = ctx;
        // Canvas just became available — start animation if we're already recording
        if ((phase === 'recording' || phase === 'edit_recording' || phase === 'meeting_recording') && !waveAnimId) {
          animateWaveform();
        }
      }
    }
  });

  // ── Window visibility ──
  // When the backend hides the overlay (e.g. after "no speech detected"),
  // no status event is emitted. Reset state so the capsule doesn't show
  // stale text (like "transcribing") when next shown.
  function handleVisibility() {
    if (document.hidden) {
      setPreparing();
    }
  }

  // ── Lifecycle ──
  onMount(async () => {
    // Init i18n from backend settings
    try {
      const s = await getSettings();
      await initLocale(s.language);
    } catch {
      await initLocale('en');
    }

    // Set initial state
    setPreparing();

    // Reset state when window is hidden by backend
    document.addEventListener('visibilitychange', handleVisibility);

    // Listen for Tauri events
    const u1 = await onRecordingStatus(handleStatus);
    const u2 = await onRecordingMaxDuration((secs) => {
      maxDuration = secs;
    });
    const u3 = await onAudioLevels((levels) => {
      targetLevels = levels;
    });
    const u4 = await onModelSwitching((p) => {
      if (p.status === 'start') {
        clearCommon();
        phase = 'switching';
        // Safety net: if backend crashes before emitting "done", reset after 30s
        switchingTimeout = setTimeout(() => {
          switchingTimeout = null;
          if (phase === 'switching') setPreparing();
        }, 30_000);
      } else if (p.status === 'done') {
        // Reset directly — do not rely on visibilitychange, which is not
        // triggered by alpha-zero hiding (alpha=0 ≠ document.hidden in WKWebView).
        clearSwitchingTimeout();
        setPreparing();
      }
    });
    const u5 = await onTranscriptionPartial((payload) => {
      if (phase === 'recording') {
        partialText = payload.text;
      }
    });
    unlisteners = [u1, u2, u3, u4, u5];
  });

  onDestroy(() => {
    document.removeEventListener('visibilitychange', handleVisibility);
    stopTimer();
    stopWaveform();
    clearUndoTimeout();
    clearEditedTimeout();
    clearSwitchingTimeout();
    for (const unlisten of unlisteners) {
      unlisten();
    }
  });
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class={capsuleClass}
  style:--rec-progress={recProgress}
  onclick={handleCapsuleClick}
>
  <!-- Recording dot (only visible in default/idle states via CSS) -->
  {#if showDot}
    <div class="dot"></div>
  {/if}

  <!-- Spinner -->
  {#if showSpinner}
    <div class="spinner" class:polish-spinner={isPolishSpinner} class:switching-spinner={isSwitchingSpinner}></div>
  {/if}

  <!-- Waveform canvas -->
  {#if showWaveform}
    <canvas
      class="waveform"
      bind:this={canvasEl}
      width={CW}
      height={CH}
    ></canvas>
  {/if}

  <!-- Result icons -->
  {#if showIconResult}
    <div class="icon-result">
      {#if isCheckIcon}
        <div class="icon-check"></div>
      {:else if isErrorIcon}
        <div class="icon-error"></div>
      {/if}
    </div>
  {/if}

  <!-- Undo icon -->
  {#if showUndoIcon}
    <svg class="icon-undo" viewBox="0 0 24 24" fill="none" stroke="rgba(255, 149, 0, 0.9)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="1 4 1 10 7 10"></polyline>
      <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
    </svg>
  {/if}

  <!-- Label -->
  <span class="label" class:partial-label={showingPartial}>{displayLabelText}</span>

  <!-- Timer -->
  {#if showTimer}
    <span class="timer">{timerText}</span>
  {/if}

  <!-- Undo countdown bar -->
  {#if showUndoBar}
    <div class="undo-bar" bind:this={undoBarEl}></div>
  {/if}
</div>

<style>
  :global(*),
  :global(*::before),
  :global(*::after) {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  :global(html),
  :global(body) {
    background: transparent;
    overflow: hidden;
    height: 100%;
    -webkit-user-select: none;
    user-select: none;
  }

  :global(body) {
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  /* ── Capsule ── */
  .capsule {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 0 18px;
    width: 200px;
    height: 40px;
    border-radius: 100px;
    background: rgb(28, 28, 32);
    -webkit-mask-image: -webkit-radial-gradient(white, black);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow:
      0 8px 32px rgba(0, 0, 0, 0.45),
      0 0 0 0.5px rgba(255, 255, 255, 0.06),
      inset 0 0.5px 0 rgba(255, 255, 255, 0.06);
    color: rgba(255, 255, 255, 0.92);
    animation: fadeIn 0.25s ease-out;
    transition: width 0.25s ease;
  }

  .capsule.has-partial {
    width: 280px;
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(6px) scale(0.96);
    }
    to {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
  }

  /* ── Recording dot ── */
  .dot {
    position: relative;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #ff3b30;
    flex-shrink: 0;
    animation: dotPulse 1.8s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  }

  .dot::after {
    content: '';
    position: absolute;
    inset: -4px;
    border-radius: 50%;
    background: rgba(255, 59, 48, 0.25);
    animation: ringPulse 1.8s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  }

  @keyframes dotPulse {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0.55;
    }
  }

  @keyframes ringPulse {
    0%,
    100% {
      transform: scale(1);
      opacity: 0.4;
    }
    50% {
      transform: scale(1.5);
      opacity: 0;
    }
  }

  /* ── Spinner ── */
  .spinner {
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255, 255, 255, 0.12);
    border-top-color: #a78bfa;
    border-radius: 50%;
    flex-shrink: 0;
    animation: spin 0.7s linear infinite;
  }

  .spinner.polish-spinner {
    border-top-color: #c084fc;
  }

  .spinner.switching-spinner {
    border-top-color: #9ca3af;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  /* ── Result icon ── */
  .icon-result {
    width: 16px;
    height: 16px;
    flex-shrink: 0;
    display: flex;
    position: relative;
  }

  /* Checkmark */
  .icon-check {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #34c759;
    position: relative;
  }

  .icon-check::after {
    content: '';
    position: absolute;
    top: 3.5px;
    left: 5.5px;
    width: 4px;
    height: 7px;
    border: solid #fff;
    border-width: 0 1.8px 1.8px 0;
    transform: rotate(45deg);
  }

  /* Error X */
  .icon-error {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #ff3b30;
    position: relative;
  }

  .icon-error::before,
  .icon-error::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 8px;
    height: 1.8px;
    background: #fff;
    border-radius: 1px;
  }

  .icon-error::before {
    transform: translate(-50%, -50%) rotate(45deg);
  }

  .icon-error::after {
    transform: translate(-50%, -50%) rotate(-45deg);
  }

  /* ── Undo icon ── */
  .icon-undo {
    width: 14px;
    height: 14px;
    flex-shrink: 0;
  }

  /* ── Undo countdown bar ── */
  .undo-bar {
    position: absolute;
    bottom: 0;
    left: 16px;
    right: 16px;
    height: 2px;
    border-radius: 1px;
    background: rgba(255, 149, 0, 0.5);
    transform-origin: left;
  }

  @keyframes undoCountdown {
    from {
      transform: scaleX(1);
    }
    to {
      transform: scaleX(0);
    }
  }

  /* ── Text ── */
  .label {
    font-size: 13px;
    font-weight: 500;
    letter-spacing: -0.01em;
    white-space: nowrap;
    line-height: 1;
  }

  .label.partial-label {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    direction: rtl;
    text-align: left;
  }

  .timer {
    font-size: 12px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.4);
    font-variant-numeric: tabular-nums;
    letter-spacing: 0.02em;
    margin-left: auto;
    flex-shrink: 0;
    line-height: 1;
  }

  /* ── Waveform canvas ── */
  .waveform {
    flex-shrink: 0;
    width: 46px;
    height: 32px;
  }

  /* ── State: Recording (orange -> red gradient via --rec-progress) ── */
  .capsule.recording {
    justify-content: flex-start;
    --rec-progress: 0;
    background: color-mix(
      in srgb,
      rgba(255, 140, 0, 0.92) calc((1 - var(--rec-progress)) * 100%),
      rgba(255, 59, 48, 0.92)
    );
    border-color: rgba(255, 200, 100, 0.15);
    box-shadow:
      0 8px 32px
        color-mix(
          in srgb,
          rgba(255, 140, 0, 0.35) calc((1 - var(--rec-progress)) * 100%),
          rgba(255, 59, 48, 0.35)
        ),
      0 0 0 0.5px rgba(255, 255, 255, 0.08),
      inset 0 0.5px 0 rgba(255, 255, 255, 0.12);
  }

  /* ── State: Meeting recording (green) ── */
  .capsule.meeting-recording {
    justify-content: flex-start;
    background: rgba(52, 199, 89, 0.92);
    border-color: rgba(120, 255, 150, 0.2);
    box-shadow:
      0 8px 32px rgba(52, 199, 89, 0.35),
      0 0 0 0.5px rgba(255, 255, 255, 0.08),
      inset 0 0.5px 0 rgba(255, 255, 255, 0.12);
  }

  /* ── State: Edit recording (blue) ── */
  .capsule.edit-recording {
    justify-content: flex-start;
    background: rgba(10, 132, 255, 0.92);
    border-color: rgba(120, 200, 255, 0.2);
    box-shadow:
      0 8px 32px rgba(10, 132, 255, 0.35),
      0 0 0 0.5px rgba(255, 255, 255, 0.08),
      inset 0 0.5px 0 rgba(255, 255, 255, 0.12);
  }

  /* ── State: Polishing (purple tint) ── */
  .capsule.polishing {
    border-color: rgba(167, 139, 250, 0.25);
    box-shadow:
      0 8px 32px rgba(167, 139, 250, 0.15),
      0 0 0 0.5px rgba(255, 255, 255, 0.06),
      inset 0 0.5px 0 rgba(255, 255, 255, 0.06);
  }

  /* ── State: Result ── */
  .capsule.result.success {
    border-color: rgba(52, 199, 89, 0.2);
  }

  .capsule.result.error-state {
    border-color: rgba(255, 59, 48, 0.2);
  }

  /* ── State: Switching (gray, neutral) ── */
  .capsule.switching {
    border-color: rgba(156, 163, 175, 0.2);
  }

  /* ── State: Undo ── */
  .capsule.undo-state {
    position: relative;
    cursor: pointer;
    border-color: rgba(255, 149, 0, 0.3);
    box-shadow:
      0 8px 32px rgba(255, 149, 0, 0.15),
      0 0 0 0.5px rgba(255, 255, 255, 0.06),
      inset 0 0.5px 0 rgba(255, 255, 255, 0.06);
    transition: all 0.15s ease;
  }

  .capsule.undo-state:hover {
    background: rgba(30, 30, 36, 0.92);
    border-color: rgba(255, 149, 0, 0.5);
  }

  .capsule.undo-state:active {
    transform: scale(0.97);
  }
</style>
