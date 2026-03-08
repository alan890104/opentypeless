<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { t } from '$lib/stores/i18n.svelte';
  import { setCurrentPage, setHighlightSection } from '$lib/stores/ui.svelte';
  import { getHotkey, getEditHotkey, getPolishConfig } from '$lib/stores/settings.svelte';
  import { hotkeyToParts, MODIFIER_SYMBOLS } from '$lib/constants';
  import {
    setTestMode,
    setContextOverride,
    setEditTextOverride,
    onTranscriptionResult,
    onHotkeyActivated,
    checkLlmModelStatus,
  } from '$lib/api';
  import Keycaps from '$lib/components/Keycaps.svelte';
  import InstructionCard from '$lib/components/InstructionCard.svelte';
  import type { UnlistenFn } from '@tauri-apps/api/event';

  // ── Shared state ──

  let step = $state(1);
  const TOTAL_STEPS = 6;

  const breadcrumbs = [
    'test.breadcrumb.mic',
    'test.breadcrumb.hotkey',
    'test.breadcrumb.general',
    'test.breadcrumb.gmail',
    'test.breadcrumb.polishCheck',
    'test.breadcrumb.editByVoice',
  ];

  let hotkey = $derived(getHotkey());
  let editHotkey = $derived(getEditHotkey());

  // ── Paste suppression ──
  // The backend inserts text via the `transcription-result` Tauri event AND
  // simulates Cmd+V (for real apps). When the paste target is our own
  // contenteditable, both paths fire and text appears twice. We suppress
  // the redundant Cmd+V paste for a short window after the event-based insert.
  let suppressPasteUntil = 0;

  function suppressNextPaste() {
    suppressPasteUntil = Date.now() + 1000;
  }

  function handlePaste(e: ClipboardEvent) {
    if (Date.now() < suppressPasteUntil) {
      e.preventDefault();
      e.stopImmediatePropagation();
    }
  }

  // ── Inline keycap HTML for instruction text ──
  function hotkeyKeycapsHtml(hotkeyStr: string): string {
    if (!hotkeyStr) return '';
    return hotkeyStr
      .split('+')
      .map((part) => {
        const sym = MODIFIER_SYMBOLS[part];
        if (sym) {
          return `<kbd>${sym}</kbd>`;
        } else {
          return `<kbd class="accent">${part.replace(/^Key/, '').replace(/^Digit/, '')}</kbd>`;
        }
      })
      .join('');
  }

  // ── Step 1: Mic Test ──

  let micStream: MediaStream | null = null;
  let audioCtx: AudioContext | null = null;
  let waveAnimId: number | null = null;
  let micCanvas = $state<HTMLCanvasElement | null>(null);

  function setupMicTest() {
    cleanupMicTest();
    if (!micCanvas) return;
    const container = micCanvas.closest('.test-right') as HTMLElement | null;
    if (!container) return;

    const dpr = window.devicePixelRatio || 1;
    const cRect = container.getBoundingClientRect();
    const drawW = Math.round(cRect.width);
    const drawH = Math.round(cRect.height);
    micCanvas.width = drawW * dpr;
    micCanvas.height = drawH * dpr;
    micCanvas.style.width = drawW + 'px';
    micCanvas.style.height = drawH + 'px';
    const ctx = micCanvas.getContext('2d')!;
    ctx.scale(dpr, dpr);

    const hPad = 20;
    const availW = drawW - hPad * 2;
    const barWidth = 4;
    const barGap = 3;
    const NUM_BARS = Math.floor((availW + barGap) / (barWidth + barGap));
    const maxBarH = Math.min(drawH * 0.4, 120);

    // Draw idle bars
    drawBars(ctx, NUM_BARS, barWidth, barGap, new Float32Array(NUM_BARS), maxBarH, drawW, drawH);

    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        micStream = stream;
        audioCtx = new AudioContext();
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);

        const dataArray = new Uint8Array(analyser.frequencyBinCount);

        function animate() {
          waveAnimId = requestAnimationFrame(animate);
          analyser.getByteFrequencyData(dataArray);

          const levels = new Float32Array(NUM_BARS);
          // Voice energy is in lower frequencies (~100-4000Hz).
          // Only map those bins across all bars so every bar responds to voice.
          const voiceBins = Math.max(NUM_BARS, Math.ceil(dataArray.length * 0.15));
          const s = Math.max(1, Math.floor(voiceBins / NUM_BARS));
          for (let i = 0; i < NUM_BARS; i++) {
            let sum = 0;
            for (let j = 0; j < s; j++) {
              const idx = i * s + j;
              if (idx < voiceBins) sum += dataArray[idx];
            }
            levels[i] = sum / s / 255;
          }

          drawBars(ctx, NUM_BARS, barWidth, barGap, levels, maxBarH, drawW, drawH);
        }
        animate();
      })
      .catch((err) => {
        console.warn('Mic access denied for test:', err);
      });
  }

  function drawBars(
    ctx: CanvasRenderingContext2D,
    numBars: number,
    barWidth: number,
    barGap: number,
    levels: Float32Array,
    maxBarH: number,
    w: number,
    h: number,
  ) {
    ctx.clearRect(0, 0, w, h);
    const totalW = numBars * barWidth + (numBars - 1) * barGap;
    const startX = (w - totalW) / 2;
    const cy = h / 2;

    for (let i = 0; i < numBars; i++) {
      const level = levels[i] || 0;
      const bh = Math.max(4, level * maxBarH);
      const x = startX + i * (barWidth + barGap);
      const y = cy - bh / 2;
      ctx.fillStyle = level > 0.05 ? '#007AFF' : '#e0e0e0';
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, bh, barWidth / 2);
      ctx.fill();
    }
  }

  function cleanupMicTest() {
    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop());
      micStream = null;
    }
    if (audioCtx) {
      audioCtx.close().catch(() => {});
      audioCtx = null;
    }
    if (waveAnimId) {
      cancelAnimationFrame(waveAnimId);
      waveAnimId = null;
    }
  }

  // ── Step 2: Hotkey Test ──

  let hotkeyUnlisten: UnlistenFn | null = null;
  let hotkeyKeycapContainer = $state<HTMLElement | null>(null);
  let hotkeyGlowing = $state(false);
  let pressedKeys = $state(new Set<string>());

  function eventToHotkeyParts(e: KeyboardEvent): Set<string> {
    const held = new Set<string>();
    if (e.altKey) held.add('Alt');
    if (e.ctrlKey) held.add('Control');
    if (e.shiftKey) held.add('Shift');
    if (e.metaKey) held.add('Super');
    const code = e.code;
    if (
      code &&
      ![
        'AltLeft',
        'AltRight',
        'ControlLeft',
        'ControlRight',
        'ShiftLeft',
        'ShiftRight',
        'MetaLeft',
        'MetaRight',
      ].includes(code)
    ) {
      held.add(code);
    }
    return held;
  }

  function hotkeyKeydown(e: KeyboardEvent) {
    e.preventDefault();
    const held = eventToHotkeyParts(e);
    const parts = hotkey.split('+');
    const newPressed = new Set(pressedKeys);
    parts.forEach((part) => {
      if (held.has(part)) newPressed.add(part);
    });
    pressedKeys = newPressed;
  }

  function hotkeyKeyup(e: KeyboardEvent) {
    e.preventDefault();
    const newPressed = new Set(pressedKeys);
    newPressed.delete(e.code);
    if (!e.altKey) newPressed.delete('Alt');
    if (!e.ctrlKey) newPressed.delete('Control');
    if (!e.shiftKey) newPressed.delete('Shift');
    if (!e.metaKey) newPressed.delete('Super');
    pressedKeys = newPressed;
  }

  async function setupHotkeyTest() {
    cleanupHotkeyTest();
    pressedKeys = new Set();

    try {
      await setTestMode(true);
    } catch (e) {
      console.warn('set_test_mode failed:', e);
    }

    document.addEventListener('keydown', hotkeyKeydown);
    document.addEventListener('keyup', hotkeyKeyup);

    hotkeyUnlisten = await onHotkeyActivated(() => {
      hotkeyGlowing = true;
      setTimeout(() => {
        hotkeyGlowing = false;
      }, 600);
    });
  }

  async function cleanupHotkeyTest() {
    document.removeEventListener('keydown', hotkeyKeydown);
    document.removeEventListener('keyup', hotkeyKeyup);
    if (hotkeyUnlisten) {
      hotkeyUnlisten();
      hotkeyUnlisten = null;
    }
    pressedKeys = new Set();
    hotkeyGlowing = false;
    try {
      await setTestMode(false);
    } catch (e) {
      console.warn('set_test_mode failed:', e);
    }
  }

  function isKeyPressed(part: string): boolean {
    return pressedKeys.has(part);
  }

  // ── Step 3: General Dictation ──

  let generalUnlisten: UnlistenFn | null = null;
  let plainBody = $state<HTMLElement | null>(null);

  async function setupGeneral() {
    cleanupGeneral();

    try {
      await setTestMode(false);
    } catch {}
    try {
      await setContextOverride('', '', '');
    } catch (e) {
      console.warn('set_context_override failed:', e);
    }

    if (plainBody) plainBody.innerHTML = '';

    generalUnlisten = await onTranscriptionResult((text: string) => {
      if (!text || !plainBody) return;
      suppressNextPaste();

      const sel = window.getSelection();
      if (sel && sel.rangeCount > 0 && plainBody.contains(sel.anchorNode)) {
        const range = sel.getRangeAt(0);
        range.deleteContents();
        range.insertNode(document.createTextNode(text));
        range.collapse(false);
        sel.removeAllRanges();
        sel.addRange(range);
      } else {
        plainBody.focus();
        plainBody.appendChild(document.createTextNode(text));
      }
    });
  }

  function cleanupGeneral() {
    if (generalUnlisten) {
      generalUnlisten();
      generalUnlisten = null;
    }
  }

  // ── Step 4: Gmail Scenario ──

  let gmailUnlisten: UnlistenFn | null = null;
  let gmailBody = $state<HTMLElement | null>(null);

  async function setupGmail() {
    cleanupGmail();

    try {
      await setTestMode(false);
    } catch {}
    try {
      await setContextOverride(
        'Google Chrome',
        'com.google.Chrome',
        'https://mail.google.com/mail/u/0/#inbox?compose=new',
      );
    } catch (e) {
      console.warn('set_context_override failed:', e);
    }

    if (gmailBody) gmailBody.innerHTML = '';

    gmailUnlisten = await onTranscriptionResult((text: string) => {
      if (!text || !gmailBody) return;
      suppressNextPaste();

      // Convert newlines to <br> for contenteditable
      const frag = document.createDocumentFragment();
      const lines = text.split('\n');
      lines.forEach((line, i) => {
        if (i > 0) frag.appendChild(document.createElement('br'));
        if (line) frag.appendChild(document.createTextNode(line));
      });

      const sel = window.getSelection();
      if (sel && sel.rangeCount > 0 && gmailBody.contains(sel.anchorNode)) {
        const range = sel.getRangeAt(0);
        range.deleteContents();
        range.insertNode(frag);
        range.collapse(false);
        sel.removeAllRanges();
        sel.addRange(range);
      } else {
        gmailBody.focus();
        gmailBody.appendChild(frag);
      }
    });
  }

  function cleanupGmail() {
    if (gmailUnlisten) {
      gmailUnlisten();
      gmailUnlisten = null;
    }
    setContextOverride('', '', '').catch(() => {});
  }

  // ── Step 5: Polish Readiness Check ──

  let polishReady = $state(false);
  let polishPressedKeys = $state(new Set<string>());

  function polishKeydown(e: KeyboardEvent) {
    e.preventDefault();
    const held = eventToHotkeyParts(e);
    const parts = (editHotkey || hotkey).split('+');
    const newPressed = new Set(polishPressedKeys);
    parts.forEach((part) => {
      if (held.has(part)) newPressed.add(part);
    });
    polishPressedKeys = newPressed;
  }

  function polishKeyup(e: KeyboardEvent) {
    e.preventDefault();
    const newPressed = new Set(polishPressedKeys);
    newPressed.delete(e.code);
    if (!e.altKey) newPressed.delete('Alt');
    if (!e.ctrlKey) newPressed.delete('Control');
    if (!e.shiftKey) newPressed.delete('Shift');
    if (!e.metaKey) newPressed.delete('Super');
    polishPressedKeys = newPressed;
  }

  function isPolishKeyPressed(part: string): boolean {
    return polishPressedKeys.has(part);
  }

  function cleanupPolishCheck() {
    document.removeEventListener('keydown', polishKeydown);
    document.removeEventListener('keyup', polishKeyup);
    polishPressedKeys = new Set();
  }

  async function checkPolishReady() {
    const pc = getPolishConfig();
    if (!pc.enabled) {
      polishReady = false;
      return;
    }
    if (pc.mode === 'cloud') {
      polishReady = pc.cloud.api_key.length > 0;
    } else {
      try {
        const status = await checkLlmModelStatus();
        polishReady = status.model_exists;
      } catch {
        polishReady = false;
      }
    }
  }

  async function setupPolishCheck() {
    cleanupPolishCheck();
    document.addEventListener('keydown', polishKeydown);
    document.addEventListener('keyup', polishKeyup);
    await checkPolishReady();
  }

  function navigateToPolishSettings() {
    cleanupAllSteps();
    setHighlightSection('polish');
    setCurrentPage('settings');
  }

  // ── Step 6: Edit by Voice ──

  let editUnlisten: UnlistenFn | null = null;
  let editBody = $state<HTMLElement | null>(null);

  function handleEditSelection() {
    if (!editBody) return;
    const sel = window.getSelection();
    const text = sel && editBody.contains(sel.anchorNode) ? sel.toString() : '';
    setEditTextOverride(text).catch(() => {});
  }

  async function setupEditTest() {
    cleanupEditTest();

    try {
      await setTestMode(false);
    } catch {}
    try {
      await setContextOverride('', '', '');
    } catch (e) {
      console.warn('set_context_override failed:', e);
    }

    if (editBody) editBody.innerText = t('test.step6.prefill');

    document.addEventListener('selectionchange', handleEditSelection);

    editUnlisten = await onTranscriptionResult((text: string) => {
      if (!text || !editBody) return;
      suppressNextPaste();
      editBody.innerText = text;
    });
  }

  function cleanupEditTest() {
    document.removeEventListener('selectionchange', handleEditSelection);
    setEditTextOverride('').catch(() => {});
    if (editUnlisten) {
      editUnlisten();
      editUnlisten = null;
    }
  }

  // ── Step navigation ──

  function goToStep(n: number) {
    // Clean up current step
    cleanupAllSteps();
    step = n;

    // Setup new step in next tick so DOM is ready
    requestAnimationFrame(() => {
      if (n === 1) setupMicTest();
      else if (n === 2) setupHotkeyTest();
      else if (n === 3) setupGeneral();
      else if (n === 4) setupGmail();
      else if (n === 5) setupPolishCheck();
      else if (n === 6) setupEditTest();
    });
  }

  function goBack(fromStep: number) {
    if (fromStep === 1) {
      finishWizard();
    } else {
      goToStep(fromStep - 1);
    }
  }

  function finishWizard() {
    cleanupAllSteps();
    setCurrentPage('settings');
  }

  function cleanupAllSteps() {
    cleanupMicTest();
    cleanupHotkeyTest();
    cleanupGeneral();
    cleanupGmail();
    cleanupPolishCheck();
    cleanupEditTest();
    setTestMode(false).catch(() => {});
  }

  // ── Lifecycle ──

  onMount(() => {
    document.addEventListener('paste', handlePaste, true);
    // Start at step 1
    requestAnimationFrame(() => {
      setupMicTest();
    });
  });

  onDestroy(() => {
    document.removeEventListener('paste', handlePaste, true);
    cleanupAllSteps();
  });

  // Helper for instruction HTML with embedded keycaps
  let generalInstruction = $derived(t('test.step3.instruction', { hotkey: hotkeyKeycapsHtml(hotkey) }));
  let gmailInstruction = $derived(t('test.step4.instruction', { hotkey: hotkeyKeycapsHtml(hotkey) }));
  let editInstruction = $derived(
    t('test.step6.instruction', { hotkey: hotkeyKeycapsHtml(editHotkey || hotkey) }),
  );
  let hotkeySubtitle = $derived(t('test.step2.subtitle', { hotkey: hotkeyToParts(hotkey).join(' + ') }));
</script>

<div class="test-page">
  <!-- Breadcrumb -->
  <div class="test-breadcrumb">
    {#each breadcrumbs as bc, i}
      {#if i > 0}
        <span class="test-breadcrumb-sep">&rsaquo;</span>
      {/if}
      <span
        class="test-breadcrumb-item"
        class:active={i + 1 === step}
        class:done={i + 1 < step}
      >
        {t(bc)}
      </span>
    {/each}
  </div>

  <!-- Progress bar -->
  <div class="test-progress-bar">
    <div class="test-progress-fill" style="width: {(step / TOTAL_STEPS) * 100}%"></div>
  </div>

  <!-- Step 1: Mic Test -->
  {#if step === 1}
    <div class="test-step active">
      <div class="test-layout">
        <div class="test-left">
          <button class="test-back" onclick={() => goBack(1)}>
            {t('test.backSettings')}
          </button>
          <div class="test-left-spacer"></div>
          <div class="test-title">{t('test.step1.title')}</div>
          <div class="test-subtitle">{t('test.step1.subtitle')}</div>
          <div class="test-question">{t('test.step1.question')}</div>
          <div class="test-actions">
            <button class="test-btn-outline" onclick={finishWizard}>
              {t('test.step1.no')}
            </button>
            <button class="test-btn-filled" onclick={() => goToStep(2)}>
              {t('test.step1.yes')}
            </button>
          </div>
          <div class="test-left-spacer"></div>
        </div>
        <div class="test-right">
          <div class="test-waveform">
            <canvas bind:this={micCanvas} width="220" height="140"></canvas>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Step 2: Hotkey Test -->
  {#if step === 2}
    <div class="test-step active">
      <div class="test-layout">
        <div class="test-left">
          <button class="test-back" onclick={() => goBack(2)}>
            {t('test.back')}
          </button>
          <div class="test-left-spacer"></div>
          <div class="test-title">{t('test.step2.title')}</div>
          <div class="test-subtitle">{hotkeySubtitle}</div>
          <div class="test-question">{t('test.step2.question')}</div>
          <div class="test-actions">
            <button class="test-btn-outline" onclick={finishWizard}>
              {t('test.step2.no')}
            </button>
            <button class="test-btn-filled" onclick={() => goToStep(3)}>
              {t('test.step2.yes')}
            </button>
          </div>
          <div class="test-left-spacer"></div>
        </div>
        <div class="test-right">
          <div
            class="test-keycap-lg"
            class:active={hotkeyGlowing}
            bind:this={hotkeyKeycapContainer}
          >
            {#each hotkey.split('+') as part, i}
              {@const sym = MODIFIER_SYMBOLS[part]}
              {@const label = sym ?? part.replace(/^Key/, '').replace(/^Digit/, '')}
              <kbd
                class:accent={!sym}
                class:pressed={isKeyPressed(part)}
              >
                {label}
              </kbd>
            {/each}
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Step 3: General Dictation -->
  {#if step === 3}
    <div class="test-step active">
      <div class="test-layout">
        <div class="test-left">
          <button class="test-back" onclick={() => goBack(3)}>
            {t('test.back')}
          </button>
          <div class="test-left-spacer"></div>
          <div class="test-title">{t('test.step3.title')}</div>
          <InstructionCard
            icon='<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="11" rx="3"/><path d="M5 10a7 7 0 0 0 14 0"/><line x1="12" y1="19" x2="12" y2="22"/></svg>'
          >
            <!-- eslint-disable-next-line svelte/no-at-html-tags -->
            {@html generalInstruction}
          </InstructionCard>
          <div class="test-sample-text">{t('test.step3.sampleText')}</div>
          <div class="test-left-spacer"></div>
          <div class="test-actions">
            <button class="test-btn-filled" onclick={() => goToStep(4)}>
              {t('test.step3.next')}
            </button>
          </div>
        </div>
        <div class="test-right">
          <div class="test-editor-wrap">
            <div class="test-plain-editor">
              <div class="test-plain-titlebar">
                <div class="test-plain-dots"><span></span><span></span><span></span></div>
                <span class="test-plain-title">{t('test.step3.editorTitle')}</span>
                <span style="width:38px"></span>
              </div>
              <!-- svelte-ignore a11y_no_static_element_interactions -->
              <div
                class="test-plain-body"
                bind:this={plainBody}
                contenteditable="true"
                spellcheck="false"
              ></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Step 4: Gmail Scenario -->
  {#if step === 4}
    <div class="test-step active">
      <div class="test-layout">
        <div class="test-left">
          <button class="test-back" onclick={() => goBack(4)}>
            {t('test.back')}
          </button>
          <div class="test-left-spacer"></div>
          <div class="test-title">{t('test.step4.title')}</div>
          <InstructionCard
            icon='<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="11" rx="3"/><path d="M5 10a7 7 0 0 0 14 0"/><line x1="12" y1="19" x2="12" y2="22"/></svg>'
          >
            <!-- eslint-disable-next-line svelte/no-at-html-tags -->
            {@html gmailInstruction}
          </InstructionCard>
          <div class="test-sample-text">{t('test.step4.sampleText')}</div>
          <div class="test-left-spacer"></div>
          <div class="test-actions">
            <button class="test-btn-filled" onclick={() => goToStep(5)}>
              {t('test.step3.next')}
            </button>
          </div>
        </div>
        <div class="test-right">
          <div class="test-editor-wrap">
            <div class="test-gmail-editor">
              <div class="test-gmail-titlebar">
                <span class="test-gmail-title">New Message</span>
                <span class="test-gmail-close">&times;</span>
              </div>
              <div class="test-gmail-fields">
                <div class="test-gmail-field">
                  <span class="test-gmail-label">To</span>
                  <span class="test-gmail-value">alice@example.com</span>
                </div>
                <div class="test-gmail-field">
                  <span class="test-gmail-label">Subject</span>
                  <span class="test-gmail-value">Lunch tomorrow</span>
                </div>
              </div>
              <!-- svelte-ignore a11y_no_static_element_interactions -->
              <div
                class="test-gmail-body"
                bind:this={gmailBody}
                contenteditable="true"
                spellcheck="false"
              ></div>
              <div class="test-gmail-toolbar">
                <span class="test-gmail-send">Send</span>
                <span class="test-gmail-toolbar-icons">
                  <span title="Formatting">A</span>
                  <span title="Attach">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
                    </svg>
                  </span>
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Step 5: Polish Readiness Check -->
  {#if step === 5}
    <div class="test-step active">
      <div class="test-layout">
        <div class="test-left">
          <button class="test-back" onclick={() => goBack(5)}>
            {t('test.back')}
          </button>
          <div class="test-left-spacer"></div>
          <div class="test-title">{t('test.step5.title')}</div>
          <div class="test-subtitle">{t('test.step5.subtitle')}</div>
          {#if polishReady}
            <div class="test-polish-status test-polish-ready">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#28c840" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
              <span>{t('test.step5.polishReady')}</span>
            </div>
            <div class="test-left-spacer"></div>
            <div class="test-actions">
              <button class="test-btn-filled" onclick={() => goToStep(6)}>
                {t('test.step3.next')}
              </button>
            </div>
          {:else}
            <div class="test-polish-status test-polish-not-ready">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ff9500" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              <span>{t('test.step5.polishNotReady')}</span>
            </div>
            <div class="test-left-spacer"></div>
            <div class="test-actions">
              <button class="test-btn-filled" onclick={navigateToPolishSettings}>
                {t('test.step5.goToSettings')}
              </button>
            </div>
          {/if}
        </div>
        <div class="test-right">
          <div class="test-keycap-lg">
            {#each (editHotkey || hotkey).split('+') as part}
              {@const sym = MODIFIER_SYMBOLS[part]}
              {@const label = sym ?? part.replace(/^Key/, '').replace(/^Digit/, '')}
              <kbd class:accent={!sym} class:pressed={isPolishKeyPressed(part)}>{label}</kbd>
            {/each}
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Step 6: Edit by Voice -->
  {#if step === 6}
    <div class="test-step active">
      <div class="test-layout">
        <div class="test-left">
          <button class="test-back" onclick={() => goBack(6)}>
            {t('test.back')}
          </button>
          <div class="test-left-spacer"></div>
          <div class="test-title">{t('test.step6.title')}</div>
          <InstructionCard
            icon='<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>'
          >
            <!-- eslint-disable-next-line svelte/no-at-html-tags -->
            {@html editInstruction}
          </InstructionCard>
          <div class="test-sample-text">{t('test.step6.sampleText')}</div>
          {#if editHotkey}
            <div class="test-edit-keycaps">
              <Keycaps hotkey={editHotkey} size="normal" />
            </div>
          {/if}
          <div class="test-left-spacer"></div>
          <div class="test-actions">
            <button class="test-btn-filled" onclick={finishWizard}>
              {t('test.step6.done')}
            </button>
          </div>
        </div>
        <div class="test-right">
          <div class="test-editor-wrap">
            <div class="test-plain-editor">
              <div class="test-plain-titlebar">
                <div class="test-plain-dots"><span></span><span></span><span></span></div>
                <span class="test-plain-title">{t('test.step6.editorTitle')}</span>
                <span style="width:38px"></span>
              </div>
              <!-- svelte-ignore a11y_no_static_element_interactions -->
              <div
                class="test-plain-body"
                bind:this={editBody}
                contenteditable="true"
                spellcheck="false"
              ></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .test-page {
    display: flex;
    flex-direction: column;
    /* 56px drag region + 44px content-scroll bottom padding */
    height: calc(100vh - 100px);
    margin: 0 -44px -44px;
    padding: 0 44px;
  }

  /* ── Breadcrumb ── */
  .test-breadcrumb {
    display: flex;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-divider);
    margin-bottom: 0;
    flex-shrink: 0;
  }

  .test-breadcrumb-item {
    flex: 1;
    text-align: center;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-tertiary);
    cursor: default;
  }

  .test-breadcrumb-item.active {
    color: var(--text-primary);
    font-weight: 600;
  }

  .test-breadcrumb-item.done {
    color: var(--accent-blue);
  }

  .test-breadcrumb-sep {
    font-size: 12px;
    color: var(--text-tertiary);
    flex-shrink: 0;
  }

  /* ── Progress bar ── */
  .test-progress-bar {
    height: 3px;
    background: var(--border-divider);
    border-radius: 2px;
    margin-bottom: 0;
    flex-shrink: 0;
    overflow: hidden;
  }

  .test-progress-fill {
    height: 100%;
    background: var(--accent-blue);
    border-radius: 2px;
    transition: width 0.3s ease;
  }

  /* ── Step container ── */
  .test-step {
    display: none;
    flex: 1;
    min-height: 0;
  }

  .test-step.active {
    display: flex;
    flex-direction: column;
  }

  /* ── Layout ── */
  .test-layout {
    display: flex;
    flex: 1;
    gap: 0;
    min-height: 0;
  }

  .test-left {
    flex: 0 0 340px;
    display: flex;
    flex-direction: column;
    padding: 24px 28px 24px 0;
  }

  .test-left-spacer {
    flex: 1;
  }

  .test-right {
    flex: 1;
    background: #f5f5f7;
    border-radius: var(--radius-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 16px 0;
    min-width: 0;
    overflow: hidden;
  }

  /* ── Back button ── */
  .test-back {
    font-size: 13px;
    color: var(--accent-blue);
    cursor: pointer;
    border: none;
    background: none;
    padding: 0;
    margin-bottom: 12px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;

    align-self: flex-start;
    flex-shrink: 0;
  }

  .test-back:hover {
    text-decoration: underline;
  }

  /* ── Title / subtitle ── */
  .test-title {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text-primary);
    margin-bottom: 8px;
    line-height: 1.3;
  }

  .test-subtitle {
    font-size: 14px;
    color: var(--text-secondary);
    margin-bottom: 24px;
    line-height: 1.5;
  }

  .test-question {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 16px;
  }

  /* ── Actions ── */
  .test-actions {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }

  .test-btn-outline {

    padding: 10px 20px;
    border: 1.5px solid rgba(0, 0, 0, 0.15);
    border-radius: 100px;
    background: none;
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .test-btn-outline:hover {
    background: var(--bg-hover);
    border-color: rgba(0, 0, 0, 0.25);
  }

  .test-btn-filled {

    padding: 10px 20px;
    border: none;
    border-radius: 100px;
    background: var(--text-primary);
    color: #fff;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .test-btn-filled:hover {
    background: #000;
  }

  /* ── Waveform ── */
  .test-waveform {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    overflow: hidden;
  }

  .test-waveform canvas {
    display: block;
  }

  /* ── Hotkey keycaps (large) ── */
  .test-keycap-lg {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: center;
  }

  .test-keycap-lg kbd {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 52px;
    height: 52px;
    padding: 0 14px;
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    background: #fff;
    border: 1px solid rgba(0, 0, 0, 0.15);
    border-bottom: 2px solid rgba(0, 0, 0, 0.15);
    border-radius: var(--radius-md);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    transition: all 0.2s ease;
  }

  .test-keycap-lg kbd.accent {
    background: var(--bg-sidebar);
    border-color: rgba(0, 0, 0, 0.12);
    border-bottom-color: rgba(0, 0, 0, 0.18);
    color: var(--accent-blue);
    font-weight: 700;
  }

  .test-keycap-lg.active kbd {
    background: var(--accent-blue);
    color: #fff;
    border-color: var(--accent-blue);
    border-bottom-color: #0062cc;
    box-shadow: 0 2px 8px rgba(0, 122, 255, 0.35);
  }

  .test-keycap-lg kbd.pressed {
    background: var(--accent-blue);
    color: #fff;
    border-color: var(--accent-blue);
    border-bottom-color: #0062cc;
    box-shadow: 0 2px 8px rgba(0, 122, 255, 0.35);
  }

  /* ── Sample text ── */
  .test-sample-text {
    border-left: 3px solid var(--accent-blue);
    padding: 12px 16px;
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.6;
    font-style: italic;
    margin-bottom: 0;
  }

  /* ── Edit keycaps display ── */
  .test-edit-keycaps {
    margin-top: 16px;
    margin-bottom: 8px;
  }

  /* ── Editor wrap ── */
  .test-editor-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 92%;
    max-width: 420px;
  }

  /* ── Plain text editor ── */
  .test-plain-editor {
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.1), 0 0 0 1px rgba(0, 0, 0, 0.06);
    width: 100%;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .test-plain-titlebar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 14px;
    background: var(--bg-sidebar);
    border-bottom: 1px solid rgba(0, 0, 0, 0.06);
  }

  .test-plain-titlebar .test-plain-dots {
    display: flex;
    gap: 6px;
  }

  .test-plain-titlebar .test-plain-dots span {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
  }

  .test-plain-titlebar .test-plain-dots span:nth-child(1) {
    background: #ff5f57;
  }
  .test-plain-titlebar .test-plain-dots span:nth-child(2) {
    background: #ffbd2e;
  }
  .test-plain-titlebar .test-plain-dots span:nth-child(3) {
    background: #28c840;
  }

  .test-plain-title {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
  }

  .test-plain-body {
    padding: 14px 16px;
    min-height: 180px;
    font-size: 13px;
    line-height: 1.6;
    color: var(--text-primary);
    outline: none;
    cursor: text;
    -webkit-user-select: text;
    user-select: text;
  }

  .test-plain-body:focus {
    background: rgba(0, 122, 255, 0.02);
  }

  /* ── Gmail editor ── */
  .test-gmail-editor {
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.1), 0 0 0 1px rgba(0, 0, 0, 0.06);
    width: 100%;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .test-gmail-titlebar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 14px;
    background: #404040;
    color: #fff;
  }

  .test-gmail-title {
    font-size: 13px;
    font-weight: 500;
  }

  .test-gmail-close {
    font-size: 18px;
    opacity: 0.6;
    cursor: default;
    line-height: 1;
  }

  .test-gmail-fields {
    border-bottom: 1px solid rgba(0, 0, 0, 0.08);
  }

  .test-gmail-field {
    display: flex;
    align-items: center;
    padding: 6px 14px;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    font-size: 13px;
  }

  .test-gmail-label {
    color: var(--text-tertiary);
    width: 54px;
    flex-shrink: 0;
  }

  .test-gmail-value {
    color: var(--text-primary);
  }

  .test-gmail-body {
    padding: 14px 16px;
    min-height: 120px;
    font-size: 13px;
    line-height: 1.6;
    color: var(--text-primary);
    outline: none;
    cursor: text;
    -webkit-user-select: text;
    user-select: text;
  }

  .test-gmail-body:focus {
    background: rgba(0, 122, 255, 0.02);
  }

  .test-gmail-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 14px;
    border-top: 1px solid rgba(0, 0, 0, 0.06);
  }

  .test-gmail-send {
    background: #1a73e8;
    color: #fff;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 18px;
    border-radius: 16px;
    cursor: default;
  }

  .test-gmail-toolbar-icons {
    display: flex;
    gap: 12px;
    font-size: 14px;
    color: var(--text-tertiary);
  }

  /* ── Inline keycap styles for instruction text ── */
  .test-left :global(kbd) {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 22px;
    height: 22px;
    padding: 0 5px;
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
    background: #fff;
    border: 1px solid rgba(0, 0, 0, 0.15);
    border-bottom: 2px solid rgba(0, 0, 0, 0.15);
    border-radius: var(--radius-sm);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    vertical-align: middle;
    margin: 0 1px;
  }

  .test-left :global(kbd.accent) {
    background: var(--bg-sidebar);
    border-color: rgba(0, 0, 0, 0.12);
    border-bottom-color: rgba(0, 0, 0, 0.18);
    color: var(--accent-blue);
    font-weight: 700;
  }

  /* ── Polish status ── */
  .test-polish-status {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 14px 16px;
    border-radius: var(--radius-md);
    margin-bottom: 16px;
  }

  .test-polish-status svg {
    flex-shrink: 0;
    margin-top: 1px;
  }

  .test-polish-status span {
    font-size: 14px;
    line-height: 1.5;
    color: var(--text-secondary);
  }

  .test-polish-ready {
    background: rgba(40, 200, 64, 0.08);
    border: 1px solid rgba(40, 200, 64, 0.25);
  }

  .test-polish-not-ready {
    background: rgba(255, 149, 0, 0.08);
    border: 1px solid rgba(255, 149, 0, 0.25);
  }

</style>
