<script lang="ts">
  import { onMount, onDestroy, tick } from 'svelte';
  import { marked } from 'marked';
  import { t } from '$lib/stores/i18n.svelte';
  import { showConfirm, setCurrentPage, setHighlightSection } from '$lib/stores/ui.svelte';
  import { getPolishConfig } from '$lib/stores/settings.svelte';
  import {
    listMeetingNotes,
    getActiveMeetingNoteId,
    getMeetingNote,
    renameMeetingNote,
    deleteMeetingNote,
    polishMeetingNote,
    exportMeetingAudio,
    deleteMeetingAudio,
    onMeetingNoteCreated,
    onMeetingNoteUpdated,
    onMeetingNoteFinalized,
    importMeetingAudio,
    cancelImport,
    onImportProgress,
  } from '$lib/api';
  import { open as openFileDialog } from '@tauri-apps/plugin-dialog';
  import { getCurrentWebviewWindow } from '@tauri-apps/api/webviewWindow';
  import type { MeetingNote } from '$lib/types';
  import type { UnlistenFn } from '@tauri-apps/api/event';

  let notes = $state<MeetingNote[]>([]);
  let selectedId = $state<string | null>(null);
  let loading = $state(true);

  // Inline rename
  let editingTitleId = $state<string | null>(null);
  let editingTitleValue = $state('');
  let titleInputEl = $state<HTMLInputElement | undefined>();

  // Auto-scroll
  let transcriptEl = $state<HTMLDivElement | undefined>();
  let userScrolledUp = $state(false);

  // Polish / tab state
  let polishing = $state(false);
  let activeTab = $state<'transcript' | 'summary'>('transcript');

  // Copy feedback
  let copied = $state(false);
  let copiedTimeout: ReturnType<typeof setTimeout> | null = null;

  // Download audio feedback
  let downloading = $state(false);

  // Split copy dropdown
  let copyMenuOpen = $state(false);

  // Context menu
  let contextMenuId = $state<string | null>(null);
  let contextMenuX = $state(0);
  let contextMenuY = $state(0);

  // Import state
  let importing = $state(false);
  let importingNoteId = $state<string | null>(null);
  let importProgress = $state(0);
  let dragOver = $state(false);

  let unlisteners: UnlistenFn[] = [];

  let selectedNote = $derived(notes.find((n) => n.id === selectedId) ?? null);

  function renderMarkdown(md: string): string {
    return marked.parse(md, { async: false }) as string;
  }

  // Format seconds-from-start as "M:SS".
  function formatSegTime(secs: number): string {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
  }

  // Speaker color palette — cycles for Speaker 1, 2, 3 …
  const SPEAKER_COLORS = ['#4A90D9', '#E07B3A', '#45A86B', '#9B59B6', '#E74C3C', '#16A085'];

  // Convert diarization label "SPEAKER_00" → "Speaker 1", etc.
  function formatSpeaker(raw: string): string {
    const m = raw.match(/^SPEAKER_(\d+)$/i);
    if (m) return `Speaker ${parseInt(m[1], 10) + 1}`;
    return raw;
  }

  // Parse JSONL WAL into structured segments for display.
  // Returns null when the transcript is legacy plain text (no JSON lines).
  function parseTranscriptSegs(
    raw: string,
  ): { speaker: string; start: number; text: string }[] | null {
    if (!raw) return null;
    const segs: { speaker: string; start: number; text: string }[] = [];
    let hasJsonl = false;
    for (const line of raw.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const s = JSON.parse(trimmed) as { speaker?: string; text?: string; start?: number };
        if (s && typeof s === 'object' && s.text) {
          hasJsonl = true;
          segs.push({ speaker: s.speaker ?? '', start: s.start ?? 0, text: s.text });
        }
      } catch { /* plain-text line, skip */ }
    }
    return hasJsonl ? segs : null;
  }

  // Group consecutive segments from the same speaker into a single block.
  function groupSegsBySpeaker(segs: { speaker: string; start: number; text: string }[]) {
    const speakerOrder: string[] = [];
    const groups: { speaker: string; speakerIdx: number; start: number; text: string }[] = [];
    for (const seg of segs) {
      if (seg.speaker && !speakerOrder.includes(seg.speaker)) speakerOrder.push(seg.speaker);
      const speakerIdx = seg.speaker ? speakerOrder.indexOf(seg.speaker) : -1;
      const last = groups[groups.length - 1];
      if (last && last.speaker === seg.speaker) {
        last.text += ' ' + seg.text;
      } else {
        groups.push({ speaker: seg.speaker, speakerIdx, start: seg.start, text: seg.text });
      }
    }
    return groups;
  }

  // Plain-text version of the transcript for copy / sidebar preview.
  function walToText(raw: string): string {
    if (!raw) return '';
    const segs = parseTranscriptSegs(raw);
    if (!segs) return raw;
    return segs
      .filter((s) => s.text)
      .map((s) => {
        const ts = `[${formatSegTime(s.start)}]`;
        const spk = s.speaker ? ` ${formatSpeaker(s.speaker)}:` : '';
        return `${ts}${spk} ${s.text}`;
      })
      .join('\n');
  }

  // Auto-set active tab when note changes
  $effect(() => {
    if (selectedNote) {
      activeTab = selectedNote.summary ? 'summary' : 'transcript';
    }
  });

  function formatDate(ts: number): string {
    const d = new Date(ts);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  }

  function formatTime(ts: number): string {
    const d = new Date(ts);
    return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  }

  function formatDuration(secs: number): string {
    if (secs < 60) return `${Math.round(secs)}s`;
    const m = Math.floor(secs / 60);
    const s = Math.round(secs % 60);
    if (m < 60) return `${m}m ${s}s`;
    const h = Math.floor(m / 60);
    const rm = m % 60;
    return `${h}h ${rm}m`;
  }

  function preview(text: string, maxLen = 80): string {
    if (!text) return '';
    const oneLine = walToText(text).replace(/\n/g, ' ');
    return oneLine.length > maxLen ? oneLine.slice(0, maxLen) + '…' : oneLine;
  }

  function defaultTitle(note: MeetingNote): string {
    if (note.title) return note.title;
    const d = new Date(note.created_at);
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    const hh = String(d.getHours()).padStart(2, '0');
    const mi = String(d.getMinutes()).padStart(2, '0');
    return `Meeting ${mm}-${dd} ${hh}:${mi}`;
  }

  // ── Auto-scroll logic ──
  function handleTranscriptScroll() {
    if (!transcriptEl) return;
    const { scrollTop, scrollHeight, clientHeight } = transcriptEl;
    userScrolledUp = scrollHeight - scrollTop - clientHeight > 50;
  }

  async function autoScroll() {
    if (!userScrolledUp && transcriptEl) {
      await tick();
      transcriptEl.scrollTop = transcriptEl.scrollHeight;
    }
  }

  // ── Rename ──
  function startRename(note: MeetingNote) {
    editingTitleId = note.id;
    editingTitleValue = defaultTitle(note);
    contextMenuId = null;
    tick().then(() => {
      titleInputEl?.select();
    });
  }

  async function commitRename() {
    if (!editingTitleId) return;
    const trimmed = editingTitleValue.trim();
    if (trimmed) {
      try {
        await renameMeetingNote(editingTitleId, trimmed);
        const n = notes.find((x) => x.id === editingTitleId);
        if (n) n.title = trimmed;
      } catch (e) {
        console.error('Rename failed:', e);
      }
    }
    editingTitleId = null;
  }

  function handleTitleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') commitRename();
    if (e.key === 'Escape') {
      editingTitleId = null;
    }
  }

  // ── Delete ──
  async function handleDelete(id: string) {
    contextMenuId = null;
    showConfirm(
      t('meeting.delete'),
      t('meeting.deleteConfirm'),
      t('meeting.delete'),
      async () => {
        try {
          await deleteMeetingNote(id);
          notes = notes.filter((n) => n.id !== id);
          if (selectedId === id) {
            selectedId = notes.length > 0 ? notes[0].id : null;
          }
        } catch (e) {
          console.error('Delete failed:', e);
        }
      },
    );
  }


  // ── Copy ──
  function stripMarkdown(md: string): string {
    return md
      .replace(/^#{1,6}\s+/gm, '')  // headings
      .replace(/\*\*(.+?)\*\*/g, '$1')  // bold
      .replace(/\*(.+?)\*/g, '$1')  // italic
      .replace(/`(.+?)`/g, '$1')  // inline code
      .replace(/^\s*[-*]\s+/gm, '- ')  // normalize bullets
      .replace(/\n{3,}/g, '\n\n')  // collapse blank lines
      .trim();
  }

  async function copyText(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      copied = true;
      copyMenuOpen = false;
      if (copiedTimeout) clearTimeout(copiedTimeout);
      copiedTimeout = setTimeout(() => (copied = false), 1500);
    } catch (e) {
      console.error('Copy failed:', e);
    }
  }

  async function handleCopy() {
    if (!selectedNote) return;
    if (activeTab === 'summary' && selectedNote.summary) {
      await copyText(stripMarkdown(selectedNote.summary));
    } else {
      await copyText(walToText(selectedNote.transcript));
    }
  }

  async function handleCopyMarkdown() {
    if (!selectedNote?.summary) return;
    await copyText(selectedNote.summary);
  }

  function toggleCopyMenu(e: MouseEvent) {
    e.stopPropagation();
    copyMenuOpen = !copyMenuOpen;
  }

  async function handleDownloadAudio() {
    if (!selectedNote?.audio_path || downloading) return;
    downloading = true;
    try {
      await exportMeetingAudio(selectedNote.id);
    } catch (e) {
      console.error('Download failed:', e);
    } finally {
      downloading = false;
    }
  }

  async function handleDeleteAudio() {
    if (!selectedNote?.audio_path) return;
    showConfirm(
      t('meeting.deleteAudio'),
      t('meeting.deleteAudioConfirm'),
      t('meeting.deleteAudio'),
      async () => {
        try {
          await deleteMeetingAudio(selectedNote!.id);
          const n = notes.find((x) => x.id === selectedNote!.id);
          if (n) n.audio_path = null;
        } catch (e) {
          console.error('Delete audio failed:', e);
        }
      },
    );
  }

  // ── Polish ──
  async function handlePolish() {
    if (!selectedNote || polishing) return;
    if (!getPolishConfig().enabled) {
      showConfirm(
        t('meeting.polish'),
        t('meeting.polishNotReady'),
        t('test.step5.goToSettings'),
        () => {
          setHighlightSection('polish');
          setCurrentPage('settings');
        },
      );
      return;
    }
    const noteId = selectedNote.id;
    polishing = true;
    try {
      const result = await polishMeetingNote(noteId);
      // Only apply if user hasn't switched to a different note
      const n = notes.find((x) => x.id === noteId);
      if (n) {
        n.title = result.title;
        n.summary = result.summary;
        notes = [...notes];
      }
      if (selectedId === noteId) {
        activeTab = 'summary';
      }
    } catch (e: any) {
      console.error('Polish failed:', e);
      const msg = typeof e === 'string' ? e : e?.message ?? t('meeting.polishError');
      // Brief inline feedback — could be a toast in the future
      alert(msg);
    }
    polishing = false;
  }

  // ── Context menu ──
  function handleContextMenu(e: MouseEvent, id: string) {
    e.preventDefault();
    contextMenuId = id;
    contextMenuX = e.clientX;
    contextMenuY = e.clientY;
  }

  function closeContextMenu() {
    contextMenuId = null;
    copyMenuOpen = false;
  }

  // ── Import ──
  const AUDIO_EXTENSIONS = ['wav', 'mp3', 'm4a', 'aac', 'ogg', 'flac'];

  async function handleImportClick() {
    if (importing) return;
    const path = await openFileDialog({
      title: t('meeting.importSelectFile'),
      filters: [{
        name: 'Audio',
        extensions: AUDIO_EXTENSIONS,
      }],
      multiple: false,
    });
    if (path) {
      startImport(path as string);
    }
  }

  async function startImport(filePath: string) {
    if (importing) return;
    importing = true;
    importProgress = 0;
    try {
      await importMeetingAudio(filePath);
    } catch (e: any) {
      const msg = typeof e === 'string' ? e : e?.message ?? 'Import failed';
      if (msg !== 'cancelled') {
        console.error('Import failed:', msg);
        alert(msg);
      }
    }
    importing = false;
    importingNoteId = null;
    importProgress = 0;
  }

  async function handleCancelImport() {
    await cancelImport();
  }

  function isAudioFile(name: string): boolean {
    const ext = name.split('.').pop()?.toLowerCase() ?? '';
    return AUDIO_EXTENSIONS.includes(ext);
  }

  // ── Lifecycle ──
  onMount(async () => {
    loading = true;
    try {
      notes = await listMeetingNotes();
      const activeId = await getActiveMeetingNoteId();
      if (activeId) {
        selectedId = activeId;
      } else if (notes.length > 0) {
        selectedId = notes[0].id;
      }
    } catch (e) {
      console.error('Failed to load meeting notes:', e);
    }
    loading = false;

    const u1 = await onMeetingNoteCreated((p) => {
      // Add to top of list and select it.
      notes = [p.note, ...notes];
      selectedId = p.id;
      userScrolledUp = false;
      // Link the new note to the active import so the processing indicator
      // appears immediately (before the first import-progress event with id).
      if (importing) importingNoteId = p.id;
    });

    const u2 = await onMeetingNoteUpdated((p) => {
      const n = notes.find((x) => x.id === p.id);
      if (n) {
        if (p.start !== undefined) {
          // Import path: build JSONL segment so walToText renders timestamps + speaker labels.
          const seg = JSON.stringify({ speaker: p.speaker ?? '', text: p.delta, start: p.start, end: p.end ?? p.start, words: [] });
          n.transcript = n.transcript ? n.transcript + '\n' + seg : seg;
        } else {
          // Live meeting feeder: plain-text delta accumulation (no timestamps yet).
          n.transcript += p.delta;
        }
        n.duration_secs = p.duration_secs;
        notes = [...notes];
      }
      if (selectedId === p.id) {
        autoScroll();
      }
    });

    const u3 = await onMeetingNoteFinalized(async (p) => {
      try {
        // Re-fetch from SQLite to get the authoritative transcript
        // (includes post-loop flush text that may not have been emitted as a delta).
        const fresh = await getMeetingNote(p.id);
        const idx = notes.findIndex((x) => x.id === p.id);
        if (idx !== -1) {
          notes[idx] = fresh;
          notes = [...notes];
        }
      } catch {
        // Fallback: just flip the flag
        const n = notes.find((x) => x.id === p.id);
        if (n) {
          n.is_recording = false;
          notes = [...notes];
        }
      }
    });

    const u4 = await onImportProgress((p) => {
      if (p.id) importingNoteId = p.id;
      importProgress = p.progress;
    });

    // Drag-and-drop: listen for file drops on this window
    const webview = getCurrentWebviewWindow();
    const u5 = await webview.onDragDropEvent((event) => {
      if (event.payload.type === 'over') {
        dragOver = true;
      } else if (event.payload.type === 'drop') {
        dragOver = false;
        const paths = event.payload.paths;
        if (paths.length > 0) {
          const file = paths[0];
          if (isAudioFile(file)) {
            startImport(file);
          }
        }
      } else {
        dragOver = false;
      }
    });

    unlisteners = [u1, u2, u3, u4, u5];

    // Close context menu on outside click
    document.addEventListener('click', closeContextMenu);
  });

  onDestroy(() => {
    for (const unlisten of unlisteners) {
      unlisten();
    }
    document.removeEventListener('click', closeContextMenu);
    if (copiedTimeout) clearTimeout(copiedTimeout);
  });
</script>

<div class="meeting-page">
  <!-- Drag-and-drop overlay -->
  {#if dragOver}
    <div class="drop-overlay">
      <div class="drop-overlay-content">
        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        <p>{t('meeting.importDrop')}</p>
      </div>
    </div>
  {/if}

  <!-- Left panel: note list -->
  <div class="note-list-panel">
    <div class="list-header">
      <h2>{t('nav.meeting')}</h2>
      <button
        class="import-btn"
        onclick={handleImportClick}
        disabled={importing}
        title={t('meeting.import')}
      >
        {#if importing && !importingNoteId}
          <span class="spinner-small" style="width:16px;height:16px;border-width:2px;"></span>
        {:else}
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        {/if}
      </button>
    </div>

    <div class="note-list">
      {#if loading}
        <div class="list-empty">
          <span class="spinner-small"></span>
        </div>
      {:else if notes.length === 0}
        <div class="list-empty">
          <p class="empty-title">{t('meeting.emptyTitle')}</p>
          <p class="empty-hint">{t('meeting.emptyHint')}</p>
          <button class="empty-import-btn" onclick={handleImportClick} disabled={importing}>
            {t('meeting.importOrImport')}
          </button>
        </div>
      {:else}
        {#each notes as note (note.id)}
          <!-- svelte-ignore a11y_no_static_element_interactions -->
          <div
            class="note-item-wrapper"
            class:active={selectedId === note.id}
          >
            <button
              class="note-item"
              class:active={selectedId === note.id}
              onclick={() => {
                selectedId = note.id;
                userScrolledUp = false;
              }}
              oncontextmenu={(e) => handleContextMenu(e, note.id)}
            >
              <div class="note-item-top">
                {#if importing && importingNoteId === note.id}
                  <span class="importing-dot"></span>
                {:else if note.is_recording}
                  <span class="recording-dot"></span>
                {/if}
                <span class="note-title">{defaultTitle(note)}</span>
              </div>
              <div class="note-item-meta">
                {#if importing && importingNoteId === note.id}
                  <span>{t('meeting.importing')} {Math.round(importProgress * 100)}%</span>
                {:else}
                  <span>{formatDate(note.created_at)}</span>
                  <span class="meta-sep">·</span>
                  <span>{formatDuration(note.duration_secs)}</span>
                {/if}
              </div>
              <div class="note-item-preview">{preview(note.transcript)}</div>
            </button>
            {#if !note.is_recording}
              <button
                class="note-delete-btn"
                onclick={(e) => { e.stopPropagation(); handleDelete(note.id); }}
                title={t('meeting.delete')}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
              </button>
            {/if}
          </div>
        {/each}
      {/if}
    </div>
  </div>

  <!-- Right panel: note content -->
  <div class="note-content-panel">
    {#if selectedNote}
      <div class="content-header">
        {#if editingTitleId === selectedNote.id}
          <input
            class="title-input"
            bind:this={titleInputEl}
            bind:value={editingTitleValue}
            onblur={commitRename}
            onkeydown={handleTitleKeydown}
          />
        {:else}
          <h1 class="content-title" ondblclick={() => startRename(selectedNote!)}>
            {defaultTitle(selectedNote)}
            <button class="rename-btn" onclick={() => startRename(selectedNote!)} title={t('meeting.rename')}>
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
            </button>
            <button
              class="polish-btn"
              onclick={handlePolish}
              disabled={polishing || selectedNote!.is_recording || !selectedNote!.transcript}
              title={t(polishing ? 'meeting.polishing' : selectedNote!.summary ? 'meeting.repolish' : 'meeting.polish')}
            >
              {#if polishing}
                <span class="spinner-inline"></span>
              {:else}
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/></svg>
              {/if}
            </button>
          </h1>
        {/if}
      </div>

      {#if importing && importingNoteId === selectedNote.id}
        <div class="import-progress-bar">
          <div class="import-progress-fill" style="width: {importProgress * 100}%"></div>
        </div>
      {/if}

      <div class="content-divider"></div>

      {#if selectedNote.summary || polishing}
        <div class="tab-bar">
          <button
            class="tab-btn"
            class:active={activeTab === 'transcript'}
            onclick={() => (activeTab = 'transcript')}
          >
            {t('meeting.tabTranscript')}
          </button>
          <button
            class="tab-btn"
            class:active={activeTab === 'summary'}
            onclick={() => (activeTab = 'summary')}
          >
            {t('meeting.tabSummary')}
          </button>
        </div>
      {/if}

      <div
        class="transcript-area"
        bind:this={transcriptEl}
        onscroll={handleTranscriptScroll}
      >
        {#if activeTab === 'summary'}
          {#if selectedNote.summary}
            <div class="summary-text markdown-body">{@html renderMarkdown(selectedNote.summary)}</div>
          {:else}
            <p class="no-content">{t('meeting.noSummaryYet')}</p>
          {/if}
        {:else if selectedNote.transcript}
          {@const segs = parseTranscriptSegs(selectedNote.transcript)}
          {#if segs}
            {@const groups = groupSegsBySpeaker(segs)}
            {@const hasSpeakers = segs.some((s) => s.speaker !== '')}
            <div class="transcript-body">
              {#each groups as group}
                <div
                  class="speaker-block"
                  style={hasSpeakers
                    ? `--spk-color: ${SPEAKER_COLORS[group.speakerIdx % SPEAKER_COLORS.length]}`
                    : ''}
                >
                  <div class="speaker-header">
                    {#if group.speaker}
                      <span class="speaker-label">{formatSpeaker(group.speaker)}</span>
                    {/if}
                    <span class="speaker-ts">{formatSegTime(group.start)}</span>
                  </div>
                  <p class="speaker-text">{group.text}</p>
                </div>
              {/each}
            </div>
          {:else}
            <pre class="transcript-text">{selectedNote.transcript}</pre>
          {/if}
        {:else if importing && importingNoteId === selectedNote.id}
          <div class="processing-state">
            <span class="spinner-small"></span>
            <p class="no-content">{t('meeting.processingAudio')}</p>
          </div>
        {:else if selectedNote.is_recording}
          <p class="no-content">{t('meeting.noSpeechDetected')}</p>
        {:else}
          <p class="no-content">{t('meeting.noContent')}</p>
        {/if}
      </div>

      <div class="content-footer">
        {#if selectedNote.audio_path && !selectedNote.is_recording}
          <div class="audio-actions">
            <button
              class="download-audio-btn"
              onclick={handleDownloadAudio}
              disabled={downloading}
              title={t('meeting.downloadAudio')}
            >
              {#if downloading}
                <span class="spinner-inline"></span>
              {:else}
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              {/if}
              {t('meeting.downloadAudio')}
            </button>
            <button
              class="delete-audio-btn"
              onclick={handleDeleteAudio}
              title={t('meeting.deleteAudio')}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            </button>
          </div>
        {/if}
        <div class="footer-meta">
          {#if importing && importingNoteId === selectedNote.id}
            <span class="meta-importing">
              <span class="importing-dot-small"></span>
              {t('meeting.importing')} {Math.round(importProgress * 100)}%
            </span>
          {:else if selectedNote.is_recording}
            <span class="meta-recording">
              <span class="recording-dot-small"></span>
              {t('meeting.recording')}
            </span>
          {/if}
          <span>{formatDuration(selectedNote.duration_secs)}</span>
          <span class="meta-sep">·</span>
          <span>{selectedNote.word_count} {t('meeting.words')}</span>
          <span class="meta-sep">·</span>
          <span>{selectedNote.stt_model}</span>
          <span class="meta-sep">·</span>
          <span>{formatDate(selectedNote.created_at)} {formatTime(selectedNote.created_at)}</span>
        </div>
        <div class="copy-btn-group">
          <button
            class="copy-btn"
            onclick={handleCopy}
            disabled={activeTab === 'summary' ? !selectedNote.summary : !selectedNote.transcript}
          >
            {#if copied}
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
              {t('meeting.copied')}
            {:else}
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
              {activeTab === 'summary' ? t('meeting.copy') : t('meeting.copyTranscript')}
            {/if}
          </button>
          {#if activeTab === 'summary' && selectedNote.summary}
            <button
              class="copy-chevron-btn"
              onclick={toggleCopyMenu}
              disabled={!selectedNote.summary}
              aria-label="Copy options"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
            </button>
            {#if copyMenuOpen}
              <!-- svelte-ignore a11y_click_events_have_key_events -->
              <!-- svelte-ignore a11y_no_static_element_interactions -->
              <div class="copy-dropdown" onclick={(e: MouseEvent) => e.stopPropagation()}>
                <button onclick={handleCopyMarkdown}>
                  {t('meeting.copyAsMarkdown')}
                </button>
              </div>
            {/if}
          {/if}
        </div>
      </div>
    {:else if !loading}
      <div class="content-empty">
        <p class="empty-title">{t('meeting.emptyTitle')}</p>
        <p class="empty-hint">{t('meeting.emptyHint')}</p>
        <button class="empty-import-btn" onclick={handleImportClick} disabled={importing}>
          {t('meeting.importOrImport')}
        </button>
      </div>
    {/if}
  </div>

  <!-- Context menu -->
  {#if contextMenuId}
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <div class="context-menu" style="left:{contextMenuX}px;top:{contextMenuY}px">
      <button onclick={() => { const n = notes.find(x => x.id === contextMenuId); if (n) startRename(n); }}>
        {t('meeting.rename')}
      </button>
      <button class="danger" onclick={() => { if (contextMenuId) handleDelete(contextMenuId); }}>
        {t('meeting.delete')}
      </button>
    </div>
  {/if}
</div>

<style>
  .meeting-page {
    display: flex;
    height: 100%;
    min-height: 0;
    position: relative;
  }

  /* ── Left panel ── */
  .note-list-panel {
    width: 260px;
    min-width: 260px;
    border-right: 1px solid var(--border-divider);
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .list-header {
    padding: 16px 16px 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }

  .list-header h2 {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
  }

  .note-list {
    flex: 1;
    overflow-y: auto;
    padding: 0 8px 8px;
  }

  .list-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
    text-align: center;
  }

  .empty-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary);
    margin: 0 0 4px;
  }

  .empty-hint {
    font-size: 12px;
    color: var(--text-tertiary);
    margin: 0;
  }

  .note-item-wrapper {
    position: relative;
    border-radius: var(--radius-md, 8px);
    transition: background 0.15s;
  }
  .note-item-wrapper:hover {
    background: var(--bg-hover);
  }
  .note-item-wrapper.active {
    background: var(--bg-active);
  }

  .note-item-wrapper:hover .note-delete-btn {
    opacity: 1;
  }

  .note-item {
    width: 100%;
    text-align: left;
    background: none;
    border: none;
    padding: 10px 12px;
    border-radius: var(--radius-md, 8px);
    cursor: pointer;
    font-family: inherit;
  }

  .note-delete-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    background: none;
    border: none;
    color: var(--text-tertiary);
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    opacity: 0;
    transition: opacity 0.15s, color 0.15s;
  }
  .note-delete-btn:hover {
    color: #ff3b30;
    background: var(--bg-hover);
  }

  .note-item-top {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .note-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .recording-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #34c759;
    flex-shrink: 0;
    animation: dotPulse 1.8s ease-in-out infinite;
  }

  @keyframes dotPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .note-item-meta {
    font-size: 11px;
    color: var(--text-tertiary);
    margin-top: 2px;
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .meta-sep {
    opacity: 0.5;
  }

  .note-item-preview {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 4px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* ── Right panel ── */
  .note-content-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    height: 100%;
  }

  .content-header {
    padding: 16px 24px 8px;
    flex-shrink: 0;
  }

  .content-title {
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .rename-btn {
    background: none;
    border: none;
    color: var(--text-tertiary);
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    display: flex;
    opacity: 0.3;
    transition: opacity 0.15s;
  }
  .content-title:hover .rename-btn {
    opacity: 1;
  }
  .rename-btn:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }

  .title-input {
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
    background: none;
    border: none;
    border-bottom: 2px solid var(--accent-primary, #007aff);
    outline: none;
    width: 100%;
    padding: 0 0 2px;
    font-family: inherit;
  }

  .content-divider {
    height: 1px;
    background: var(--border-divider);
    margin: 0 24px;
    flex-shrink: 0;
  }

  .transcript-area {
    flex: 1;
    overflow-y: auto;
    padding: 16px 24px;
    min-height: 0;
  }

  /* ── Speaker-differentiated transcript blocks ── */
  .transcript-body {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .speaker-block {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding-left: 12px;
    border-left: 2.5px solid var(--spk-color, var(--border-divider));
  }

  .speaker-header {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .speaker-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--spk-color, var(--text-tertiary));
    line-height: 1.4;
    letter-spacing: 0.01em;
  }

  .speaker-ts {
    font-size: 11px;
    color: var(--text-tertiary);
    font-variant-numeric: tabular-nums;
    line-height: 1.4;
  }

  .speaker-text {
    font-size: 14px;
    line-height: 1.75;
    color: var(--text-primary);
    margin: 0;
    word-break: break-word;
  }

  /* Legacy plain-text fallback */
  .transcript-text {
    font-size: 14px;
    line-height: 1.7;
    color: var(--text-primary);
    white-space: pre-wrap;
    word-break: break-word;
    margin: 0;
    font-family: inherit;
  }

  .no-content {
    font-size: 14px;
    color: var(--text-tertiary);
    font-style: italic;
    margin: 0;
  }

  .processing-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding-top: 40px;
  }

  .content-footer {
    flex-shrink: 0;
    padding: 12px 24px;
    border-top: 1px solid var(--border-divider);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .footer-meta {
    font-size: 11px;
    color: var(--text-tertiary);
    display: flex;
    align-items: center;
    gap: 4px;
    flex-wrap: wrap;
    min-width: 0;
  }

  .meta-recording {
    display: flex;
    align-items: center;
    gap: 4px;
    color: #34c759;
    font-weight: 600;
  }

  .recording-dot-small {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #34c759;
    animation: dotPulse 1.8s ease-in-out infinite;
  }

  .audio-actions {
    display: flex;
    align-items: center;
    gap: 4px;
    flex-shrink: 0;
  }

  .download-audio-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid var(--border-subtle);
    background: var(--bg-hover);
    color: var(--text-secondary);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    font-family: inherit;
    transition: all 0.15s;
    flex-shrink: 0;
  }
  .download-audio-btn:hover:not(:disabled) {
    background: var(--bg-active);
    color: var(--text-primary);
  }
  .download-audio-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .delete-audio-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 6px 8px;
    border-radius: 6px;
    border: 1px solid var(--border-subtle);
    background: var(--bg-hover);
    color: var(--text-tertiary);
    cursor: pointer;
    transition: all 0.15s;
    flex-shrink: 0;
  }
  .delete-audio-btn:hover {
    background: rgba(255, 59, 48, 0.1);
    color: #ff3b30;
    border-color: rgba(255, 59, 48, 0.3);
  }

  .copy-btn-group {
    position: relative;
    display: flex;
    align-items: center;
    flex-shrink: 0;
  }

  .copy-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid var(--border-subtle);
    background: var(--bg-hover);
    color: var(--text-secondary);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    font-family: inherit;
    transition: all 0.15s;
    flex-shrink: 0;
  }
  .copy-btn-group .copy-btn {
    border-top-right-radius: 0;
    border-bottom-right-radius: 0;
    border-right: none;
  }
  .copy-btn-group .copy-btn:only-child {
    border-radius: 6px;
    border-right: 1px solid var(--border-subtle);
  }
  .copy-btn:hover:not(:disabled) {
    background: var(--bg-active);
    color: var(--text-primary);
  }
  .copy-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .copy-chevron-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 6px 6px;
    border-radius: 0 6px 6px 0;
    border: 1px solid var(--border-subtle);
    border-left: 1px solid var(--border-divider);
    background: var(--bg-hover);
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
    align-self: stretch;
  }
  .copy-chevron-btn:hover:not(:disabled) {
    background: var(--bg-active);
    color: var(--text-primary);
  }
  .copy-chevron-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .copy-dropdown {
    position: absolute;
    bottom: 100%;
    right: 0;
    margin-bottom: 4px;
    background: var(--bg-sidebar, #1c1c20);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 4px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    z-index: 100;
    min-width: 160px;
  }
  .copy-dropdown button {
    width: 100%;
    text-align: left;
    background: none;
    border: none;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
    color: var(--text-primary);
    cursor: pointer;
    font-family: inherit;
    white-space: nowrap;
  }
  .copy-dropdown button:hover {
    background: var(--bg-hover);
  }

  .polish-btn {
    background: none;
    border: none;
    color: var(--color-warning, #f5a623);
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    display: flex;
    opacity: 0.5;
    transition: opacity 0.15s, color 0.15s;
  }
  .content-title:hover .polish-btn {
    opacity: 1;
  }
  .polish-btn:hover:not(:disabled) {
    color: var(--color-warning, #f5a623);
    background: var(--bg-hover);
  }
  .polish-btn:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }

  .spinner-inline {
    width: 14px;
    height: 14px;
    border: 2px solid var(--border-subtle);
    border-top-color: var(--color-warning, #f5a623);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: block;
  }

  .tab-bar {
    display: flex;
    gap: 0;
    padding: 8px 24px 0;
    flex-shrink: 0;
  }

  .tab-btn {
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 6px 16px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-tertiary);
    cursor: pointer;
    font-family: inherit;
    transition: color 0.15s, border-color 0.15s;
  }
  .tab-btn:hover {
    color: var(--text-secondary);
  }
  .tab-btn.active {
    color: var(--text-primary);
    border-bottom-color: var(--accent-primary, #007aff);
  }

  .summary-text {
    font-size: 14px;
    line-height: 1.7;
    color: var(--text-primary);
    word-break: break-word;
  }

  /* ── Markdown rendered summary ── */
  .summary-text.markdown-body :global(h2) {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 20px 0 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border-divider);
  }
  .summary-text.markdown-body :global(h2:first-child) {
    margin-top: 0;
  }
  .summary-text.markdown-body :global(h3) {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 16px 0 6px;
  }
  .summary-text.markdown-body :global(ul) {
    margin: 4px 0 12px;
    padding-left: 20px;
  }
  .summary-text.markdown-body :global(li) {
    margin-bottom: 4px;
    line-height: 1.6;
  }
  .summary-text.markdown-body :global(p) {
    margin: 8px 0;
  }
  .summary-text.markdown-body :global(strong) {
    font-weight: 600;
    color: var(--text-primary);
  }
  .summary-text.markdown-body :global(code) {
    font-size: 13px;
    background: var(--bg-hover);
    padding: 1px 4px;
    border-radius: 3px;
  }

  .content-empty {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 40px;
  }

  /* ── Context menu ── */
  .context-menu {
    position: fixed;
    background: var(--bg-sidebar, #1c1c20);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 4px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    z-index: 1000;
    min-width: 140px;
  }
  .context-menu button {
    width: 100%;
    text-align: left;
    background: none;
    border: none;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
    color: var(--text-primary);
    cursor: pointer;
    font-family: inherit;
  }
  .context-menu button:hover {
    background: var(--bg-hover);
  }
  .context-menu button.danger {
    color: #ff3b30;
  }

  .spinner-small {
    width: 20px;
    height: 20px;
    border: 2px solid var(--border-subtle);
    border-top-color: var(--text-tertiary);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* ── Import button ── */
  .import-btn {
    background: none;
    border: none;
    color: var(--text-tertiary);
    cursor: pointer;
    padding: 4px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    transition: color 0.15s, background 0.15s;
  }
  .import-btn:hover:not(:disabled) {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  .import-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .empty-import-btn {
    margin-top: 12px;
    background: none;
    border: 1px solid var(--border-subtle);
    color: var(--accent-primary, #007aff);
    font-size: 12px;
    font-weight: 500;
    padding: 6px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-family: inherit;
    transition: all 0.15s;
  }
  .empty-import-btn:hover:not(:disabled) {
    background: var(--bg-hover);
  }
  .empty-import-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* ── Importing indicator ── */
  .importing-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-primary, #007aff);
    flex-shrink: 0;
    animation: dotPulse 1.8s ease-in-out infinite;
  }

  .importing-dot-small {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-primary, #007aff);
    animation: dotPulse 1.8s ease-in-out infinite;
  }

  .meta-importing {
    display: flex;
    align-items: center;
    gap: 4px;
    color: var(--accent-primary, #007aff);
    font-weight: 600;
  }

  /* ── Import progress bar ── */
  .import-progress-bar {
    height: 3px;
    background: var(--bg-hover);
    margin: 0 24px;
    border-radius: 2px;
    overflow: hidden;
    flex-shrink: 0;
  }

  .import-progress-fill {
    height: 100%;
    background: var(--accent-primary, #007aff);
    border-radius: 2px;
    transition: width 0.3s ease;
  }

  /* ── Drag-and-drop overlay ── */
  .drop-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0, 122, 255, 0.08);
    border: 2px dashed var(--accent-primary, #007aff);
    border-radius: 12px;
    z-index: 500;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: none;
  }

  .drop-overlay-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    color: var(--accent-primary, #007aff);
  }

  .drop-overlay-content p {
    font-size: 14px;
    font-weight: 600;
    margin: 0;
  }
</style>
