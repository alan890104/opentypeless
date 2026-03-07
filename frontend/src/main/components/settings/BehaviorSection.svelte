<script lang="ts">
  import { t } from '$lib/stores/i18n.svelte';
  import { getSettings, setAutoPaste, setIdleMicTimeout, setRecordMeetingAudio, save } from '$lib/stores/settings.svelte';
  import SettingRow from '$lib/components/SettingRow.svelte';
  import SectionHeader from '$lib/components/SectionHeader.svelte';
  import Toggle from '$lib/components/Toggle.svelte';
  import Select from '$lib/components/Select.svelte';

  const settings = $derived(getSettings());

  const micIdleOptions = $derived([
    { value: '0', label: t('settings.behavior.micIdle.off') },
    { value: '30', label: t('settings.behavior.micIdle.30s') },
    { value: '60', label: t('settings.behavior.micIdle.1min') },
    { value: '300', label: t('settings.behavior.micIdle.5min') },
    { value: '600', label: t('settings.behavior.micIdle.10min') },
    { value: '1800', label: t('settings.behavior.micIdle.30min') },
  ]);

  function onToggleAutoPaste(checked: boolean) {
    setAutoPaste(checked);
    save();
  }

  function onMicIdleChange(value: string) {
    setIdleMicTimeout(parseInt(value, 10));
    save();
  }

  function onToggleRecordMeetingAudio(checked: boolean) {
    setRecordMeetingAudio(checked);
    save();
  }
</script>

<div class="section">
  <SectionHeader title={t('settings.behavior')}>
    {#snippet icon()}
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>
        <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>
        <path d="m9 14 2 2 4-4"/>
      </svg>
    {/snippet}
  </SectionHeader>

  <SettingRow name={t('settings.behavior.autoPaste')} desc={t('settings.behavior.autoPasteDesc')}>
    <Toggle checked={settings.auto_paste} onchange={onToggleAutoPaste} />
  </SettingRow>

  <SettingRow name={t('settings.behavior.micIdle')} desc={t('settings.behavior.micIdleDesc')}>
    <Select
      options={micIdleOptions}
      value={String(settings.idle_mic_timeout_secs)}
      onchange={onMicIdleChange}
    />
  </SettingRow>

  <SettingRow name={t('settings.behavior.recordMeetingAudio')} desc={t('settings.behavior.recordMeetingAudioDesc')}>
    <Toggle checked={settings.record_meeting_audio} onchange={onToggleRecordMeetingAudio} />
  </SettingRow>
</div>

<style>
  .section {
    margin-bottom: 32px;
  }
</style>
