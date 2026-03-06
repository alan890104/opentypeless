use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::polisher;
use crate::stt::SttConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub hotkey: String,
    pub auto_paste: bool,
    #[serde(default)]
    pub polish: polisher::PolishConfig,
    /// 0 = keep forever, otherwise number of days to retain history entries.
    #[serde(default)]
    pub history_retention_days: u32,
    /// UI language override. None = auto-detect from system.
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub stt: SttConfig,
    /// Optional hotkey for "Edit by Voice" — select text, speak editing instruction.
    #[serde(default)]
    pub edit_hotkey: Option<String>,
    /// Whether the onboarding wizard has been completed. `false` triggers the setup overlay.
    #[serde(default)]
    pub onboarding_completed: bool,
    /// Preferred microphone input device name. None = use system default.
    #[serde(default)]
    pub mic_device: Option<String>,
    /// Optional hotkey for meeting transcription mode. None = disabled.
    #[serde(default)]
    pub meeting_hotkey: Option<String>,
    /// Idle mic timeout in seconds. 0 = never close (always-on).
    /// When > 0, the mic stream is closed after this many seconds of inactivity
    /// to prevent CoreAudio DSP (echo cancellation, AGC) from affecting other apps.
    #[serde(default = "default_idle_mic_timeout_secs")]
    pub idle_mic_timeout_secs: u32,
}

fn default_idle_mic_timeout_secs() -> u32 {
    0
}

impl Default for Settings {
    fn default() -> Self {
        let (hotkey, edit_hotkey, meeting_hotkey) = if is_debug() {
            (
                "Alt+Super+KeyV".to_string(),
                Some("Alt+Super+KeyE".to_string()),
                Some("Alt+Super+KeyM".to_string()),
            )
        } else {
            (
                "Alt+KeyV".to_string(),
                Some("Alt+KeyE".to_string()),
                Some("Alt+KeyM".to_string()),
            )
        };
        Self {
            hotkey,
            auto_paste: true,
            polish: polisher::PolishConfig::default(),
            history_retention_days: 0,
            language: None,
            stt: SttConfig::default(),
            edit_hotkey,
            onboarding_completed: false,
            mic_device: None,
            meeting_hotkey,
            idle_mic_timeout_secs: default_idle_mic_timeout_secs(),
        }
    }
}

// ── Consolidated data directory: ~/.sumi (release) or ~/.sumi-dev (debug) ────

pub const fn is_debug() -> bool {
    cfg!(debug_assertions)
}

pub fn base_dir() -> PathBuf {
    let dir_name = if is_debug() { ".sumi-dev" } else { ".sumi" };
    dirs::home_dir()
        .unwrap_or_else(|| {
            tracing::warn!("Home directory not found, using /tmp as fallback");
            PathBuf::from("/tmp")
        })
        .join(dir_name)
}

pub fn config_dir() -> PathBuf {
    base_dir().join("config")
}

pub fn models_dir() -> PathBuf {
    base_dir().join("models")
}

pub fn history_dir() -> PathBuf {
    base_dir().join("history")
}

pub fn audio_dir() -> PathBuf {
    base_dir().join("audio")
}

pub fn logs_dir() -> PathBuf {
    base_dir().join("logs")
}

pub fn settings_path() -> PathBuf {
    config_dir().join("settings.json")
}

/// Load settings from disk. Pure file I/O — no locale detection.
pub fn load_settings() -> Settings {
    let path = settings_path();
    let mut settings = if path.exists() {
        match std::fs::read_to_string(&path) {
            Ok(contents) => match serde_json::from_str(&contents) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("Settings file corrupted ({}), using defaults", e);
                    Settings::default()
                }
            },
            Err(e) => {
                tracing::warn!("Failed to read settings file ({}), using defaults", e);
                Settings::default()
            }
        }
    } else {
        Settings::default()
    };
    settings.stt.migrate_language();
    // Migrate old local polish model names to new ones
    if settings.polish.model == polisher::PolishModel::Unknown {
        settings.polish.model = polisher::recommend_polish_model(settings.language.as_deref());
        save_settings_to_disk(&settings);
    }
    settings
}

/// Fill in any missing locale-dependent fields (UI language, STT language,
/// polish model_id) by detecting the system locale.  Call this once during
/// app setup.
pub fn apply_locale_defaults(settings: &mut Settings) {
    let mut changed = false;
    let is_new_install = settings.stt.language == "auto" || settings.stt.language.is_empty();

    if let Some(locale) = crate::whisper_models::detect_system_language() {
        // STT language & prompt rules
        if is_new_install {
            let lang = crate::stt::locale_to_stt_language(&locale);
            if lang != "auto" {
                // Regenerate prompt rules with detected locale
                let localized_rules = polisher::default_prompt_rules_for_lang(Some(&lang));
                let mut map = std::collections::HashMap::new();
                map.insert("auto".to_string(), localized_rules);
                settings.polish.prompt_rules = map;

                settings.stt.language = lang.clone();
                settings.stt.cloud.language = lang;
                changed = true;
            }
        }

        // Polish cloud model_id for new installs
        if settings.polish.cloud.model_id.is_empty() {
            settings.polish.cloud.model_id =
                polisher::CloudConfig::default_model_id_for_locale(&locale).to_string();
            changed = true;
        }

        // UI language
        if settings.language.is_none() {
            let lang = crate::stt::locale_to_stt_language(&locale);
            if lang != "auto" {
                settings.language = Some(lang);
                changed = true;
            }
        }
    }

    if changed {
        save_settings_to_disk(settings);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Old config (e.g. v0.2) that lacks new fields like `edit_hotkey`,
    /// `meeting_hotkey`, `mic_device`, `onboarding_completed`.
    /// Must deserialize without error, filling in defaults.
    #[test]
    fn deserialize_legacy_config_missing_new_fields() {
        let json = r#"{
            "hotkey": "Alt+KeyZ",
            "auto_paste": true,
            "history_retention_days": 7
        }"#;
        let s: Settings = serde_json::from_str(json).unwrap();
        assert_eq!(s.hotkey, "Alt+KeyZ");
        assert!(s.auto_paste);
        assert_eq!(s.history_retention_days, 7);
        // New fields should fall back to defaults
        assert!(s.edit_hotkey.is_none());
        assert!(s.meeting_hotkey.is_none());
        assert!(s.mic_device.is_none());
        assert!(!s.onboarding_completed);
        assert!(s.language.is_none());
    }

    /// Config with unknown extra fields (forward compat: newer config opened
    /// by older binary). `serde(deny_unknown_fields)` is NOT set, so this
    /// must succeed.
    #[test]
    fn deserialize_config_with_extra_fields() {
        let json = r#"{
            "hotkey": "Alt+KeyZ",
            "auto_paste": false,
            "some_future_field": 42,
            "another_new_thing": "hello"
        }"#;
        let s: Settings = serde_json::from_str(json).unwrap();
        assert_eq!(s.hotkey, "Alt+KeyZ");
        assert!(!s.auto_paste);
    }

    /// Round-trip: serialize then deserialize should produce equivalent settings.
    #[test]
    fn settings_round_trip() {
        let original = Settings::default();
        let json = serde_json::to_string(&original).unwrap();
        let restored: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(original.hotkey, restored.hotkey);
        assert_eq!(original.auto_paste, restored.auto_paste);
        assert_eq!(original.history_retention_days, restored.history_retention_days);
        assert_eq!(original.edit_hotkey, restored.edit_hotkey);
        assert_eq!(original.meeting_hotkey, restored.meeting_hotkey);
        assert_eq!(original.onboarding_completed, restored.onboarding_completed);
        assert_eq!(original.mic_device, restored.mic_device);
        assert_eq!(original.language, restored.language);
    }

    /// Empty JSON object should produce valid defaults for all fields.
    #[test]
    fn deserialize_empty_object() {
        // hotkey and auto_paste have no #[serde(default)] — they must be present.
        // But Settings::default() is used on parse failure, so let's test that path.
        let json = "{}";
        let result: Result<Settings, _> = serde_json::from_str(json);
        // `hotkey` is required (no default attr), so this should fail.
        assert!(result.is_err());
    }

    /// Corrupt JSON falls back to defaults in load_settings.
    /// (We can't call load_settings directly since it reads from disk,
    /// but we can verify the serde fallback behavior.)
    #[test]
    fn corrupt_json_is_err() {
        let result: Result<Settings, _> = serde_json::from_str("not json at all");
        assert!(result.is_err());
    }
}

pub fn save_settings_to_disk(settings: &Settings) {
    let path = settings_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match serde_json::to_string_pretty(settings) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                tracing::error!("Failed to write settings to disk ({}): {}", path.display(), e);
            }
        }
        Err(e) => {
            tracing::error!("Failed to serialize settings: {}", e);
        }
    }
}
