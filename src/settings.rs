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
}

impl Default for Settings {
    fn default() -> Self {
        let (hotkey, edit_hotkey) = if is_debug() {
            ("Alt+Super+KeyZ".to_string(), Some("Control+Alt+Super+KeyZ".to_string()))
        } else {
            ("Alt+KeyZ".to_string(), Some("Control+Alt+KeyZ".to_string()))
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
            Err(_) => Settings::default(),
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

pub fn save_settings_to_disk(settings: &Settings) {
    let path = settings_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(settings) {
        let _ = std::fs::write(&path, json);
    }
}
