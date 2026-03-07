use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

use crate::polisher::truncate_for_error;
use crate::settings::models_dir;
use crate::whisper_models::WhisperModel;


#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SttMode {
    #[default]
    Local,
    Cloud,
}

// ── Local STT engine selection ────────────────────────────────────────────────

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalSttEngine {
    #[default]
    Whisper,
    Qwen3Asr,
}

// ── Qwen3-ASR model variants ──────────────────────────────────────────────────

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen3AsrModel {
    #[default]
    Qwen3Asr1_7B,
    Qwen3Asr0_6B,
}

impl Qwen3AsrModel {
    pub fn model_dir_name(&self) -> &'static str {
        match self {
            Self::Qwen3Asr1_7B => "qwen3-asr-1.7b",
            Self::Qwen3Asr0_6B => "qwen3-asr-0.6b",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Qwen3Asr1_7B => "Qwen3-ASR 1.7B",
            Self::Qwen3Asr0_6B => "Qwen3-ASR 0.6B",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Qwen3Asr1_7B => "Highest accuracy, BF16. 30 languages + 22 Chinese dialects.",
            Self::Qwen3Asr0_6B => "Fast and efficient. Greatly outperforms Whisper on Chinese and noisy speech. 30 languages + 22 Chinese dialects.",
        }
    }

    pub fn size_bytes(&self) -> u64 {
        match self {
            Self::Qwen3Asr1_7B => 4_700_000_000, // BF16 safetensors (sharded)
            Self::Qwen3Asr0_6B => 1_876_000_000, // BF16 safetensors (single)
        }
    }

    /// Files that must be present for the model to be considered downloaded.
    pub fn required_files(&self) -> Vec<&'static str> {
        match self {
            Self::Qwen3Asr1_7B => vec![
                "config.json",
                "tokenizer.json",
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
            Self::Qwen3Asr0_6B => vec![
                "config.json",
                "tokenizer.json",
                "model.safetensors",
            ],
        }
    }

    /// Returns `(filename, hf_repo)` pairs for downloading.
    pub fn download_files(&self) -> Vec<(&'static str, &'static str)> {
        match self {
            Self::Qwen3Asr1_7B => vec![
                ("config.json",                      "Qwen/Qwen3-ASR-1.7B"),
                ("tokenizer.json",                   "Qwen/Qwen3-1.7B"),
                ("model.safetensors.index.json",     "Qwen/Qwen3-ASR-1.7B"),
                ("model-00001-of-00002.safetensors", "Qwen/Qwen3-ASR-1.7B"),
                ("model-00002-of-00002.safetensors", "Qwen/Qwen3-ASR-1.7B"),
            ],
            Self::Qwen3Asr0_6B => vec![
                ("config.json",       "Qwen/Qwen3-ASR-0.6B"),
                ("tokenizer.json",    "Qwen/Qwen3-0.6B"),
                ("model.safetensors", "Qwen/Qwen3-ASR-0.6B"),
            ],
        }
    }

    /// Sorted large → small by size_bytes.
    pub fn all() -> &'static [Self] {
        &[Self::Qwen3Asr1_7B, Self::Qwen3Asr0_6B]
    }
}

/// Returns the on-disk directory for a Qwen3-ASR model.
pub fn qwen3_asr_model_dir(model: &Qwen3AsrModel) -> PathBuf {
    models_dir().join(model.model_dir_name())
}

/// Returns true if all required files for the model are present on disk.
pub fn is_qwen3_asr_downloaded(model: &Qwen3AsrModel) -> bool {
    let dir = qwen3_asr_model_dir(model);
    model.required_files().iter().all(|f| dir.join(f).exists())
}

#[derive(Debug, Clone, Serialize)]
pub struct Qwen3AsrModelInfo {
    pub id: Qwen3AsrModel,
    pub display_name: &'static str,
    pub description: &'static str,
    pub size_bytes: u64,
    pub downloaded: bool,
    pub file_size_on_disk: u64,
    pub is_active: bool,
}

impl Qwen3AsrModelInfo {
    pub fn from_model(model: &Qwen3AsrModel, active: &Qwen3AsrModel) -> Self {
        let dir = qwen3_asr_model_dir(model);
        let file_size_on_disk: u64 = model
            .required_files()
            .iter()
            .filter_map(|f| std::fs::metadata(dir.join(f)).ok())
            .map(|m| m.len())
            .sum();
        let downloaded = is_qwen3_asr_downloaded(model);
        Self {
            id: model.clone(),
            display_name: model.display_name(),
            description: model.description(),
            size_bytes: model.size_bytes(),
            downloaded,
            file_size_on_disk,
            is_active: model == active,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SttProvider {
    #[default]
    Deepgram,
    Groq,
    OpenAi,
    Azure,
    Custom,
}

impl SttProvider {
    pub fn as_key(&self) -> &'static str {
        match self {
            Self::Deepgram => "stt_deepgram",
            Self::Groq => "stt_groq",
            Self::OpenAi => "stt_open_ai",
            Self::Azure => "stt_azure",
            Self::Custom => "stt_custom",
        }
    }

    pub fn default_endpoint(&self) -> &'static str {
        match self {
            Self::Deepgram => "https://api.deepgram.com/v1/listen",
            Self::Groq => "https://api.groq.com/openai/v1/audio/transcriptions",
            Self::OpenAi => "https://api.openai.com/v1/audio/transcriptions",
            Self::Azure => "",
            Self::Custom => "",
        }
    }

    pub fn default_model(&self) -> &'static str {
        match self {
            Self::Deepgram => "whisper",
            Self::Groq => "whisper-large-v3-turbo",
            Self::OpenAi => "whisper-1",
            Self::Azure => "",
            Self::Custom => "",
        }
    }

    /// Whether this provider uses the OpenAI-compatible multipart API.
    pub fn is_openai_compatible(&self) -> bool {
        matches!(self, Self::Groq | Self::OpenAi | Self::Custom)
    }

    /// Whether the provider requires an endpoint URL from the user.
    pub fn requires_endpoint(&self) -> bool {
        matches!(self, Self::Azure | Self::Custom)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttCloudConfig {
    #[serde(default)]
    pub provider: SttProvider,
    #[serde(skip)]
    pub api_key: String,
    #[serde(default)]
    pub endpoint: String,
    #[serde(default = "default_stt_model_id")]
    pub model_id: String,
    /// BCP-47 language code for STT (e.g. "zh-TW", "en", "ja").
    /// Empty string means auto-detect (provider-dependent).
    #[serde(default = "default_stt_language")]
    pub language: String,
}

fn default_stt_model_id() -> String {
    "whisper".to_string()
}

fn default_stt_language() -> String {
    "auto".to_string()
}

impl Default for SttCloudConfig {
    fn default() -> Self {
        Self {
            provider: SttProvider::default(),
            api_key: String::new(),
            endpoint: String::new(),
            model_id: default_stt_model_id(),
            language: default_stt_language(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttConfig {
    #[serde(default)]
    pub mode: SttMode,
    #[serde(default)]
    pub cloud: SttCloudConfig,
    #[serde(default)]
    pub whisper_model: WhisperModel,
    /// Which local STT engine to use when mode is Local.
    #[serde(default)]
    pub local_engine: LocalSttEngine,
    /// Which Qwen3-ASR model variant to use.
    #[serde(default)]
    pub qwen3_asr_model: Qwen3AsrModel,
    /// BCP-47 language code shared by both local and cloud STT.
    /// Migrated from `cloud.language` for older settings files.
    #[serde(default = "default_stt_language")]
    pub language: String,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            mode: SttMode::default(),
            cloud: SttCloudConfig::default(),
            whisper_model: WhisperModel::default(),
            local_engine: LocalSttEngine::default(),
            qwen3_asr_model: Qwen3AsrModel::default(),
            language: default_stt_language(),
        }
    }
}

impl SttConfig {
    /// Migrate: if top-level `language` is the default but `cloud.language`
    /// was customised, pull it up.  Called once on settings load.
    pub fn migrate_language(&mut self) {
        // Treat empty string as "auto"
        if self.language.is_empty() {
            self.language = default_stt_language();
        }
        if self.cloud.language.is_empty() {
            self.cloud.language = default_stt_language();
        }
        if self.language == default_stt_language() && self.cloud.language != default_stt_language() {
            self.language = self.cloud.language.clone();
        }
        // Keep cloud.language in sync so cloud providers still work.
        self.cloud.language = self.language.clone();
    }
}

/// Map a raw system locale identifier (e.g. `"zh_tw"`, `"en_us"`) to a
/// BCP-47 language code recognised by Whisper.  Returns `"auto"` when the
/// locale cannot be mapped.
pub fn locale_to_stt_language(locale: &str) -> String {
    let lower = locale.to_lowercase();
    // Strip encoding suffix (e.g. "en_us.utf-8" → "en_us")
    let base = lower.split('.').next().unwrap_or(&lower);

    // Chinese: region/script matters
    if base.starts_with("zh") {
        if base.contains("tw") || base.contains("hant") {
            return "zh-TW".to_string();
        }
        return "zh-CN".to_string();
    }

    // Extract the language part (before _ or -)
    let lang = base.split(['_', '-']).next().unwrap_or(base);

    const VALID: &[&str] = &[
        "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs",
        "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi",
        "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy",
        "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "lb", "ln",
        "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my",
        "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa",
        "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta",
        "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi",
        "yo", "yue",
    ];

    if VALID.contains(&lang) {
        lang.to_string()
    } else {
        "auto".to_string()
    }
}

/// Transcribe audio via a cloud STT API.
///
/// `prompt`: optional context text (e.g. previous transcript) for Groq/OpenAI
/// compatible APIs. Ignored by Deepgram/Azure.
pub fn run_cloud_stt(stt_cloud: &SttCloudConfig, samples_16k: &[f32], client: &reqwest::blocking::Client, prompt: Option<&str>) -> Result<String, String> {
    if stt_cloud.api_key.is_empty() {
        return Err("Cloud STT API key is not set. Please configure it in Settings.".to_string());
    }

    let endpoint = if stt_cloud.provider == SttProvider::Azure {
        let region = stt_cloud.endpoint.trim();
        if region.is_empty() {
            return Err("Azure region is not configured. Please set it in Settings.".to_string());
        }
        // Azure region only allows lowercase letters, digits, and hyphens (e.g. "westus", "east-us-2").
        // Must not begin/end with a dash or contain consecutive dashes (RFC 952 DNS label rules).
        if !region.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
            return Err("Azure region contains invalid characters. Use only lowercase letters, digits, and hyphens (e.g. \"westus\", \"east-us-2\").".to_string());
        }
        if region.starts_with('-') || region.ends_with('-') || region.contains("--") {
            return Err("Azure region must not begin or end with a hyphen, or contain consecutive hyphens.".to_string());
        }
        format!(
            "https://{}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1",
            region
        )
    } else if stt_cloud.provider == SttProvider::Custom {
        if stt_cloud.endpoint.is_empty() {
            return Err("Cloud STT endpoint is not configured.".to_string());
        }
        crate::polisher::validate_custom_endpoint(&stt_cloud.endpoint)?;
        stt_cloud.endpoint.clone()
    } else {
        let default_ep = stt_cloud.provider.default_endpoint();
        if default_ep.is_empty() {
            if !stt_cloud.endpoint.is_empty() {
                crate::polisher::validate_custom_endpoint(&stt_cloud.endpoint)?;
            }
            stt_cloud.endpoint.clone()
        } else {
            default_ep.to_string()
        }
    };
    if endpoint.is_empty() {
        return Err("Cloud STT endpoint is not configured.".to_string());
    }

    let model_id = {
        let default = stt_cloud.provider.default_model();
        if default.is_empty() {
            stt_cloud.model_id.clone()
        } else {
            default.to_string()
        }
    };

    // Encode f32 samples → 16-bit PCM WAV in-memory
    let wav_bytes = {
        let num_samples = samples_16k.len();
        let data_size = (num_samples * 2) as u32;
        let file_size = 36 + data_size;
        let mut buf = Vec::with_capacity(44 + data_size as usize);

        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&16000u32.to_le_bytes());
        buf.extend_from_slice(&32000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());
        for &s in samples_16k {
            let clamped = s.clamp(-1.0, 1.0);
            let val = (clamped * 32767.0) as i16;
            buf.extend_from_slice(&val.to_le_bytes());
        }
        buf
    };

    let language = if stt_cloud.language == "auto" { "" } else { &stt_cloud.language };

    let resp = match stt_cloud.provider {
        SttProvider::Deepgram => {
            let lang_param = if language.is_empty() { "multi".to_string() } else { language.to_string() };
            client
                .post(endpoint)
                .query(&[
                    ("model", model_id.as_str()),
                    ("language", lang_param.as_str()),
                    ("punctuate", "true"),
                    ("smart_format", "true"),
                ])
                .header("Authorization", format!("Token {}", stt_cloud.api_key))
                .header("Content-Type", "audio/wav")
                .body(wav_bytes)
                .send()
                .map_err(|e| format!("Cloud STT request failed: {}", e))?
        }
        SttProvider::Azure => {
            let lang_param = if language.is_empty() { "en-US".to_string() } else { language.to_string() };
            let url = format!("{}?language={}&format=simple", endpoint, lang_param);
            client
                .post(&url)
                .header("Ocp-Apim-Subscription-Key", &stt_cloud.api_key)
                .header("Content-Type", "audio/wav; codecs=audio/pcm; samplerate=16000")
                .header("Accept", "application/json")
                .body(wav_bytes)
                .send()
                .map_err(|e| format!("Cloud STT request failed: {}", e))?
        }
        _ => {
            let file_part = reqwest::blocking::multipart::Part::bytes(wav_bytes)
                .file_name("audio.wav")
                .mime_str("audio/wav")
                .map_err(|e| format!("Failed to create multipart part: {}", e))?;

            let mut form = reqwest::blocking::multipart::Form::new()
                .part("file", file_part)
                .text("model", model_id)
                .text("response_format", "json");

            if !language.is_empty() {
                let iso_lang = language.split('-').next().unwrap_or("").to_string();
                if !iso_lang.is_empty() {
                    form = form.text("language", iso_lang);
                }
            }

            if let Some(p) = prompt {
                if !p.is_empty() {
                    form = form.text("prompt", p.to_string());
                }
            }

            client
                .post(&endpoint)
                .header("Authorization", format!("Bearer {}", stt_cloud.api_key))
                .multipart(form)
                .send()
                .map_err(|e| format!("Cloud STT request failed: {}", e))?
        }
    };

    let status = resp.status();
    let body = resp
        .text()
        .map_err(|e| format!("Failed to read Cloud STT response: {}", e))?;

    if !status.is_success() {
        let preview = truncate_for_error(&body, 200);
        return Err(format!("Cloud STT returned HTTP {}: {}", status, preview));
    }

    let json: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| {
            let preview = truncate_for_error(&body, 200);
            format!("Failed to parse Cloud STT response: {} — body: {}", e, preview)
        })?;

    let text = match stt_cloud.provider {
        SttProvider::Deepgram => {
            json["results"]["channels"]
                .as_array()
                .and_then(|ch| ch.first())
                .and_then(|c| c["alternatives"].as_array())
                .and_then(|alts| alts.first())
                .and_then(|a| a["transcript"].as_str())
                .unwrap_or("")
                .trim()
                .to_string()
        }
        SttProvider::Azure => {
            json["DisplayText"]
                .as_str()
                .unwrap_or("")
                .trim()
                .to_string()
        }
        _ => {
            json["text"]
                .as_str()
                .unwrap_or("")
                .trim()
                .to_string()
        }
    };

    if text.is_empty() {
        Err("no_speech".to_string())
    } else {
        Ok(text)
    }
}

// ── Cloud meeting feeder ─────────────────────────────────────────────────────

/// Meeting-mode feeder for continuous long-form transcription via Cloud STT.
///
/// Mirrors `whisper_streaming::run_whisper_meeting_feeder_loop` but uses
/// stateless `run_cloud_stt` HTTP calls for each silence-separated segment.
/// API failures are logged and skipped (the meeting continues), preventing
/// transient network issues from terminating the entire session.
/// Meeting-mode feeder for continuous long-form transcription via cloud STT.
///
/// Delegates to `meeting_feeder::run_meeting_feeder` with a cloud STT transcription
/// closure that uses the WAL context as a prompt hint.
pub(crate) fn run_cloud_meeting_feeder_loop(
    app: AppHandle,
    cloud_config: SttCloudConfig,
    language: String,
    session_id: u64,
    record_audio: bool,
) {
    let mut cloud_config = cloud_config;
    cloud_config.language = language;

    let app_for_closure = app.clone();
    let transcribe: Box<dyn FnMut(&[f32], &str) -> String + Send + 'static> =
        Box::new(move |samples, prev_text| {
            let state = app_for_closure.state::<crate::AppState>();
            let prompt = if prev_text.is_empty() { None } else { Some(prev_text) };
            match run_cloud_stt(&cloud_config, samples, &state.http_client, prompt) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!("[cloud-meeting] transcription failed, skipping: {e}");
                    String::new()
                }
            }
        });
    // Cap each segment at 120 s to bound per-segment cloud STT cost and keep
    // stop_meeting_mode well within the 5-min timeout even on slow networks.
    crate::meeting_feeder::run_meeting_feeder(app, session_id, "cloud-meeting", Some(120 * 16_000), transcribe, record_audio);
}

