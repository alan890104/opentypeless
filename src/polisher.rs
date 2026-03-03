use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use unicode_segmentation::UnicodeSegmentation;

use candle_core::quantized::gguf_file;
use candle_core::quantized::tokenizer::TokenizerFromGguf;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_llama::ModelWeights as LlamaWeights;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2Weights;
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3Weights;

use crate::context_detect::AppContext;

// ── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolishConfig {
    pub enabled: bool,
    #[serde(default)]
    pub model: PolishModel,
    #[serde(default)]
    pub custom_prompt: Option<String>,
    #[serde(default)]
    pub mode: PolishMode,
    #[serde(default)]
    pub cloud: CloudConfig,
    #[serde(
        default = "default_prompt_rules_map",
        deserialize_with = "deserialize_prompt_rules"
    )]
    pub prompt_rules: HashMap<String, Vec<PromptRule>>,
    #[serde(default)]
    pub dictionary: DictionaryConfig,
    /// Enable model reasoning / chain-of-thought (e.g. Qwen3 `<think>` blocks).
    /// When false, `/no_think` is prepended to suppress reasoning.
    #[serde(default)]
    pub reasoning: bool,
}

impl Default for PolishConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: PolishModel::default(),
            custom_prompt: None,
            mode: PolishMode::default(),
            cloud: CloudConfig::default(),
            prompt_rules: default_prompt_rules_map(),
            dictionary: DictionaryConfig::default(),
            reasoning: false,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PolishMode {
    Local,
    #[default]
    Cloud,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CloudProvider {
    #[serde(rename = "github_models")]
    GitHubModels,
    Groq,
    OpenRouter,
    OpenAi,
    Gemini,
    SambaNova,
    Custom,
}

impl Default for CloudProvider {
    fn default() -> Self {
        Self::Groq
    }
}

impl CloudProvider {
    /// Returns the snake_case identifier matching the serde serialization.
    pub fn as_key(&self) -> &'static str {
        match self {
            CloudProvider::GitHubModels => "github_models",
            CloudProvider::Groq => "groq",
            CloudProvider::OpenRouter => "open_router",
            CloudProvider::OpenAi => "open_ai",
            CloudProvider::Gemini => "gemini",
            CloudProvider::SambaNova => "samba_nova",
            CloudProvider::Custom => "custom",
        }
    }

    pub fn default_endpoint(&self) -> &'static str {
        match self {
            CloudProvider::GitHubModels => "https://models.github.ai/inference/chat/completions",
            CloudProvider::Groq => "https://api.groq.com/openai/v1/chat/completions",
            CloudProvider::OpenRouter => "https://openrouter.ai/api/v1/chat/completions",
            CloudProvider::OpenAi => "https://api.openai.com/v1/chat/completions",
            CloudProvider::Gemini => "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            CloudProvider::SambaNova => "https://api.sambanova.ai/v1/chat/completions",
            CloudProvider::Custom => "",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudConfig {
    #[serde(default)]
    pub provider: CloudProvider,
    #[serde(skip)]
    pub api_key: String,
    #[serde(default)]
    pub endpoint: String,
    #[serde(default)]
    pub model_id: String,
}

impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            provider: CloudProvider::default(),
            api_key: String::new(),
            endpoint: String::new(),
            model_id: String::new(),
        }
    }
}

impl CloudConfig {
    /// Returns the default model ID for the given locale.
    /// Chinese locales → Qwen 3 32B; others → GPT-oss 120B.
    pub fn default_model_id_for_locale(locale: &str) -> &'static str {
        let lower = locale.to_lowercase();
        if lower.starts_with("zh") {
            "qwen/qwen3-32b"
        } else {
            "openai/gpt-oss-120b"
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PolishModel {
    #[default]
    LlamaTaiwan,
    Qwen25,
    Qwen3,
    #[serde(rename = "qwen3_0_6b")]
    Qwen3_4B,
}

impl PolishModel {
    pub fn filename(&self) -> &'static str {
        match self {
            PolishModel::LlamaTaiwan => "Llama-3-Taiwan-8B-Instruct.Q4_K_M.gguf",
            PolishModel::Qwen25 => "qwen2.5-7b-instruct-q4_k_m.gguf",
            PolishModel::Qwen3 => "Qwen3-8B-Q4_K_M.gguf",
            PolishModel::Qwen3_4B => "Qwen3-4B-Q4_K_M.gguf",
        }
    }

    pub fn download_url(&self) -> &'static str {
        match self {
            PolishModel::LlamaTaiwan => {
                "https://huggingface.co/QuantFactory/Llama-3-Taiwan-8B-Instruct-GGUF/resolve/main/Llama-3-Taiwan-8B-Instruct.Q4_K_M.gguf"
            }
            PolishModel::Qwen25 => {
                "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
            }
            PolishModel::Qwen3 => {
                "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
            }
            PolishModel::Qwen3_4B => {
                "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
            }
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            PolishModel::LlamaTaiwan => "Llama 3 Taiwan 8B",
            PolishModel::Qwen25 => "Qwen 2.5 7B",
            PolishModel::Qwen3 => "Qwen 3 8B",
            PolishModel::Qwen3_4B => "Qwen 3 4B (Demo)",
        }
    }

    pub fn size_bytes(&self) -> u64 {
        match self {
            PolishModel::LlamaTaiwan => 4_920_000_000,
            PolishModel::Qwen25 => 4_680_000_000,
            PolishModel::Qwen3 => 5_030_000_000,
            PolishModel::Qwen3_4B => 2_500_000_000,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            PolishModel::LlamaTaiwan => "Best for Traditional Chinese",
            PolishModel::Qwen25 => "Multilingual",
            PolishModel::Qwen3 => "Latest multilingual, thinking/non-thinking",
            PolishModel::Qwen3_4B => "Compact model for quick demo/testing",
        }
    }

    pub fn all() -> &'static [PolishModel] {
        if cfg!(debug_assertions) {
            &[PolishModel::LlamaTaiwan, PolishModel::Qwen25, PolishModel::Qwen3, PolishModel::Qwen3_4B]
        } else {
            &[PolishModel::LlamaTaiwan, PolishModel::Qwen25, PolishModel::Qwen3]
        }
    }

    fn eos_token(&self) -> &'static str {
        match self {
            PolishModel::LlamaTaiwan => "<|eot_id|>",
            PolishModel::Qwen25 | PolishModel::Qwen3 | PolishModel::Qwen3_4B => "<|im_end|>",
        }
    }
}

// ── PolishModelInfo (for frontend serialization) ─────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct PolishModelInfo {
    pub id: PolishModel,
    pub display_name: &'static str,
    pub description: &'static str,
    pub size_bytes: u64,
    pub downloaded: bool,
    pub file_size_on_disk: u64,
    pub is_active: bool,
}

impl PolishModelInfo {
    pub fn from_model(model: &PolishModel, active_model: &PolishModel) -> Self {
        let dir = crate::settings::models_dir();
        let (downloaded, file_size_on_disk) = model_file_status(&dir, model);
        Self {
            id: model.clone(),
            display_name: model.display_name(),
            description: model.description(),
            size_bytes: model.size_bytes(),
            downloaded,
            file_size_on_disk,
            is_active: model == active_model,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MatchType {
    AppName,
    BundleId,
    Url,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchCondition {
    pub match_type: MatchType,
    pub match_value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptRule {
    pub name: String,
    pub match_type: MatchType,
    pub match_value: String,
    pub prompt: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Optional icon key for the frontend (e.g. "terminal", "slack"). Auto-detected if None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,
    /// Alternative match conditions (OR logic). Rule triggers if primary OR any alt matches.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alt_matches: Vec<MatchCondition>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryEntry {
    pub term: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub entries: Vec<DictionaryEntry>,
}

impl Default for DictionaryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            entries: Vec::new(),
        }
    }
}

impl DictionaryConfig {
    pub fn enabled_terms(&self) -> Vec<String> {
        self.entries
            .iter()
            .filter(|e| e.enabled && !e.term.is_empty())
            .map(|e| e.term.clone())
            .collect()
    }
}

/// Format a system+user prompt using the correct chat template.
fn format_chat_prompt(model: &PolishModel, system: &str, user: &str) -> String {
    match model {
        PolishModel::LlamaTaiwan => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
             {system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
             {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        PolishModel::Qwen25 | PolishModel::Qwen3 | PolishModel::Qwen3_4B => format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{user}<|im_end|>\n\
             <|im_start|>assistant\n"
        ),
    }
}

// ── Cached model ────────────────────────────────────────────────────────────

/// Wraps the three supported quantized model architectures with a unified interface.
enum QuantizedModel {
    Llama(LlamaWeights),
    Qwen2(Qwen2Weights),
    Qwen3(Qwen3Weights),
}

impl QuantizedModel {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(input, index_pos),
            Self::Qwen2(m) => m.forward(input, index_pos),
            Self::Qwen3(m) => m.forward(input, index_pos),
        }
    }

    fn clear_kv_cache(&mut self) {
        match self {
            Self::Llama(_) => {}
            Self::Qwen2(_) => {}
            Self::Qwen3(m) => m.clear_kv_cache(),
        }
    }
}

pub struct LlmModelCache {
    model: QuantizedModel,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    loaded_path: PathBuf,
}

// All candle types and tokenizers::Tokenizer are Send.
// Safety: LlmModelCache is only accessed behind a Mutex.
unsafe impl Send for LlmModelCache {}

/// Returns a per-language map with built-in preset prompt rules.
/// Used by serde `#[serde(default = ...)]` and `PolishConfig::default()`.
fn default_prompt_rules_map() -> HashMap<String, Vec<PromptRule>> {
    let mut map = HashMap::new();
    map.insert("auto".to_string(), default_prompt_rules());
    map
}

/// Backwards-compatible deserializer: accepts either a per-language map (new/old format)
/// or a flat array.
fn deserialize_prompt_rules<'de, D>(
    deserializer: D,
) -> Result<HashMap<String, Vec<PromptRule>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Format {
        Map(HashMap<String, Vec<PromptRule>>),
        List(Vec<PromptRule>),
    }

    match Format::deserialize(deserializer)? {
        Format::Map(map) => Ok(map),
        Format::List(list) => {
            let mut map = HashMap::new();
            map.insert("auto".to_string(), list);
            Ok(map)
        }
    }
}

/// Returns built-in preset prompt rules.
pub fn default_prompt_rules() -> Vec<PromptRule> {
    default_prompt_rules_for_lang(None)
}

/// Returns built-in preset prompt rules localised to `lang` (BCP-47).
/// Falls back to English when a translation is not available.
pub fn default_prompt_rules_for_lang(lang: Option<&str>) -> Vec<PromptRule> {
    let lower = lang.map(|l| l.to_lowercase());
    let is_zh_tw = lower.as_deref().map_or(false, |l| {
        l.starts_with("zh-tw") || l.starts_with("zh_tw") || l.starts_with("zh-hant")
    });
    let is_zh = lower.as_deref().map_or(false, |l| l.starts_with("zh"));

    let (code_editor_prompt, ai_cli_prompt, chat_prompt, email_prompt, notion_prompt, slack_prompt, github_prompt, twitter_prompt) = if is_zh_tw {
        (
            "使用者正在程式碼編輯器中工作（可能在寫程式碼、註解、commit 訊息，或與 AI 程式助手對話）。\n\
             完整保留所有程式碼、指令、路徑、變數名稱和技術術語。\n\
             技術術語保留英文原文，不要翻譯（如 function、commit、merge、deploy）。\n\
             輸出簡潔精確的文字，不要額外解釋。".to_string(),

            "使用者正在終端機中對 AI 程式助手口述提示或訊息。\n\
             語音內容會直接作為 AI 的輸入。\n\
             完整保留所有技術術語、程式碼引用、檔案路徑、變數名稱和指令。\n\
             輸出清晰、結構良好的文字。\n\
             只回覆整理後的文字，不要附加任何其他內容。".to_string(),

            "使用者正在傳送聊天訊息。\n\
             保持輕鬆、自然、口語化的語氣。\n\
             修正語法和贅詞，但保留說話者的個性和語意。\n\
             如果說話者語氣中帶有情緒表達，可以適當保留或加入 emoji。\n\
             只回覆整理後的訊息文字，不要附加任何其他內容。".to_string(),

            "將口述內容整理成正式的電子郵件格式（問候語、正文、結尾）。\n\
             使用專業、清晰、有禮貌的語氣。\n\
             只回覆郵件文字，不要附加任何其他內容。".to_string(),

            "使用者正在 Notion 中撰寫內容（筆記、文件或 Wiki）。\n\
             產出乾淨、結構良好的文字，適合用於文件。\n\
             適當將口述內容轉換為條列式、表格、標題等結構化格式，符合 Notion 的排版風格。\n\
             只回覆整理後的文字，不要附加任何其他內容。".to_string(),

            "使用者正在傳送 Slack 訊息。\n\
             保持專業但親切的語氣。\n\
             修正語法和贅詞，保持簡潔。\n\
             適當使用 Slack 支援的格式：*粗體*、> 引用、`程式碼`、條列式等。\n\
             只回覆整理後的訊息文字，不要附加任何其他內容。".to_string(),

            "使用者正在 GitHub 上工作（如 PR 說明、Issue、Code Review 留言、Commit 訊息、README 或討論區）。\n\
             重要：無論口述使用什麼語言，一律以英文輸出。\n\
             使用清晰、專業、簡潔的語言，適合軟體協作場景。\n\
             完整保留所有技術術語、程式碼引用、檔案路徑和變數名稱。\n\
             當內容暗示有結構時（列表、標題、程式碼區塊），使用 Markdown 格式。\n\
             只回覆整理後的文字，不要附加任何其他內容。".to_string(),

            "使用者正在 X（Twitter）上撰寫貼文或回覆。\n\
             保持簡潔有力，在短篇幅中追求清晰。\n\
             注意控制文字長度，盡量精簡，適合社群貼文的篇幅。\n\
             如果口述提到 hashtag，保留或整理為 #標籤 格式。\n\
             修正語法但保留說話者的語調和風格。\n\
             只回覆整理後的文字，不要附加任何其他內容。".to_string(),
        )
    } else if is_zh {
        (
            "用户正在代码编辑器中工作（可能在写代码、注释、commit 消息，或与 AI 编程助手对话）。\n\
             完整保留所有代码、指令、路径、变量名和技术术语。\n\
             技术术语保留英文原文，不要翻译（如 function、commit、merge、deploy）。\n\
             输出简洁精确的文字，不要额外解释。".to_string(),

            "用户正在终端中对 AI 编程助手口述提示或消息。\n\
             语音内容会直接作为 AI 的输入。\n\
             完整保留所有技术术语、代码引用、文件路径、变量名和指令。\n\
             输出清晰、结构良好的文字。\n\
             只回复整理后的文字，不要附加任何其他内容。".to_string(),

            "用户正在发送聊天消息。\n\
             保持轻松、自然、口语化的语气。\n\
             修正语法和赘词，但保留说话者的个性和语意。\n\
             如果说话者语气中带有情绪表达，可以适当保留或加入 emoji。\n\
             只回复整理后的消息文字，不要附加任何其他内容。".to_string(),

            "将口述内容整理成正式的电子邮件格式（问候语、正文、结尾）。\n\
             使用专业、清晰、有礼貌的语气。\n\
             只回复邮件文字，不要附加任何其他内容。".to_string(),

            "用户正在 Notion 中撰写内容（笔记、文档或 Wiki）。\n\
             产出干净、结构良好的文字，适合用于文档。\n\
             适当将口述内容转换为列表、表格、标题等结构化格式，符合 Notion 的排版风格。\n\
             只回复整理后的文字，不要附加任何其他内容。".to_string(),

            "用户正在发送 Slack 消息。\n\
             保持专业但亲切的语气。\n\
             修正语法和赘词，保持简洁。\n\
             适当使用 Slack 支持的格式：*粗体*、> 引用、`代码`、列表等。\n\
             只回复整理后的消息文字，不要附加任何其他内容。".to_string(),

            "用户正在 GitHub 上工作（如 PR 描述、Issue、Code Review 评论、Commit 消息、README 或讨论区）。\n\
             重要：无论口述使用什么语言，一律以英文输出。\n\
             使用清晰、专业、简洁的语言，适合软件协作场景。\n\
             完整保留所有技术术语、代码引用、文件路径和变量名。\n\
             当内容暗示有结构时（列表、标题、代码块），使用 Markdown 格式。\n\
             只回复整理后的文字，不要附加任何其他内容。".to_string(),

            "用户正在 X（Twitter）上撰写帖子或回复。\n\
             保持简洁有力，在短篇幅中追求清晰。\n\
             注意控制文字长度，尽量精简，适合社交媒体帖子的篇幅。\n\
             如果口述提到 hashtag，保留或整理为 #标签 格式。\n\
             修正语法但保留说话者的语调和风格。\n\
             只回复整理后的文字，不要附加任何其他内容。".to_string(),
        )
    } else {
        (
            "The user is working in a code editor (possibly writing code, comments, commit messages, or chatting with an AI coding assistant).\n\
             Preserve all code, commands, paths, variable names, and technical terms exactly as spoken.\n\
             Never translate technical terms — keep them in English as spoken (e.g. function, commit, merge, deploy).\n\
             Output concise, precise text. No extra explanation.".to_string(),

            "The user is dictating a prompt or message to an AI coding assistant running in the terminal. \
             The spoken text will be sent as input to the AI. \
             Preserve all technical terms, code references, file paths, variable names, and commands exactly. \
             Output clear, well-structured text. \
             Reply with ONLY the cleaned text, nothing else.".to_string(),

            "The user is writing a chat message.\n\
             Keep a casual, natural, and conversational tone.\n\
             Fix grammar and filler words but preserve the speaker's personality and intent.\n\
             If the speaker's tone conveys emotions, feel free to keep or add appropriate emoji.\n\
             Reply with ONLY the cleaned message text, nothing else.".to_string(),

            "Restructure the spoken content into proper email format (greeting, body, sign-off).\n\
             Use a professional, clear, and polite tone.\n\
             Reply with ONLY the email text, nothing else.".to_string(),

            "The user is writing in Notion (notes, docs, or wiki).\n\
             Produce clean, well-structured text suitable for documentation.\n\
             Convert spoken content into structured formats where appropriate: bullet lists, tables, headings, and other Notion-style formatting.\n\
             Reply with ONLY the cleaned text, nothing else.".to_string(),

            "The user is writing a Slack message.\n\
             Keep a professional but approachable tone.\n\
             Fix grammar and filler words. Keep it concise.\n\
             Use Slack-supported formatting where appropriate: *bold*, > quotes, `code`, and bullet lists.\n\
             Reply with ONLY the cleaned message text, nothing else.".to_string(),

            "The user is working on GitHub (e.g. PR description, issue, code review comment, commit message, README, or discussion).\n\
             IMPORTANT: Always output in English, regardless of the language spoken.\n\
             Use clear, professional, and concise language appropriate for software collaboration.\n\
             Preserve all technical terms, code references, file paths, and variable names exactly as spoken.\n\
             Use Markdown formatting when the content implies structure (lists, headings, code blocks).\n\
             Reply with ONLY the cleaned text, nothing else.".to_string(),

            "The user is composing a post or reply on X (Twitter).\n\
             Keep it concise and punchy. Aim for clarity within a short format.\n\
             Keep text short and suitable for social media.\n\
             If the speaker mentions hashtags, preserve or format them as #hashtags.\n\
             Fix grammar but preserve the speaker's voice and tone.\n\
             Reply with ONLY the cleaned text, nothing else.".to_string(),
        )
    };

    vec![
        // ── Email ──
        PromptRule {
            name: "Gmail".to_string(),
            match_type: MatchType::Url,
            match_value: "mail.google.com".to_string(),
            prompt: email_prompt,
            enabled: true,
            icon: None,
            alt_matches: vec![],
        },
        // ── AI CLI tools (detected via terminal subprocess enrichment) ──
        PromptRule {
            name: "Claude Code".to_string(),
            match_type: MatchType::AppName,
            match_value: "Claude Code".to_string(),
            prompt: ai_cli_prompt.clone(),
            enabled: true,
            icon: None,
            alt_matches: vec![],
        },
        PromptRule {
            name: "Gemini CLI".to_string(),
            match_type: MatchType::AppName,
            match_value: "Gemini CLI".to_string(),
            prompt: ai_cli_prompt.clone(),
            enabled: true,
            icon: None,
            alt_matches: vec![],
        },
        PromptRule {
            name: "Codex CLI".to_string(),
            match_type: MatchType::AppName,
            match_value: "Codex CLI".to_string(),
            prompt: ai_cli_prompt.clone(),
            enabled: true,
            icon: None,
            alt_matches: vec![],
        },
        PromptRule {
            name: "Aider".to_string(),
            match_type: MatchType::AppName,
            match_value: "Aider".to_string(),
            prompt: ai_cli_prompt,
            enabled: true,
            icon: None,
            alt_matches: vec![],
        },
        // ── Code editors & terminals ──
        PromptRule {
            name: "Terminal".to_string(),
            match_type: MatchType::AppName,
            match_value: "Terminal".to_string(),
            prompt: code_editor_prompt.clone(),
            enabled: true,
            icon: None,
            alt_matches: vec![],
        },
        PromptRule {
            name: "VSCode".to_string(),
            match_type: MatchType::AppName,
            match_value: "Code".to_string(),
            prompt: code_editor_prompt.clone(),
            enabled: true,
            icon: None,
            alt_matches: vec![],
        },
        PromptRule {
            name: "Cursor".to_string(),
            match_type: MatchType::AppName,
            match_value: "Cursor".to_string(),
            prompt: code_editor_prompt.clone(),
            enabled: true,
            icon: None,
            alt_matches: vec![],
        },
        PromptRule {
            name: "Antigravity".to_string(),
            match_type: MatchType::AppName,
            match_value: "Antigravity".to_string(),
            prompt: code_editor_prompt.clone(),
            enabled: true,
            icon: None,
            alt_matches: vec![],
        },
        PromptRule {
            name: "iTerm2".to_string(),
            match_type: MatchType::AppName,
            match_value: "iTerm2".to_string(),
            prompt: code_editor_prompt,
            enabled: true,
            icon: None,
            alt_matches: vec![],
        },
        // ── Notes & docs ──
        PromptRule {
            name: "Notion".to_string(),
            match_type: MatchType::Url,
            match_value: "notion.so".to_string(),
            prompt: notion_prompt,
            enabled: true,
            icon: None,
            alt_matches: vec![MatchCondition {
                match_type: MatchType::AppName,
                match_value: "Notion".to_string(),
            }],
        },
        // ── Chat & messaging ──
        PromptRule {
            name: "WhatsApp".to_string(),
            match_type: MatchType::AppName,
            match_value: "WhatsApp".to_string(),
            prompt: chat_prompt.clone(),
            enabled: true,
            icon: None,
            alt_matches: vec![MatchCondition {
                match_type: MatchType::Url,
                match_value: "web.whatsapp.com".to_string(),
            }],
        },
        PromptRule {
            name: "Telegram".to_string(),
            match_type: MatchType::AppName,
            match_value: "Telegram".to_string(),
            prompt: chat_prompt.clone(),
            enabled: true,
            icon: None,
            alt_matches: vec![MatchCondition {
                match_type: MatchType::Url,
                match_value: "web.telegram.org".to_string(),
            }],
        },
        PromptRule {
            name: "Slack".to_string(),
            match_type: MatchType::AppName,
            match_value: "Slack".to_string(),
            prompt: slack_prompt,
            enabled: true,
            icon: None,
            alt_matches: vec![MatchCondition {
                match_type: MatchType::Url,
                match_value: "app.slack.com".to_string(),
            }],
        },
        PromptRule {
            name: "Discord".to_string(),
            match_type: MatchType::AppName,
            match_value: "Discord".to_string(),
            prompt: chat_prompt.clone(),
            enabled: true,
            icon: None,
            alt_matches: vec![MatchCondition {
                match_type: MatchType::Url,
                match_value: "discord.com".to_string(),
            }],
        },
        PromptRule {
            name: "LINE".to_string(),
            match_type: MatchType::AppName,
            match_value: "LINE".to_string(),
            prompt: chat_prompt,
            enabled: true,
            icon: None,
            alt_matches: vec![],
        },
        // ── Developer platforms ──
        PromptRule {
            name: "GitHub".to_string(),
            match_type: MatchType::Url,
            match_value: "github.com".to_string(),
            prompt: github_prompt,
            enabled: true,
            icon: None,
            alt_matches: vec![],
        },
        // ── Social media ──
        PromptRule {
            name: "X (Twitter)".to_string(),
            match_type: MatchType::Url,
            match_value: "x.com".to_string(),
            prompt: twitter_prompt,
            enabled: true,
            icon: None,
            alt_matches: vec![],
        },
    ]
}

/// Returns the base prompt template for polishing speech-to-text output.
pub fn base_prompt_template() -> String {
    "Clean up the speech-to-text output inside the <speech> tags. Fix recognition errors, grammar, and punctuation. \
     Remove fillers and repetitions. If the speaker corrects themselves, keep only the final intent. \
     Preserve meaning and tone. Output in the same language the user spoke in. \
     NEVER answer questions or generate new content — only correct the original text. \
     Reply with ONLY the cleaned text."
        .to_string()
}

/// Resolve a prompt template by replacing the legacy `{language}` placeholder.
pub fn resolve_prompt(template: &str) -> String {
    template.replace("{language}", "the same language the user spoke in").trim().to_string()
}

/// Extract reasoning from `<think>…</think>` blocks and return (cleaned_text, reasoning).
fn extract_think_tags(text: &str) -> (String, Option<String>) {
    if let Some(start) = text.find("<think>") {
        if let Some(end) = text.find("</think>") {
            if start < end {
                let reasoning = text[start + "<think>".len()..end].trim().to_string();
                let cleaned = text[end + "</think>".len()..].to_string();
                let reasoning = if reasoning.is_empty() { None } else { Some(reasoning) };
                return (cleaned, reasoning);
            }
        }
    }
    (text.to_string(), None)
}

/// Result of AI polishing, containing the cleaned text and optional reasoning.
pub struct PolishResult {
    pub text: String,
    pub reasoning: Option<String>,
}

/// Format app context information into a single descriptive line.
fn format_app_context(context: &AppContext) -> String {
    if context.app_name.is_empty() {
        return String::new();
    }
    let mut line = format!("App: {}", context.app_name);
    if !context.terminal_host.is_empty() {
        line.push_str(&format!(" (in {})", context.terminal_host));
    } else if !context.url.is_empty() {
        line.push_str(&format!(" ({})", context.url));
    }
    line
}

/// Find the first matching prompt rule for the given app context.
fn matches_condition(
    match_type: &MatchType,
    match_value: &str,
    app_lower: &str,
    url_lower: &str,
    bundle_id: &str,
) -> bool {
    if match_value.is_empty() {
        return false;
    }
    let val_lower = match_value.to_lowercase();
    match match_type {
        MatchType::AppName => app_lower.contains(&val_lower),
        MatchType::BundleId => bundle_id == match_value,
        MatchType::Url => !url_lower.is_empty() && url_lower.contains(&val_lower),
    }
}

fn find_matching_rule<'a>(rules: &[&'a PromptRule], context: &AppContext) -> Option<&'a str> {
    let app_lower = context.app_name.to_lowercase();
    let url_lower = context.url.to_lowercase();

    for rule in rules {
        if !rule.enabled {
            continue;
        }
        let matched = matches_condition(
            &rule.match_type,
            &rule.match_value,
            &app_lower,
            &url_lower,
            &context.bundle_id,
        ) || rule.alt_matches.iter().any(|alt| {
            matches_condition(
                &alt.match_type,
                &alt.match_value,
                &app_lower,
                &url_lower,
                &context.bundle_id,
            )
        });
        if matched {
            tracing::info!("Prompt rule matched: \"{}\"", rule.name);
            return Some(&rule.prompt);
        }
    }
    tracing::info!("No prompt rule matched (app: {:?}, url: {:?})", context.app_name, context.url);
    None
}

/// Format dictionary entries into a prompt block for the AI model.
fn format_dictionary_prompt(dictionary: &DictionaryConfig) -> String {
    if !dictionary.enabled {
        return String::new();
    }
    let active: Vec<&str> = dictionary
        .entries
        .iter()
        .filter(|e| e.enabled && !e.term.is_empty())
        .map(|e| e.term.as_str())
        .collect();
    if active.is_empty() {
        return String::new();
    }
    let header = "\n\nThe following are user-defined proper nouns. \
         When you encounter homophones or similar-sounding words, \
         automatically apply the correct form based on context:";
    let mut block = String::from(header);
    for term in &active {
        block.push_str(&format!("\n• {}", term));
    }
    block
}

/// Build the system prompt for polishing.
///
/// Composition: base prompt (or custom override) + matched rule context
/// + dictionary block + app context info.
fn build_system_prompt(config: &PolishConfig, context: &AppContext) -> String {
    // 1. Base prompt (or custom_prompt override)
    let base_tmpl = base_prompt_template();
    let base = config.custom_prompt.as_deref().unwrap_or(&base_tmpl);
    let mut prompt = resolve_prompt(base);

    // 2. Append matched rule's context prompt (search all language keys)
    let all_rules: Vec<&PromptRule> = config.prompt_rules.values()
        .flat_map(|rules| rules.iter())
        .collect();
    if let Some(rule_prompt) = find_matching_rule(&all_rules, context) {
        prompt.push_str("\n\n");
        prompt.push_str(rule_prompt);
    }

    // 3. Append dictionary block
    prompt.push_str(&format_dictionary_prompt(&config.dictionary));

    // 4. Append app context info
    let context_line = format_app_context(context);
    if !context_line.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(&context_line);
    }

    prompt
}

/// Polish transcribed text using a local LLM.
///
/// This function is meant to be called from a background thread.
/// It lazy-loads the model on first use and reuses it across calls.
///
/// On any error, returns the original text unchanged (graceful fallback).
pub fn polish_text(
    llm_cache: &Mutex<Option<LlmModelCache>>,
    model_dir: &std::path::Path,
    config: &PolishConfig,
    context: &AppContext,
    raw_text: &str,
    client: &reqwest::blocking::Client,
) -> PolishResult {
    if raw_text.trim().is_empty() {
        return PolishResult { text: raw_text.to_string(), reasoning: None };
    }

    match polish_text_inner(llm_cache, model_dir, config, context, raw_text, client) {
        Ok(raw_output) => {
            // Extract reasoning from <think> blocks
            let (polished, reasoning) = extract_think_tags(&raw_output);
            // Strip any <speech> tags the LLM may have echoed back
            let polished = polished
                .replace("<speech>", "")
                .replace("</speech>", "");
            let polished = polished.trim().to_string();

            // Safety: if output is empty or suspiciously long, use original
            if polished.is_empty() {
                tracing::warn!("Polish returned empty, using original");
                return PolishResult { text: raw_text.to_string(), reasoning };
            }
            let raw_chars = raw_text.graphemes(true).count();
            let polished_chars = polished.graphemes(true).count();
            if polished_chars > raw_chars * 3 + 200 {
                tracing::warn!(
                    "Polish output too long ({} vs {} graphemes), likely hallucination — using original",
                    polished_chars,
                    raw_chars
                );
                return PolishResult { text: raw_text.to_string(), reasoning };
            }
            PolishResult { text: polished, reasoning }
        }
        Err(e) => {
            tracing::error!("Polish error: {} — using original text", e);
            PolishResult { text: raw_text.to_string(), reasoning: None }
        }
    }
}

fn polish_text_inner(
    llm_cache: &Mutex<Option<LlmModelCache>>,
    model_dir: &std::path::Path,
    config: &PolishConfig,
    context: &AppContext,
    raw_text: &str,
    client: &reqwest::blocking::Client,
) -> Result<String, String> {
    let system_prompt = build_system_prompt(config, context);

    // Wrap user speech in XML tags so the LLM can clearly distinguish
    // instructions from the actual speech content to be polished.
    let wrapped = format!("<speech>\n{}\n</speech>", raw_text);

    // Prepend /no_think to suppress model reasoning (Qwen3 convention)
    let user_text = if config.reasoning {
        wrapped
    } else {
        format!("/no_think\n{}", wrapped)
    };

    match config.mode {
        PolishMode::Cloud => run_cloud_inference(&config.cloud, &system_prompt, &user_text, client),
        PolishMode::Local => run_llm_inference(llm_cache, model_dir, config, &system_prompt, &user_text),
    }
}

/// Run cloud LLM inference via an OpenAI-compatible chat completions API.
fn run_cloud_inference(
    cloud: &CloudConfig,
    system_prompt: &str,
    raw_text: &str,
    client: &reqwest::blocking::Client,
) -> Result<String, String> {
    if cloud.api_key.is_empty() {
        return Err("Cloud API key is not set".to_string());
    }

    let endpoint = if cloud.endpoint.is_empty() {
        cloud.provider.default_endpoint().to_string()
    } else {
        validate_custom_endpoint(&cloud.endpoint)?;
        cloud.endpoint.clone()
    };

    if endpoint.is_empty() {
        return Err("Cloud API endpoint is not set".to_string());
    }

    let model_id = if cloud.model_id.is_empty() {
        return Err("Cloud model ID is not set".to_string());
    } else {
        &cloud.model_id
    };

    let mut body = serde_json::json!({
        "model": model_id,
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": raw_text }
        ],
        "max_completion_tokens": 1024
    });
    // GPT-5 series does not support temperature; only set it for other models
    if !model_id.contains("gpt-5") {
        body["temperature"] = serde_json::json!(0.1);
    }

    tracing::info!("Cloud polish: {} via {}", model_id, sanitize_url_for_log(&endpoint));
    let start = std::time::Instant::now();

    let body_str = serde_json::to_string(&body).map_err(|e| format!("Serialize body: {}", e))?;

    let resp = client
        .post(&endpoint)
        .header("Authorization", format!("Bearer {}", cloud.api_key))
        .header("Content-Type", "application/json")
        .body(body_str)
        .send()
        .map_err(|e| format!("Cloud API request failed: {}", e))?;

    let status = resp.status();
    let resp_text = resp.text().map_err(|e| format!("Read response: {}", e))?;

    if !status.is_success() {
        let preview = truncate_for_error(&resp_text, 200);
        return Err(format!("Cloud API returned HTTP {}: {}", status, preview));
    }

    let json: serde_json::Value =
        serde_json::from_str(&resp_text).map_err(|e| format!("Parse response JSON: {}", e))?;

    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| {
            let preview = truncate_for_error(&resp_text, 200);
            format!("Unexpected response format: {}", preview)
        })?;

    tracing::info!(
        "Cloud polish done: {:.0?}, {} graphemes",
        start.elapsed(),
        content.graphemes(true).count()
    );

    Ok(content.trim().to_string())
}

/// Run LLM inference with the given system prompt and user text.
/// Handles model loading/caching, tokenization, and sampling.
fn run_llm_inference(
    llm_cache: &Mutex<Option<LlmModelCache>>,
    model_dir: &std::path::Path,
    config: &PolishConfig,
    system_prompt: &str,
    raw_text: &str,
) -> Result<String, String> {
    let model_path = model_dir.join(config.model.filename());
    if !model_path.exists() {
        return Err(format!(
            "Model file not found: {}",
            model_path.display()
        ));
    }

    // Validate the GGUF file before loading to prevent issues on corrupted files
    validate_gguf_file(&model_path, &config.model)?;

    // Ensure model is loaded (lazy init / reuse pre-warmed cache)
    ensure_llm_loaded(llm_cache, &model_path, config.model.display_name(), &config.model)?;

    // Mutable lock (candle forward() mutates internal KV cache)
    let mut cache = llm_cache.lock().map_err(|e| e.to_string())?;
    let cache_ref = cache.as_mut().ok_or("LLM not loaded")?;

    // Format prompt
    let formatted = format_chat_prompt(&config.model, system_prompt, raw_text);

    // Tokenize
    let tokenize_start = std::time::Instant::now();
    let encoding = cache_ref
        .tokenizer
        .encode(formatted.as_str(), false)
        .map_err(|e| format!("Tokenize: {}", e))?;
    let tokens: Vec<u32> = encoding.get_ids().to_vec();
    tracing::info!(
        "LLM tokenized: {} tokens ({:.0?})",
        tokens.len(),
        tokenize_start.elapsed()
    );

    if tokens.is_empty() {
        return Err("Empty tokenization result".to_string());
    }

    // Resolve EOS token
    let eos_token_id = cache_ref
        .tokenizer
        .token_to_id(config.model.eos_token())
        .ok_or_else(|| format!("EOS token '{}' not found", config.model.eos_token()))?;

    // Clear KV cache for fresh inference
    cache_ref.model.clear_kv_cache();

    // Prompt eval — feed all tokens at once
    let prompt_start = std::time::Instant::now();
    let input = Tensor::new(tokens.as_slice(), &cache_ref.device)
        .and_then(|t| t.unsqueeze(0))
        .map_err(|e| format!("Input tensor: {}", e))?;
    let logits = cache_ref
        .model
        .forward(&input, 0)
        .map_err(|e| format!("Prompt eval: {}", e))?;
    let logits = logits
        .squeeze(0)
        .map_err(|e| format!("Squeeze: {}", e))?;
    tracing::info!(
        "LLM prompt eval: {:.0?} ({} tokens, {:.1} t/s)",
        prompt_start.elapsed(),
        tokens.len(),
        tokens.len() as f64 / prompt_start.elapsed().as_secs_f64()
    );

    // Greedy sampling
    let mut logits_processor = LogitsProcessor::from_sampling(42, Sampling::ArgMax);
    let mut next_token = logits_processor
        .sample(&logits)
        .map_err(|e| format!("Sample: {}", e))?;

    // Generation loop
    let max_tokens: usize = 512;
    let gen_start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(15);
    let mut output_token_ids: Vec<u32> = Vec::new();

    for i in 0..max_tokens {
        if gen_start.elapsed() > timeout {
            tracing::warn!("Polish inference timeout (15s)");
            break;
        }
        if next_token == eos_token_id {
            break;
        }

        output_token_ids.push(next_token);

        let input = Tensor::new(&[next_token], &cache_ref.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| format!("Token tensor: {}", e))?;
        let logits = cache_ref
            .model
            .forward(&input, tokens.len() + i)
            .map_err(|e| format!("Decode step {}: {}", i, e))?;
        let logits = logits
            .squeeze(0)
            .map_err(|e| format!("Squeeze: {}", e))?;

        next_token = logits_processor
            .sample(&logits)
            .map_err(|e| format!("Sample: {}", e))?;
    }

    let gen_elapsed = gen_start.elapsed();
    tracing::info!(
        "LLM generation: {} tokens in {:.0?} ({:.1} t/s)",
        output_token_ids.len(),
        gen_elapsed,
        output_token_ids.len() as f64 / gen_elapsed.as_secs_f64()
    );

    // Decode to string
    let output = cache_ref
        .tokenizer
        .decode(&output_token_ids, true)
        .map_err(|e| format!("Decode output: {}", e))?;

    Ok(output.trim().to_string())
}

/// Polish text using a specific system prompt (for testing/comparison).
pub fn polish_with_prompt(
    llm_cache: &Mutex<Option<LlmModelCache>>,
    model_dir: &std::path::Path,
    config: &PolishConfig,
    system_prompt: &str,
    raw_text: &str,
    client: &reqwest::blocking::Client,
) -> Result<String, String> {
    let raw_output = match config.mode {
        PolishMode::Cloud => run_cloud_inference(&config.cloud, system_prompt, raw_text, client)?,
        PolishMode::Local => run_llm_inference(llm_cache, model_dir, config, system_prompt, raw_text)?,
    };
    let (cleaned, _) = extract_think_tags(&raw_output);
    Ok(cleaned.trim().to_string())
}

/// Build the system prompt for edit-by-instruction mode.
fn build_edit_system_prompt() -> String {
    "You are a text editing assistant. The user provides selected text and an editing instruction.\n\
     Modify the selected text according to the instruction and output ONLY the modified result.\n\
     Do not add any explanation, prefix, or extra text. Output only the final result."
        .to_string()
}

/// Edit text by applying a voice instruction using the LLM.
///
/// Takes the selected text and a spoken instruction (e.g. "translate to English",
/// "rewrite in formal tone"), and returns the modified text.
pub fn edit_text_by_instruction(
    llm_cache: &Mutex<Option<LlmModelCache>>,
    model_dir: &std::path::Path,
    config: &PolishConfig,
    selected_text: &str,
    instruction: &str,
    client: &reqwest::blocking::Client,
) -> Result<String, String> {
    if selected_text.trim().is_empty() {
        return Err("Selected text is empty".to_string());
    }
    if instruction.trim().is_empty() {
        return Err("Instruction is empty".to_string());
    }

    let system_prompt = build_edit_system_prompt();

    let user_text = format!(
        "<selected_text>\n{}\n</selected_text>\n\n<instruction>\n{}\n</instruction>",
        selected_text, instruction
    );

    // Prepend /no_think to suppress model reasoning unless enabled
    let user_text = if config.reasoning {
        user_text
    } else {
        format!("/no_think\n{}", user_text)
    };

    let raw_output = match config.mode {
        PolishMode::Cloud => run_cloud_inference(&config.cloud, &system_prompt, &user_text, client)?,
        PolishMode::Local => run_llm_inference(llm_cache, model_dir, config, &system_prompt, &user_text)?,
    };

    let (cleaned, _reasoning) = extract_think_tags(&raw_output);

    // Strip any XML tags the LLM may have echoed back
    let cleaned = cleaned
        .replace("<selected_text>", "")
        .replace("</selected_text>", "")
        .replace("<instruction>", "")
        .replace("</instruction>", "");
    let cleaned = cleaned.trim().to_string();

    if cleaned.is_empty() {
        return Err("LLM returned empty result".to_string());
    }

    Ok(cleaned)
}

/// Validate a GGUF model file by checking magic bytes, version, and file size.
/// Returns `Ok(())` if the file appears valid, or an error describing the problem.
pub fn validate_gguf_file(path: &std::path::Path, expected_model: &PolishModel) -> Result<(), String> {
    use std::io::Read;

    let mut f = std::fs::File::open(path).map_err(|e| format!("Cannot open model file: {}", e))?;

    // Check GGUF magic bytes
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)
        .map_err(|e| format!("Cannot read GGUF header: {}", e))?;
    if &magic != b"GGUF" {
        return Err(format!(
            "Invalid GGUF magic: expected 'GGUF', got {:?}",
            magic
        ));
    }

    // Check GGUF version (2 or 3 are valid)
    let mut version_bytes = [0u8; 4];
    f.read_exact(&mut version_bytes)
        .map_err(|e| format!("Cannot read GGUF version: {}", e))?;
    let version = u32::from_le_bytes(version_bytes);
    if version < 2 || version > 3 {
        return Err(format!("Unsupported GGUF version: {}", version));
    }

    // Check file size is at least 90% of the expected size (catch truncated downloads)
    let file_size = std::fs::metadata(path)
        .map_err(|e| format!("Cannot stat model file: {}", e))?
        .len();
    let expected_size = expected_model.size_bytes();
    let min_size = expected_size * 9 / 10;
    if file_size < min_size {
        return Err(format!(
            "Model file too small: {} bytes (expected ~{} bytes, min {}). File may be corrupted or incomplete.",
            file_size, expected_size, min_size
        ));
    }

    Ok(())
}

/// Check if polishing is ready to run (either local model exists or cloud API key is set).
pub fn is_polish_ready(model_dir: &std::path::Path, config: &PolishConfig) -> bool {
    match config.mode {
        PolishMode::Cloud => !config.cloud.api_key.is_empty(),
        PolishMode::Local => model_dir.join(config.model.filename()).exists(),
    }
}

/// Check existence and size in a single metadata call.
pub fn model_file_status(model_dir: &std::path::Path, model: &PolishModel) -> (bool, u64) {
    let path = model_dir.join(model.filename());
    match std::fs::metadata(&path) {
        Ok(m) => (true, m.len()),
        Err(_) => (false, 0),
    }
}

/// Invalidate the cached LLM model so it gets reloaded on next use.
pub fn invalidate_cache(llm_cache: &Mutex<Option<LlmModelCache>>) {
    if let Ok(mut cache) = llm_cache.lock() {
        *cache = None;
        tracing::info!("LLM model cache invalidated");
    }
}

/// Pre-warm the LLM model cache so the first polish request is instant.
pub fn warm_llm_cache(
    llm_cache: &Mutex<Option<LlmModelCache>>,
    model_dir: &std::path::Path,
    model: &PolishModel,
) -> Result<(), String> {
    let model_path = model_dir.join(model.filename());
    if !model_path.exists() {
        return Err(format!("Model file not found: {}", model_path.display()));
    }
    validate_gguf_file(&model_path, model)?;
    ensure_llm_loaded(llm_cache, &model_path, model.display_name(), model)?;
    Ok(())
}

/// Shared helper: ensure the LLM is loaded into `llm_cache`, reloading only
/// when the cached path differs from `model_path` (or the cache is empty).
fn ensure_llm_loaded(
    llm_cache: &Mutex<Option<LlmModelCache>>,
    model_path: &std::path::Path,
    display_name: &str,
    polish_model: &PolishModel,
) -> Result<(), String> {
    let mut cache = llm_cache.lock().map_err(|e| e.to_string())?;
    let needs_reload = match cache.as_ref() {
        Some(c) => c.loaded_path != model_path,
        None => true,
    };
    if needs_reload {
        let load_start = std::time::Instant::now();
        tracing::info!("Loading LLM: {} ...", display_name);

        let device = Device::new_metal(0)
            .or_else(|_| Device::new_cuda(0))
            .unwrap_or(Device::Cpu);

        let mut file = std::fs::File::open(model_path)
            .map_err(|e| format!("Cannot open model: {}", e))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| format!("Read GGUF: {}", e))?;

        // Extract tokenizer from GGUF metadata
        let tokenizer = tokenizers::Tokenizer::from_gguf(&content)
            .map_err(|e| format!("Load tokenizer from GGUF: {}", e))?;

        // Load model weights (consumes content; file reader is positioned at tensor data)
        let model = match polish_model {
            PolishModel::LlamaTaiwan => QuantizedModel::Llama(
                LlamaWeights::from_gguf(content, &mut file, &device)
                    .map_err(|e| format!("Load Llama: {}", e))?,
            ),
            PolishModel::Qwen25 => QuantizedModel::Qwen2(
                Qwen2Weights::from_gguf(content, &mut file, &device)
                    .map_err(|e| format!("Load Qwen2: {}", e))?,
            ),
            PolishModel::Qwen3 | PolishModel::Qwen3_4B => QuantizedModel::Qwen3(
                Qwen3Weights::from_gguf(content, &mut file, &device)
                    .map_err(|e| format!("Load Qwen3: {}", e))?,
            ),
        };

        *cache = Some(LlmModelCache {
            model,
            tokenizer,
            device,
            loaded_path: model_path.to_path_buf(),
        });
        tracing::info!("LLM loaded (took {:.0?})", load_start.elapsed());
    }
    Ok(())
}

/// Extract only the host from a URL for safe logging (strips path, query params, credentials).
fn sanitize_url_for_log(url: &str) -> String {
    match url::Url::parse(url) {
        Ok(parsed) => {
            let host = parsed.host_str().unwrap_or("unknown");
            let port = parsed.port().map(|p| format!(":{}", p)).unwrap_or_default();
            format!("{}://{}{}", parsed.scheme(), host, port)
        }
        Err(_) => "invalid-url".to_string(),
    }
}

/// Truncate a string for inclusion in error messages to avoid leaking large response bodies.
pub fn truncate_for_error(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        // Find a valid UTF-8 boundary
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

/// Validate a custom cloud endpoint URL.
/// Allows localhost/private IPs (needed for local model servers like Ollama, LM Studio)
/// but blocks known dangerous targets (cloud metadata endpoints) and requires http(s).
pub fn validate_custom_endpoint(url_str: &str) -> Result<(), String> {
    if url_str.is_empty() {
        return Err("Endpoint URL is empty".to_string());
    }

    let parsed = url::Url::parse(url_str)
        .map_err(|e| format!("Invalid endpoint URL: {}", e))?;

    // Only allow HTTP and HTTPS schemes (block file://, ftp://, etc.)
    if parsed.scheme() != "https" && parsed.scheme() != "http" {
        return Err(format!(
            "Endpoint must use HTTP or HTTPS (got \"{}://\")",
            parsed.scheme()
        ));
    }

    let host = parsed.host_str().unwrap_or("");
    if host.is_empty() {
        return Err("Endpoint URL has no host".to_string());
    }

    // Block cloud metadata endpoints
    const BLOCKED_METADATA_IPS: &[std::net::Ipv4Addr] = &[
        std::net::Ipv4Addr::new(169, 254, 169, 254), // AWS / GCP / Azure IMDS
        std::net::Ipv4Addr::new(168,  63, 129,  16), // Azure Wire Server (IMDS v2)
        std::net::Ipv4Addr::new(100, 100, 100, 200), // Alibaba Cloud IMDS
    ];
    if let Ok(ip) = host.parse::<std::net::Ipv4Addr>() {
        if BLOCKED_METADATA_IPS.contains(&ip) {
            return Err("Endpoint must not target a cloud metadata address".to_string());
        }
    }
    // IPv6: link-local (fe80::/10) and IPv4-mapped IPv6 (::ffff:x.x.x.x) checks
    if let Ok(std::net::IpAddr::V6(v6)) = host.parse::<std::net::IpAddr>() {
        if (v6.segments()[0] & 0xffc0) == 0xfe80 {
            return Err("Endpoint must not target a link-local address".to_string());
        }
        // IPv4-mapped IPv6 (e.g. ::ffff:169.254.169.254) bypasses the IPv4 check above
        if let Some(mapped_v4) = v6.to_ipv4_mapped() {
            if BLOCKED_METADATA_IPS.contains(&mapped_v4) {
                return Err("Endpoint must not target a cloud metadata address".to_string());
            }
        }
    }
    // GCP metadata hostname
    if host == "metadata.google.internal" {
        return Err("Endpoint must not target a cloud metadata address".to_string());
    }

    // Reject credentials embedded in URL (e.g. https://user:pass@host/)
    if parsed.username() != "" || parsed.password().is_some() {
        return Err("Endpoint URL must not contain embedded credentials".to_string());
    }

    // Warn via log (not block) if using plain HTTP to a non-local host
    if parsed.scheme() == "http" {
        let is_local = host == "localhost"
            || host == "127.0.0.1"
            || host == "::1"
            || host == "0.0.0.0"
            || host.parse::<std::net::Ipv4Addr>().map_or(false, |ip| ip.is_private())
            || host.parse::<std::net::IpAddr>().map_or(false, |ip| match ip {
                // IPv6 ULA (fc00::/7) — private routable IPv6
                std::net::IpAddr::V6(v6) => (v6.segments()[0] & 0xfe00) == 0xfc00,
                _ => false,
            });
        if !is_local {
            tracing::warn!(
                "Warning: custom endpoint uses plain HTTP to remote host ({}). Data will be sent unencrypted.",
                host
            );
        }
    }

    Ok(())
}
