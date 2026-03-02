use serde::{Deserialize, Serialize};

use crate::settings::models_dir;

// ── WhisperModel enum ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WhisperModel {
    #[default]
    LargeV3Turbo,
    LargeV3TurboQ5,
    Medium,
    Small,
    Base,
    LargeV3TurboZhTw,
}

impl WhisperModel {
    pub fn filename(&self) -> &'static str {
        match self {
            Self::LargeV3Turbo => "ggml-large-v3-turbo.bin",
            Self::LargeV3TurboQ5 => "ggml-large-v3-turbo-q5_0.bin",
            Self::Medium => "ggml-medium.bin",
            Self::Small => "ggml-small.bin",
            Self::Base => "ggml-base.bin",
            Self::LargeV3TurboZhTw => "ggml-large-v3-turbo-zh-TW.bin",
        }
    }

    /// Returns the download URL for this model, or `None` if it's a custom/legacy model
    /// with no public URL.
    pub fn download_url(&self) -> Option<&'static str> {
        match self {
            Self::LargeV3Turbo => Some(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
            ),
            Self::LargeV3TurboQ5 => Some(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
            ),
            Self::Medium => Some(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
            ),
            Self::Small => Some(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
            ),
            Self::Base => Some(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
            ),
            Self::LargeV3TurboZhTw => Some(
                "https://huggingface.co/Alkd/whisper-large-v3-turbo-zh-TW/resolve/main/ggml-model.bin",
            ),
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::LargeV3Turbo => "Whisper Turbo",
            Self::LargeV3TurboQ5 => "Whisper Turbo Lite",
            Self::Medium => "Whisper Medium",
            Self::Small => "Whisper Small",
            Self::Base => "Whisper Base",
            Self::LargeV3TurboZhTw => "Whisper Turbo TW",
        }
    }

    pub fn size_bytes(&self) -> u64 {
        match self {
            Self::LargeV3Turbo => 1_620_000_000,
            Self::LargeV3TurboQ5 => 547_000_000,
            Self::Medium => 1_530_000_000,
            Self::Small => 488_000_000,
            Self::Base => 148_000_000,
            Self::LargeV3TurboZhTw => 1_600_000_000,
        }
    }

    pub fn languages(&self) -> &'static [&'static str] {
        match self {
            Self::LargeV3TurboZhTw => &["zh-TW"],
            _ => &["multilingual"],
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::LargeV3Turbo => "Highest multilingual accuracy",
            Self::LargeV3TurboQ5 => "High quality, compact size (quantized)",
            Self::Medium => "Balanced speed and quality",
            Self::Small => "Lightweight and fast",
            Self::Base => "Fastest, smallest footprint",
            Self::LargeV3TurboZhTw => "Best for Traditional Chinese",
        }
    }

    pub fn all() -> &'static [WhisperModel] {
        &[
            Self::LargeV3Turbo,
            Self::LargeV3TurboQ5,
            Self::Base,
            Self::LargeV3TurboZhTw,
        ]
    }
}

// ── WhisperModelInfo (for frontend serialization) ────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct WhisperModelInfo {
    pub id: WhisperModel,
    pub display_name: &'static str,
    pub description: &'static str,
    pub size_bytes: u64,
    pub languages: &'static [&'static str],
    pub downloaded: bool,
    pub file_size_on_disk: u64,
    pub is_active: bool,
}

impl WhisperModelInfo {
    pub fn from_model(model: &WhisperModel, active_model: &WhisperModel) -> Self {
        let dir = models_dir();
        let path = dir.join(model.filename());
        let (downloaded, file_size_on_disk) = match std::fs::metadata(&path) {
            Ok(m) => (true, m.len()),
            Err(_) => (false, 0),
        };
        Self {
            id: model.clone(),
            display_name: model.display_name(),
            description: model.description(),
            size_bytes: model.size_bytes(),
            languages: model.languages(),
            downloaded,
            file_size_on_disk,
            is_active: model == active_model,
        }
    }
}

// ── SystemInfo ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct SystemInfo {
    pub total_ram_bytes: u64,
    pub available_disk_bytes: u64,
    pub is_apple_silicon: bool,
    pub gpu_vram_bytes: u64,
    pub has_cuda: bool,
    pub os: String,
    pub arch: String,
}

/// Detect system information (RAM, disk space, CPU architecture, GPU VRAM).
pub fn detect_system_info() -> SystemInfo {
    let total_ram_bytes = get_total_ram();
    let available_disk_bytes = get_available_disk_space();
    let gpu_vram_bytes = get_gpu_vram();
    let arch = std::env::consts::ARCH.to_string();
    let is_apple_silicon = cfg!(target_os = "macos") && arch == "aarch64";
    let has_cuda = cfg!(feature = "cuda");

    SystemInfo {
        total_ram_bytes,
        available_disk_bytes,
        is_apple_silicon,
        gpu_vram_bytes,
        has_cuda,
        os: std::env::consts::OS.to_string(),
        arch,
    }
}

/// Recommend a model based on system info and language preference.
///
/// Effective memory selection:
/// - Apple Silicon → system RAM (unified memory shared with GPU)
/// - CUDA enabled + discrete GPU with >= 2 GB VRAM → GPU VRAM
/// - Otherwise → system RAM
pub fn recommend_model(system: &SystemInfo, settings_language: Option<&str>) -> WhisperModel {
    let lang = settings_language
        .map(|l| l.to_lowercase())
        .or_else(detect_system_language)
        .unwrap_or_default();

    let prefers_zh_tw = lang.starts_with("zh-tw") || lang.starts_with("zh_tw")
        || lang.starts_with("zh-hant") || lang.starts_with("zh_hant");
    let _prefers_zh = lang.starts_with("zh") || lang == "chinese";

    let ram_gb = system.total_ram_bytes as f64 / 1_073_741_824.0;
    let vram_gb = system.gpu_vram_bytes as f64 / 1_073_741_824.0;
    let disk_gb = system.available_disk_bytes as f64 / 1_073_741_824.0;

    let effective_gb = if system.is_apple_silicon {
        ram_gb
    } else if system.has_cuda && vram_gb >= 2.0 {
        vram_gb
    } else {
        ram_gb
    };

    if effective_gb >= 8.0 && disk_gb >= 3.0 {
        if prefers_zh_tw {
            return WhisperModel::LargeV3TurboZhTw;
        }
        WhisperModel::LargeV3Turbo
    } else if effective_gb >= 4.0 && disk_gb >= 1.0 {
        WhisperModel::LargeV3TurboQ5
    } else {
        WhisperModel::Base
    }
}

// ── System language detection ─────────────────────────────────────────────────

/// Detect the system language via `tauri-plugin-os` (cross-platform).
/// Returns a lowercased BCP-47 tag, e.g. `"zh-tw"`, `"ja"`, `"en-us"`.
pub fn detect_system_language() -> Option<String> {
    tauri_plugin_os::locale().map(|s| s.to_lowercase())
}

// ── Platform-specific system info helpers ─────────────────────────────────────

#[cfg(unix)]
fn get_total_ram() -> u64 {
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        let mut size: u64 = 0;
        let mut len = mem::size_of::<u64>();
        let mib = [libc::CTL_HW, libc::HW_MEMSIZE];
        let ret = unsafe {
            libc::sysctl(
                mib.as_ptr() as *mut _,
                2,
                &mut size as *mut u64 as *mut _,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if ret == 0 {
            size
        } else {
            0
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        unsafe {
            let info: libc::sysinfo = std::mem::zeroed();
            if libc::sysinfo(&info as *const _ as *mut _) == 0 {
                info.totalram as u64 * info.mem_unit as u64
            } else {
                0
            }
        }
    }
}

#[cfg(target_os = "windows")]
fn get_total_ram() -> u64 {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    unsafe {
        let mut mem_info = MEMORYSTATUSEX {
            dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
            ..std::mem::zeroed()
        };
        if GlobalMemoryStatusEx(&mut mem_info).is_ok() {
            mem_info.ullTotalPhys
        } else {
            0
        }
    }
}

#[cfg(not(any(unix, target_os = "windows")))]
fn get_total_ram() -> u64 {
    0
}

fn get_available_disk_space() -> u64 {
    let models = models_dir();
    // Ensure the directory exists
    let _ = std::fs::create_dir_all(&models);

    #[cfg(unix)]
    {
        use std::ffi::CString;
        let path_c = match CString::new(models.to_string_lossy().as_bytes()) {
            Ok(c) => c,
            Err(_) => return 0,
        };
        unsafe {
            let mut stat: libc::statvfs = std::mem::zeroed();
            if libc::statvfs(path_c.as_ptr(), &mut stat) == 0 {
                stat.f_bavail as u64 * stat.f_frsize as u64
            } else {
                0
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        use windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
        use windows::core::HSTRING;
        let path_str = models.to_string_lossy().to_string();
        let path_w = HSTRING::from(path_str);
        let mut free_bytes_available: u64 = 0;
        unsafe {
            if GetDiskFreeSpaceExW(
                &path_w,
                Some(&mut free_bytes_available),
                None,
                None,
            )
            .is_ok()
            {
                free_bytes_available
            } else {
                0
            }
        }
    }

    #[cfg(not(any(unix, target_os = "windows")))]
    {
        let _ = models;
        0
    }
}

/// Detect the largest dedicated GPU VRAM via DXGI (Windows only).
/// Returns 0 on non-Windows platforms or if no discrete GPU is found.
fn get_gpu_vram() -> u64 {
    #[cfg(target_os = "windows")]
    {
        use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
        let factory: IDXGIFactory1 = match unsafe { CreateDXGIFactory1() } {
            Ok(f) => f,
            Err(_) => return 0,
        };
        let mut max_vram: u64 = 0;
        let mut i = 0u32;
        loop {
            let adapter = match unsafe { factory.EnumAdapters(i) } {
                Ok(a) => a,
                Err(_) => break,
            };
            if let Ok(desc) = unsafe { adapter.GetDesc() } {
                let dedicated = desc.DedicatedVideoMemory as u64;
                if dedicated > max_vram {
                    max_vram = dedicated;
                }
            }
            i += 1;
        }
        max_vram
    }

    #[cfg(not(target_os = "windows"))]
    {
        0
    }
}
