//! Cross-platform audio device querying, virtual device filtering, and
//! Bluetooth avoidance logic.
//!
//! Platform-specific FFI lives in `platform/macos.rs` (CoreAudio) and
//! `platform/windows.rs`. This module provides the unified API consumed
//! by the rest of the crate.

/// Known virtual audio device name patterns (case-insensitive substring match).
/// Used as a cross-platform fallback when OS-level transport type detection is
/// unavailable (e.g. Windows, or macOS 16+ where CoreAudio transport type API
/// returns errors for all devices).
const VIRTUAL_DEVICE_PATTERNS: &[&str] = &[
    "blackhole",
    "loopback audio",
    "soundflower",
    "vb-cable",
    "vb-audio",
    "voicemeeter",
    "cable input",
    "cable output",
    "speaker audio recorder",
    "obs-",
    "virtual audio",
    "zoom audio",
];

/// Returns `true` if `name` matches a known virtual audio device pattern.
pub fn is_known_virtual_device(name: &str) -> bool {
    let lower = name.to_lowercase();
    VIRTUAL_DEVICE_PATTERNS.iter().any(|p| lower.contains(p))
}

/// Returns `true` if the system default audio input device is Bluetooth.
/// Always `false` on non-macOS platforms.
pub fn is_default_input_bluetooth() -> bool {
    #[cfg(target_os = "macos")]
    { crate::platform::macos::is_default_input_bluetooth() }
    #[cfg(not(target_os = "macos"))]
    { false }
}

/// Returns the name of the first built-in audio input device, or `None`.
/// Always `None` on non-macOS platforms.
pub fn get_builtin_input_device_name() -> Option<String> {
    #[cfg(target_os = "macos")]
    { crate::platform::macos::get_builtin_input_device_name() }
    #[cfg(not(target_os = "macos"))]
    { None }
}

/// List names of all physical (non-virtual) audio input devices.
///
/// On macOS, uses CoreAudio to filter out virtual loopback drivers.
/// On other platforms, returns an empty Vec (caller falls back to cpal
/// with [`is_known_virtual_device`] name filtering).
pub fn list_physical_input_device_names() -> Vec<String> {
    #[cfg(target_os = "macos")]
    { crate::platform::macos::list_physical_input_device_names() }
    #[cfg(not(target_os = "macos"))]
    { Vec::new() }
}

/// Register a permanent listener that fires when the system default audio
/// input device changes. No-op on non-macOS platforms.
pub fn add_default_input_listener(callback: impl Fn() + Send + 'static) {
    #[cfg(target_os = "macos")]
    crate::platform::macos::add_default_input_listener(callback);
    #[cfg(not(target_os = "macos"))]
    { let _ = callback; }
}

/// Resolve the effective microphone device name, applying Bluetooth avoidance
/// when the user has not made an explicit device selection.
///
/// - `preferred = Some(name)` → user chose a specific device; always honoured.
/// - `preferred = None` (Auto) → if the system default input is a Bluetooth
///   device, return the name of the built-in microphone instead. This prevents
///   macOS from switching the Bluetooth headset from A2DP → HFP (which causes
///   the microphone to sound dramatically louder and degrades audio output
///   quality). If no built-in mic exists, falls back to `None` (cpal default).
pub fn resolve_input_device(preferred: Option<String>) -> Option<String> {
    if preferred.is_some() {
        return preferred;
    }
    if is_default_input_bluetooth() {
        let builtin = get_builtin_input_device_name();
        if builtin.is_none() {
            tracing::warn!("Default input is Bluetooth but no built-in mic found — recording from BT device");
        } else {
            tracing::info!("Default input is Bluetooth — routing to built-in mic: {:?}", builtin);
        }
        return builtin; // None = let cpal pick system default (safe fallback)
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify all patterns match their canonical device names.
    #[test]
    fn detects_all_known_virtual_devices() {
        let cases = [
            ("BlackHole 2ch", true),
            ("Loopback Audio", true),
            ("Soundflower (2ch)", true),
            ("VB-Cable", true),
            ("VoiceMeeter Output", true),
            ("Speaker Audio Recorder", true),
            ("OBS-Monitor", true),
            ("Zoom Audio Device", true),
        ];
        for (name, expected) in cases {
            assert_eq!(is_known_virtual_device(name), expected, "failed for: {}", name);
        }
    }

    /// Real hardware devices must not be filtered.
    #[test]
    fn allows_real_devices() {
        let real = [
            "MacBook Pro Microphone",
            "External Microphone",
            "USB Audio Device",
            "Blue Yeti",
            "AirPods Pro",
        ];
        for name in real {
            assert!(!is_known_virtual_device(name), "false positive: {}", name);
        }
    }

    #[test]
    fn case_insensitive() {
        assert!(is_known_virtual_device("BLACKHOLE 2CH"));
        assert!(is_known_virtual_device("blackhole 2ch"));
    }

    #[test]
    fn preferred_device_always_honoured() {
        let result = resolve_input_device(Some("My USB Mic".to_string()));
        assert_eq!(result, Some("My USB Mic".to_string()));
    }
}
