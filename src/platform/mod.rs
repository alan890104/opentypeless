#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "windows")]
pub mod windows;
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub mod fallback;

/// Hide the Dock icon (macOS) or equivalent.
pub fn set_app_accessory_mode() {
    #[cfg(target_os = "macos")]
    unsafe {
        macos::set_accessory_policy();
    }
    #[cfg(target_os = "windows")]
    windows::set_accessory_policy();
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    fallback::set_accessory_policy();
}

/// Make the main window draggable by its background (macOS overlay title bar fix).
pub fn set_main_window_movable(window: &tauri::WebviewWindow) {
    #[cfg(target_os = "macos")]
    if let Ok(ns_win) = window.ns_window() {
        unsafe { macos::set_movable_by_background(ns_win); }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = window;
    }
}

/// One-time overlay setup (non-activating, always-on-top, all Spaces).
pub fn setup_overlay_window(overlay: &tauri::WebviewWindow) {
    #[cfg(target_os = "macos")]
    if let Ok(ns_win) = overlay.ns_window() {
        unsafe { macos::setup_overlay(ns_win); }
    }
    #[cfg(target_os = "windows")]
    if let Ok(hwnd) = overlay.hwnd() {
        unsafe { windows::setup_overlay(hwnd.0); }
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = overlay;
    }
}

/// Show overlay without activating.
pub fn show_overlay(overlay: &tauri::WebviewWindow) {
    #[cfg(target_os = "macos")]
    if let Ok(ns_win) = overlay.ns_window() {
        unsafe { macos::show_no_activate(ns_win); }
    }
    #[cfg(target_os = "windows")]
    if let Ok(hwnd) = overlay.hwnd() {
        unsafe { windows::show_no_activate(hwnd.0); }
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = overlay.show();
    }
}

/// Hide overlay.
pub fn hide_overlay(overlay: &tauri::WebviewWindow) {
    #[cfg(target_os = "macos")]
    if let Ok(ns_win) = overlay.ns_window() {
        unsafe { macos::hide_window(ns_win); }
    }
    #[cfg(target_os = "windows")]
    if let Ok(hwnd) = overlay.hwnd() {
        unsafe { windows::hide_window(hwnd.0); }
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = overlay.hide();
    }
}

/// Simulate paste (Cmd+V on macOS, Ctrl+V on Windows).
pub fn simulate_paste() -> bool {
    #[cfg(target_os = "macos")]
    { unsafe { macos::simulate_cmd_v() } }
    #[cfg(target_os = "windows")]
    { unsafe { windows::simulate_paste() } }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    { false }
}

/// Simulate copy (Cmd+C on macOS, Ctrl+C on Windows).
pub fn simulate_copy() -> bool {
    #[cfg(target_os = "macos")]
    { unsafe { macos::simulate_cmd_c() } }
    #[cfg(target_os = "windows")]
    { unsafe { windows::simulate_copy() } }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    { false }
}

/// Simulate undo (Cmd+Z on macOS, Ctrl+Z on Windows).
pub fn simulate_undo() -> bool {
    #[cfg(target_os = "macos")]
    { unsafe { macos::simulate_cmd_z() } }
    #[cfg(target_os = "windows")]
    { unsafe { windows::simulate_undo() } }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    { false }
}

/// Returns `true` if any media is actively playing system-wide.
/// Used to guard the play/pause key — avoids launching Apple Music when idle.
pub fn is_now_playing() -> bool {
    #[cfg(target_os = "macos")]
    { macos::is_now_playing() }
    #[cfg(not(target_os = "macos"))]
    { false }
}

/// Send the Play/Pause media key to pause currently playing media.
/// Call `resume_now_playing` when done to restore playback.
pub fn pause_now_playing() {
    #[cfg(target_os = "macos")]
    macos::simulate_media_play_pause();
}

/// Send the Play/Pause media key to resume media paused by `pause_now_playing`.
pub fn resume_now_playing() {
    #[cfg(target_os = "macos")]
    macos::simulate_media_play_pause();
}

/// Returns the clipboard change sequence number if the platform supports it.
/// macOS: NSPasteboard.changeCount, Windows: GetClipboardSequenceNumber.
/// Returns None on Linux/other (caller falls back to sentinel approach).
pub fn clipboard_change_count() -> Option<u32> {
    #[cfg(target_os = "macos")]
    { macos::clipboard_change_count() }
    #[cfg(target_os = "windows")]
    { windows::clipboard_change_count() }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    { None }
}
