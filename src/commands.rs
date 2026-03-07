use crate::audio;
use crate::credentials;
use crate::hotkey::{hotkey_display_label, parse_hotkey_string};
use crate::platform;
use crate::polisher::{self, PolishModelInfo};
use crate::qwen3_asr as qwen3;
use crate::settings::{self, Settings};
use crate::stt::{Qwen3AsrModel, Qwen3AsrModelInfo, SttMode};
use crate::whisper_models::{self, WhisperModel, WhisperModelInfo, SystemInfo};
use crate::{history, meeting_notes, AppState};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::time::Instant;
use tauri::{AppHandle, Emitter, Manager, State};

/// Load an API key, checking the in-memory cache first before falling back
/// to the credential store.
pub fn get_cached_api_key(cache: &Mutex<HashMap<String, String>>, provider: &str) -> String {
    if let Ok(map) = cache.lock() {
        if let Some(key) = map.get(provider) {
            return key.clone();
        }
    }
    match credentials::load(provider) {
        Ok(key) if !key.is_empty() => {
            if let Ok(mut map) = cache.lock() {
                map.insert(provider.to_string(), key.clone());
            }
            key
        }
        // Empty key or error: return empty without caching, so the next call retries
        // the credential store rather than hitting a poisoned cache entry.
        Ok(_) | Err(_) => String::new(),
    }
}

#[tauri::command]
pub fn get_settings(state: State<'_, AppState>) -> Settings {
    state.settings.lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

#[tauri::command]
pub fn save_settings(
    state: State<'_, AppState>,
    new_settings: Settings,
) -> Result<(), String> {
    let mut current = state.settings.lock().map_err(|e| e.to_string())?;
    current.auto_paste = new_settings.auto_paste;
    current.polish = new_settings.polish;
    current.history_retention_days = new_settings.history_retention_days;
    current.language = new_settings.language;
    current.stt = new_settings.stt;
    // Keep cloud.language in sync with top-level language
    current.stt.cloud.language = current.stt.language.clone();
    current.edit_hotkey = new_settings.edit_hotkey;
    current.meeting_hotkey = new_settings.meeting_hotkey;
    current.onboarding_completed = new_settings.onboarding_completed;
    current.idle_mic_timeout_secs = new_settings.idle_mic_timeout_secs;
    settings::save_settings_to_disk(&current);
    Ok(())
}

#[tauri::command]
pub fn update_hotkey(
    app: AppHandle,
    state: State<'_, AppState>,
    hotkey: String,
) -> Result<(), String> {
    use tauri_plugin_global_shortcut::GlobalShortcutExt;

    let shortcut =
        parse_hotkey_string(&hotkey).ok_or_else(|| "Invalid hotkey string".to_string())?;

    // Check for conflicts with existing edit/meeting hotkeys before touching shortcuts.
    {
        let settings = state.settings.lock().map_err(|e| e.to_string())?;
        if settings.edit_hotkey.as_deref() == Some(hotkey.as_str()) {
            return Err("Primary hotkey must differ from edit hotkey".to_string());
        }
        if settings.meeting_hotkey.as_deref() == Some(hotkey.as_str()) {
            return Err("Primary hotkey must differ from meeting hotkey".to_string());
        }
    }

    app.global_shortcut()
        .unregister_all()
        .map_err(|e| format!("Failed to unregister shortcuts: {}", e))?;

    app.global_shortcut()
        .register(shortcut)
        .map_err(|e| format!("Failed to register shortcut: {}", e))?;

    let mut settings = state.settings.lock().map_err(|e| e.to_string())?;
    settings.hotkey = hotkey.clone();
    settings::save_settings_to_disk(&settings);

    if let Some(ref edit_hk) = settings.edit_hotkey {
        if let Some(edit_shortcut) = parse_hotkey_string(edit_hk) {
            if let Err(e) = app.global_shortcut().register(edit_shortcut) {
                tracing::warn!("Failed to re-register edit hotkey: {}", e);
            }
        }
    }

    if let Some(ref meeting_hk) = settings.meeting_hotkey {
        if let Some(meeting_shortcut) = parse_hotkey_string(meeting_hk) {
            if let Err(e) = app.global_shortcut().register(meeting_shortcut) {
                tracing::warn!("Failed to re-register meeting hotkey: {}", e);
            }
        }
    }

    let label = hotkey_display_label(&hotkey);
    if let Some(tray) = app.tray_by_id("main-tray") {
        let tooltip = if settings::is_debug() {
            format!("Sumi [Dev] – {} to record", label)
        } else {
            format!("Sumi – {} to record", label)
        };
        let _ = tray.set_tooltip(Some(&tooltip));
    }

    tracing::info!("Hotkey updated to: {} ({})", hotkey, label);
    Ok(())
}

#[tauri::command]
pub fn update_edit_hotkey(
    app: AppHandle,
    state: State<'_, AppState>,
    hotkey: Option<String>,
) -> Result<(), String> {
    use tauri_plugin_global_shortcut::GlobalShortcutExt;

    // Validate BEFORE unregistering, so a failed validation cannot leave
    // all shortcuts permanently unregistered (mirrors update_meeting_hotkey).
    let mut settings = state.settings.lock().map_err(|e| e.to_string())?;

    if let Some(ref hk) = hotkey {
        if !hk.is_empty() {
            let _ = parse_hotkey_string(hk)
                .ok_or_else(|| "Invalid edit hotkey string".to_string())?;
            // Symmetric conflict check: edit hotkey must not match primary or meeting hotkey.
            if *hk == settings.hotkey {
                return Err("Edit hotkey must differ from primary hotkey".to_string());
            }
            if settings.meeting_hotkey.as_deref() == Some(hk.as_str()) {
                return Err("Edit hotkey must differ from meeting hotkey".to_string());
            }
        }
    }
    settings.edit_hotkey = hotkey.filter(|s| !s.is_empty());

    // Unregister only after validation succeeds.
    app.global_shortcut()
        .unregister_all()
        .map_err(|e| format!("Failed to unregister shortcuts: {}", e))?;

    let primary = parse_hotkey_string(&settings.hotkey)
        .ok_or_else(|| "Invalid primary hotkey".to_string())?;
    app.global_shortcut()
        .register(primary)
        .map_err(|e| format!("Failed to register primary shortcut: {}", e))?;

    if let Some(ref edit_hk) = settings.edit_hotkey {
        if let Some(shortcut) = parse_hotkey_string(edit_hk) {
            app.global_shortcut()
                .register(shortcut)
                .map_err(|e| format!("Failed to register edit shortcut: {}", e))?;
            tracing::info!("Edit hotkey registered: {}", edit_hk);
        }
    }

    if let Some(ref meeting_hk) = settings.meeting_hotkey {
        if let Some(shortcut) = parse_hotkey_string(meeting_hk) {
            if let Err(e) = app.global_shortcut().register(shortcut) {
                tracing::warn!("Failed to re-register meeting hotkey: {}", e);
            }
        }
    }

    *state.registered_edit_shortcut.lock().map_err(|e| e.to_string())? =
        settings.edit_hotkey.as_deref().and_then(parse_hotkey_string);

    settings::save_settings_to_disk(&settings);
    tracing::info!("Edit hotkey updated to: {:?}", settings.edit_hotkey);
    Ok(())
}

#[tauri::command]
pub fn update_meeting_hotkey(
    app: AppHandle,
    state: State<'_, AppState>,
    hotkey: Option<String>,
) -> Result<(), String> {
    use tauri_plugin_global_shortcut::GlobalShortcutExt;

    // Refuse if a meeting is in progress — unregistering the active hotkey
    // would make it impossible to stop the meeting via keyboard.
    if state.meeting_active.load(Ordering::SeqCst) {
        return Err("Cannot change meeting hotkey while a meeting is in progress".to_string());
    }

    // Validate before making any changes so a bad hotkey string does not leave
    // the app with no shortcuts registered (unregister_all already called).
    let mut settings = state.settings.lock().map_err(|e| e.to_string())?;

    if let Some(ref hk) = hotkey {
        if !hk.is_empty() {
            let _ = parse_hotkey_string(hk)
                .ok_or_else(|| "Invalid meeting hotkey string".to_string())?;
            // Must include at least one modifier to avoid swallowing bare keypresses.
            let has_modifier = ["Alt+", "Control+", "Shift+", "Super+"]
                .iter()
                .any(|m| hk.contains(m));
            if !has_modifier {
                return Err("Meeting hotkey must include at least one modifier key".to_string());
            }
            // Must not conflict with primary or edit hotkeys.
            if *hk == settings.hotkey {
                return Err("Meeting hotkey must differ from primary hotkey".to_string());
            }
            if settings.edit_hotkey.as_deref() == Some(hk.as_str()) {
                return Err("Meeting hotkey must differ from edit hotkey".to_string());
            }
        }
    }
    settings.meeting_hotkey = hotkey.filter(|s| !s.is_empty());

    // Unregister only after validation succeeds.
    app.global_shortcut()
        .unregister_all()
        .map_err(|e| format!("Failed to unregister shortcuts: {}", e))?;

    // Re-register primary hotkey.
    let primary = parse_hotkey_string(&settings.hotkey)
        .ok_or_else(|| "Invalid primary hotkey".to_string())?;
    app.global_shortcut()
        .register(primary)
        .map_err(|e| format!("Failed to register primary shortcut: {}", e))?;

    // Re-register edit hotkey if set.
    if let Some(ref edit_hk) = settings.edit_hotkey {
        if let Some(shortcut) = parse_hotkey_string(edit_hk) {
            if let Err(e) = app.global_shortcut().register(shortcut) {
                tracing::warn!("Failed to re-register edit hotkey: {}", e);
            }
        }
    }

    // Register new meeting hotkey if set.
    if let Some(ref meeting_hk) = settings.meeting_hotkey {
        if let Some(shortcut) = parse_hotkey_string(meeting_hk) {
            app.global_shortcut()
                .register(shortcut)
                .map_err(|e| format!("Failed to register meeting shortcut: {}", e))?;
            tracing::info!("Meeting hotkey registered: {}", meeting_hk);
        }
    }

    *state.registered_meeting_shortcut.lock().map_err(|e| e.to_string())? =
        settings.meeting_hotkey.as_deref().and_then(parse_hotkey_string);

    settings::save_settings_to_disk(&settings);
    tracing::info!("Meeting hotkey updated to: {:?}", settings.meeting_hotkey);
    Ok(())
}

#[tauri::command]
pub fn trigger_undo(app: AppHandle) -> Result<(), String> {
    let app_handle = app.clone();
    std::thread::spawn(move || {
        platform::simulate_undo();
        tracing::info!("↩️ Undo triggered from overlay");
        let app_for_hide = app_handle.clone();
        let _ = app_handle.run_on_main_thread(move || {
            if let Some(overlay) = app_for_hide.get_webview_window("overlay") {
                platform::hide_overlay(&overlay);
            }
        });
    });
    Ok(())
}

#[tauri::command]
pub fn reset_settings(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    use tauri_plugin_global_shortcut::GlobalShortcutExt;

    // Save bare defaults to disk, then re-load and apply locale defaults.
    settings::save_settings_to_disk(&Settings::default());
    let mut fresh = settings::load_settings();
    settings::apply_locale_defaults(&mut fresh);
    let default_hotkey = fresh.hotkey.clone();
    let default_edit_hotkey = fresh.edit_hotkey.clone();
    let default_meeting_hotkey = fresh.meeting_hotkey.clone();

    {
        let mut current = state.settings.lock().map_err(|e| e.to_string())?;
        *current = fresh;
    }

    let shortcut = parse_hotkey_string(&default_hotkey)
        .ok_or_else(|| "Invalid default hotkey string".to_string())?;

    app.global_shortcut()
        .unregister_all()
        .map_err(|e| format!("Failed to unregister shortcuts: {}", e))?;

    app.global_shortcut()
        .register(shortcut)
        .map_err(|e| format!("Failed to register shortcut: {}", e))?;

    // Re-register edit hotkey if the default settings include one.
    if let Some(ref edit_hk) = default_edit_hotkey {
        if let Some(s) = parse_hotkey_string(edit_hk) {
            if let Err(e) = app.global_shortcut().register(s) {
                tracing::warn!("Failed to re-register edit hotkey after reset: {}", e);
            }
        }
    }

    // Re-register meeting hotkey if the default settings include one.
    if let Some(ref meeting_hk) = default_meeting_hotkey {
        if let Some(s) = parse_hotkey_string(meeting_hk) {
            if let Err(e) = app.global_shortcut().register(s) {
                tracing::warn!("Failed to re-register meeting hotkey after reset: {}", e);
            }
        }
    }

    // Update shortcut identity caches to match the newly registered shortcuts.
    *state.registered_edit_shortcut.lock().map_err(|e| e.to_string())? =
        default_edit_hotkey.as_deref().and_then(parse_hotkey_string);
    *state.registered_meeting_shortcut.lock().map_err(|e| e.to_string())? =
        default_meeting_hotkey.as_deref().and_then(parse_hotkey_string);

    let label = hotkey_display_label(&default_hotkey);
    if let Some(tray) = app.tray_by_id("main-tray") {
        let tooltip = if settings::is_debug() {
            format!("Sumi [Dev] – {} to record", label)
        } else {
            format!("Sumi – {} to record", label)
        };
        let _ = tray.set_tooltip(Some(&tooltip));
    }

    tracing::info!("Settings reset to defaults (hotkey: {})", label);
    Ok(())
}

#[tauri::command]
pub fn get_default_prompt() -> String {
    polisher::base_prompt_template()
}

#[tauri::command]
pub fn get_default_prompt_rules(language: Option<String>) -> Vec<polisher::PromptRule> {
    polisher::default_prompt_rules_for_lang(language.as_deref())
}

#[tauri::command]
pub fn save_api_key(state: State<'_, AppState>, provider: String, key: String) -> Result<(), String> {
    if key.is_empty() {
        credentials::delete(&provider)?;
        if let Ok(mut map) = state.api_key_cache.lock() {
            map.remove(&provider);
        }
    } else {
        credentials::save(&provider, &key)?;
        if let Ok(mut map) = state.api_key_cache.lock() {
            map.insert(provider, key);
        }
    }
    Ok(())
}

#[tauri::command]
pub fn get_api_key(state: State<'_, AppState>, provider: String) -> Result<String, String> {
    Ok(get_cached_api_key(&state.api_key_cache, &provider))
}

#[derive(Serialize)]
pub struct HistoryPage {
    pub entries: Vec<history::HistoryEntry>,
    pub has_more: bool,
}

#[tauri::command]
pub async fn get_history_page(
    before_timestamp: Option<i64>,
    limit: Option<u32>,
) -> Result<HistoryPage, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let limit = limit.unwrap_or(10);
        let (entries, has_more) =
            history::load_history_page(&settings::history_dir(), before_timestamp, limit);
        HistoryPage { entries, has_more }
    })
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_history_stats() -> Result<history::HistoryStats, String> {
    tauri::async_runtime::spawn_blocking(move || history::get_stats(&settings::history_dir()))
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_history() -> Result<Vec<history::HistoryEntry>, String> {
    tauri::async_runtime::spawn_blocking(move || history::load_history(&settings::history_dir()))
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_history_entry(id: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        history::delete_entry(&settings::history_dir(), &settings::audio_dir(), &id);
    })
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn export_history_audio(id: String) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let dest = history::export_audio(&settings::audio_dir(), &id)?;
        Ok(dest.to_string_lossy().to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn clear_all_history() -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        history::clear_all(&settings::history_dir(), &settings::audio_dir());
    })
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_history_storage_path() -> String {
    settings::base_dir().to_string_lossy().to_string()
}

#[tauri::command]
pub fn get_app_icon(bundle_id: String) -> Result<String, String> {
    #[cfg(target_os = "macos")]
    {
        get_app_icon_macos(&bundle_id)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = bundle_id;
        Err("Not supported on this platform".to_string())
    }
}

#[cfg(target_os = "macos")]
fn get_app_icon_macos(bundle_id: &str) -> Result<String, String> {
    use base64::Engine;
    use std::ffi::{c_char, c_void};

    extern "C" {
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_getClass(name: *const c_char) -> *mut c_void;
        fn objc_msgSend();
    }

    unsafe {
        let send_void: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let send_obj_obj: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());

        let ws_cls = objc_getClass(c"NSWorkspace".as_ptr());
        if ws_cls.is_null() {
            return Err("NSWorkspace not found".to_string());
        }
        let workspace = send_void(ws_cls, sel_registerName(c"sharedWorkspace".as_ptr()));
        if workspace.is_null() {
            return Err("sharedWorkspace is null".to_string());
        }

        let nsstring_cls = objc_getClass(c"NSString".as_ptr());
        if nsstring_cls.is_null() {
            return Err("NSString class not found".to_string());
        }
        let c_bundle = std::ffi::CString::new(bundle_id.as_bytes())
            .map_err(|_| "Invalid bundle_id".to_string())?;
        let send_cstr: unsafe extern "C" fn(*mut c_void, *mut c_void, *const i8) -> *mut c_void =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let ns_bundle_id = send_cstr(
            nsstring_cls,
            sel_registerName(c"stringWithUTF8String:".as_ptr()),
            c_bundle.as_ptr(),
        );
        if ns_bundle_id.is_null() {
            return Err("Failed to create NSString".to_string());
        }

        let app_url = send_obj_obj(
            workspace,
            sel_registerName(c"URLForApplicationWithBundleIdentifier:".as_ptr()),
            ns_bundle_id,
        );
        if app_url.is_null() {
            return Err("App not found for bundle_id".to_string());
        }

        let app_path = send_void(app_url, sel_registerName(c"path".as_ptr()));
        if app_path.is_null() {
            return Err("Failed to get app path".to_string());
        }

        let icon = send_obj_obj(
            workspace,
            sel_registerName(c"iconForFile:".as_ptr()),
            app_path,
        );
        if icon.is_null() {
            return Err("Failed to get icon".to_string());
        }

        #[repr(C)]
        struct NSSize {
            width: f64,
            height: f64,
        }
        let send_set_size: unsafe extern "C" fn(*mut c_void, *mut c_void, NSSize) =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        send_set_size(
            icon,
            sel_registerName(c"setSize:".as_ptr()),
            NSSize {
                width: 32.0,
                height: 32.0,
            },
        );

        let tiff_data = send_void(icon, sel_registerName(c"TIFFRepresentation".as_ptr()));
        if tiff_data.is_null() {
            return Err("Failed to get TIFF data".to_string());
        }

        let bitmap_cls = objc_getClass(c"NSBitmapImageRep".as_ptr());
        if bitmap_cls.is_null() {
            return Err("NSBitmapImageRep not found".to_string());
        }
        let bitmap_rep = send_obj_obj(
            bitmap_cls,
            sel_registerName(c"imageRepWithData:".as_ptr()),
            tiff_data,
        );
        if bitmap_rep.is_null() {
            return Err("Failed to create bitmap rep".to_string());
        }

        let send_png: unsafe extern "C" fn(*mut c_void, *mut c_void, u64, *mut c_void) -> *mut c_void =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let empty_dict_cls = objc_getClass(c"NSDictionary".as_ptr());
        let empty_dict = send_void(empty_dict_cls, sel_registerName(c"dictionary".as_ptr()));
        let png_data = send_png(
            bitmap_rep,
            sel_registerName(c"representationUsingType:properties:".as_ptr()),
            4,
            empty_dict,
        );
        if png_data.is_null() {
            return Err("Failed to create PNG data".to_string());
        }

        let send_len: unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64 =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let send_bytes: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *const u8 =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());

        let len = send_len(png_data, sel_registerName(c"length".as_ptr())) as usize;
        let bytes_ptr = send_bytes(png_data, sel_registerName(c"bytes".as_ptr()));
        if bytes_ptr.is_null() || len == 0 {
            return Err("PNG data is empty".to_string());
        }

        let bytes = std::slice::from_raw_parts(bytes_ptr, len);
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        Ok(format!("data:image/png;base64,{}", b64))
    }
}

#[derive(Serialize)]
pub struct TestPolishResult {
    current_result: String,
    edited_result: String,
}

#[tauri::command]
pub async fn test_polish(
    app: AppHandle,
    test_text: String,
    custom_prompt: String,
) -> Result<TestPolishResult, String> {
    let config = {
        let state = app.state::<AppState>();
        let mut config = state.settings.lock().map_err(|e| e.to_string())?.polish.clone();
        if config.mode == polisher::PolishMode::Cloud {
            let key = get_cached_api_key(&state.api_key_cache, config.cloud.provider.as_key());
            if !key.is_empty() {
                config.cloud.api_key = key;
            }
        }
        config
    };

    let model_dir = settings::models_dir();
    let default_instructions = polisher::resolve_prompt(&polisher::base_prompt_template());
    let custom_instructions = polisher::resolve_prompt(&custom_prompt);

    let app_clone = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let state = app_clone.state::<AppState>();
        let system_prompt = "You are a speech-to-text post-processor.";

        let default_user = format!("<speech>\n{}\n</speech>\n\n{}", test_text, default_instructions);
        let default_result = polisher::polish_with_prompt(
            &state.llm_model,
            &model_dir,
            &config,
            system_prompt,
            &default_user,
            &state.http_client,
            None,
        )?;

        let custom_user = format!("<speech>\n{}\n</speech>\n\n{}", test_text, custom_instructions);
        let custom_result = polisher::polish_with_prompt(
            &state.llm_model,
            &model_dir,
            &config,
            system_prompt,
            &custom_user,
            &state.http_client,
            None,
        )?;

        Ok(TestPolishResult {
            current_result: default_result,
            edited_result: custom_result,
        })
    })
    .await
    .map_err(|e| format!("Test polish task failed: {}", e))?
}

// ── Voice Add Rule ────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct GeneratedRule {
    name: String,
    match_type: String,
    match_value: String,
    prompt: String,
}

fn parse_generated_rule(raw: &str) -> Result<GeneratedRule, String> {
    let stripped = raw.trim();
    let stripped = if stripped.starts_with("```") {
        let s = stripped
            .trim_start_matches("```json")
            .trim_start_matches("```");
        s.strip_suffix("```").unwrap_or(s)
    } else {
        stripped
    }
    .trim();

    let start = stripped.find('{').ok_or("No JSON object found in LLM response")?;
    let end = stripped.rfind('}').ok_or("No closing brace found in LLM response")?;
    if end <= start {
        return Err("Invalid JSON structure".to_string());
    }
    let json_str = &stripped[start..=end];

    let val: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

    let name = val
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let match_type = val
        .get("match_type")
        .and_then(|v| v.as_str())
        .unwrap_or("app_name")
        .to_string();
    let match_value = val
        .get("match_value")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let prompt = val
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let match_type = match match_type.as_str() {
        "app_name" | "bundle_id" | "url" => match_type,
        _ => "app_name".to_string(),
    };

    Ok(GeneratedRule {
        name,
        match_type,
        match_value,
        prompt,
    })
}

#[tauri::command]
pub async fn generate_rule_from_description(
    app: AppHandle,
    description: String,
) -> Result<GeneratedRule, String> {
    let (config, model_dir) = {
        let state = app.state::<AppState>();
        let mut config = state
            .settings
            .lock()
            .map_err(|e| e.to_string())?
            .polish
            .clone();

        if config.mode == polisher::PolishMode::Cloud {
            let key = get_cached_api_key(&state.api_key_cache, config.cloud.provider.as_key());
            if !key.is_empty() {
                config.cloud.api_key = key;
            }
        }

        let model_dir = settings::models_dir();
        if !polisher::is_polish_ready(&model_dir, &config) {
            return Err("LLM not configured".to_string());
        }
        (config, model_dir)
    };

    let app_clone = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let state = app_clone.state::<AppState>();

        let lang_hint = "the same language the user uses";
        let system_prompt = "You are a JSON generator.";
        let user_text = format!(
            r#"{description}

Convert the description above into a JSON object with these fields:
- "name": short descriptive name (max 30 chars)
- "match_type": one of "app_name", "bundle_id", or "url"
- "match_value": the value to match (e.g. app name, bundle ID, or URL pattern)
- "prompt": detailed instruction for AI polishing when this rule matches

If a specific app is mentioned, use "app_name" as match_type.
If a website or URL is mentioned, use "url" as match_type.
If the match target is unclear, use "app_name" with empty match_value.

Write "name" and "prompt" in {lang_hint}.
Return ONLY the JSON object."#
        );

        let result = polisher::polish_with_prompt(
            &state.llm_model,
            &model_dir,
            &config,
            system_prompt,
            &user_text,
            &state.http_client,
            None,
        )?;

        parse_generated_rule(&result)
    })
    .await
    .map_err(|e| format!("Generate rule task failed: {}", e))?
}

#[tauri::command]
pub fn start_recording(state: State<'_, AppState>) -> Result<(), String> {
    if state.meeting_active.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("meeting_mode_active".to_string());
    }
    let device_name = state.settings.lock().ok().and_then(|s| s.mic_device.clone());
    audio::do_start_recording(
        &state.is_recording,
        &state.mic_available,
        &state.reconnecting,
        &state.sample_rate,
        &state.buffer,
        &state.is_recording,
        &state.audio_thread,
        device_name,
    )
}

#[tauri::command]
pub fn stop_recording(state: State<'_, AppState>) -> Result<String, String> {
    // Refuse to stop if meeting mode is active — the meeting hotkey must be
    // used to end a meeting session so the transcript is handled correctly.
    if state.meeting_active.load(std::sync::atomic::Ordering::SeqCst) {
        return Err("meeting_mode_active".to_string());
    }

    let mut stt_config = state.settings.lock().map_err(|e| e.to_string())?.stt.clone();
    if stt_config.mode == SttMode::Cloud {
        let key = get_cached_api_key(&state.api_key_cache, stt_config.cloud.provider.as_key());
        if !key.is_empty() {
            stt_config.cloud.api_key = key;
        }
    }
    let stt_language = stt_config.language.clone();
    let dictionary_terms: Vec<String> = state
        .settings
        .lock()
        .map(|s| {
            s.polish
                .dictionary
                .entries
                .iter()
                .filter(|e| e.enabled && !e.term.is_empty())
                .map(|e| e.term.clone())
                .collect()
        })
        .unwrap_or_default();
    audio::do_stop_recording(
        &state,
        &stt_config,
        &stt_language,
        &dictionary_terms,
    )
    .map(|(text, _samples)| text)
}

#[tauri::command]
pub fn set_test_mode(state: State<'_, AppState>, enabled: bool) {
    state.test_mode.store(enabled, Ordering::SeqCst);
}

#[tauri::command]
pub fn set_voice_rule_mode(state: State<'_, AppState>, enabled: bool) {
    state.voice_rule_mode.store(enabled, Ordering::SeqCst);
}

#[tauri::command]
pub fn set_context_override(
    state: State<'_, AppState>,
    app_name: String,
    bundle_id: String,
    url: String,
) -> Result<(), String> {
    use crate::context_detect;
    if let Ok(mut ctx) = state.context_override.lock() {
        if app_name.is_empty() && bundle_id.is_empty() && url.is_empty() {
            *ctx = None;
        } else {
            *ctx = Some(context_detect::AppContext { app_name, bundle_id, url, terminal_host: String::new() });
        }
    }
    Ok(())
}

#[tauri::command]
pub fn set_edit_text_override(state: State<'_, AppState>, text: String) {
    if let Ok(mut ov) = state.edit_text_override.lock() {
        *ov = if text.is_empty() { None } else { Some(text) };
    }
}

#[tauri::command]
pub fn cancel_recording(app: AppHandle, state: State<'_, AppState>) {
    // If meeting mode is active, signal the feeder to discard its work
    // (transcript, clipboard copy, history save) before hiding the overlay.
    if state.meeting_active.load(Ordering::SeqCst) {
        state.meeting_cancelled.store(true, Ordering::SeqCst);
        state.meeting_active.store(false, Ordering::SeqCst);
    }
    state.is_recording.store(false, Ordering::SeqCst);
    // Intentionally not updating last_recording_end: a cancelled recording
    // does not count as "real use" for the idle mic timeout.
    // Wake any sleeping feeder immediately.
    state.feeder_stop_cv.notify_all();
    // Resume music if we paused it when recording started.
    if state.media_paused_by_sumi.swap(false, Ordering::SeqCst) {
        crate::platform::resume_now_playing();
    }
    if let Some(overlay) = app.get_webview_window("overlay") {
        platform::hide_overlay(&overlay);
    }
}

#[derive(Serialize)]
pub struct MicStatus {
    connected: bool,
    default_device: Option<String>,
    devices: Vec<String>,
}

#[tauri::command]
pub fn get_mic_status(_state: State<'_, AppState>) -> MicStatus {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    let default_device = host.default_input_device().and_then(|d| d.name().ok());
    // Use CoreAudio to filter out virtual audio devices (e.g. BlackHole,
    // Loopback, Speaker Audio Recorder) on macOS. Fall back to cpal's
    // list with name-based filtering on other platforms.
    let physical = crate::audio_devices::list_physical_input_device_names();
    let devices: Vec<String> = if !physical.is_empty() {
        physical
    } else {
        host.input_devices()
            .map(|devs| {
                devs.filter_map(|d| d.name().ok())
                    .filter(|name| !crate::audio_devices::is_known_virtual_device(name))
                    .collect()
            })
            .unwrap_or_default()
    };

    // On-demand model: stream is closed between recordings.
    // Report physical device availability, not stream state.
    let connected = !devices.is_empty() || default_device.is_some();

    MicStatus {
        connected,
        default_device,
        devices,
    }
}

#[tauri::command]
pub fn set_mic_device(device_name: Option<String>, state: State<'_, AppState>) -> Result<(), String> {
    if state.is_recording.load(Ordering::SeqCst) {
        return Err("Cannot change input device while recording".to_string());
    }

    // Save to settings
    {
        let mut settings = state.settings.lock().map_err(|e| e.to_string())?;
        settings.mic_device = device_name.clone();
        settings::save_settings_to_disk(&settings);
    }

    // Close the old stream (if any) and clear the buffer.
    // On-demand model: the new device will be used on the next recording start.
    audio::close_audio_stream(&state.audio_thread, &state.mic_available);
    if let Ok(mut buf) = state.buffer.lock() {
        buf.clear();
    }
    Ok(())
}

// ── Model download ──────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ModelStatus {
    engine: String,
    model_exists: bool,
}

#[tauri::command]
pub fn check_model_status() -> ModelStatus {
    let model_exists = settings::models_dir()
        .join("ggml-large-v3-turbo-zh-TW.bin")
        .exists();
    ModelStatus {
        engine: "whisper".to_string(),
        model_exists,
    }
}

#[tauri::command]
pub fn download_model(app: AppHandle) -> Result<(), String> {
    use std::io::Read as _;

    let dir = settings::models_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let model_path = dir.join("ggml-large-v3-turbo-zh-TW.bin");
    if model_path.exists() {
        let _ = app.emit("model-download-progress", serde_json::json!({
            "status": "complete",
            "downloaded": 0u64,
            "total": 0u64,
            "percent": 100.0
        }));
        return Ok(());
    }

    {
        let state = app.state::<AppState>();
        if state.downloading.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
            return Err("A model download is already in progress".to_string());
        }
    }

    let tmp_path = model_path.with_extension("bin.part");
    let _ = std::fs::remove_file(&tmp_path);

    std::thread::spawn(move || {
        (|| {
        let url = "https://huggingface.co/Alkd/whisper-large-v3-turbo-zh-TW/resolve/main/ggml-model.bin";
        let client = match reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                let _ = app.emit("model-download-progress", serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to create HTTP client: {}", e)
                }));
                return;
            }
        };

        let resp = match client.get(url).send() {
            Ok(r) => r,
            Err(e) => {
                let _ = app.emit("model-download-progress", serde_json::json!({
                    "status": "error",
                    "message": format!("Download request failed: {}", e)
                }));
                return;
            }
        };

        if !resp.status().is_success() {
            let _ = app.emit("model-download-progress", serde_json::json!({
                "status": "error",
                "message": format!("Download returned HTTP {}", resp.status())
            }));
            return;
        }

        let total = resp.content_length().unwrap_or(0);

        let mut file = match std::fs::File::create(&tmp_path) {
            Ok(f) => f,
            Err(e) => {
                let _ = app.emit("model-download-progress", serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to create temp file: {}", e)
                }));
                return;
            }
        };

        let mut downloaded: u64 = 0;
        let mut buf = [0u8; 65536];
        let mut last_emit = Instant::now();
        let mut reader = resp;

        loop {
            let n = match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => {
                    let _ = app.emit("model-download-progress", serde_json::json!({
                        "status": "error",
                        "message": format!("Download read error: {}", e)
                    }));
                    return;
                }
            };

            if let Err(e) = std::io::Write::write_all(&mut file, &buf[..n]) {
                let _ = app.emit("model-download-progress", serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to write to disk: {}", e)
                }));
                return;
            }

            downloaded += n as u64;

            if last_emit.elapsed() >= std::time::Duration::from_millis(100) {
                let percent = if total > 0 {
                    (downloaded as f64 / total as f64) * 100.0
                } else {
                    0.0
                };
                let _ = app.emit("model-download-progress", serde_json::json!({
                    "status": "downloading",
                    "downloaded": downloaded,
                    "total": total,
                    "percent": percent
                }));
                last_emit = Instant::now();
            }
        }

        drop(file);
        if let Err(e) = std::fs::rename(&tmp_path, &model_path) {
            let _ = app.emit("model-download-progress", serde_json::json!({
                "status": "error",
                "message": format!("Failed to rename temp file: {}", e)
            }));
            return;
        }

        if let Some(app_state) = app.try_state::<AppState>() {
            if let Ok(mut ctx) = app_state.whisper_ctx.lock() {
                *ctx = None;
                tracing::info!("Whisper context cache invalidated after model download");
            }
        }

        let _ = app.emit("model-download-progress", serde_json::json!({
            "status": "complete",
            "downloaded": downloaded,
            "total": total,
            "percent": 100.0
        }));
        tracing::info!("Whisper model downloaded: {:?}", model_path);
        })();
        if let Some(state) = app.try_state::<AppState>() {
            state.downloading.store(false, Ordering::SeqCst);
        }
    });

    Ok(())
}

// ── LLM Model management ────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct LlmModelStatus {
    model: String,
    model_exists: bool,
    model_size_bytes: u64,
}

#[tauri::command]
pub fn check_llm_model_status(state: State<'_, AppState>) -> LlmModelStatus {
    let settings = state.settings.lock()
        .unwrap_or_else(|e| e.into_inner());
    let model = &settings.polish.model;
    let dir = settings::models_dir();
    let (exists, size) = polisher::model_file_status(&dir, model);
    LlmModelStatus {
        model: model.display_name().to_string(),
        model_exists: exists,
        model_size_bytes: size,
    }
}

#[tauri::command]
pub fn download_llm_model(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    use std::io::Read as _;

    let dir = settings::models_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let settings = state.settings.lock().map_err(|e| e.to_string())?;
    let model = settings.polish.model.clone();
    drop(settings);

    let model_path = dir.join(model.filename());
    if model_path.exists() {
        let _ = app.emit("llm-model-download-progress", serde_json::json!({
            "status": "complete",
            "downloaded": 0u64,
            "total": 0u64,
            "percent": 100.0
        }));
        return Ok(());
    }

    if state.downloading.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
        return Err("A model download is already in progress".to_string());
    }

    let tmp_path = model_path.with_extension("gguf.part");
    let _ = std::fs::remove_file(&tmp_path);

    let url = model.download_url().to_string();
    let downloaded_model = model;

    std::thread::spawn(move || {
        (|| {
        let client = match reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(1800))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                let _ = app.emit("llm-model-download-progress", serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to create HTTP client: {}", e)
                }));
                return;
            }
        };

        let resp = match client.get(&url).send() {
            Ok(r) => r,
            Err(e) => {
                let _ = app.emit("llm-model-download-progress", serde_json::json!({
                    "status": "error",
                    "message": format!("Download request failed: {}", e)
                }));
                return;
            }
        };

        if !resp.status().is_success() {
            let _ = app.emit("llm-model-download-progress", serde_json::json!({
                "status": "error",
                "message": format!("Download returned HTTP {}", resp.status())
            }));
            return;
        }

        let total = resp.content_length().unwrap_or(0);

        let mut file = match std::fs::File::create(&tmp_path) {
            Ok(f) => f,
            Err(e) => {
                let _ = app.emit("llm-model-download-progress", serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to create temp file: {}", e)
                }));
                return;
            }
        };

        let mut downloaded: u64 = 0;
        let mut buf = [0u8; 65536];
        let mut last_emit = Instant::now();
        let mut reader = resp;

        loop {
            let n = match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => {
                    let _ = app.emit("llm-model-download-progress", serde_json::json!({
                        "status": "error",
                        "message": format!("Download read error: {}", e)
                    }));
                    return;
                }
            };

            if let Err(e) = std::io::Write::write_all(&mut file, &buf[..n]) {
                let _ = app.emit("llm-model-download-progress", serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to write to disk: {}", e)
                }));
                return;
            }

            downloaded += n as u64;

            if last_emit.elapsed() >= std::time::Duration::from_millis(100) {
                let percent = if total > 0 {
                    (downloaded as f64 / total as f64) * 100.0
                } else {
                    0.0
                };
                let _ = app.emit("llm-model-download-progress", serde_json::json!({
                    "status": "downloading",
                    "downloaded": downloaded,
                    "total": total,
                    "percent": percent
                }));
                last_emit = Instant::now();
            }
        }

        drop(file);
        if let Err(e) = std::fs::rename(&tmp_path, &model_path) {
            let _ = app.emit("llm-model-download-progress", serde_json::json!({
                "status": "error",
                "message": format!("Failed to rename temp file: {}", e)
            }));
            return;
        }

        if let Some(app_state) = app.try_state::<AppState>() {
            if let Err(e) = polisher::warm_llm_cache(&app_state.llm_model, &settings::models_dir(), &downloaded_model) {
                tracing::error!("Failed to pre-warm LLM after download: {}", e);
                polisher::invalidate_cache(&app_state.llm_model); // fallback to lazy load
            }
        }

        let _ = app.emit("llm-model-download-progress", serde_json::json!({
            "status": "complete",
            "downloaded": downloaded,
            "total": total,
            "percent": 100.0
        }));
        tracing::info!("LLM model downloaded: {:?}", model_path);
        })();
        if let Some(state) = app.try_state::<AppState>() {
            state.downloading.store(false, Ordering::SeqCst);
        }
    });

    Ok(())
}

// ── Polish model management ──────────────────────────────────────────────────

#[tauri::command]
pub fn list_polish_models(state: State<'_, AppState>) -> Vec<PolishModelInfo> {
    let (active_model, recommended) = state
        .settings
        .lock()
        .map(|s| {
            let lang = if !s.stt.language.is_empty() && s.stt.language != "auto" {
                Some(s.stt.language.clone())
            } else {
                s.language.clone()
            };
            (s.polish.model.clone(), polisher::recommend_polish_model(lang.as_deref()))
        })
        .unwrap_or_default();
    polisher::PolishModel::all()
        .iter()
        .map(|m| PolishModelInfo::from_model(m, &active_model, &recommended))
        .collect()
}

#[tauri::command]
pub async fn switch_polish_model(app: AppHandle, model: polisher::PolishModel) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        // Guard: refuse if recording or processing is already in progress.
        if state.is_recording.load(Ordering::SeqCst) || state.is_processing.load(Ordering::SeqCst) {
            return Err("Cannot switch model while recording or processing".to_string());
        }

        {
            let mut settings = state.settings.lock().map_err(|e| e.to_string())?;
            settings.polish.model = model.clone();
            settings::save_settings_to_disk(&settings);
        }

        // Invalidate and pre-warm the new model if it's already downloaded
        polisher::invalidate_cache(&state.llm_model);
        let model_dir = settings::models_dir();
        if model_dir.join(model.filename()).exists() {
            if let Err(e) = polisher::warm_llm_cache(&state.llm_model, &model_dir, &model) {
                tracing::error!("Failed to pre-warm LLM: {}", e);
            }
        }
        tracing::info!(
            "Polish model switched to {}",
            model.display_name()
        );

        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub fn download_polish_model(app: AppHandle, model: polisher::PolishModel) -> Result<(), String> {
    use std::io::Read as _;

    let dir = settings::models_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let model_path = dir.join(model.filename());
    let tokenizer_path = model.tokenizer_filename().map(|f| dir.join(f));
    let already_complete = model_path.exists()
        && tokenizer_path.as_ref().is_none_or(|p| p.exists());
    if already_complete {
        let _ = app.emit("polish-model-download-progress", serde_json::json!({
            "status": "complete",
            "downloaded": 0u64,
            "total": 0u64,
            "percent": 100.0
        }));
        return Ok(());
    }

    {
        let state = app.state::<AppState>();
        if state.downloading.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
            return Err("A model download is already in progress".to_string());
        }
    }

    let tmp_path = model_path.with_extension("gguf.part");
    let _ = std::fs::remove_file(&tmp_path);

    let url = model.download_url().to_string();

    std::thread::spawn(move || {
        (|| {
        let client = match reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(1800))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                let _ = app.emit("polish-model-download-progress", serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to create HTTP client: {}", e)
                }));
                return;
            }
        };

        let resp = match client.get(&url).send() {
            Ok(r) => r,
            Err(e) => {
                let _ = app.emit("polish-model-download-progress", serde_json::json!({
                    "status": "error",
                    "message": format!("Download request failed: {}", e)
                }));
                return;
            }
        };

        if !resp.status().is_success() {
            let _ = app.emit("polish-model-download-progress", serde_json::json!({
                "status": "error",
                "message": format!("Download returned HTTP {}", resp.status())
            }));
            return;
        }

        let total = resp.content_length().unwrap_or(0);

        let mut file = match std::fs::File::create(&tmp_path) {
            Ok(f) => f,
            Err(e) => {
                let _ = app.emit("polish-model-download-progress", serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to create temp file: {}", e)
                }));
                return;
            }
        };

        let mut downloaded: u64 = 0;
        let mut buf = [0u8; 65536];
        let mut last_emit = Instant::now();
        let mut reader = resp;

        loop {
            let n = match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => {
                    let _ = app.emit("polish-model-download-progress", serde_json::json!({
                        "status": "error",
                        "message": format!("Download read error: {}", e)
                    }));
                    return;
                }
            };

            if let Err(e) = std::io::Write::write_all(&mut file, &buf[..n]) {
                let _ = app.emit("polish-model-download-progress", serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to write to disk: {}", e)
                }));
                return;
            }

            downloaded += n as u64;

            if last_emit.elapsed() >= std::time::Duration::from_millis(100) {
                let percent = if total > 0 {
                    (downloaded as f64 / total as f64) * 100.0
                } else {
                    0.0
                };
                let _ = app.emit("polish-model-download-progress", serde_json::json!({
                    "status": "downloading",
                    "downloaded": downloaded,
                    "total": total,
                    "percent": percent
                }));
                last_emit = Instant::now();
            }
        }

        drop(file);
        if let Err(e) = std::fs::rename(&tmp_path, &model_path) {
            let _ = app.emit("polish-model-download-progress", serde_json::json!({
                "status": "error",
                "message": format!("Failed to rename temp file: {}", e)
            }));
            return;
        }

        // Download external tokenizer JSON if required (e.g. Phi-4-mini).
        if let (Some(tok_url), Some(tok_path)) = (model.tokenizer_url(), tokenizer_path.as_ref()) {
            if !tok_path.exists() {
                match client.get(tok_url).send() {
                    Ok(r) => {
                        if !r.status().is_success() {
                            let _ = app.emit("polish-model-download-progress", serde_json::json!({
                                "status": "error",
                                "message": format!("Failed to download tokenizer: HTTP {}", r.status())
                            }));
                            return;
                        }
                        match r.bytes() {
                            Ok(bytes) => {
                                if let Err(e) = std::fs::write(tok_path, &bytes) {
                                    let _ = app.emit("polish-model-download-progress", serde_json::json!({
                                        "status": "error",
                                        "message": format!("Failed to write tokenizer: {}", e)
                                    }));
                                    return;
                                }
                            }
                            Err(e) => {
                                let _ = app.emit("polish-model-download-progress", serde_json::json!({
                                    "status": "error",
                                    "message": format!("Failed to download tokenizer: {}", e)
                                }));
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = app.emit("polish-model-download-progress", serde_json::json!({
                            "status": "error",
                            "message": format!("Failed to download tokenizer: {}", e)
                        }));
                        return;
                    }
                }
            }
        }

        if let Some(app_state) = app.try_state::<AppState>() {
            polisher::invalidate_cache(&app_state.llm_model);
        }

        let _ = app.emit("polish-model-download-progress", serde_json::json!({
            "status": "complete",
            "downloaded": downloaded,
            "total": total,
            "percent": 100.0
        }));
        tracing::info!("Polish model downloaded: {:?}", model_path);
        })();
        if let Some(state) = app.try_state::<AppState>() {
            state.downloading.store(false, Ordering::SeqCst);
        }
    });

    Ok(())
}

// ── Whisper model management ─────────────────────────────────────────────────

#[tauri::command]
pub fn list_whisper_models(state: State<'_, AppState>) -> Vec<WhisperModelInfo> {
    let active_model = state
        .settings
        .lock()
        .map(|s| s.stt.whisper_model.clone())
        .unwrap_or_default();
    WhisperModel::all()
        .iter()
        .map(|m| WhisperModelInfo::from_model(m, &active_model))
        .collect()
}

#[tauri::command]
pub fn get_system_info() -> SystemInfo {
    whisper_models::detect_system_info()
}

#[tauri::command]
pub fn get_whisper_model_recommendation(state: State<'_, AppState>) -> WhisperModel {
    let system = whisper_models::detect_system_info();
    // When stt.language is "auto", resolve to proper BCP-47 code via system locale
    let stt_language = state
        .settings
        .lock()
        .ok()
        .map(|s| s.stt.language.clone())
        .filter(|l| !l.is_empty() && l != "auto")
        .or_else(|| {
            whisper_models::detect_system_language()
                .map(|locale| crate::stt::locale_to_stt_language(&locale))
                .filter(|l| l != "auto")
        });
    whisper_models::recommend_model(&system, stt_language.as_deref())
}

#[tauri::command]
pub async fn switch_whisper_model(app: AppHandle, model: WhisperModel) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        // Atomically claim the switching slot — prevents concurrent calls from racing.
        if state.model_switching.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
            return Err("Model switch already in progress".to_string());
        }

        // Guard: refuse if recording or processing is already in progress.
        if state.is_recording.load(Ordering::SeqCst) || state.is_processing.load(Ordering::SeqCst) {
            state.model_switching.store(false, Ordering::SeqCst);
            return Err("Cannot switch model while recording or processing".to_string());
        }

        match state.settings.lock() {
            Ok(mut s) => {
                s.stt.whisper_model = model.clone();
                settings::save_settings_to_disk(&s);
            }
            Err(e) => {
                state.model_switching.store(false, Ordering::SeqCst);
                return Err(e.to_string());
            }
        }

        // Show overlay on main thread (Cocoa APIs are thread-affine).
        {
            let app2 = app.clone();
            let _ = app.run_on_main_thread(move || {
                if let Some(ov) = app2.get_webview_window("overlay") {
                    platform::show_overlay(&ov);
                    let _ = ov.emit("model-switching", serde_json::json!({"status": "start"}));
                }
            });
        }

        // Pre-warm the new model. Use catch_unwind so model_switching is always cleared
        // even if whisper-rs panics on a corrupt model file.
        let warm_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::transcribe::warm_whisper_cache(&state.whisper_ctx, &model)
        }));

        // Emit "done" and hide overlay on the main thread, then clear the flag —
        // keeping model_switching=true until the UI transition is finished so the
        // hotkey cannot fire in the gap between warm completion and overlay hide.
        {
            let app2 = app.clone();
            let _ = app.run_on_main_thread(move || {
                if let Some(ov) = app2.get_webview_window("overlay") {
                    let _ = ov.emit("model-switching", serde_json::json!({"status": "done"}));
                    platform::hide_overlay(&ov);
                }
                app2.state::<AppState>().model_switching.store(false, Ordering::SeqCst);
            });
        }

        match warm_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::error!("Failed to pre-warm whisper model: {}", e),
            Err(_) => tracing::error!("Whisper model pre-warm panicked"),
        }

        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub fn download_whisper_model(app: AppHandle, model: WhisperModel) -> Result<(), String> {
    use std::io::Read as _;

    let url = model
        .download_url()
        .ok_or_else(|| format!("No download URL for model: {}", model.display_name()))?
        .to_string();

    let dir = settings::models_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let model_path = dir.join(model.filename());
    if model_path.exists() {
        let _ = app.emit(
            "whisper-model-download-progress",
            serde_json::json!({
                "status": "complete",
                "downloaded": 0u64,
                "total": 0u64,
                "percent": 100.0
            }),
        );
        return Ok(());
    }

    {
        let state = app.state::<AppState>();
        if state.downloading.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
            return Err("A model download is already in progress".to_string());
        }
    }

    let tmp_path = model_path.with_extension("bin.part");
    let _ = std::fs::remove_file(&tmp_path);

    // LargeV3TurboZhTw downloads as ggml-model.bin; all models are renamed from
    // tmp_path to model_path at the end of the download.

    std::thread::spawn(move || {
        (|| {
        let client = match reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(1800))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                let _ = app.emit(
                    "whisper-model-download-progress",
                    serde_json::json!({
                        "status": "error",
                        "message": format!("Failed to create HTTP client: {}", e)
                    }),
                );
                return;
            }
        };

        let resp = match client.get(&url).send() {
            Ok(r) => r,
            Err(e) => {
                let _ = app.emit(
                    "whisper-model-download-progress",
                    serde_json::json!({
                        "status": "error",
                        "message": format!("Download request failed: {}", e)
                    }),
                );
                return;
            }
        };

        if !resp.status().is_success() {
            let _ = app.emit(
                "whisper-model-download-progress",
                serde_json::json!({
                    "status": "error",
                    "message": format!("Download returned HTTP {}", resp.status())
                }),
            );
            return;
        }

        let total = resp.content_length().unwrap_or(0);

        let mut file = match std::fs::File::create(&tmp_path) {
            Ok(f) => f,
            Err(e) => {
                let _ = app.emit(
                    "whisper-model-download-progress",
                    serde_json::json!({
                        "status": "error",
                        "message": format!("Failed to create temp file: {}", e)
                    }),
                );
                return;
            }
        };

        let mut downloaded: u64 = 0;
        let mut buf = [0u8; 65536];
        let mut last_emit = Instant::now();
        let mut reader = resp;

        loop {
            let n = match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => {
                    let _ = app.emit(
                        "whisper-model-download-progress",
                        serde_json::json!({
                            "status": "error",
                            "message": format!("Download read error: {}", e)
                        }),
                    );
                    return;
                }
            };

            if let Err(e) = std::io::Write::write_all(&mut file, &buf[..n]) {
                let _ = app.emit(
                    "whisper-model-download-progress",
                    serde_json::json!({
                        "status": "error",
                        "message": format!("Failed to write to disk: {}", e)
                    }),
                );
                return;
            }

            downloaded += n as u64;

            if last_emit.elapsed() >= std::time::Duration::from_millis(100) {
                let percent = if total > 0 {
                    (downloaded as f64 / total as f64) * 100.0
                } else {
                    0.0
                };
                let _ = app.emit(
                    "whisper-model-download-progress",
                    serde_json::json!({
                        "status": "downloading",
                        "downloaded": downloaded,
                        "total": total,
                        "percent": percent
                    }),
                );
                last_emit = Instant::now();
            }
        }

        drop(file);
        if let Err(e) = std::fs::rename(&tmp_path, &model_path) {
            let _ = app.emit(
                "whisper-model-download-progress",
                serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to rename temp file: {}", e)
                }),
            );
            return;
        }

        // Invalidate whisper context cache
        if let Some(app_state) = app.try_state::<AppState>() {
            if let Ok(mut ctx) = app_state.whisper_ctx.lock() {
                *ctx = None;
                tracing::info!("Whisper context cache invalidated after model download");
            }
        }

        let _ = app.emit(
            "whisper-model-download-progress",
            serde_json::json!({
                "status": "complete",
                "downloaded": downloaded,
                "total": total,
                "percent": 100.0
            }),
        );
        tracing::info!("Whisper model downloaded: {:?}", model_path);
        })();
        if let Some(state) = app.try_state::<AppState>() {
            state.downloading.store(false, Ordering::SeqCst);
        }
    });

    Ok(())
}

// ── VAD model commands ──────────────────────────────────────────────────────

#[tauri::command]
pub fn check_vad_model_status() -> Result<serde_json::Value, String> {
    let downloaded = crate::transcribe::vad_model_path().exists();
    Ok(serde_json::json!({ "downloaded": downloaded }))
}

#[tauri::command]
pub fn download_vad_model(app: AppHandle) -> Result<(), String> {
    use std::io::Read as _;

    let url = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin";
    let dir = settings::models_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let model_path = crate::transcribe::vad_model_path();
    if model_path.exists() {
        let _ = app.emit(
            "vad-model-download-progress",
            serde_json::json!({ "status": "complete" }),
        );
        return Ok(());
    }

    let tmp_path = model_path.with_extension("bin.part");
    let _ = std::fs::remove_file(&tmp_path);

    std::thread::spawn(move || {
        let client = match reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                let _ = app.emit(
                    "vad-model-download-progress",
                    serde_json::json!({
                        "status": "error",
                        "message": format!("Failed to create HTTP client: {}", e)
                    }),
                );
                return;
            }
        };

        let resp = match client.get(url).send() {
            Ok(r) => r,
            Err(e) => {
                let _ = app.emit(
                    "vad-model-download-progress",
                    serde_json::json!({
                        "status": "error",
                        "message": format!("Download request failed: {}", e)
                    }),
                );
                return;
            }
        };

        if !resp.status().is_success() {
            let _ = app.emit(
                "vad-model-download-progress",
                serde_json::json!({
                    "status": "error",
                    "message": format!("Download returned HTTP {}", resp.status())
                }),
            );
            return;
        }

        let total = resp.content_length().unwrap_or(0);

        let mut file = match std::fs::File::create(&tmp_path) {
            Ok(f) => f,
            Err(e) => {
                let _ = app.emit(
                    "vad-model-download-progress",
                    serde_json::json!({
                        "status": "error",
                        "message": format!("Failed to create temp file: {}", e)
                    }),
                );
                return;
            }
        };

        let mut downloaded: u64 = 0;
        let mut buf = [0u8; 65536];
        let mut reader = resp;

        loop {
            let n = match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => {
                    let _ = app.emit(
                        "vad-model-download-progress",
                        serde_json::json!({
                            "status": "error",
                            "message": format!("Download read error: {}", e)
                        }),
                    );
                    return;
                }
            };

            if let Err(e) = std::io::Write::write_all(&mut file, &buf[..n]) {
                let _ = app.emit(
                    "vad-model-download-progress",
                    serde_json::json!({
                        "status": "error",
                        "message": format!("Failed to write to disk: {}", e)
                    }),
                );
                return;
            }

            downloaded += n as u64;
        }

        drop(file);
        if let Err(e) = std::fs::rename(&tmp_path, &model_path) {
            let _ = app.emit(
                "vad-model-download-progress",
                serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to rename temp file: {}", e)
                }),
            );
            return;
        }

        // Invalidate VAD context cache so it reloads on next use
        if let Some(app_state) = app.try_state::<AppState>() {
            if let Ok(mut ctx) = app_state.vad_ctx.lock() {
                *ctx = None;
                tracing::info!("VAD context cache invalidated after model download");
            }
        }

        let _ = app.emit(
            "vad-model-download-progress",
            serde_json::json!({
                "status": "complete",
                "downloaded": downloaded,
                "total": total
            }),
        );
        tracing::info!("VAD model downloaded: {:?}", model_path);
    });

    Ok(())
}

// ── Clipboard image ─────────────────────────────────────────────────────────

#[tauri::command]
pub fn copy_image_to_clipboard(png_bytes: Vec<u8>) -> Result<(), String> {
    let img = image::load_from_memory_with_format(&png_bytes, image::ImageFormat::Png)
        .map_err(|e| format!("Failed to decode PNG: {}", e))?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let mut clipboard = arboard::Clipboard::new().map_err(|e| format!("Clipboard error: {}", e))?;
    clipboard
        .set_image(arboard::ImageData {
            width: w as usize,
            height: h as usize,
            bytes: std::borrow::Cow::Owned(rgba.into_raw()),
        })
        .map_err(|e| format!("Failed to set clipboard image: {}", e))?;
    Ok(())
}

#[tauri::command]
pub fn is_dev_mode() -> bool {
    crate::settings::is_debug()
}

// ── Diagnostic log export ────────────────────────────────────────────────────

#[tauri::command]
pub fn export_diagnostic_log(state: State<'_, AppState>) -> Result<String, String> {
    use std::fmt::Write as _;
    use std::path::PathBuf;
    let mut report = String::new();

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    writeln!(report, "=== Sumi Diagnostic Report ===").ok();
    writeln!(report, "Version: {}", env!("CARGO_PKG_VERSION")).ok();
    writeln!(report, "Build: {}", if settings::is_debug() { "Dev" } else { "Release" }).ok();
    writeln!(report, "Timestamp (unix): {}", ts).ok();
    writeln!(report).ok();

    let sys = get_system_info();
    writeln!(report, "--- System ---").ok();
    writeln!(report, "OS: {} ({})", sys.os, sys.arch).ok();
    writeln!(report, "Apple Silicon: {}", sys.is_apple_silicon).ok();
    writeln!(report, "RAM: {} MB", sys.total_ram_bytes / 1024 / 1024).ok();
    writeln!(report, "Disk Free: {} MB", sys.available_disk_bytes / 1024 / 1024).ok();
    writeln!(report).ok();

    let s = state.settings.lock().unwrap_or_else(|e| e.into_inner());
    writeln!(report, "--- Settings ---").ok();
    writeln!(report, "STT Mode: {:?}", s.stt.mode).ok();
    writeln!(report, "Whisper Model: {}", s.stt.whisper_model.display_name()).ok();
    writeln!(report, "Language: {}", s.stt.language).ok();
    writeln!(report, "VAD: always enabled (model downloaded: {})", crate::transcribe::vad_model_path().exists()).ok();
    writeln!(report, "Polish: {}", s.polish.enabled).ok();
    writeln!(report, "Polish Mode: {:?}", s.polish.mode).ok();
    writeln!(report, "Auto Paste: {}", s.auto_paste).ok();
    drop(s);
    writeln!(report).ok();

    // Collect sumi.log.* files (daily rotation), sort by mtime, take 2 most recent
    // in chronological order so context is preserved across the midnight rollover.
    let log_dir = settings::logs_dir();
    let mut log_files: Vec<(std::time::SystemTime, std::path::PathBuf)> =
        std::fs::read_dir(&log_dir)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| {
                e.path()
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("sumi.log"))
                    .unwrap_or(false)
            })
            .filter_map(|e| {
                let path = e.path();
                let mtime = std::fs::metadata(&path).ok()?.modified().ok()?;
                Some((mtime, path))
            })
            .collect();

    // Sort ascending then take last 2 (most recent), reversing back to chronological order.
    log_files.sort_by_key(|(m, _)| *m);
    let recent: Vec<_> = log_files.into_iter().rev().take(2).rev().collect();

    let mut all_lines: Vec<String> = Vec::new();
    for (_, path) in &recent {
        if let Ok(content) = std::fs::read_to_string(path) {
            all_lines.extend(content.lines().map(|l| l.to_string()));
        }
    }

    if all_lines.is_empty() {
        writeln!(report, "--- App Log ---\n(no log file yet)").ok();
    } else {
        let start = all_lines.len().saturating_sub(200);
        writeln!(report, "--- App Log (last {} lines) ---", all_lines.len() - start).ok();
        for line in &all_lines[start..] {
            writeln!(report, "{}", line).ok();
        }
    }

    let downloads = dirs::download_dir()
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join("Downloads")
        });
    let _ = std::fs::create_dir_all(&downloads);
    let dest = downloads.join(format!("sumi-diagnostic-{}.txt", ts));
    std::fs::write(&dest, &report).map_err(|e| format!("Failed to write: {}", e))?;
    tracing::info!("Diagnostic report saved to {:?}", dest);
    Ok(dest.to_string_lossy().to_string())
}

// ── Qwen3-ASR commands ────────────────────────────────────────────────────────

#[tauri::command]
pub fn list_qwen3_asr_models(state: State<'_, AppState>) -> Vec<Qwen3AsrModelInfo> {
    let active = state.settings.lock()
        .map(|s| s.stt.qwen3_asr_model.clone())
        .unwrap_or_default();
    Qwen3AsrModel::all()
        .iter()
        .map(|m| Qwen3AsrModelInfo::from_model(m, &active))
        .collect()
}

#[tauri::command]
pub async fn switch_qwen3_asr_model(
    app: AppHandle,
    model: Qwen3AsrModel,
) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        // Atomically claim the switching slot — prevents concurrent calls from racing.
        if state.model_switching.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
            return Err("Model switch already in progress".to_string());
        }

        // Guard: refuse if recording or processing is already in progress.
        if state.is_recording.load(Ordering::SeqCst) || state.is_processing.load(Ordering::SeqCst) {
            state.model_switching.store(false, Ordering::SeqCst);
            return Err("Cannot switch model while recording or processing".to_string());
        }

        match state.settings.lock() {
            Ok(mut s) => {
                s.stt.qwen3_asr_model = model.clone();
                settings::save_settings_to_disk(&s);
            }
            Err(e) => {
                state.model_switching.store(false, Ordering::SeqCst);
                return Err(e.to_string());
            }
        }

        // Reset the readiness flag BEFORE invalidating the cache so the flag
        // always accurately reflects whether the cache holds a warm engine.
        // (flag=false → cache state unknown; flag=true → cache is valid)
        *state.qwen3_ready_mu.lock().unwrap_or_else(|e| e.into_inner()) = false;
        // Invalidate stale cache so next transcription loads the new model.
        qwen3::invalidate_qwen3_asr_cache(&state.qwen3_asr_ctx);

        // Pre-warm inline if already downloaded.
        if crate::stt::is_qwen3_asr_downloaded(&model) {
            // Show overlay on main thread (Cocoa APIs are thread-affine).
            {
                let app2 = app.clone();
                let _ = app.run_on_main_thread(move || {
                    if let Some(ov) = app2.get_webview_window("overlay") {
                        platform::show_overlay(&ov);
                        let _ = ov.emit("model-switching", serde_json::json!({"status": "start"}));
                    }
                });
            }

            // Pre-warm. Use catch_unwind so model_switching is always cleared on panic.
            let warm_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                qwen3::warm_qwen3_asr(&state.qwen3_asr_ctx, &model, Some((&state.qwen3_ready_cv, &state.qwen3_ready_mu)))
            }));

            // Emit "done" and hide overlay on the main thread, then clear the flag.
            {
                let app2 = app.clone();
                let _ = app.run_on_main_thread(move || {
                    if let Some(ov) = app2.get_webview_window("overlay") {
                        let _ = ov.emit("model-switching", serde_json::json!({"status": "done"}));
                        platform::hide_overlay(&ov);
                    }
                    app2.state::<AppState>().model_switching.store(false, Ordering::SeqCst);
                });
            }

            match warm_result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!("switch_qwen3_asr_model: pre-warm failed: {}", e),
                Err(_) => tracing::error!("switch_qwen3_asr_model: pre-warm panicked"),
            }
        } else {
            // Model not yet downloaded — clear the flag immediately (no overlay needed).
            state.model_switching.store(false, Ordering::SeqCst);
        }
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub fn download_qwen3_asr_model(
    app: AppHandle,
    state: State<'_, AppState>,
    model: Qwen3AsrModel,
) -> Result<(), String> {
    // Only one download at a time.
    if state.downloading.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
        return Err("A model download is already in progress".to_string());
    }

    let model_dir = crate::stt::qwen3_asr_model_dir(&model);
    let _ = std::fs::create_dir_all(&model_dir);

    let files: Vec<(&'static str, &'static str)> = model.download_files();
    let total_size = model.size_bytes();

    std::thread::spawn(move || {
        let state = app.state::<AppState>();

        let client = match reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(3600))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                let _ = app.emit("qwen3-asr-download-progress", serde_json::json!({
                    "status": "error", "message": e.to_string()
                }));
                state.downloading.store(false, Ordering::SeqCst);
                return;
            }
        };

        let mut downloaded_total: u64 = 0;
        let num_files = files.len();

        for (file_idx, (filename, hf_repo)) in files.iter().enumerate() {
            let url = format!("https://huggingface.co/{}/resolve/main/{}", hf_repo, filename);
            let dest = model_dir.join(filename);

            // Skip already-downloaded files.
            if dest.exists() {
                if let Ok(m) = std::fs::metadata(&dest) {
                    downloaded_total += m.len();
                    let _ = app.emit("qwen3-asr-download-progress", serde_json::json!({
                        "downloaded": downloaded_total,
                        "total": total_size,
                        "percent": (downloaded_total as f64 / total_size as f64 * 100.0) as u64,
                        "current_file": filename,
                        "file_index": file_idx,
                        "file_count": num_files,
                    }));
                    continue;
                }
            }

            let tmp = dest.with_extension("part");
            let resp = match client.get(&url).send() {
                Ok(r) => r,
                Err(e) => {
                    let _ = app.emit("qwen3-asr-download-progress", serde_json::json!({
                        "status": "error", "message": format!("Download failed for {}: {}", filename, e)
                    }));
                    state.downloading.store(false, Ordering::SeqCst);
                    return;
                }
            };

            if !resp.status().is_success() {
                let _ = app.emit("qwen3-asr-download-progress", serde_json::json!({
                    "status": "error",
                    "message": format!("HTTP {} for {}", resp.status(), filename)
                }));
                state.downloading.store(false, Ordering::SeqCst);
                return;
            }

            let mut file = match std::fs::File::create(&tmp) {
                Ok(f) => f,
                Err(e) => {
                    let _ = app.emit("qwen3-asr-download-progress", serde_json::json!({
                        "status": "error", "message": format!("Cannot create {}: {}", filename, e)
                    }));
                    state.downloading.store(false, Ordering::SeqCst);
                    return;
                }
            };

            let mut stream = resp;
            let mut file_downloaded: u64 = 0;
            let mut buf = vec![0u8; 65536];
            loop {
                use std::io::{Read, Write};
                match stream.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        if let Err(e) = file.write_all(&buf[..n]) {
                            let _ = app.emit("qwen3-asr-download-progress", serde_json::json!({
                                "status": "error", "message": format!("Write error for {}: {}", filename, e)
                            }));
                            state.downloading.store(false, Ordering::SeqCst);
                            return;
                        }
                        file_downloaded += n as u64;
                        downloaded_total += n as u64;
                        let _ = app.emit("qwen3-asr-download-progress", serde_json::json!({
                            "downloaded": downloaded_total,
                            "total": total_size,
                            "percent": (downloaded_total as f64 / total_size as f64 * 100.0) as u64,
                            "current_file": filename,
                            "file_index": file_idx,
                            "file_count": num_files,
                            "file_downloaded": file_downloaded,
                        }));
                    }
                    Err(e) => {
                        let _ = app.emit("qwen3-asr-download-progress", serde_json::json!({
                            "status": "error", "message": format!("Read error for {}: {}", filename, e)
                        }));
                        state.downloading.store(false, Ordering::SeqCst);
                        return;
                    }
                }
            }

            drop(file);
            if let Err(e) = std::fs::rename(&tmp, &dest) {
                let _ = app.emit("qwen3-asr-download-progress", serde_json::json!({
                    "status": "error", "message": format!("Rename failed for {}: {}", filename, e)
                }));
                state.downloading.store(false, Ordering::SeqCst);
                return;
            }
        }

        // All files downloaded — warm the engine.
        let active_model = state.settings.lock()
            .map(|s| s.stt.qwen3_asr_model.clone())
            .unwrap_or_default();
        if active_model == model {
            // Reset flag before warm (consistent with switch_qwen3_asr_model /
            // delete_qwen3_asr_model) so warm_qwen3_asr always transitions
            // flag false → true, making wait_engine_ready's flag-check reliable.
            *state.qwen3_ready_mu.lock().unwrap_or_else(|e| e.into_inner()) = false;
            if let Err(e) = qwen3::warm_qwen3_asr(&state.qwen3_asr_ctx, &model, Some((&state.qwen3_ready_cv, &state.qwen3_ready_mu))) {
                tracing::warn!("download_qwen3_asr_model: post-download warm failed: {}", e);
            }
        }

        let _ = app.emit("qwen3-asr-download-progress", serde_json::json!({
            "status": "complete",
            "downloaded": downloaded_total,
            "total": total_size,
            "percent": 100u64,
        }));

        state.downloading.store(false, Ordering::SeqCst);
    });

    Ok(())
}

// ── Model deletion ─────────────────────────────────────────────────────────

/// Guard helper: returns Err if recording, processing, downloading, or switching.
fn guard_model_op(state: &AppState) -> Result<(), String> {
    if state.is_recording.load(Ordering::SeqCst) || state.is_processing.load(Ordering::SeqCst) {
        return Err("Cannot delete model while recording or processing".to_string());
    }
    if state.downloading.load(Ordering::SeqCst) {
        return Err("Cannot delete model while a download is in progress".to_string());
    }
    if state.model_switching.load(Ordering::SeqCst) {
        return Err("Cannot delete model while switching models".to_string());
    }
    Ok(())
}

#[tauri::command]
pub fn delete_whisper_model(
    state: State<'_, AppState>,
    model: WhisperModel,
) -> Result<u64, String> {
    guard_model_op(&state)?;

    let path = settings::models_dir().join(model.filename());
    if !path.exists() {
        return Err("Model file not found".to_string());
    }

    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    // Invalidate whisper cache if this model is currently loaded.
    if let Ok(mut cache) = state.whisper_ctx.lock() {
        if let Some(ref c) = *cache {
            if c.loaded_path == path {
                *cache = None;
                tracing::info!("Whisper cache invalidated for deleted model");
            }
        }
    }

    std::fs::remove_file(&path).map_err(|e| format!("Failed to delete model: {}", e))?;
    tracing::info!("Deleted whisper model {:?}, freed {} bytes", model.display_name(), size);
    Ok(size)
}

#[tauri::command]
pub fn delete_polish_model(
    state: State<'_, AppState>,
    model: polisher::PolishModel,
) -> Result<u64, String> {
    guard_model_op(&state)?;

    let dir = settings::models_dir();
    let path = dir.join(model.filename());
    if !path.exists() {
        return Err("Model file not found".to_string());
    }

    let mut freed = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    // Invalidate LLM cache if this model is currently loaded.
    polisher::invalidate_cache(&state.llm_model);

    std::fs::remove_file(&path).map_err(|e| format!("Failed to delete model: {}", e))?;

    // Also remove companion tokenizer file if present.
    if let Some(tok) = model.tokenizer_filename() {
        let tok_path = dir.join(tok);
        if tok_path.exists() {
            freed += std::fs::metadata(&tok_path).map(|m| m.len()).unwrap_or(0);
            let _ = std::fs::remove_file(&tok_path);
        }
    }

    tracing::info!("Deleted polish model {:?}, freed {} bytes", model.display_name(), freed);
    Ok(freed)
}

#[tauri::command]
pub fn delete_qwen3_asr_model(
    state: State<'_, AppState>,
    model: Qwen3AsrModel,
) -> Result<u64, String> {
    guard_model_op(&state)?;

    let model_dir = crate::stt::qwen3_asr_model_dir(&model);
    if !model_dir.exists() {
        return Err("Model directory not found".to_string());
    }

    // Calculate total size before deletion.
    let freed: u64 = model
        .required_files()
        .iter()
        .filter_map(|f| std::fs::metadata(model_dir.join(f)).ok())
        .map(|m| m.len())
        .sum();

    // Reset flag before invalidating cache (same ordering as switch_qwen3_asr_model).
    *state.qwen3_ready_mu.lock().unwrap_or_else(|e| e.into_inner()) = false;
    // Invalidate Qwen3-ASR cache.
    qwen3::invalidate_qwen3_asr_cache(&state.qwen3_asr_ctx);
    // Wake any wait_engine_ready waiter so it re-evaluates the deadline
    // immediately rather than sleeping the full remaining timeout.
    // (flag=false → it will not return true, but avoids an unnecessary 8 s stall.)
    // Nudge any wait_engine_ready waiter so it skips the rest of its current
    // wait_timeout slice and re-evaluates the deadline immediately.
    // flag=false means it will loop back to sleep (or break if deadline passed),
    // not return true. This is called without holding qwen3_ready_mu, which is
    // valid per std::sync::Condvar: notify_all does not require the lock.
    state.qwen3_ready_cv.notify_all();

    std::fs::remove_dir_all(&model_dir)
        .map_err(|e| format!("Failed to delete model directory: {}", e))?;

    tracing::info!("Deleted Qwen3-ASR model {:?}, freed {} bytes", model.display_name(), freed);
    Ok(freed)
}

#[tauri::command]
pub fn delete_vad_model(state: State<'_, AppState>) -> Result<u64, String> {
    guard_model_op(&state)?;

    let path = crate::transcribe::vad_model_path();
    if !path.exists() {
        return Err("VAD model not found".to_string());
    }

    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    // Invalidate VAD cache.
    if let Ok(mut cache) = state.vad_ctx.lock() {
        *cache = None;
    }

    std::fs::remove_file(&path).map_err(|e| format!("Failed to delete VAD model: {}", e))?;
    tracing::info!("Deleted VAD model, freed {} bytes", size);
    Ok(size)
}

// ── Meeting Notes ──

#[tauri::command]
pub fn list_meeting_notes() -> Vec<meeting_notes::MeetingNote> {
    meeting_notes::list_notes(&settings::history_dir())
}

#[tauri::command]
pub fn get_meeting_note(id: String) -> Result<meeting_notes::MeetingNote, String> {
    meeting_notes::get_note(&settings::history_dir(), &id)
}

#[tauri::command]
pub fn rename_meeting_note(id: String, title: String) -> Result<(), String> {
    meeting_notes::rename_note(&settings::history_dir(), &id, &title)
}

#[tauri::command]
pub fn delete_meeting_note(
    state: State<'_, AppState>,
    id: String,
) -> Result<(), String> {
    let active_id = state
        .active_meeting_note_id
        .lock()
        .ok()
        .and_then(|n| n.clone());
    if active_id.as_deref() == Some(&id) {
        return Err("Cannot delete the currently-recording meeting note".to_string());
    }
    meeting_notes::delete_note(&settings::history_dir(), &id)
}

#[tauri::command]
pub fn delete_all_meeting_notes() -> Result<(), String> {
    meeting_notes::delete_all_notes(&settings::history_dir())
}

#[tauri::command]
pub fn get_active_meeting_note_id(state: State<'_, AppState>) -> Option<String> {
    state
        .active_meeting_note_id
        .lock()
        .ok()
        .and_then(|nid| nid.clone())
}

#[derive(Serialize)]
pub struct PolishedMeetingNote {
    pub title: String,
    pub summary: String,
}

#[tauri::command]
pub async fn polish_meeting_note(
    app: AppHandle,
    id: String,
) -> Result<PolishedMeetingNote, String> {
    let history_dir = settings::history_dir();
    let note = meeting_notes::get_note(&history_dir, &id)?;
    if note.transcript.is_empty() {
        return Err("Transcript is empty".to_string());
    }

    // Extract config and validate before spawning the blocking task.
    let (config, model_dir, stt_language) = {
        let state = app.state::<AppState>();
        let settings = state.settings.lock().map_err(|e| e.to_string())?;
        let mut config = settings.polish.clone();
        let stt_language = settings.stt.language.clone();
        drop(settings);

        if config.mode == polisher::PolishMode::Cloud {
            let key = get_cached_api_key(&state.api_key_cache, config.cloud.provider.as_key());
            if !key.is_empty() {
                config.cloud.api_key = key;
            }
        }

        let model_dir = settings::models_dir();
        if !polisher::is_polish_ready(&model_dir, &config) {
            return Err("LLM not configured".to_string());
        }
        (config, model_dir, stt_language)
    };

    let transcript = note.transcript.clone();
    let fallback_title = note.title.clone();

    // Run the heavy LLM inference on a blocking thread so the UI stays responsive.
    let app_clone = app.clone();
    let parsed = tauri::async_runtime::spawn_blocking(move || {
        let state = app_clone.state::<AppState>();

        let lang_name = language_display_name(&stt_language);
        let system_prompt = "You are a meeting notes assistant.";
        let user_text = format!(
            r#"<transcript>
{transcript}
</transcript>

Summarize the transcript above. Generate:
1. A concise title (max 60 chars)
2. A Markdown summary with sections: Key Points, Action Items, Decisions. Omit empty sections.

Return ONLY a JSON object: {{"title": "...", "summary": "..."}}
Write in {lang_name}."#
        );

        let result = polisher::polish_with_prompt(
            &state.llm_model,
            &model_dir,
            &config,
            system_prompt,
            &user_text,
            &state.http_client,
            Some(4096),
        )?;

        Ok::<_, String>(parse_polish_json(&result, &fallback_title))
    })
    .await
    .map_err(|e| format!("Polish task failed: {}", e))??;

    meeting_notes::save_summary(&history_dir, &id, &parsed.title, &parsed.summary)?;

    Ok(parsed)
}

fn language_display_name(bcp47: &str) -> &'static str {
    match bcp47 {
        "zh-TW" => "繁體中文 (Traditional Chinese)",
        "zh-CN" | "zh" => "简体中文 (Simplified Chinese)",
        "en" => "English",
        "ja" => "日本語 (Japanese)",
        "ko" => "한국어 (Korean)",
        "es" => "Español (Spanish)",
        "fr" => "Français (French)",
        "de" => "Deutsch (German)",
        "pt" => "Português (Portuguese)",
        "it" => "Italiano (Italian)",
        "ru" => "Русский (Russian)",
        "ar" => "العربية (Arabic)",
        "hi" => "हिन्दी (Hindi)",
        "th" => "ไทย (Thai)",
        "vi" => "Tiếng Việt (Vietnamese)",
        "id" => "Bahasa Indonesia (Indonesian)",
        "ms" => "Bahasa Melayu (Malay)",
        "nl" => "Nederlands (Dutch)",
        "pl" => "Polski (Polish)",
        "tr" => "Türkçe (Turkish)",
        "uk" => "Українська (Ukrainian)",
        "sv" => "Svenska (Swedish)",
        "da" => "Dansk (Danish)",
        "fi" => "Suomi (Finnish)",
        "no" => "Norsk (Norwegian)",
        "cs" => "Čeština (Czech)",
        "ro" => "Română (Romanian)",
        "hu" => "Magyar (Hungarian)",
        "el" => "Ελληνικά (Greek)",
        "he" => "עברית (Hebrew)",
        "auto" | "" => "the same language as the transcript",
        _ => "the same language as the transcript",
    }
}

fn parse_polish_json(raw: &str, fallback_title: &str) -> PolishedMeetingNote {
    // Strip markdown code fences if present
    let cleaned = raw
        .trim()
        .strip_prefix("```json")
        .or_else(|| raw.trim().strip_prefix("```"))
        .unwrap_or(raw.trim());
    let cleaned = cleaned
        .strip_suffix("```")
        .unwrap_or(cleaned)
        .trim();

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(cleaned) {
        let title = v
            .get("title")
            .and_then(|t| t.as_str())
            .unwrap_or(fallback_title)
            .to_string();
        let summary = v
            .get("summary")
            .and_then(|s| s.as_str())
            .unwrap_or(cleaned)
            .to_string();
        PolishedMeetingNote { title, summary }
    } else {
        // Fallback: use raw output as summary, keep existing title
        PolishedMeetingNote {
            title: fallback_title.to_string(),
            summary: raw.trim().to_string(),
        }
    }
}

// ── Audio Import ──

#[tauri::command]
pub async fn import_meeting_audio(
    app: AppHandle,
    file_path: String,
) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        crate::audio_import::run_import(&app, &file_path)
    })
    .await
    .map_err(|e| format!("Import thread panicked: {e}"))?
}

#[tauri::command]
pub fn cancel_import(state: State<'_, AppState>) {
    state
        .import_cancelled
        .store(true, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_polish_json_valid() {
        let raw = r#"{"title": "Meeting Title", "summary": "Key Points"}"#;
        let result = parse_polish_json(raw, "Fallback");
        assert_eq!(result.title, "Meeting Title");
        assert_eq!(result.summary, "Key Points");
    }

    #[test]
    fn parse_polish_json_code_fence() {
        let raw = "```json\n{\"title\": \"Fenced\", \"summary\": \"Content\"}\n```";
        let result = parse_polish_json(raw, "Fallback");
        assert_eq!(result.title, "Fenced");
        assert_eq!(result.summary, "Content");
    }

    #[test]
    fn parse_polish_json_bare_code_fence() {
        let raw = "```\n{\"title\": \"Bare\", \"summary\": \"Content\"}\n```";
        let result = parse_polish_json(raw, "Fallback");
        assert_eq!(result.title, "Bare");
        assert_eq!(result.summary, "Content");
    }

    #[test]
    fn parse_polish_json_invalid_falls_back() {
        let raw = "This is not JSON at all";
        let result = parse_polish_json(raw, "My Fallback Title");
        assert_eq!(result.title, "My Fallback Title");
        assert_eq!(result.summary, "This is not JSON at all");
    }

    #[test]
    fn parse_polish_json_truncated_falls_back() {
        let raw = "{\"title\": \"Truncated\", \"summary\": \"## Key Points";
        let result = parse_polish_json(raw, "Fallback");
        assert_eq!(result.title, "Fallback");
        assert_eq!(result.summary, raw.trim());
    }

    #[test]
    fn parse_polish_json_missing_title_uses_fallback() {
        let raw = r#"{"summary": "Only a summary"}"#;
        let result = parse_polish_json(raw, "Fallback");
        assert_eq!(result.title, "Fallback");
        assert_eq!(result.summary, "Only a summary");
    }

    #[test]
    fn parse_polish_json_missing_summary_uses_raw() {
        let raw = r#"{"title": "Only a title"}"#;
        let result = parse_polish_json(raw, "Fallback");
        assert_eq!(result.title, "Only a title");
        assert_eq!(result.summary, raw);
    }

    #[test]
    fn parse_polish_json_whitespace_trimmed() {
        let raw = "  \n  {\"title\": \"Trimmed\", \"summary\": \"Content\"}  \n  ";
        let result = parse_polish_json(raw, "Fallback");
        assert_eq!(result.title, "Trimmed");
        assert_eq!(result.summary, "Content");
    }
}