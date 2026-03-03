use windows::Win32::System::DataExchange::GetClipboardSequenceNumber;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP, VIRTUAL_KEY,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetWindowLongPtrW, SetWindowLongPtrW, SetWindowPos, ShowWindow, GWL_EXSTYLE, HWND_TOPMOST,
    SWP_NOMOVE, SWP_NOSIZE, SW_HIDE, SW_SHOWNOACTIVATE,
    WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW,
};
use windows::Win32::Foundation::HWND;

// Virtual-Key codes
const VK_CONTROL: u16 = 0x11;
const VK_V: u16 = 0x56;
const VK_C: u16 = 0x43;
const VK_Z: u16 = 0x5A;

/// Set app accessory mode — no-op on Windows (no Dock equivalent).
pub fn set_accessory_policy() {}

/// Configure a window as a non-activating, always-on-top overlay.
pub unsafe fn setup_overlay(hwnd: *mut std::ffi::c_void) {
    let hwnd = HWND(hwnd);
    // OR in the required flags instead of replacing all extended styles.
    // tao sets up DWM blur-behind during window creation for `transparent: true`
    // windows. A full EXSTYLE replacement triggers WM_STYLECHANGED, which causes
    // DWM to re-evaluate compositing and can reset the blur-behind transparency
    // state, producing a visible opaque rectangle behind the capsule on Windows.
    let current = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
    let ex_style = current | (WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW).0 as isize;
    SetWindowLongPtrW(hwnd, GWL_EXSTYLE, ex_style);
    let _ = SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
}

/// Show the overlay without activating it.
pub unsafe fn show_no_activate(hwnd: *mut std::ffi::c_void) {
    let _ = ShowWindow(HWND(hwnd), SW_SHOWNOACTIVATE);
}

/// Hide the overlay.
pub unsafe fn hide_window(hwnd: *mut std::ffi::c_void) {
    let _ = ShowWindow(HWND(hwnd), SW_HIDE);
}

/// Simulate Ctrl+V (paste) via SendInput.
pub unsafe fn simulate_paste() -> bool {
    send_key_combo(VK_CONTROL, VK_V)
}

/// Simulate Ctrl+C (copy) via SendInput.
pub unsafe fn simulate_copy() -> bool {
    send_key_combo(VK_CONTROL, VK_C)
}

/// Simulate Ctrl+Z (undo) via SendInput.
pub unsafe fn simulate_undo() -> bool {
    send_key_combo(VK_CONTROL, VK_Z)
}

/// Returns the clipboard sequence number, which increments each time the clipboard is written.
/// Used to detect whether a Ctrl+C actually updated the clipboard.
pub fn clipboard_change_count() -> Option<u32> {
    Some(unsafe { GetClipboardSequenceNumber() })
}

/// Send a modifier+key combo via SendInput (4 events: mod↓ key↓ key↑ mod↑).
unsafe fn send_key_combo(modifier: u16, key: u16) -> bool {
    let inputs = [
        make_key_input(modifier, false),
        make_key_input(key, false),
        make_key_input(key, true),
        make_key_input(modifier, true),
    ];
    SendInput(&inputs, std::mem::size_of::<INPUT>() as i32) == 4
}

fn make_key_input(vk: u16, key_up: bool) -> INPUT {
    let flags = if key_up { KEYEVENTF_KEYUP } else { Default::default() };
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(vk),
                wScan: 0,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}
