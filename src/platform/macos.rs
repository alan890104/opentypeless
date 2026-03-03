use std::ffi::c_void;

extern "C" {
    fn sel_registerName(name: *const u8) -> *mut c_void;
    fn objc_msgSend();
    fn objc_getClass(name: *const u8) -> *mut c_void;
    fn objc_allocateClassPair(
        superclass: *mut c_void,
        name: *const u8,
        extra_bytes: usize,
    ) -> *mut c_void;
    fn objc_registerClassPair(cls: *mut c_void);
    fn object_setClass(obj: *mut c_void, cls: *mut c_void) -> *mut c_void;
}

/// Hide the Dock icon by setting the activation policy to Accessory.
/// NSApplicationActivationPolicyAccessory = 1
pub unsafe fn set_accessory_policy() {
    let cls_name = b"NSApplication\0";
    let sel_shared = sel_registerName(b"sharedApplication\0".as_ptr());
    let send_shared: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());

    let class_sel = sel_registerName(b"class\0".as_ptr());
    let _ = class_sel;

    let ns_app_class = objc_getClass(cls_name.as_ptr());
    if ns_app_class.is_null() {
        return;
    }
    let ns_app = send_shared(ns_app_class, sel_shared);
    if ns_app.is_null() {
        return;
    }

    let sel_policy = sel_registerName(b"setActivationPolicy:\0".as_ptr());
    let send_policy: unsafe extern "C" fn(*mut c_void, *mut c_void, i64) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send_policy(ns_app, sel_policy, 1); // 1 = Accessory
}

/// Enable dragging the main window by its background.
///
/// With `titleBarStyle: "Overlay"` + `resizable: true`, macOS no longer
/// provides a native title-bar drag region because the web content covers
/// the full window.  Setting `isMovableByWindowBackground = YES` lets the
/// user drag the window by clicking anywhere that is not an interactive
/// control.
pub unsafe fn set_movable_by_background(ns_window: *mut c_void) {
    let sel = sel_registerName(b"setMovableByWindowBackground:\0".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, i8) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 1); // YES
}

/// Collection behavior flags for the overlay window.
const OVERLAY_BEHAVIOR: u64 = 1    // canJoinAllSpaces
                            | 8    // transient
                            | 64   // ignoresCycle
                            | 256; // fullScreenAuxiliary

/// Swizzle the Tauri NSWindow into an NSPanel subclass.
///
/// macOS fullscreen Spaces only allow **NSPanel** (not NSWindow) to
/// appear alongside the fullscreen app.  We create a one-off
/// runtime class that inherits from NSPanel and swap the window's
/// isa pointer so the window server treats it as a panel.
unsafe fn make_panel(ns_window: *mut c_void) {
    let panel_class_name = b"SumiOverlayPanel\0".as_ptr();
    let mut cls = objc_getClass(panel_class_name);
    if cls.is_null() {
        let ns_panel = objc_getClass(b"NSPanel\0".as_ptr());
        if ns_panel.is_null() {
            return;
        }
        cls = objc_allocateClassPair(ns_panel, panel_class_name, 0);
        if cls.is_null() {
            return;
        }
        objc_registerClassPair(cls);
    }
    object_setClass(ns_window, cls);

    // NSPanel-specific: don't become key unless user explicitly clicks
    let sel = sel_registerName(b"setBecomesKeyOnlyIfNeeded:\0".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, i8) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 1);

    // NSPanel-specific: treat as a floating panel
    let sel = sel_registerName(b"setFloatingPanel:\0".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, i8) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 1);

    // Add non-activating panel to style mask (bit 7 = 128)
    let sel_mask = sel_registerName(b"styleMask\0".as_ptr());
    let get_mask: unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64 =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    let mask = get_mask(ns_window, sel_mask);

    let sel_set = sel_registerName(b"setStyleMask:\0".as_ptr());
    let set_mask: unsafe extern "C" fn(*mut c_void, *mut c_void, u64) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    set_mask(ns_window, sel_set, mask | (1 << 7)); // NSWindowStyleMaskNonactivatingPanel
}

/// One-time setup: convert to NSPanel, floating level, stays visible
/// when app deactivates, joins all Spaces (including fullscreen).
pub unsafe fn setup_overlay(ns_window: *mut c_void) {
    // Convert NSWindow → NSPanel so it can appear in fullscreen Spaces
    make_panel(ns_window);

    // setLevel: kCGPopUpMenuWindowLevel (101) — above fullscreen windows
    let sel = sel_registerName(b"setLevel:\0".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, i64) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 101);

    // setHidesOnDeactivate: NO
    let sel = sel_registerName(b"setHidesOnDeactivate:\0".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, i8) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 0);

    // setCollectionBehavior
    let sel = sel_registerName(b"setCollectionBehavior:\0".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, u64) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, OVERLAY_BEHAVIOR);

    // Register with window server immediately (alpha=0 so invisible),
    // ensuring the window joins all Spaces from the start.
    let sel = sel_registerName(b"setAlphaValue:\0".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, f64) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 0.0);

    let sel = sel_registerName(b"setIgnoresMouseEvents:\0".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, i8) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 1);

    // Order front while invisible to register with all Spaces immediately
    let sel = sel_registerName(b"orderFrontRegardless\0".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel);

}

/// Show without activating the application.
pub unsafe fn show_no_activate(ns_window: *mut c_void) {
    // Accept mouse events
    let sel = sel_registerName(b"setIgnoresMouseEvents:\0".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, i8) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 0);

    // Make visible
    let sel = sel_registerName(b"setAlphaValue:\0".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, f64) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 1.0);

    // Bring to front without activating
    let sel = sel_registerName(b"orderFrontRegardless\0".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel);
}

/// Hide the overlay (alpha-based, stays in window server for all Spaces).
pub unsafe fn hide_window(ns_window: *mut c_void) {
    let sel = sel_registerName(b"setAlphaValue:\0".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, f64) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 0.0);

    let sel = sel_registerName(b"setIgnoresMouseEvents:\0".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, i8) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 1);
}

// ── CGEvent: keyboard simulation ────────────────────────────────────

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGEventSourceCreate(state_id: i32) -> *mut c_void;
    fn CGEventCreateKeyboardEvent(
        source: *mut c_void,
        virtual_key: u16,
        key_down: bool,
    ) -> *mut c_void;
    fn CGEventSetFlags(event: *mut c_void, flags: u64);
    fn CGEventPost(tap: u32, event: *mut c_void);
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(cf: *mut c_void);
}

/// Simulate Cmd+<key> via CGEvent.
unsafe fn simulate_cmd_key(virtual_key: u16) -> bool {
    const COMBINED_STATE: i32 = 0;
    const HID_EVENT_TAP: u32 = 0;
    const FLAG_CMD: u64 = 0x100000;

    let source = CGEventSourceCreate(COMBINED_STATE);
    if source.is_null() {
        return false;
    }

    let key_down = CGEventCreateKeyboardEvent(source, virtual_key, true);
    CGEventSetFlags(key_down, FLAG_CMD);
    CGEventPost(HID_EVENT_TAP, key_down);

    let key_up = CGEventCreateKeyboardEvent(source, virtual_key, false);
    CGEventSetFlags(key_up, FLAG_CMD);
    CGEventPost(HID_EVENT_TAP, key_up);

    CFRelease(key_down);
    CFRelease(key_up);
    CFRelease(source);

    true
}

/// Convert an NSString pointer to a Rust String.
pub unsafe fn nsstring_to_string(nsstr: *mut c_void) -> String {
    if nsstr.is_null() {
        return String::new();
    }

    let sel_utf8 = sel_registerName(b"UTF8String\0".as_ptr());
    let send_cstr: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *const i8 =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    let cstr_ptr = send_cstr(nsstr, sel_utf8);
    if cstr_ptr.is_null() {
        return String::new();
    }
    std::ffi::CStr::from_ptr(cstr_ptr)
        .to_str()
        .unwrap_or("")
        .to_string()
}

/// Simulate Cmd+V (paste).
pub unsafe fn simulate_cmd_v() -> bool { simulate_cmd_key(9) }
/// Simulate Cmd+C (copy).
pub unsafe fn simulate_cmd_c() -> bool { simulate_cmd_key(8) }
/// Simulate Cmd+Z (undo).
pub unsafe fn simulate_cmd_z() -> bool { simulate_cmd_key(6) }

// ── CoreAudio: Bluetooth input detection & device-change listener ────────────

/// Packed selector / scope / element address used by CoreAudio property queries.
#[repr(C)]
struct AudioObjectPropertyAddress {
    selector: u32,
    scope: u32,
    element: u32,
}

#[link(name = "CoreAudio", kind = "framework")]
extern "C" {
    fn AudioObjectGetPropertyDataSize(
        object_id: u32,
        address: *const AudioObjectPropertyAddress,
        qualifier_data_size: u32,
        qualifier_data: *const c_void,
        out_data_size: *mut u32,
    ) -> i32;
    fn AudioObjectGetPropertyData(
        object_id: u32,
        address: *const AudioObjectPropertyAddress,
        qualifier_data_size: u32,
        qualifier_data: *const c_void,
        io_data_size: *mut u32,
        out_data: *mut c_void,
    ) -> i32;
    fn AudioObjectAddPropertyListener(
        object_id: u32,
        address: *const AudioObjectPropertyAddress,
        listener: unsafe extern "C" fn(u32, u32, *const AudioObjectPropertyAddress, *mut c_void) -> i32,
        client_data: *mut c_void,
    ) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFStringGetCStringPtr(the_string: *mut c_void, encoding: u32) -> *const i8;
    fn CFStringGetCString(
        the_string: *mut c_void,
        buffer: *mut i8,
        buffer_size: i64,
        encoding: u32,
    ) -> bool;
}

// CoreAudio system object and property selectors (fourCC literals)
const K_AUDIO_OBJECT_SYSTEM: u32        = 1;
const K_AUDIO_PROP_DEVICES: u32         = 0x64657623; // 'dev#'
const K_AUDIO_PROP_DEFAULT_INPUT: u32   = 0x64496e20; // 'dIn '
const K_AUDIO_PROP_TRANSPORT_TYPE: u32  = 0x74726e73; // 'trns'
const K_AUDIO_PROP_STREAMS: u32         = 0x73746d23; // 'stm#'
const K_AUDIO_PROP_NAME: u32            = 0x6c6e616d; // 'lnam'
const K_SCOPE_GLOBAL: u32               = 0x676c6f62; // 'glob'
const K_SCOPE_INPUT: u32                = 0x696e7074; // 'inpt'
const K_ELEMENT_MAIN: u32              = 0;
const K_TRANSPORT_BLUETOOTH: u32        = 0x626c7565; // 'blue'
const K_TRANSPORT_BLUETOOTH_LE: u32     = 0x626c6574; // 'blet'
const K_TRANSPORT_BUILT_IN: u32         = 0x626c746e; // 'bltn'
const K_CF_STRING_UTF8: u32             = 0x08000100;

/// Convert a CFStringRef to a Rust `String`.
unsafe fn cfstring_to_string(cf_str: *mut c_void) -> String {
    if cf_str.is_null() { return String::new(); }
    // Fast path: zero-copy C-string pointer (only works for ASCII-backed strings)
    let ptr = CFStringGetCStringPtr(cf_str, K_CF_STRING_UTF8);
    if !ptr.is_null() {
        return std::ffi::CStr::from_ptr(ptr).to_str().unwrap_or("").to_string();
    }
    // Slow path: copy into a stack buffer
    let mut buf = [0i8; 512];
    if CFStringGetCString(cf_str, buf.as_mut_ptr(), buf.len() as i64, K_CF_STRING_UTF8) {
        return std::ffi::CStr::from_ptr(buf.as_ptr()).to_str().unwrap_or("").to_string();
    }
    String::new()
}

/// Returns `true` if the system default audio input device is Bluetooth (classic or LE).
pub fn is_default_input_bluetooth() -> bool {
    unsafe {
        // Get default input device ID
        let addr = AudioObjectPropertyAddress {
            selector: K_AUDIO_PROP_DEFAULT_INPUT,
            scope: K_SCOPE_GLOBAL,
            element: K_ELEMENT_MAIN,
        };
        let mut device_id: u32 = 0;
        let mut size: u32 = std::mem::size_of::<u32>() as u32;
        let status = AudioObjectGetPropertyData(
            K_AUDIO_OBJECT_SYSTEM, &addr,
            0, std::ptr::null(),
            &mut size, &mut device_id as *mut u32 as *mut c_void,
        );
        if status != 0 || device_id == 0 { return false; }

        // Get transport type of that device
        let addr_t = AudioObjectPropertyAddress {
            selector: K_AUDIO_PROP_TRANSPORT_TYPE,
            scope: K_SCOPE_GLOBAL,
            element: K_ELEMENT_MAIN,
        };
        let mut transport: u32 = 0;
        let mut size2: u32 = std::mem::size_of::<u32>() as u32;
        let status2 = AudioObjectGetPropertyData(
            device_id, &addr_t,
            0, std::ptr::null(),
            &mut size2, &mut transport as *mut u32 as *mut c_void,
        );
        if status2 != 0 { return false; }
        transport == K_TRANSPORT_BLUETOOTH || transport == K_TRANSPORT_BLUETOOTH_LE
    }
}

/// Find the name of the first built-in audio input device (transport type `'bltn'`
/// with at least one input stream). Returns `None` if none exists.
pub fn get_builtin_input_device_name() -> Option<String> {
    unsafe {
        // Enumerate all device IDs
        let addr_devs = AudioObjectPropertyAddress {
            selector: K_AUDIO_PROP_DEVICES,
            scope: K_SCOPE_GLOBAL,
            element: K_ELEMENT_MAIN,
        };
        let mut data_size: u32 = 0;
        let s = AudioObjectGetPropertyDataSize(
            K_AUDIO_OBJECT_SYSTEM, &addr_devs,
            0, std::ptr::null(), &mut data_size,
        );
        if s != 0 || data_size == 0 { return None; }

        let count = data_size as usize / std::mem::size_of::<u32>();
        let mut ids: Vec<u32> = vec![0u32; count];
        let mut actual = data_size;
        let s2 = AudioObjectGetPropertyData(
            K_AUDIO_OBJECT_SYSTEM, &addr_devs,
            0, std::ptr::null(),
            &mut actual, ids.as_mut_ptr() as *mut c_void,
        );
        if s2 != 0 { return None; }

        for &dev_id in &ids {
            // Must be built-in transport
            let addr_t = AudioObjectPropertyAddress {
                selector: K_AUDIO_PROP_TRANSPORT_TYPE,
                scope: K_SCOPE_GLOBAL,
                element: K_ELEMENT_MAIN,
            };
            let mut transport: u32 = 0;
            let mut ts = std::mem::size_of::<u32>() as u32;
            let r = AudioObjectGetPropertyData(
                dev_id, &addr_t,
                0, std::ptr::null(),
                &mut ts, &mut transport as *mut u32 as *mut c_void,
            );
            if r != 0 || transport != K_TRANSPORT_BUILT_IN { continue; }

            // Must have at least one input stream
            let addr_streams = AudioObjectPropertyAddress {
                selector: K_AUDIO_PROP_STREAMS,
                scope: K_SCOPE_INPUT,
                element: K_ELEMENT_MAIN,
            };
            let mut stream_size: u32 = 0;
            let rs = AudioObjectGetPropertyDataSize(
                dev_id, &addr_streams,
                0, std::ptr::null(), &mut stream_size,
            );
            if rs != 0 || stream_size == 0 { continue; }

            // Get device name (returned as CFStringRef)
            let addr_name = AudioObjectPropertyAddress {
                selector: K_AUDIO_PROP_NAME,
                scope: K_SCOPE_GLOBAL,
                element: K_ELEMENT_MAIN,
            };
            let mut cf_str: *mut c_void = std::ptr::null_mut();
            let mut name_size = std::mem::size_of::<*mut c_void>() as u32;
            let rn = AudioObjectGetPropertyData(
                dev_id, &addr_name,
                0, std::ptr::null(),
                &mut name_size, &mut cf_str as *mut *mut c_void as *mut c_void,
            );
            if rn != 0 || cf_str.is_null() { continue; }

            let name = cfstring_to_string(cf_str);
            CFRelease(cf_str);
            if !name.is_empty() { return Some(name); }
        }
        None
    }
}

/// Static C-ABI callback invoked by CoreAudio on an arbitrary thread when
/// the system default input device changes.
unsafe extern "C" fn on_default_input_changed(
    _object_id: u32,
    _num_addresses: u32,
    _addresses: *const AudioObjectPropertyAddress,
    client_data: *mut c_void,
) -> i32 {
    if client_data.is_null() { return 0; }
    // Reconstruct a shared reference to the Box<dyn Fn() + Send> without taking ownership.
    // The pointer was leaked intentionally and lives for the process lifetime.
    let cb = &*(client_data as *const Box<dyn Fn() + Send>);
    cb();
    0
}

/// Register a permanent listener that fires whenever the system default audio
/// input device changes. The `callback` is called on a CoreAudio HAL thread
/// and must be `Send + 'static`.
pub fn add_default_input_listener(callback: impl Fn() + Send + 'static) {
    // Double-box so we get a thin pointer to a fat-pointer trait object.
    // `Box::into_raw` leaks intentionally — the listener is permanent.
    let boxed: Box<dyn Fn() + Send> = Box::new(callback);
    let raw = Box::into_raw(Box::new(boxed)) as *mut c_void;

    let addr = AudioObjectPropertyAddress {
        selector: K_AUDIO_PROP_DEFAULT_INPUT,
        scope: K_SCOPE_GLOBAL,
        element: K_ELEMENT_MAIN,
    };
    unsafe {
        AudioObjectAddPropertyListener(
            K_AUDIO_OBJECT_SYSTEM, &addr,
            on_default_input_changed, raw,
        );
    }
}

/// Returns the NSPasteboard changeCount, which increments each time the clipboard is written.
/// Used to detect whether a Cmd+C actually updated the clipboard (avoids false negatives
/// when the selected text is identical to the previously saved clipboard content).
pub fn clipboard_change_count() -> Option<u32> {
    unsafe {
        let cls = objc_getClass(b"NSPasteboard\0".as_ptr());
        if cls.is_null() { return None; }

        let sel_general = sel_registerName(b"generalPasteboard\0".as_ptr());
        type MsgSendPb = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let pb = std::mem::transmute::<_, MsgSendPb>(objc_msgSend as *const ())(cls, sel_general);
        if pb.is_null() { return None; }

        let sel_count = sel_registerName(b"changeCount\0".as_ptr());
        type MsgSendCount = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i64;
        let count = std::mem::transmute::<_, MsgSendCount>(objc_msgSend as *const ())(pb, sel_count);
        Some(count as u32)
    }
}
