use std::ffi::{c_char, c_void};

extern "C" {
    fn sel_registerName(name: *const c_char) -> *mut c_void;
    fn objc_msgSend();
    fn objc_getClass(name: *const c_char) -> *mut c_void;
    fn objc_allocateClassPair(
        superclass: *mut c_void,
        name: *const c_char,
        extra_bytes: usize,
    ) -> *mut c_void;
    fn objc_registerClassPair(cls: *mut c_void);
    fn object_setClass(obj: *mut c_void, cls: *mut c_void) -> *mut c_void;
}

/// Hide the Dock icon by setting the activation policy to Accessory.
/// NSApplicationActivationPolicyAccessory = 1
///
/// # Safety
/// Calls ObjC runtime through raw pointers. Must be called from the main thread.
pub unsafe fn set_accessory_policy() {
    let sel_shared = sel_registerName(c"sharedApplication".as_ptr());
    let send_shared: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());

    let ns_app_class = objc_getClass(c"NSApplication".as_ptr());
    if ns_app_class.is_null() {
        return;
    }
    let ns_app = send_shared(ns_app_class, sel_shared);
    if ns_app.is_null() {
        return;
    }

    let sel_policy = sel_registerName(c"setActivationPolicy:".as_ptr());
    let send_policy: unsafe extern "C" fn(*mut c_void, *mut c_void, i64) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send_policy(ns_app, sel_policy, 1); // 1 = Accessory
}

#[cfg(target_arch = "x86_64")]
extern "C" {
    fn objc_msgSend_stret();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CGRect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

/// Read an NSScreen's `frame` (CGRect) portably across ARM64 / x86-64.
unsafe fn get_screen_frame(screen: *mut c_void) -> CGRect {
    let sel = sel_registerName(c"frame".as_ptr());
    #[cfg(target_arch = "aarch64")]
    {
        let f: unsafe extern "C" fn(*mut c_void, *mut c_void) -> CGRect =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f(screen, sel)
    }
    #[cfg(target_arch = "x86_64")]
    {
        let f: unsafe extern "C" fn(*mut CGRect, *mut c_void, *mut c_void) =
            std::mem::transmute(objc_msgSend_stret as unsafe extern "C" fn());
        let mut r = CGRect { x: 0.0, y: 0.0, w: 0.0, h: 0.0 };
        f(&mut r, screen, sel);
        r
    }
}

/// Read an NSScreen's `backingScaleFactor`.
unsafe fn get_screen_scale(screen: *mut c_void) -> f64 {
    let sel = sel_registerName(c"backingScaleFactor".as_ptr());
    let f: unsafe extern "C" fn(*mut c_void, *mut c_void) -> f64 =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    f(screen, sel)
}

/// Returns `(tauri_x, tauri_y, width, height, scale)` of `NSScreen.mainScreen`
/// (the screen that currently has keyboard focus) in Tauri logical coordinates
/// (y=0 at top of primary screen, y increases downward).
///
/// Returns `None` if `NSScreen.mainScreen` or the primary screen is unavailable.
pub fn focused_screen_logical_frame() -> Option<(f64, f64, f64, f64, f64)> {
    unsafe {
        let cls = objc_getClass(c"NSScreen".as_ptr());
        if cls.is_null() {
            return None;
        }
        type Send1 = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let send: Send1 = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());

        // [NSScreen mainScreen] — the screen with the key window (keyboard focus)
        let main_screen = send(cls, sel_registerName(c"mainScreen".as_ptr()));
        if main_screen.is_null() {
            return None;
        }
        let main_frame = get_screen_frame(main_screen);
        let scale = get_screen_scale(main_screen);

        // [NSScreen screens].firstObject — the primary screen (has menu bar)
        // Its frame is always {0, 0, w, h}; we need its height to flip the Y axis.
        let screens = send(cls, sel_registerName(c"screens".as_ptr()));
        if screens.is_null() {
            return None;
        }
        type Send2 = unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> *mut c_void;
        let send2: Send2 = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let primary = send2(screens, sel_registerName(c"objectAtIndex:".as_ptr()), 0);
        if primary.is_null() {
            return None;
        }
        let primary_h = get_screen_frame(primary).h;

        // macOS: origin bottom-left of primary screen, Y increases upward.
        // Tauri: origin top-left of primary screen, Y increases downward.
        // tauri_y = primary_h - macos_y - screen_h
        let tauri_x = main_frame.x;
        let tauri_y = primary_h - main_frame.y - main_frame.h;
        Some((tauri_x, tauri_y, main_frame.w, main_frame.h, scale))
    }
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
    let mut cls = objc_getClass(c"SumiOverlayPanel".as_ptr());
    if cls.is_null() {
        let ns_panel = objc_getClass(c"NSPanel".as_ptr());
        if ns_panel.is_null() {
            return;
        }
        cls = objc_allocateClassPair(ns_panel, c"SumiOverlayPanel".as_ptr(), 0);
        if cls.is_null() {
            return;
        }
        objc_registerClassPair(cls);
    }
    object_setClass(ns_window, cls);

    // NSPanel-specific: don't become key unless user explicitly clicks
    let sel = sel_registerName(c"setBecomesKeyOnlyIfNeeded:".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, i8) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 1);

    // NSPanel-specific: treat as a floating panel
    let sel = sel_registerName(c"setFloatingPanel:".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, i8) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 1);

    // Add non-activating panel to style mask (bit 7 = 128)
    let sel_mask = sel_registerName(c"styleMask".as_ptr());
    let get_mask: unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64 =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    let mask = get_mask(ns_window, sel_mask);

    let sel_set = sel_registerName(c"setStyleMask:".as_ptr());
    let set_mask: unsafe extern "C" fn(*mut c_void, *mut c_void, u64) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    set_mask(ns_window, sel_set, mask | (1 << 7)); // NSWindowStyleMaskNonactivatingPanel
}

/// One-time setup: convert to NSPanel, floating level, stays visible
/// when app deactivates, joins all Spaces (including fullscreen).
///
/// # Safety
/// `ns_window` must be a valid, non-null NSWindow pointer. Must be called
/// on the main thread during window setup.
pub unsafe fn setup_overlay(ns_window: *mut c_void) {
    // Convert NSWindow → NSPanel so it can appear in fullscreen Spaces
    make_panel(ns_window);

    // setLevel: kCGPopUpMenuWindowLevel (101) — above fullscreen windows
    let sel = sel_registerName(c"setLevel:".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, i64) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 101);

    // setHidesOnDeactivate: NO
    let sel = sel_registerName(c"setHidesOnDeactivate:".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, i8) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 0);

    // setCollectionBehavior
    let sel = sel_registerName(c"setCollectionBehavior:".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, u64) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, OVERLAY_BEHAVIOR);

    // Register with window server immediately (alpha=0 so invisible),
    // ensuring the window joins all Spaces from the start.
    let sel = sel_registerName(c"setAlphaValue:".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, f64) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 0.0);

    let sel = sel_registerName(c"setIgnoresMouseEvents:".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, i8) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 1);

    // Order front while invisible to register with all Spaces immediately
    let sel = sel_registerName(c"orderFrontRegardless".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel);
}

/// Show without activating the application.
///
/// # Safety
/// `ns_window` must be a valid, non-null NSWindow pointer.
pub unsafe fn show_no_activate(ns_window: *mut c_void) {
    // Accept mouse events
    let sel = sel_registerName(c"setIgnoresMouseEvents:".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, i8) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 0);

    // Make visible
    let sel = sel_registerName(c"setAlphaValue:".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, f64) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 1.0);

    // Bring to front without activating
    let sel = sel_registerName(c"orderFrontRegardless".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel);
}

/// Hide the overlay (alpha-based, stays in window server for all Spaces).
///
/// # Safety
/// `ns_window` must be a valid, non-null NSWindow pointer.
pub unsafe fn hide_window(ns_window: *mut c_void) {
    let sel = sel_registerName(c"setAlphaValue:".as_ptr());
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, f64) =
        std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
    send(ns_window, sel, 0.0);

    let sel = sel_registerName(c"setIgnoresMouseEvents:".as_ptr());
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
///
/// # Safety
/// `nsstr` must be a valid NSString pointer or null. A null pointer returns an
/// empty string.
pub unsafe fn nsstring_to_string(nsstr: *mut c_void) -> String {
    if nsstr.is_null() {
        return String::new();
    }

    let sel_utf8 = sel_registerName(c"UTF8String".as_ptr());
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

/// Returns `true` if any media is **actively playing** system-wide.
///
/// Uses `MRMediaRemoteGetNowPlayingApplicationIsPlaying` from the MediaRemote
/// private framework — the same API the Control Centre uses.  Works for every
/// player (Apple Music, Spotify, YouTube Music in Chrome, etc.).
///
/// Falls back to `false` (safe: don't send key) if the framework is missing
/// or the call times out after 300 ms.
pub fn is_now_playing() -> bool {
    use std::sync::OnceLock;

    // Symbols from libdispatch (part of libSystem, always available on macOS).
    extern "C" {
        fn dispatch_semaphore_create(value: i64) -> *mut c_void;
        fn dispatch_semaphore_signal(dsema: *mut c_void) -> i64;
        fn dispatch_semaphore_wait(dsema: *mut c_void, timeout: u64) -> i64;
        fn dispatch_release(obj: *mut c_void);
        fn dispatch_get_global_queue(identifier: i64, flags: usize) -> *mut c_void;
        fn dispatch_time(when: u64, delta: i64) -> u64;
        fn dlopen(filename: *const c_char, flag: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        // Objective-C block ISA for stack-allocated blocks.
        static _NSConcreteStackBlock: u8;
    }

    // Objective-C block layout for `^(BOOL playing)`.
    #[repr(C)]
    struct BlockDesc { reserved: u64, size: u64 }
    #[repr(C)]
    struct Block {
        isa: *const u8,
        flags: i32,
        reserved: i32,
        invoke: unsafe extern "C" fn(*mut Block, u8),
        descriptor: *const BlockDesc,
        // Captured variables
        out: *mut u8,
        sema: *mut c_void,
    }

    static DESC: BlockDesc = BlockDesc {
        reserved: 0,
        size: std::mem::size_of::<Block>() as u64,
    };

    unsafe extern "C" fn invoke(b: *mut Block, playing: u8) {
        *(*b).out = playing;
        dispatch_semaphore_signal((*b).sema);
    }

    // Cache the resolved fn pointer so dlopen/dlsym run only once per process.
    // Storing as usize to satisfy OnceLock's Send+Sync requirement.
    static GET_PLAYING_FN: OnceLock<Option<usize>> = OnceLock::new();

    let fn_ptr = *GET_PLAYING_FN.get_or_init(|| unsafe {
        let handle = dlopen(
            c"/System/Library/PrivateFrameworks/MediaRemote.framework/MediaRemote".as_ptr(),
            1, // RTLD_LAZY
        );
        if handle.is_null() { return None; }
        let sym = dlsym(handle, c"MRMediaRemoteGetNowPlayingApplicationIsPlaying".as_ptr());
        if sym.is_null() { return None; }
        // Intentionally skip dlclose: the framework stays loaded for the process
        // lifetime; dlclose would only decrement the refcount, never unload it.
        Some(sym as usize)
    });

    let fn_ptr = match fn_ptr {
        Some(p) => p,
        None => return false,
    };

    unsafe {
        type GetPlayingFn = unsafe extern "C" fn(*mut c_void, *mut Block);
        let get_playing: GetPlayingFn = std::mem::transmute(fn_ptr);

        let sema = dispatch_semaphore_create(0);

        // Heap-allocate `result` so the captured pointer stays valid even if
        // the 300 ms deadline fires before the GCD callback does.
        // `MRMediaRemoteGetNowPlayingApplicationIsPlaying` copies the block to
        // the heap (standard GCD contract), but `out` inside the copied block
        // still points to the original allocation — it must be heap-stable.
        let result: *mut u8 = Box::into_raw(Box::new(0u8));

        let mut block = Block {
            isa: &_NSConcreteStackBlock,
            flags: 0,
            reserved: 0,
            invoke,
            descriptor: &DESC,
            out: result,
            sema,
        };

        let queue = dispatch_get_global_queue(0, 0);
        get_playing(queue, &mut block);

        // Wait up to 100 ms; MediaRemote responds in single-digit ms when a
        // player is active — 100 ms is only reached when the framework is
        // frozen, which is extremely rare. Keeping this short minimises the
        // delay between overlay-show and do_start_recording on the hotkey thread.
        let deadline = dispatch_time(0 /* DISPATCH_TIME_NOW */, 100_000_000);
        let timed_out = dispatch_semaphore_wait(sema, deadline) != 0;

        if timed_out {
            // The callback may still fire later and signal sema — do NOT call
            // dispatch_release here or that signal races a freed object.
            // Leak both `result` and `sema`; this path is hit only when the
            // framework is frozen, which is extremely rare in practice.
            false
        } else {
            let val = *result;
            drop(Box::from_raw(result));
            dispatch_release(sema);
            val != 0
        }
    }
}

/// Simulate the Play/Pause media key (NX_KEYTYPE_PLAY = 16).
///
/// Works with all players that honour system media keys: Apple Music, Spotify,
/// YouTube Music in Chrome/Safari, etc.  Sends key-down + key-up via
/// `NSEvent otherEventWithType:NSEventTypeSystemDefined`.
pub fn simulate_media_play_pause() {
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct NSPoint { x: f64, y: f64 }

    const NX_KEYTYPE_PLAY: i64 = 16;
    const NS_EVENT_TYPE_SYSTEM_DEFINED: u64 = 14;
    const NX_SUBTYPE_AUX_CONTROL_BUTTONS: i16 = 8;

    // NSEvent +otherEventWithType:location:modifierFlags:timestamp:
    //         windowNumber:context:subtype:data1:data2:
    type MakeEventFn = unsafe extern "C" fn(
        *mut c_void,  // class
        *mut c_void,  // selector
        u64,          // NSEventType
        NSPoint,      // location  (HFA on ARM64: passed in d0/d1)
        u64,          // modifierFlags
        f64,          // timestamp
        i64,          // windowNumber
        *mut c_void,  // context (NSGraphicsContext*, nil)
        i16,          // subtype
        i64,          // data1
        i64,          // data2
    ) -> *mut c_void;

    type GetCgEventFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;

    unsafe {
        let ns_event_class = objc_getClass(c"NSEvent".as_ptr());
        if ns_event_class.is_null() {
            tracing::warn!("simulate_media_play_pause: NSEvent class not found");
            return;
        }

        let sel_make = sel_registerName(
            c"otherEventWithType:location:modifierFlags:timestamp:windowNumber:context:subtype:data1:data2:".as_ptr(),
        );
        let sel_cg = sel_registerName(c"CGEvent".as_ptr());

        let make_event: MakeEventFn = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let get_cg: GetCgEventFn = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());

        let zero = NSPoint { x: 0.0, y: 0.0 };

        // key-down  (NX_KEYDOWN = 0x0a → flags byte in data1)
        let ev_down = make_event(
            ns_event_class, sel_make,
            NS_EVENT_TYPE_SYSTEM_DEFINED, zero,
            0, 0.0, 0, std::ptr::null_mut(),
            NX_SUBTYPE_AUX_CONTROL_BUTTONS,
            (NX_KEYTYPE_PLAY << 16) | 0x0a00,
            -1,
        );
        if !ev_down.is_null() {
            let cg = get_cg(ev_down, sel_cg);
            if !cg.is_null() { CGEventPost(0 /* kCGHIDEventTap */, cg); }
        }

        // key-up  (NX_KEYUP = 0x0b, released-bit 0x01)
        let ev_up = make_event(
            ns_event_class, sel_make,
            NS_EVENT_TYPE_SYSTEM_DEFINED, zero,
            0, 0.0, 0, std::ptr::null_mut(),
            NX_SUBTYPE_AUX_CONTROL_BUTTONS,
            (NX_KEYTYPE_PLAY << 16) | 0x0b01,
            -1,
        );
        if !ev_up.is_null() {
            let cg = get_cg(ev_up, sel_cg);
            if !cg.is_null() { CGEventPost(0, cg); }
        }
    }
}

/// Simulate Cmd+V (paste).
///
/// # Safety
/// Posts CGEvents; must be called from a context where CGEvent posting is allowed.
pub unsafe fn simulate_cmd_v() -> bool { simulate_cmd_key(9) }
/// Simulate Cmd+C (copy).
///
/// # Safety
/// Posts CGEvents; must be called from a context where CGEvent posting is allowed.
pub unsafe fn simulate_cmd_c() -> bool { simulate_cmd_key(8) }
/// Simulate Cmd+Z (undo).
///
/// # Safety
/// Posts CGEvents; must be called from a context where CGEvent posting is allowed.
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
    ) -> u8; // Boolean = unsigned char, not C99 _Bool
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
const K_TRANSPORT_VIRTUAL: u32          = 0x7672746c; // 'vrtl'
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
    if CFStringGetCString(cf_str, buf.as_mut_ptr(), buf.len() as i64, K_CF_STRING_UTF8) != 0 {
        return std::ffi::CStr::from_ptr(buf.as_ptr()).to_str().unwrap_or("").to_string();
    }
    tracing::warn!("cfstring_to_string: name did not fit in 512-byte buffer, skipping device");
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

        // Method 1: transport type (works on macOS ≤ 15).
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
        if status2 == 0 {
            return transport == K_TRANSPORT_BLUETOOTH || transport == K_TRANSPORT_BLUETOOTH_LE;
        }

        // Method 2: UID heuristic (macOS 16+ where transport type is unavailable).
        // Bluetooth devices have UIDs like "XX-XX-XX-XX-XX-XX:input" where the
        // prefix is a MAC address (6 colon-separated hex pairs when decoded).
        let uid = get_cfstring_property(device_id, K_AUDIO_PROP_UID);
        if let Some(prefix) = uid.split(':').next() {
            // MAC address format: 12 hex chars with dashes, e.g. "50-F3-51-E6-A1-09"
            let parts: Vec<&str> = prefix.split('-').collect();
            if parts.len() == 6 && parts.iter().all(|p| p.len() == 2 && p.chars().all(|c| c.is_ascii_hexdigit())) {
                return true;
            }
        }
        false
    }
}

/// Find the name of the first built-in audio input device (transport type `'bltn'`
/// with at least one input stream). Returns `None` if none exists.
pub fn get_builtin_input_device_name() -> Option<String> {
    unsafe {
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
            if !is_builtin_device(dev_id) { continue; }

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

            let name = get_cfstring_property(dev_id, K_AUDIO_PROP_NAME);
            if !name.is_empty() { return Some(name); }
        }
        None
    }
}

/// Check whether a CoreAudio device is built-in.
unsafe fn is_builtin_device(dev_id: u32) -> bool {
    // Method 1: transport type (works on macOS ≤ 15).
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
    if r == 0 {
        return transport == K_TRANSPORT_BUILT_IN;
    }

    // Method 2: UID heuristic (macOS 16+).
    let uid = get_cfstring_property(dev_id, K_AUDIO_PROP_UID);
    uid.starts_with("BuiltIn")
}

const K_AUDIO_PROP_UID: u32             = 0x75696420; // 'uid '
const K_AUDIO_PROP_MANUFACTURER: u32    = 0x6c6d616b; // 'lmak'

/// Check whether a CoreAudio device is virtual (loopback driver).
///
/// Uses a two-tier strategy:
/// 1. `kAudioDevicePropertyTransportType == 'vrtl'` (works on macOS ≤ 15)
/// 2. Fallback (macOS 16+, where transport type is unavailable): checks that
///    the manufacturer is NOT "Apple Inc." and the UID lacks hardware
///    identifiers (MAC-address pattern, "BuiltIn" prefix, or UUID dashes).
unsafe fn is_virtual_device(dev_id: u32) -> bool {
    // Method 1: transport type (reliable on older macOS).
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
    if r == 0 {
        return transport == K_TRANSPORT_VIRTUAL;
    }

    // Method 2: manufacturer + UID heuristic (macOS 16+).
    let uid = get_cfstring_property(dev_id, K_AUDIO_PROP_UID);
    let mfr = get_cfstring_property(dev_id, K_AUDIO_PROP_MANUFACTURER);

    let is_apple = mfr == "Apple Inc.";
    // Physical devices typically have hardware UIDs: "XX-XX-XX:input", "BuiltIn*", or UUID-style dashes.
    let has_hw_uid = uid.contains(':') || uid.starts_with("BuiltIn") || uid.contains('-');

    !is_apple && !has_hw_uid
}

/// Read a CFString property from a CoreAudio object.
unsafe fn get_cfstring_property(dev_id: u32, selector: u32) -> String {
    let addr = AudioObjectPropertyAddress {
        selector,
        scope: K_SCOPE_GLOBAL,
        element: K_ELEMENT_MAIN,
    };
    let mut cf_str: *mut c_void = std::ptr::null_mut();
    let mut sz = std::mem::size_of::<*mut c_void>() as u32;
    let r = AudioObjectGetPropertyData(
        dev_id, &addr,
        0, std::ptr::null(),
        &mut sz, &mut cf_str as *mut *mut c_void as *mut c_void,
    );
    if r != 0 || cf_str.is_null() { return String::new(); }
    let s = cfstring_to_string(cf_str);
    CFRelease(cf_str);
    s
}

/// List names of all non-virtual audio input devices (i.e. physical mics).
///
/// Filters out virtual loopback drivers (BlackHole, Loopback, Speaker Audio
/// Recorder, etc.) using transport type on macOS ≤ 15 and a manufacturer/UID
/// heuristic on macOS 16+ where transport type is unavailable.
pub fn list_physical_input_device_names() -> Vec<String> {
    unsafe {
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
        if s != 0 || data_size == 0 { return Vec::new(); }

        let count = data_size as usize / std::mem::size_of::<u32>();
        let mut ids: Vec<u32> = vec![0u32; count];
        let mut actual = data_size;
        let s2 = AudioObjectGetPropertyData(
            K_AUDIO_OBJECT_SYSTEM, &addr_devs,
            0, std::ptr::null(),
            &mut actual, ids.as_mut_ptr() as *mut c_void,
        );
        if s2 != 0 { return Vec::new(); }

        let mut names = Vec::new();
        for &dev_id in &ids {
            if is_virtual_device(dev_id) { continue; }

            // Must have at least one input stream.
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

            // Get device name.
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
            if !name.is_empty() { names.push(name); }
        }
        names
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
        let status = AudioObjectAddPropertyListener(
            K_AUDIO_OBJECT_SYSTEM, &addr,
            on_default_input_changed, raw,
        );
        if status != 0 {
            tracing::warn!("AudioObjectAddPropertyListener failed: OSStatus {}", status);
        }
    }
}

/// Returns the NSPasteboard changeCount, which increments each time the clipboard is written.
/// Used to detect whether a Cmd+C actually updated the clipboard (avoids false negatives
/// when the selected text is identical to the previously saved clipboard content).
pub fn clipboard_change_count() -> Option<u32> {
    unsafe {
        let cls = objc_getClass(c"NSPasteboard".as_ptr());
        if cls.is_null() { return None; }

        let sel_general = sel_registerName(c"generalPasteboard".as_ptr());
        type MsgSendPb = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
        let pb = std::mem::transmute::<*const (), MsgSendPb>(objc_msgSend as *const ())(cls, sel_general);
        if pb.is_null() { return None; }

        let sel_count = sel_registerName(c"changeCount".as_ptr());
        type MsgSendCount = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i64;
        let count = std::mem::transmute::<*const (), MsgSendCount>(objc_msgSend as *const ())(pb, sel_count);
        Some(count as u32)
    }
}
