//! Credential storage — platform-specific implementations.
//!
//! macOS: uses the `security-framework` crate (direct Keychain C API calls).
//!        API keys are never exposed in process argv.
//! Other: uses the `keyring` crate (Windows Credential Manager, etc.).

const SERVICE: &str = if cfg!(debug_assertions) { "sumi-dev" } else { "sumi" };

fn keychain_service(provider: &str) -> String {
    format!("{}-api-key-{}", SERVICE, provider)
}

// ── macOS: `security-framework` crate ─────────────────────────────

#[cfg(target_os = "macos")]
pub fn save(provider: &str, key: &str) -> Result<(), String> {
    security_framework::passwords::set_generic_password(
        &keychain_service(provider),
        SERVICE,
        key.as_bytes(),
    )
    .map_err(|e| format!("Keychain save failed: {}", e))
}

#[cfg(target_os = "macos")]
pub fn load(provider: &str) -> Result<String, String> {
    match security_framework::passwords::get_generic_password(
        &keychain_service(provider),
        SERVICE,
    ) {
        Ok(bytes) => String::from_utf8(bytes)
            .map_err(|e| format!("Keychain value is not valid UTF-8: {}", e)),
        Err(e) if e.code() == -25300 => Ok(String::new()), // errSecItemNotFound
        Err(e) => Err(format!("Keychain load failed: {}", e)),
    }
}

#[cfg(target_os = "macos")]
pub fn delete(provider: &str) -> Result<(), String> {
    match security_framework::passwords::delete_generic_password(
        &keychain_service(provider),
        SERVICE,
    ) {
        Ok(()) => Ok(()),
        Err(e) if e.code() == -25300 => Ok(()), // errSecItemNotFound → already gone
        Err(e) => Err(format!("Keychain delete failed: {}", e)),
    }
}

// ── Non-macOS: `keyring` crate ─────────────────────────────────────

#[cfg(not(target_os = "macos"))]
pub fn save(provider: &str, key: &str) -> Result<(), String> {
    entry(provider)?
        .set_password(key)
        .map_err(|e| format!("Keyring save failed: {}", e))
}

#[cfg(not(target_os = "macos"))]
pub fn load(provider: &str) -> Result<String, String> {
    match entry(provider)?.get_password() {
        Ok(key) => Ok(key),
        Err(keyring::Error::NoEntry) => Ok(String::new()),
        Err(e) => Err(format!("Keyring load failed: {}", e)),
    }
}

#[cfg(not(target_os = "macos"))]
pub fn delete(provider: &str) -> Result<(), String> {
    match entry(provider)?.delete_credential() {
        Ok(()) => Ok(()),
        Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(format!("Keyring delete failed: {}", e)),
    }
}

#[cfg(not(target_os = "macos"))]
fn entry(provider: &str) -> Result<keyring::Entry, String> {
    keyring::Entry::new(&keychain_service(provider), SERVICE)
        .map_err(|e| format!("Keyring error: {}", e))
}
