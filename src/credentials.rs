//! Credential storage — platform-specific implementations.
//!
//! macOS: hybrid approach — good UX AND no argv leaks:
//!   save()   → security-framework + Data Protection Keychain
//!              (kSecUseDataProtectionKeychain=true, no per-app ACL, no prompts ever)
//!   load()   → Data Protection Keychain first; falls back to `security` CLI for
//!              legacy items (key returned via stdout, never in argv), then auto-migrates
//!   delete() → Data Protection Keychain + `security` CLI legacy cleanup
//!
//! Non-macOS: `keyring` crate (Windows Credential Manager, etc.)

const SERVICE: &str = if cfg!(debug_assertions) { "sumi-dev" } else { "sumi" };

fn keychain_service(provider: &str) -> String {
    format!("{}-api-key-{}", SERVICE, provider)
}

// ── macOS ──────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
pub fn save(provider: &str, key: &str) -> Result<(), String> {
    use security_framework::passwords::{delete_generic_password_options, set_generic_password_options, PasswordOptions};
    const ERR_DUPLICATE: i32 = -25299;        // errSecDuplicateItem
    const ERR_MISSING_ENTITLEMENT: i32 = -34018; // errSecMissingEntitlement

    let mut opts = PasswordOptions::new_generic_password(&keychain_service(provider), SERVICE);
    opts.use_protected_keychain();
    match set_generic_password_options(key.as_bytes(), opts) {
        Ok(()) => return Ok(()),
        Err(e) if e.code() == ERR_DUPLICATE => {
            // Item already exists — delete then re-add with the updated value.
            let mut del = PasswordOptions::new_generic_password(&keychain_service(provider), SERVICE);
            del.use_protected_keychain();
            let _ = delete_generic_password_options(del);
            let mut opts2 = PasswordOptions::new_generic_password(&keychain_service(provider), SERVICE);
            opts2.use_protected_keychain();
            return set_generic_password_options(key.as_bytes(), opts2)
                .map_err(|e| format!("Keychain update failed: {}", e));
        }
        Err(e) if e.code() == ERR_MISSING_ENTITLEMENT => {
            // App isn't signed with keychain-access-groups yet (e.g. raw cargo build).
            // Fall back to security CLI so the key is never silently dropped.
            tracing::warn!("Data Protection Keychain unavailable, falling back to security CLI");
        }
        Err(e) => return Err(format!("Keychain save failed: {}", e)),
    }

    // Fallback: security CLI. The key appears in argv briefly — acceptable only
    // because this path is only reached when the app lacks proper entitlements.
    let service = keychain_service(provider);
    let _ = std::process::Command::new("/usr/bin/security")
        .args(["delete-generic-password", "-s", &service, "-a", SERVICE])
        .output();
    let out = std::process::Command::new("/usr/bin/security")
        .args(["add-generic-password", "-s", &service, "-a", SERVICE, "-w", key, "-U"])
        .output()
        .map_err(|e| format!("Failed to run security CLI: {}", e))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "security add-generic-password failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

#[cfg(target_os = "macos")]
pub fn load(provider: &str) -> Result<String, String> {
    use security_framework::passwords::{generic_password, PasswordOptions};
    const ERR_NOT_FOUND: i32 = -25300;         // errSecItemNotFound
    const ERR_MISSING_ENTITLEMENT: i32 = -34018; // errSecMissingEntitlement

    // Try Data Protection Keychain first (no per-app ACL → no prompts).
    // Track whether the entitlement is present so we know if migration is safe.
    let mut opts = PasswordOptions::new_generic_password(&keychain_service(provider), SERVICE);
    opts.use_protected_keychain();
    let has_entitlement = match generic_password(opts) {
        Ok(bytes) => return String::from_utf8(bytes)
            .map_err(|e| format!("Keychain value is not valid UTF-8: {}", e)),
        Err(e) if e.code() == ERR_NOT_FOUND => true,          // entitlement ok, item just absent
        Err(e) if e.code() == ERR_MISSING_ENTITLEMENT => false, // no entitlement, skip migration
        Err(e) => return Err(format!("Keychain load failed: {}", e)),
    };

    // Fallback: read legacy item via `security` CLI.
    // The key is returned via stdout — it never appears in argv.
    let service = keychain_service(provider);
    let out = std::process::Command::new("/usr/bin/security")
        .args(["find-generic-password", "-s", &service, "-a", SERVICE, "-w"])
        .output()
        .map_err(|e| format!("Failed to run security CLI: {}", e))?;

    if !out.status.success() {
        return Ok(String::new()); // nothing in legacy either
    }

    let key = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if key.is_empty() {
        return Ok(String::new());
    }

    // Only migrate if entitlement is present — otherwise save() would fall back to
    // security CLI (recreating the legacy item), and the delete below would erase it.
    if has_entitlement {
        match save(provider, &key) {
            Ok(()) => {
                let out = std::process::Command::new("/usr/bin/security")
                    .args(["delete-generic-password", "-s", &service, "-a", SERVICE])
                    .output();
                if let Ok(o) = out {
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    if !stderr.trim().is_empty() {
                        tracing::warn!("Legacy Keychain entry cleanup may have failed: {}", stderr.trim());
                    }
                }
            }
            Err(e) => tracing::warn!("Keychain migration failed: {}", e),
        }
    }

    Ok(key)
}

#[cfg(target_os = "macos")]
pub fn delete(provider: &str) -> Result<(), String> {
    use security_framework::passwords::{delete_generic_password_options, PasswordOptions};
    const ERR_NOT_FOUND: i32 = -25300;         // errSecItemNotFound
    const ERR_MISSING_ENTITLEMENT: i32 = -34018; // errSecMissingEntitlement

    // Remove from Data Protection Keychain.
    let mut opts = PasswordOptions::new_generic_password(&keychain_service(provider), SERVICE);
    opts.use_protected_keychain();
    let result = match delete_generic_password_options(opts) {
        Ok(()) => Ok(()),
        Err(e) if e.code() == ERR_NOT_FOUND => Ok(()),
        Err(e) if e.code() == ERR_MISSING_ENTITLEMENT => Ok(()), // unsigned build — nothing to delete
        Err(e) => Err(format!("Keychain delete failed: {}", e)),
    };

    // Also clean up any legacy item.
    let service = keychain_service(provider);
    let _ = std::process::Command::new("/usr/bin/security")
        .args(["delete-generic-password", "-s", &service, "-a", SERVICE])
        .output();

    result
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
