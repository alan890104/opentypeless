//! Credential storage — platform-specific implementations.
//!
//! macOS: hybrid approach — Data Protection Keychain primary + CLI backup:
//!   save()   → security-framework + Data Protection Keychain (primary,
//!              kSecUseDataProtectionKeychain=true, no per-app ACL, no prompts ever)
//!              + `security` CLI backup written only when value changes (checked via
//!              stdout read first; the key appears in argv only on actual key change)
//!   load()   → Data Protection Keychain first; falls back to `security` CLI for
//!              legacy/backup items (key returned via stdout, never in argv),
//!              then migrates to Data Protection Keychain while keeping CLI backup
//!   delete() → Data Protection Keychain + `security` CLI cleanup
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

    // Try Data Protection Keychain. Track whether the entitlement is available so
    // we know whether the CLI write below is the primary store or just a backup.
    let dp_ok = match {
        let mut opts = PasswordOptions::new_generic_password(&keychain_service(provider), SERVICE);
        opts.use_protected_keychain();
        set_generic_password_options(key.as_bytes(), opts)
    } {
        Ok(()) => true,
        Err(e) if e.code() == ERR_DUPLICATE => {
            // Item already exists — delete then re-add with the updated value.
            let mut del = PasswordOptions::new_generic_password(&keychain_service(provider), SERVICE);
            del.use_protected_keychain();
            if let Err(e) = delete_generic_password_options(del) {
                tracing::warn!("Keychain delete (before update) failed: {} — subsequent add may fail", e);
            }
            let mut opts2 = PasswordOptions::new_generic_password(&keychain_service(provider), SERVICE);
            opts2.use_protected_keychain();
            set_generic_password_options(key.as_bytes(), opts2)
                .map_err(|e| format!("Keychain update failed: {}", e))?;
            true
        }
        Err(e) if e.code() == ERR_MISSING_ENTITLEMENT => {
            // App lacks keychain-access-groups entitlement (e.g. unsigned build).
            tracing::warn!("Data Protection Keychain unavailable, falling back to security CLI");
            false
        }
        Err(e) => return Err(format!("Keychain save failed: {}", e)),
    };

    // Always mirror the key in the security CLI store as a fallback.  If the
    // keychain-access-groups entitlement changes between app versions (e.g.
    // Team ID prefix added or removed), the Data Protection Keychain items
    // become inaccessible to the new binary.  The CLI copy ensures load() can
    // still recover the key via its fallback path.
    //
    // Read the current CLI value first (stdout only, no argv exposure) and skip
    // the write when the stored value already matches — this avoids passing the
    // key through argv on every save when nothing has changed.
    let service = keychain_service(provider);
    let existing_cli = std::process::Command::new("/usr/bin/security")
        .args(["find-generic-password", "-s", &service, "-a", SERVICE, "-w"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string());

    let cli_out = if existing_cli.as_deref() == Some(key) {
        // CLI backup is already up to date; skip the argv-visible write.
        None
    } else {
        let _ = std::process::Command::new("/usr/bin/security")
            .args(["delete-generic-password", "-s", &service, "-a", SERVICE])
            .output();
        Some(
            std::process::Command::new("/usr/bin/security")
                .args(["add-generic-password", "-s", &service, "-a", SERVICE, "-w", key, "-U"])
                .output()
                .map_err(|e| format!("Failed to run security CLI: {}", e))?,
        )
    };

    if dp_ok {
        // Data Protection Keychain is the primary store; CLI is just a backup.
        // Ignore CLI failures — they are non-fatal when the primary succeeded.
        if let Some(out) = cli_out {
            if !out.status.success() {
                tracing::warn!(
                    "CLI keychain backup write failed (non-fatal): {}",
                    String::from_utf8_lossy(&out.stderr).trim()
                );
            }
        }
        Ok(())
    } else {
        // CLI is the only store — a skipped write means it was already current.
        match cli_out {
            None => Ok(()),
            Some(out) if out.status.success() => Ok(()),
            Some(out) => Err(format!(
                "security add-generic-password failed: {}",
                String::from_utf8_lossy(&out.stderr)
            )),
        }
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

    // Only migrate if entitlement is present. save() writes both Data Protection
    // Keychain (primary) and CLI (backup via dual-write), so after migration the
    // CLI item is already the backup — do NOT delete it, or the backup is lost.
    if has_entitlement {
        if let Err(e) = save(provider, &key) {
            tracing::warn!("Keychain migration failed: {}", e);
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
