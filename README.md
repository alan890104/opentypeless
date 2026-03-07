# Sumi

![GitHub Release](https://img.shields.io/github/v/release/alan890104/sumi)
![License](https://img.shields.io/github/license/alan890104/sumi)
![GitHub stars](https://img.shields.io/github/stars/alan890104/sumi?style=social)
![GitHub forks](https://img.shields.io/github/forks/alan890104/sumi?style=social)
![Rust](https://img.shields.io/badge/Rust-black?style=flat-square&logo=rust)
![Tauri](https://img.shields.io/badge/Tauri_v2-FFC131?style=flat-square&logo=tauri&logoColor=white)
![Svelte](https://img.shields.io/badge/Svelte_5-FF3E00?style=flat-square&logo=svelte&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-000000?style=flat-square&logo=apple&logoColor=white)

English | [繁體中文](README_TW.md) | [简体中文](README_CN.md)

System-wide speech-to-text for macOS. Free and open source.

Sumi is a macOS dictation app. Press a hotkey anywhere, speak, and the transcription pastes at your cursor. It runs the text through an LLM before pasting — and the LLM prompt changes depending on which app you're in. That last part is why I built it; none of the other tools I tried did it.

<table>
<tr>
<td align="center"><img src="assets/demo-gmail.gif" width="240"/><br/>Gmail</td>
<td align="center"><img src="assets/demo-notion.gif" width="240"/><br/>Notion</td>
<td align="center"><img src="assets/demo-telegram.gif" width="240"/><br/>Telegram</td>
</tr>
</table>

## You might need this if…

- You switch between WhatsApp, Slack, Gmail, and Teams all day and every dictation tool you've tried outputs the same flat text in all of them. You're still fixing it yourself.
- Your wrists hurt. You don't want another subscription.
- You work in a terminal or VSCode. Voice input that sounds like texting is useless there.
- "We take privacy seriously" is not something you find convincing. You want to read the code.
- You leave every meeting having written nothing down.

## The per-app rule thing

Most dictation apps transcribe and paste. Sumi does that too, but the text goes through an LLM rewrite first, and the prompt the LLM sees depends on what app has focus.

There are 18 built-in rules for common apps. You can write your own, or just describe what you want in plain text and the LLM generates the rule. Rules match by app name, bundle ID, or URL — so the Slack desktop app and `app.slack.com` in a browser both pick up the same rule automatically.

Same sentence, three apps:

> You say: *"um I think the project is kind of behind schedule and we should probably have a meeting to figure out what to do next"*

LINE (casual):
> I think the project is behind schedule, we should have a meeting to figure out what to do next

Slack (professional, concise):
> I think the project is behind schedule. We should have a meeting to discuss next steps.

Gmail (email format):
> Hi,
>
> I believe the project is currently behind schedule. Could we schedule a meeting to discuss the next steps?
>
> Best regards

## What it runs on

Speech recognition is either local (Whisper with Metal GPU acceleration, 7 model sizes from 148 MB to 1.6 GB — or Qwen3-ASR as an alternative) or cloud (Groq, OpenAI, Deepgram, Azure, any custom endpoint). You bring your own API key; there's no Sumi account.

LLM rewriting works the same way: local models via candle with Metal/CUDA (Llama 3 Taiwan 8B, Qwen 2.5 7B, Qwen 3 8B) or cloud (OpenAI, Groq, Gemini, GitHub Models, OpenRouter, SambaNova, any OpenAI-compatible endpoint).

In full local mode, audio never leaves your Mac. You can verify that because the code is here.

When idle: about 130 MB and 0% CPU. The first local transcription loads Whisper into GPU memory and RSS climbs to around 730 MB, but Metal runs the inference so CPU stays under 20%. Cloud mode barely registers — 7 MB during the recording, then back to 0% the moment it's done.

Other things worth knowing:

- Edit by Voice: select text, press `Ctrl+Option+Z`, say what you want done. The AI rewrites and pastes it back.
- Meeting mode: runs in the background and transcribes into a note file. No babysitting.
- Silero VAD for silence filtering before transcription (optional, needs a separate model download)
- Custom dictionary: words you add show up in both the Whisper prompt and LLM context, so names and terms don't get mangled
- Transcription history with audio playback, 58 UI languages, customizable hotkeys

## How to use it

1. Open the app. It lives in the menu bar — nothing else.
2. Click into any text field, anywhere on your Mac.
3. Press `Option+Z`. A small floating capsule appears with the waveform.
4. Speak.
5. Press `Option+Z` again. The text pastes.

To edit text by voice: select it, press `Ctrl+Option+Z`, say what you want done with it.

## Comparison

> [!NOTE]
> This reflects our best understanding at the time of writing. Competitors update frequently — corrections welcome via issues or PRs.

| | **Sumi** | Built-in Dictation | Typeless | Wispr Flow | VoiceInk | SuperWhisper |
|---|---|---|---|---|---|---|
| **Price** | **Free** | Free | 4K words/wk free, $12-30/mo | 2K words/wk free, $12-15/mo | $25-49 (one-time) | Free trial, ~$8/mo |
| **Open Source** | ✅ GPLv3 | ❌ | ❌ | ❌ | ✅ GPLv3 | ❌ |
| **Local STT** | ✅ | ✅ Apple Silicon | ❌ Cloud only | ❌ Cloud only | ✅ | ✅ |
| **Cloud STT** | ✅ BYOK | ❌ | ✅ | ✅ | ✅ Optional | ✅ |
| **AI Polish** | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Local LLM Polish** | ✅ 3 models | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Per-App Rules** | ✅ 18 presets + custom | ❌ | ❌ | ✅ Styles | ✅ Power Modes | ✅ Custom modes |
| **Context-Aware** | ✅ App + URL | ❌ | ✅ App | ✅ App | ✅ App | ✅ Super Mode |
| **Edit by Voice** | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Dictionary** | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **History** | ✅ + audio export | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Meeting notes** | ✅ | ❌ | - | ❌ | - | ✅ |
| **Platforms** | macOS | macOS, iOS | macOS, Win, iOS, Android | macOS, Win, iOS, Android | macOS | macOS, Win, iOS |

## Installation

### Homebrew

```bash
brew tap alan890104/sumi
brew install --cask sumi
```

### Download

1. Download the latest DMG from [GitHub Releases](https://github.com/alan890104/sumi/releases/latest).
2. Open the DMG, drag Sumi into `/Applications`.
3. The app isn't notarized yet, so macOS will block it. Run this first:

   ```bash
   xattr -cr /Applications/Sumi.app
   ```

4. On first launch: grant Microphone access, and turn on Accessibility under System Settings > Privacy & Security > Accessibility. The second one is needed for auto-paste.

### Build from source

Requires [Rust](https://rustup.rs/) and `cargo install tauri-cli --version "^2"`.

macOS:

```bash
git clone https://github.com/alan890104/sumi.git
cd sumi
cargo tauri dev      # dev mode
cargo tauri build    # production .dmg
```

Windows:

Metal is macOS-only. On Windows, you need to disable it:

```bash
# CPU only
cargo tauri dev --no-default-features

# With NVIDIA CUDA (requires CUDA Toolkit, LLVM, Ninja, CMake)
bash dev-cuda.sh
bash dev-cuda.sh --release
```

## License

GPLv3
