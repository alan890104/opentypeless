# Sumi

![GitHub Release](https://img.shields.io/github/v/release/alan890104/sumi)
![License](https://img.shields.io/github/license/alan890104/sumi)
![GitHub stars](https://img.shields.io/github/stars/alan890104/sumi?style=social)
![GitHub forks](https://img.shields.io/github/forks/alan890104/sumi?style=social)
![Rust](https://img.shields.io/badge/Rust-black?style=flat-square&logo=rust)
![Tauri](https://img.shields.io/badge/Tauri_v2-FFC131?style=flat-square&logo=tauri&logoColor=white)
![Svelte](https://img.shields.io/badge/Svelte_5-FF3E00?style=flat-square&logo=svelte&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-000000?style=flat-square&logo=apple&logoColor=white)

[English](README.md) | 繁體中文 | [简体中文](README_CN.md)

macOS 全域語音輸入工具。免費開源。

Sumi 是一款 macOS 語音輸入 App。在任何地方按下快捷鍵、說話，文字就會貼到游標位置。它會在貼上前先讓 LLM 改寫一遍，而且 LLM 用的提示詞會根據你當下開著哪個 App 自動調整。這個功能我在其他工具都找不到，所以就自己做了。

<table>
<tr>
<td align="center"><img src="assets/demo-gmail.gif" width="240"/><br/>Gmail</td>
<td align="center"><img src="assets/demo-notion.gif" width="240"/><br/>Notion</td>
<td align="center"><img src="assets/demo-telegram.gif" width="240"/><br/>Telegram</td>
</tr>
</table>

## 你可能需要它，如果…

- 你一天要在 LINE、Slack、Gmail 之間切換，每個工具的語音輸入出來都長一樣。語氣和格式還是你自己在改。
- 你的手腕開始不舒服了。不想再多一個訂閱費。
- 你在 VSCode 或 Terminal 裡工作。語音輸入出來像在傳 LINE 訊息，沒有用。
- 廠商說「我們非常重視隱私」這種話你不信。你要的是可以自己看程式碼確認的東西。
- 你每次開完會什麼都沒記下來。

## 依 App 套用規則這件事

大多數語音輸入工具就是轉錄完貼上。Sumi 也會這樣做，但文字會先過 LLM 改寫，而且 LLM 看到的提示詞會依照當下前景 App（或瀏覽器分頁）而不同。

內建 18 組規則涵蓋常用 App。你可以自己寫規則，或直接用白話描述你想要的效果，LLM 會幫你生成一條。規則可以用 App 名稱、Bundle ID 或網址來比對 — 所以 Slack 桌面版和瀏覽器裡的 `app.slack.com` 會自動套用同一條規則。

同一段話，三個 App：

> 你說：*「嗯就是我覺得這個專案的進度有點落後，我們需要開個會討論一下接下來要怎麼做」*

LINE（輕鬆隨意）：
> 我覺得專案進度有點落後，我們開個會討論一下接下來怎麼做吧

Slack（專業簡潔）：
> 我覺得專案進度有些落後，我們需要開個會討論接下來的計畫。

Gmail（信件格式）：
> 您好，
>
> 我注意到目前專案進度略有落後，想請大家安排一次會議，討論接下來的工作規劃。期待您的回覆。

## 跑什麼東西

語音辨識可以跑本地（Whisper 搭配 Metal GPU 加速，7 種模型大小從 148 MB 到 1.6 GB，或改用 Qwen3-ASR）或雲端（Groq、OpenAI、Deepgram、Azure、任何自訂端點）。帶自己的 API Key 來就好，Sumi 沒有帳號系統。

LLM 改寫也一樣：本地模型透過 candle 跑 Metal/CUDA 加速（Llama 3 Taiwan 8B、Qwen 2.5 7B、Qwen 3 8B），或接雲端（OpenAI、Groq、Gemini、GitHub Models、OpenRouter、SambaNova、任何相容 OpenAI 格式的端點）。

完全本地模式下，音訊不會離開你的 Mac。程式碼都在這裡，可以自己驗證。

待機時約 130 MB、CPU 0%。第一次本地轉錄把 Whisper 模型丟進 GPU 記憶體，RSS 會升到約 730 MB，但 Metal 跑推理，CPU 峰值還是在 20% 以下。雲端模式幾乎沒有感覺：錄音期間多約 7 MB，傳完立刻歸零。

其他值得知道的：

- 語音編輯：選取文字，按 `Ctrl+Option+Z`，說你想怎麼改。AI 改寫後自動貼回。
- 會議模式：在背景持續轉錄，存成筆記檔案。不需要人盯著。
- Silero VAD 靜音過濾（選配，需另外下載模型）
- 自訂詞典：加入的詞彙會出現在 Whisper 提示詞和 LLM 上下文中，人名和術語不再被亂改
- 轉錄歷史含音訊回放、58 種介面語言、快捷鍵可自訂

## 怎麼用

1. 打開 App，它住在選單列，沒有別的東西。
2. 在 Mac 上任何文字欄位點一下。
3. 按 `Option+Z`，畫面出現帶波形的浮動膠囊。
4. 說話。
5. 再按一次 `Option+Z`，文字貼上。

語音編輯文字：先選取，按 `Ctrl+Option+Z`，說你想怎麼改。

## 競品比較

> [!NOTE]
> 此表為撰寫當時的資訊，各產品功能可能隨時更新，歡迎透過 Issue 或 PR 更正。

| | **Sumi** | 系統內建聽寫 | Typeless | Wispr Flow | VoiceInk | SuperWhisper |
|---|---|---|---|---|---|---|
| **價格** | **免費** | 免費 | 每週 4K 字免費, $12-30/月 | 每週 2K 字免費, $12-15/月 | $25-49（買斷） | 免費試用, ~$8/月 |
| **開源** | ✅ GPLv3 | ❌ | ❌ | ❌ | ✅ GPLv3 | ❌ |
| **本地語音辨識** | ✅ | ✅ Apple Silicon | ❌ 僅雲端 | ❌ 僅雲端 | ✅ | ✅ |
| **雲端語音辨識** | ✅ 自帶 Key | ❌ | ✅ | ✅ | ✅ 可選 | ✅ |
| **AI 潤飾** | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **本地 LLM 潤飾** | ✅ 3 種模型 | ❌ | ❌ | ❌ | ❌ | ✅ |
| **依 App 規則** | ✅ 18 預設 + 自訂 | ❌ | ❌ | ✅ Styles | ✅ Power Modes | ✅ 自訂模式 |
| **情境感知** | ✅ App + URL | ❌ | ✅ App | ✅ App | ✅ App | ✅ Super Mode |
| **語音編輯文字** | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **詞典** | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **歷史紀錄** | ✅ 含音訊匯出 | ❌ | ✅ | ✅ | ✅ | ✅ |
| **會議記錄** | ✅ | ❌ | - | ❌ | - | ✅ |
| **平台** | macOS | macOS, iOS | macOS, Win, iOS, Android | macOS, Win, iOS, Android | macOS | macOS, Win, iOS |

## 安裝

### Homebrew

```bash
brew tap alan890104/sumi
brew install --cask sumi
```

### 下載安裝

1. 從 [GitHub Releases](https://github.com/alan890104/sumi/releases/latest) 下載最新的 DMG。
2. 打開 DMG，把 Sumi 拖進 `/Applications`。
3. 這個 App 還沒有 Apple 公證，macOS 第一次會擋住。先跑這個：

   ```bash
   xattr -cr /Applications/Sumi.app
   ```

4. 第一次開啟：給麥克風權限，然後到系統設定 > 隱私權與安全性 > 輔助功能 開啟 Sumi。後者是自動貼上功能需要的。

### 從原始碼編譯

需要 [Rust](https://rustup.rs/) 和 `cargo install tauri-cli --version "^2"`。

macOS：

```bash
git clone https://github.com/alan890104/sumi.git
cd sumi
cargo tauri dev      # 開發模式
cargo tauri build    # 正式編譯（輸出 .dmg）
```

Windows：

Metal 是 macOS 專屬的，在 Windows 上要關掉：

```bash
# 純 CPU
cargo tauri dev --no-default-features

# 搭配 NVIDIA CUDA（需要 CUDA Toolkit、LLVM、Ninja、CMake）
bash dev-cuda.sh
bash dev-cuda.sh --release
```

## 授權

GPLv3
