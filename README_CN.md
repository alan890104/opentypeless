# Sumi

![GitHub Release](https://img.shields.io/github/v/release/alan890104/sumi)
![License](https://img.shields.io/github/license/alan890104/sumi)
![GitHub stars](https://img.shields.io/github/stars/alan890104/sumi?style=social)
![GitHub forks](https://img.shields.io/github/forks/alan890104/sumi?style=social)
![Rust](https://img.shields.io/badge/Rust-black?style=flat-square&logo=rust)
![Tauri](https://img.shields.io/badge/Tauri_v2-FFC131?style=flat-square&logo=tauri&logoColor=white)
![Svelte](https://img.shields.io/badge/Svelte_5-FF3E00?style=flat-square&logo=svelte&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-000000?style=flat-square&logo=apple&logoColor=white)

[English](README.md) | [繁體中文](README_TW.md) | 简体中文

macOS 全局语音输入工具。免费开源。

Sumi 是一款 macOS 语音输入 App。在任何地方按下快捷键、说话，文字就会粘贴到光标位置。它会在粘贴前先让 LLM 改写一遍，而且 LLM 用的提示词会根据你当下打开的 App 自动调整。这个功能我在其他工具里都找不到，所以就自己做了。

<table>
<tr>
<td align="center"><img src="assets/demo-gmail.gif" width="240"/><br/>Gmail</td>
<td align="center"><img src="assets/demo-notion.gif" width="240"/><br/>Notion</td>
<td align="center"><img src="assets/demo-telegram.gif" width="240"/><br/>Telegram</td>
</tr>
</table>

## 你可能需要它，如果…

- 你一天要在微信、钉钉、Gmail 之间切换，每个工具的语音输入出来都长一样。语气和格式还是你自己在改。
- 你的手腕开始不舒服了。不想再多一个订阅费。
- 你在 VSCode 或终端里工作。语音输入出来像在发微信，没有用。
- 厂商说"我们非常重视隐私"这种话你不信。你要的是可以自己看代码确认的东西。
- 你每次开完会什么都没记下来。

## 按 App 套用规则这件事

大多数语音输入工具就是转录完粘贴。Sumi 也会这样做，但文字会先经过 LLM 改写，而且 LLM 看到的提示词会依照当前前景 App（或浏览器标签页）而不同。

内置 18 条规则涵盖常用 App。你可以自己写规则，或直接用白话描述你想要的效果，LLM 会帮你生成一条。规则可以用 App 名称、Bundle ID 或网址来匹配 — 所以 Slack 桌面版和浏览器里的 `app.slack.com` 会自动套用同一条规则。

同一段话，三个 App：

> 你说：*"嗯就是我觉得这个项目的进度有点落后，我们需要开个会讨论一下接下来要怎么做"*

LINE（轻松随意）：
> 我觉得项目进度有点落后，我们开个会讨论一下接下来怎么做吧

Slack（专业简洁）：
> 我觉得项目进度有些落后，我们需要开个会讨论接下来的计划。

Gmail（邮件格式）：
> 您好，
>
> 我注意到目前项目进度略有落后，想请大家安排一次会议，讨论接下来的工作规划。期待您的回复。

## 运行什么东西

语音识别可以跑本地（Whisper 搭配 Metal GPU 加速，7 种模型大小从 148 MB 到 1.6 GB，或改用 Qwen3-ASR）或云端（Groq、OpenAI、Deepgram、Azure、任何自定义端点）。带自己的 API Key 来就好，Sumi 没有账号系统。

LLM 改写也一样：本地模型通过 candle 跑 Metal/CUDA 加速（Llama 3 Taiwan 8B、Qwen 2.5 7B、Qwen 3 8B），或接云端（OpenAI、Groq、Gemini、GitHub Models、OpenRouter、SambaNova、任何兼容 OpenAI 格式的端点）。

完全本地模式下，音频不会离开你的 Mac。代码都在这里，可以自己验证。

待机时约 130 MB、CPU 0%。第一次本地转录把 Whisper 模型丢进 GPU 内存，RSS 会升到约 730 MB，但 Metal 跑推理，CPU 峰值还是在 20% 以下。云端模式几乎没有感觉：录音期间多约 7 MB，传完立刻归零。

其他值得知道的：

- 语音编辑：选中文字，按 `Ctrl+Option+Z`，说你想怎么改。AI 改写后自动粘贴回去。
- 会议模式：在后台持续转录，存成笔记文件。不需要人盯着。
- Silero VAD 静音过滤（可选，需另外下载模型）
- 自定义词典：加入的词汇会出现在 Whisper 提示词和 LLM 上下文中，人名和术语不再被乱改
- 转录历史含音频回放、58 种界面语言、快捷键可自定义

## 怎么用

1. 打开 App，它住在菜单栏，没有别的东西。
2. 在 Mac 上任何文字输入框点一下。
3. 按 `Option+Z`，画面出现带波形的浮动胶囊。
4. 说话。
5. 再按一次 `Option+Z`，文字粘贴上去。

语音编辑文字：先选中，按 `Ctrl+Option+Z`，说你想怎么改。

## 竞品对比

> [!NOTE]
> 此表为撰写时的信息，各产品功能可能随时更新，欢迎通过 Issue 或 PR 更正。

| | **Sumi** | 系统内置听写 | Typeless | Wispr Flow | VoiceInk | SuperWhisper |
|---|---|---|---|---|---|---|
| **价格** | **免费** | 免费 | 每周 4K 字免费, $12-30/月 | 每周 2K 字免费, $12-15/月 | $25-49（买断） | 免费试用, ~$8/月 |
| **开源** | ✅ GPLv3 | ❌ | ❌ | ❌ | ✅ GPLv3 | ❌ |
| **本地语音识别** | ✅ | ✅ Apple Silicon | ❌ 仅云端 | ❌ 仅云端 | ✅ | ✅ |
| **云端语音识别** | ✅ 自带 Key | ❌ | ✅ | ✅ | ✅ 可选 | ✅ |
| **AI 润色** | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **本地 LLM 润色** | ✅ 3 种模型 | ❌ | ❌ | ❌ | ❌ | ✅ |
| **按 App 规则** | ✅ 18 预设 + 自定义 | ❌ | ❌ | ✅ Styles | ✅ Power Modes | ✅ 自定义模式 |
| **情境感知** | ✅ App + URL | ❌ | ✅ App | ✅ App | ✅ App | ✅ Super Mode |
| **语音编辑文字** | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **词典** | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **历史记录** | ✅ 含音频导出 | ❌ | ✅ | ✅ | ✅ | ✅ |
| **会议记录** | ✅ | ❌ | - | ❌ | - | ✅ |
| **平台** | macOS | macOS, iOS | macOS, Win, iOS, Android | macOS, Win, iOS, Android | macOS | macOS, Win, iOS |

## 安装

### Homebrew

```bash
brew tap alan890104/sumi
brew install --cask sumi
```

### 下载安装

1. 从 [GitHub Releases](https://github.com/alan890104/sumi/releases/latest) 下载最新的 DMG。
2. 打开 DMG，把 Sumi 拖进 `/Applications`。
3. 这个 App 还没有 Apple 公证，macOS 第一次会拦住。先运行这个：

   ```bash
   xattr -cr /Applications/Sumi.app
   ```

4. 第一次打开：给麦克风权限，然后到系统设置 > 隐私与安全 > 辅助功能 开启 Sumi。后者是自动粘贴功能需要的。

### 从源码编译

需要 [Rust](https://rustup.rs/) 和 `cargo install tauri-cli --version "^2"`。

macOS：

```bash
git clone https://github.com/alan890104/sumi.git
cd sumi
cargo tauri dev      # 开发模式
cargo tauri build    # 正式编译（输出 .dmg）
```

Windows：

Metal 是 macOS 专属的，在 Windows 上需要关掉：

```bash
# 纯 CPU
cargo tauri dev --no-default-features

# 搭配 NVIDIA CUDA（需要 CUDA Toolkit、LLVM、Ninja、CMake）
bash dev-cuda.sh
bash dev-cuda.sh --release
```

## 许可证

GPLv3
