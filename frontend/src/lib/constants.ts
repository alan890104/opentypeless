import type { CloudProvider, SttProvider, MatchType } from './types';

// ── Default hotkeys (must match backend Settings::default) ──

export const DEFAULT_HOTKEY = 'Alt+KeyV';
export const DEFAULT_EDIT_HOTKEY = 'Alt+KeyE';
export const DEFAULT_MEETING_HOTKEY = 'Alt+KeyM';

// ── Modifier display symbols (platform-aware) ──

const isMac = typeof navigator !== 'undefined' && navigator.platform?.toLowerCase().includes('mac');

export const MODIFIER_SYMBOLS: Record<string, string> = isMac
  ? { Alt: '⌥', Control: '⌃', Shift: '⇧', Super: '⌘' }
  : { Alt: 'Alt', Control: 'Ctrl', Shift: 'Shift', Super: 'Win' };

export { isMac };

export function formatHotkeyDisplay(hotkeyStr: string): string {
  return hotkeyStr
    .split('+')
    .map((p) => MODIFIER_SYMBOLS[p] ?? p.replace(/^Key/, '').replace(/^Digit/, ''))
    .join(' ');
}

export function hotkeyToParts(hotkeyStr: string): string[] {
  return hotkeyStr.split('+').map((p) => {
    if (MODIFIER_SYMBOLS[p]) return MODIFIER_SYMBOLS[p];
    return p.replace(/^Key/, '').replace(/^Digit/, '');
  });
}

// ── Cloud providers (Polish) ──

export interface CloudModelOption {
  id: string;
  name: string;
}

export interface CloudProviderMeta {
  models: CloudModelOption[];
  apiKeyUrl?: string;
}

export const CLOUD_PROVIDERS: Record<CloudProvider, CloudProviderMeta> = {
  github_models: {
    models: [
      { id: 'openai/gpt-5-nano', name: 'GPT-5 Nano' },
      { id: 'openai/gpt-5-mini', name: 'GPT-5 Mini' },
      { id: 'openai/gpt-4.1-mini', name: 'GPT-4.1 Mini' },
      { id: 'openai/gpt-4o', name: 'GPT-4o' },
      { id: 'microsoft/Phi-4', name: 'Phi-4' },
    ],
    apiKeyUrl: 'https://github.com/settings/personal-access-tokens/new',
  },
  groq: {
    models: [
      { id: 'qwen/qwen3-32b', name: 'Qwen 3 32B' },
      { id: 'openai/gpt-oss-120b', name: 'GPT-oss 120B' },
      { id: 'llama-3.3-70b-versatile', name: 'Llama 3.3 70B' },
    ],
    apiKeyUrl: 'https://console.groq.com/keys',
  },
  open_router: {
    models: [],
    apiKeyUrl: 'https://openrouter.ai/settings/keys',
  },
  open_ai: {
    models: [],
    apiKeyUrl: 'https://platform.openai.com/api-keys',
  },
  gemini: {
    models: [],
    apiKeyUrl: 'https://aistudio.google.com/apikey',
  },
  samba_nova: {
    models: [
      { id: 'Qwen3-32B', name: 'Qwen 3 32B' },
      { id: 'DeepSeek-V3.1', name: 'DeepSeek V3.1' },
      { id: 'Meta-Llama-3.3-70B-Instruct', name: 'Llama 3.3 70B' },
      { id: 'Llama-4-Maverick-17B-128E-Instruct', name: 'Llama 4 Maverick' },
      { id: 'DeepSeek-R1-0528', name: 'DeepSeek R1' },
    ],
    apiKeyUrl: 'https://cloud.sambanova.ai/apis',
  },
  custom: {
    models: [],
  },
};

// ── Cloud providers (STT) ──

export interface SttProviderMeta {
  model: { id: string; name: string } | null;
  apiKeyUrl?: string;
}

export const STT_CLOUD_PROVIDERS: Record<SttProvider, SttProviderMeta> = {
  deepgram: {
    model: { id: 'whisper', name: 'Whisper' },
    apiKeyUrl: 'https://console.deepgram.com/api-keys',
  },
  groq: {
    model: { id: 'whisper-large-v3-turbo', name: 'Whisper' },
    apiKeyUrl: 'https://console.groq.com/keys',
  },
  open_ai: {
    model: { id: 'whisper-1', name: 'Whisper' },
    apiKeyUrl: 'https://platform.openai.com/api-keys',
  },
  azure: {
    model: null,
    apiKeyUrl: 'https://ai.azure.com/nextgen',
  },
  custom: {
    model: { id: 'whisper', name: 'Whisper' },
  },
};

// ── Provider display names ──

export const CLOUD_PROVIDER_LABELS: Record<CloudProvider, string> = {
  github_models: 'GitHub Models',
  groq: 'Groq',
  open_router: 'OpenRouter',
  open_ai: 'OpenAI',
  gemini: 'Gemini',
  samba_nova: 'SambaNova',
  custom: 'Custom',
};

export const STT_PROVIDER_LABELS: Record<SttProvider, string> = {
  deepgram: 'Deepgram',
  groq: 'Groq',
  open_ai: 'OpenAI',
  azure: 'Azure',
  custom: 'Custom',
};

// ── Rule icon SVGs ──

export const RULE_ICON_SVG: Record<string, string> = {
  gmail:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#EA4335" d="M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 0 1 0 19.366V5.457c0-2.023 2.309-3.178 3.927-1.964L5.455 4.64 12 9.548l6.545-4.91 1.528-1.145C21.69 2.28 24 3.434 24 5.457z"/></svg>',
  notion:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M4.459 4.208c.746.606 1.026.56 2.428.466l13.215-.793c.28 0 .047-.28-.046-.326L17.86 1.968c-.42-.326-.981-.7-2.055-.607L3.01 2.295c-.466.046-.56.28-.374.466zm.793 3.08v13.904c0 .747.373 1.027 1.214.98l14.523-.84c.841-.046.935-.56.935-1.167V6.354c0-.606-.233-.933-.748-.887l-15.177.887c-.56.047-.747.327-.747.933zm14.337.745c.093.42 0 .84-.42.888l-.7.14v10.264c-.608.327-1.168.514-1.635.514-.748 0-.935-.234-1.495-.933l-4.577-7.186v6.952L12.21 19s0 .84-1.168.84l-3.222.186c-.093-.186 0-.653.327-.746l.84-.233V9.854L7.822 9.76c-.094-.42.14-1.026.793-1.073l3.456-.233 4.764 7.279v-6.44l-1.215-.139c-.093-.514.28-.887.747-.933zM1.936 1.035l13.31-.98c1.634-.14 2.055-.047 3.082.7l4.249 2.986c.7.513.934.653.934 1.213v16.378c0 1.026-.373 1.634-1.68 1.726l-15.458.934c-.98.047-1.448-.093-1.962-.747l-3.129-4.06c-.56-.747-.793-1.306-.793-1.96V2.667c0-.839.374-1.54 1.447-1.632z"/></svg>',
  chrome:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#4285F4" d="M12 0C8.21 0 4.831 1.757 2.632 4.501l3.953 6.848A5.454 5.454 0 0 1 12 6.545h10.691A12 12 0 0 0 12 0zM1.931 5.47A11.943 11.943 0 0 0 0 12c0 6.012 4.42 10.991 10.189 11.864l3.953-6.847a5.45 5.45 0 0 1-6.865-2.29zm13.342 2.166a5.446 5.446 0 0 1 1.45 7.09l.002.001h-.002l-5.344 9.257c.206.01.413.016.621.016 6.627 0 12-5.373 12-12 0-1.54-.29-3.011-.818-4.364zM12 16.364a4.364 4.364 0 1 1 0-8.728 4.364 4.364 0 0 1 0 8.728Z"/></svg>',
  slack:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#4A154B" d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zM6.313 15.165a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313zM8.834 5.042a2.528 2.528 0 0 1-2.521-2.52A2.528 2.528 0 0 1 8.834 0a2.528 2.528 0 0 1 2.521 2.522v2.52H8.834zM8.834 6.313a2.528 2.528 0 0 1 2.521 2.521 2.528 2.528 0 0 1-2.521 2.521H2.522A2.528 2.528 0 0 1 0 8.834a2.528 2.528 0 0 1 2.522-2.521h6.312zM18.956 8.834a2.528 2.528 0 0 1 2.522-2.521A2.528 2.528 0 0 1 24 8.834a2.528 2.528 0 0 1-2.522 2.521h-2.522V8.834zM17.688 8.834a2.528 2.528 0 0 1-2.523 2.521 2.527 2.527 0 0 1-2.52-2.521V2.522A2.527 2.527 0 0 1 15.165 0a2.528 2.528 0 0 1 2.523 2.522v6.312zM15.165 18.956a2.528 2.528 0 0 1 2.523 2.522A2.528 2.528 0 0 1 15.165 24a2.527 2.527 0 0 1-2.52-2.522v-2.522h2.52zM15.165 17.688a2.527 2.527 0 0 1-2.52-2.523 2.526 2.526 0 0 1 2.52-2.52h6.313A2.527 2.527 0 0 1 24 15.165a2.528 2.528 0 0 1-2.522 2.523h-6.313z"/></svg>',
  discord:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#5865F2" d="M20.317 4.3698a19.7913 19.7913 0 00-4.8851-1.5152.0741.0741 0 00-.0785.0371c-.211.3753-.4447.8648-.6083 1.2495-1.8447-.2762-3.68-.2762-5.4868 0-.1636-.3933-.4058-.8742-.6177-1.2495a.077.077 0 00-.0785-.037 19.7363 19.7363 0 00-4.8852 1.515.0699.0699 0 00-.0321.0277C.5334 9.0458-.319 13.5799.0992 18.0578a.0824.0824 0 00.0312.0561c2.0528 1.5076 4.0413 2.4228 5.9929 3.0294a.0777.0777 0 00.0842-.0276c.4616-.6304.8731-1.2952 1.226-1.9942a.076.076 0 00-.0416-.1057c-.6528-.2476-1.2743-.5495-1.8722-.8923a.077.077 0 01-.0076-.1277c.1258-.0943.2517-.1923.3718-.2914a.0743.0743 0 01.0776-.0105c3.9278 1.7933 8.18 1.7933 12.0614 0a.0739.0739 0 01.0785.0095c.1202.099.246.1981.3728.2924a.077.077 0 01-.0066.1276 12.2986 12.2986 0 01-1.873.8914.0766.0766 0 00-.0407.1067c.3604.698.7719 1.3628 1.225 1.9932a.076.076 0 00.0842.0286c1.961-.6067 3.9495-1.5219 6.0023-3.0294a.077.077 0 00.0313-.0552c.5004-5.177-.8382-9.6739-3.5485-13.6604a.061.061 0 00-.0312-.0286zM8.02 15.3312c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9555-2.4189 2.157-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.9555 2.4189-2.1569 2.4189zm7.9748 0c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9554-2.4189 2.1569-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.946 2.4189-2.1568 2.4189Z"/></svg>',
  vscode:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#007ACC" d="M23.15 2.587L18.21.21a1.494 1.494 0 0 0-1.705.29l-9.46 8.63-4.12-3.128a.999.999 0 0 0-1.276.057L.327 7.261A1 1 0 0 0 .326 8.74L3.899 12 .326 15.26a1 1 0 0 0 .001 1.479L1.65 17.94a.999.999 0 0 0 1.276.057l4.12-3.128 9.46 8.63a1.492 1.492 0 0 0 1.704.29l4.942-2.377A1.5 1.5 0 0 0 24 20.06V3.939a1.5 1.5 0 0 0-.85-1.352zm-5.146 14.861L10.826 12l7.178-5.448v10.896z"/></svg>',
  firefox:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#FF7139" d="M20.452 3.445a11.002 11.002 0 00-2.482-1.908C16.944.997 15.098.093 12.477.032c-.734-.017-1.457.03-2.174.144-.72.114-1.398.292-2.118.56-1.017.377-1.996.975-2.574 1.554.583-.349 1.476-.733 2.55-.992a10.083 10.083 0 013.729-.167c2.341.34 4.178 1.381 5.48 2.625a8.066 8.066 0 011.298 1.587c1.468 2.382 1.33 5.376.184 7.142-.85 1.312-2.67 2.544-4.37 2.53-.583-.023-1.438-.152-2.25-.566-2.629-1.343-3.021-4.688-1.118-6.306-.632-.136-1.82.13-2.646 1.363-.742 1.107-.7 2.816-.242 4.028a6.473 6.473 0 01-.59-1.895 7.695 7.695 0 01.416-3.845A8.212 8.212 0 019.45 5.399c.896-1.069 1.908-1.72 2.75-2.005-.54-.471-1.411-.738-2.421-.767C8.31 2.583 6.327 3.061 4.7 4.41a8.148 8.148 0 00-1.976 2.414c-.455.836-.691 1.659-.697 1.678.122-1.445.704-2.994 1.248-4.055-.79.413-1.827 1.668-2.41 3.042C.095 9.37-.2 11.608.14 13.989c.966 5.668 5.9 9.982 11.843 9.982C18.62 23.971 24 18.591 24 11.956a11.93 11.93 0 00-3.548-8.511z"/></svg>',
  github:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>',
  twitter:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932ZM17.61 20.644h2.039L6.486 3.24H4.298Z"/></svg>',
  youtube:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#FF0000" d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>',
  telegram:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#26A5E4" d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.48.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z"/></svg>',
  whatsapp:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#25D366" d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413Z"/></svg>',
  wechat:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#07C160" d="M8.691 2.188C3.891 2.188 0 5.476 0 9.53c0 2.212 1.17 4.203 3.002 5.55a.59.59 0 0 1 .213.665l-.39 1.48c-.019.07-.048.141-.048.213 0 .163.13.295.29.295a.326.326 0 0 0 .167-.054l1.903-1.114a.864.864 0 0 1 .717-.098 10.16 10.16 0 0 0 2.837.403c.276 0 .543-.027.811-.05-.857-2.578.157-4.972 1.932-6.446 1.703-1.415 3.882-1.98 5.853-1.838-.576-3.583-4.196-6.348-8.596-6.348zM5.785 5.991c.642 0 1.162.529 1.162 1.18a1.17 1.17 0 0 1-1.162 1.178A1.17 1.17 0 0 1 4.623 7.17c0-.651.52-1.18 1.162-1.18zm5.813 0c.642 0 1.162.529 1.162 1.18a1.17 1.17 0 0 1-1.162 1.178 1.17 1.17 0 0 1-1.162-1.178c0-.651.52-1.18 1.162-1.18zm5.34 2.867c-1.797-.052-3.746.512-5.28 1.786-1.72 1.428-2.687 3.72-1.78 6.22.942 2.453 3.666 4.229 6.884 4.229.826 0 1.622-.12 2.361-.336a.722.722 0 0 1 .598.082l1.584.926a.272.272 0 0 0 .14.047c.134 0 .24-.111.24-.247 0-.06-.023-.12-.038-.177l-.327-1.233a.582.582 0 0 1-.023-.156.49.49 0 0 1 .201-.398C23.024 18.48 24 16.82 24 14.98c0-3.21-2.931-5.837-6.656-6.088V8.89c-.135-.01-.27-.027-.407-.03zm-2.53 3.274c.535 0 .969.44.969.982a.976.976 0 0 1-.969.983.976.976 0 0 1-.969-.983c0-.542.434-.982.97-.982zm4.844 0c.535 0 .969.44.969.982a.976.976 0 0 1-.969.983.976.976 0 0 1-.969-.983c0-.542.434-.982.969-.982z"/></svg>',
  line: '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#00C300" d="M19.365 9.863c.349 0 .63.285.63.631 0 .345-.281.63-.63.63H17.61v1.125h1.755c.349 0 .63.283.63.63 0 .344-.281.629-.63.629h-2.386c-.345 0-.627-.285-.627-.629V8.108c0-.345.282-.63.63-.63h2.386c.346 0 .627.285.627.63 0 .349-.281.63-.63.63H17.61v1.125h1.755zm-3.855 3.016c0 .27-.174.51-.432.596-.064.021-.133.031-.199.031-.211 0-.391-.09-.51-.25l-2.443-3.317v2.94c0 .344-.279.629-.631.629-.346 0-.626-.285-.626-.629V8.108c0-.27.173-.51.43-.595.06-.023.136-.033.194-.033.195 0 .375.104.495.254l2.462 3.33V8.108c0-.345.282-.63.63-.63.345 0 .63.285.63.63v4.771zm-5.741 0c0 .344-.282.629-.631.629-.345 0-.627-.285-.627-.629V8.108c0-.345.282-.63.63-.63.346 0 .628.285.628.63v4.771zm-2.466.629H4.917c-.345 0-.63-.285-.63-.629V8.108c0-.345.285-.63.63-.63.348 0 .63.285.63.63v4.141h1.756c.348 0 .629.283.629.63 0 .344-.282.629-.629.629M24 10.314C24 4.943 18.615.572 12 .572S0 4.943 0 10.314c0 4.811 4.27 8.842 10.035 9.608.391.082.923.258 1.058.59.12.301.079.766.038 1.08l-.164 1.02c-.045.301-.24 1.186 1.049.645 1.291-.539 6.916-4.078 9.436-6.975C23.176 14.393 24 12.458 24 10.314"/></svg>',
  figma:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#F24E1E" d="M15.852 8.981h-4.588V0h4.588c2.476 0 4.49 2.014 4.49 4.49s-2.014 4.491-4.49 4.491zM12.735 7.51h3.117c1.665 0 3.019-1.355 3.019-3.019s-1.355-3.019-3.019-3.019h-3.117V7.51zm0 1.471H8.148c-2.476 0-4.49-2.014-4.49-4.49S5.672 0 8.148 0h4.588v8.981zm-4.587-7.51c-1.665 0-3.019 1.355-3.019 3.019s1.354 3.02 3.019 3.02h3.117V1.471H8.148zm4.587 15.019H8.148c-2.476 0-4.49-2.014-4.49-4.49s2.014-4.49 4.49-4.49h4.588v8.98zM8.148 8.981c-1.665 0-3.019 1.355-3.019 3.019s1.355 3.019 3.019 3.019h3.117V8.981H8.148zM8.172 24c-2.489 0-4.515-2.014-4.515-4.49s2.014-4.49 4.49-4.49h4.588v4.441c0 2.503-2.047 4.539-4.563 4.539zm-.024-7.51a3.023 3.023 0 0 0-3.019 3.019c0 1.665 1.365 3.019 3.044 3.019 1.705 0 3.093-1.376 3.093-3.068v-2.97H8.148zm7.704 0h-.098c-2.476 0-4.49-2.014-4.49-4.49s2.014-4.49 4.49-4.49h.098c2.476 0 4.49 2.014 4.49 4.49s-2.014 4.49-4.49 4.49zm-.097-7.509c-1.665 0-3.019 1.355-3.019 3.019s1.355 3.019 3.019 3.019h.098c1.665 0 3.019-1.355 3.019-3.019s-1.355-3.019-3.019-3.019h-.098z"/></svg>',
  apple:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M12.152 6.896c-.948 0-2.415-1.078-3.96-1.04-2.04.027-3.91 1.183-4.961 3.014-2.117 3.675-.546 9.103 1.519 12.09 1.013 1.454 2.208 3.09 3.792 3.039 1.52-.065 2.09-.987 3.935-.987 1.831 0 2.35.987 3.96.948 1.637-.026 2.676-1.48 3.676-2.948 1.156-1.688 1.636-3.325 1.662-3.415-.039-.013-3.182-1.221-3.22-4.857-.026-3.04 2.48-4.494 2.597-4.559-1.429-2.09-3.623-2.324-4.39-2.376-2-.156-3.675 1.09-4.61 1.09zM15.53 3.83c.843-1.012 1.4-2.427 1.245-3.83-1.207.052-2.662.805-3.532 1.818-.78.896-1.454 2.338-1.273 3.714 1.338.104 2.715-.688 3.559-1.701"/></svg>',
  word: '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#2B579A" d="M23.004 1.5q.41 0 .703.293t.293.703v19.008q0 .41-.293.703t-.703.293H6.996q-.41 0-.703-.293T6 21.504V18H.996q-.41 0-.703-.293T0 17.004V6.996q0-.41.293-.703T.996 6H6V2.496q0-.41.293-.703t.703-.293zM6.035 11.203l1.442 4.735h1.64l1.57-7.876H9.036l-.937 4.653-1.325-4.5H5.38l-1.406 4.523-.938-4.675H1.312l1.57 7.874h1.641zM22.5 21v-3h-15v3zm0-4.5v-3.75H12v3.75zm0-5.25V7.5H12v3.75zm0-5.25V3h-15v3Z"/></svg>',
  excel:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#217346" d="M23 1.5q.41 0 .7.3.3.29.3.7v19q0 .41-.3.7-.29.3-.7.3H7q-.41 0-.7-.3-.3-.29-.3-.7V18H1q-.41 0-.7-.3-.3-.29-.3-.7V7q0-.41.3-.7Q.58 6 1 6h5V2.5q0-.41.3-.7.29-.3.7-.3zM6 13.28l1.42 2.66h2.14l-2.38-3.87 2.34-3.8H7.46l-1.3 2.4-.05.08-.04.09-.64-1.28-.66-1.29H2.59l2.27 3.82-2.48 3.85h2.16zM14.25 21v-3H7.5v3zm0-4.5v-3.75H12v3.75zm0-5.25V7.5H12v3.75zm0-5.25V3H7.5v3zm8.25 15v-3h-6.75v3zm0-4.5v-3.75h-6.75v3.75zm0-5.25V7.5h-6.75v3.75zm0-5.25V3h-6.75v3Z"/></svg>',
  teams:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#6264A7" d="M20.625 8.127q-.55 0-1.025-.205-.475-.205-.832-.563-.358-.357-.563-.832Q18 6.053 18 5.502q0-.54.205-1.02t.563-.837q.357-.358.832-.563.474-.205 1.025-.205.54 0 1.02.205t.837.563q.358.357.563.837.205.48.205 1.02 0 .55-.205 1.025-.205.475-.563.832-.357.358-.837.563-.48.205-1.02.205zm0-3.75q-.469 0-.797.328-.328.328-.328.797 0 .469.328.797.328.328.797.328.469 0 .797-.328.328-.328.328-.797 0-.469-.328-.797-.328-.328-.797-.328zM24 10.002v5.578q0 .774-.293 1.46-.293.685-.803 1.194-.51.51-1.195.803-.686.293-1.459.293-.445 0-.908-.105-.463-.106-.85-.329-.293.95-.855 1.729-.563.78-1.319 1.336-.756.557-1.67.861-.914.305-1.898.305-1.148 0-2.162-.398-1.014-.399-1.805-1.102-.79-.703-1.312-1.664t-.674-2.086h-5.8q-.411 0-.704-.293T0 16.881V6.873q0-.41.293-.703t.703-.293h8.59q-.34-.715-.34-1.5 0-.727.275-1.365.276-.639.75-1.114.475-.474 1.114-.75.638-.275 1.365-.275t1.365.275q.639.276 1.114.75.474.475.75 1.114.275.638.275 1.365t-.275 1.365q-.276.639-.75 1.113-.475.475-1.114.75-.638.276-1.365.276-.188 0-.375-.024-.188-.023-.375-.058v1.078h10.875q.469 0 .797.328.328.328.328.797zM12.75 2.373q-.41 0-.78.158-.368.158-.638.434-.27.275-.428.639-.158.363-.158.773 0 .41.158.78.159.368.428.638.27.27.639.428.369.158.779.158.41 0 .773-.158.364-.159.64-.428.274-.27.433-.639.158-.369.158-.779 0-.41-.158-.773-.159-.364-.434-.64-.275-.275-.639-.433-.363-.158-.773-.158zM6.937 9.814h2.25V7.94H2.814v1.875h2.25v6h1.875zm10.313 7.313v-6.75H12v6.504q0 .41-.293.703t-.703.293H8.309q.152.809.556 1.5.405.691.985 1.19.58.497 1.318.779.738.281 1.582.281.926 0 1.746-.352.82-.351 1.436-.966.615-.616.966-1.43.352-.815.352-1.752zm5.25-1.547v-5.203h-3.75v6.855q.305.305.691.452.387.146.809.146.469 0 .879-.176.41-.175.715-.48.304-.305.48-.715t.176-.879Z"/></svg>',
  linear:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#5E6AD2" d="M2.886 4.18A11.982 11.982 0 0 1 11.99 0C18.624 0 24 5.376 24 12.009c0 3.64-1.62 6.903-4.18 9.105L2.887 4.18ZM1.817 5.626l16.556 16.556c-.524.33-1.075.62-1.65.866L.951 7.277c.247-.575.537-1.126.866-1.65ZM.322 9.163l14.515 14.515c-.71.172-1.443.282-2.195.322L0 11.358a12 12 0 0 1 .322-2.195Zm-.17 4.862 9.823 9.824a12.02 12.02 0 0 1-9.824-9.824Z"/></svg>',
  anthropic:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#D97757" d="M13.827 3.52h3.603L24 20.48h-3.603l-6.57-16.96zm-7.258 0h3.767L16.906 20.48h-3.674l-1.665-4.32H5.212L3.548 20.48H0l6.57-16.96zm1.04 3.79L5.2 13.18h4.818l-2.41-5.87z"/></svg>',
  google:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>',
  openai:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#000" d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z"/></svg>',
  claude:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#D97757" d="m4.7144 15.9555 4.7174-2.6471.079-.2307-.079-.1275h-.2307l-.7893-.0486-2.6956-.0729-2.3375-.0971-2.2646-.1214-.5707-.1215-.5343-.7042.0546-.3522.4797-.3218.686.0608 1.5179.1032 2.2767.1578 1.6514.0972 2.4468.255h.3886l.0546-.1579-.1336-.0971-.1032-.0972L6.973 9.8356l-2.55-1.6879-1.3356-.9714-.7225-.4918-.3643-.4614-.1578-1.0078.6557-.7225.8803.0607.2246.0607.8925.686 1.9064 1.4754 2.4893 1.8336.3643.3035.1457-.1032.0182-.0728-.164-.2733-1.3539-2.4467-1.445-2.4893-.6435-1.032-.17-.6194c-.0607-.255-.1032-.4674-.1032-.7285L6.287.1335 6.6997 0l.9957.1336.419.3642.6192 1.4147 1.0018 2.2282 1.5543 3.0296.4553.8985.2429.8318.091.255h.1579v-.1457l.1275-1.706.2368-2.0947.2307-2.6957.0789-.7589.3764-.9107.7468-.4918.5828.2793.4797.686-.0668.4433-.2853 1.8517-.5586 2.9021-.3643 1.9429h.2125l.2429-.2429.9835-1.3053 1.6514-2.0643.7286-.8196.85-.9046.5464-.4311h1.0321l.759 1.1293-.34 1.1657-1.0625 1.3478-.8804 1.1414-1.2628 1.7-.7893 1.36.0729.1093.1882-.0183 2.8535-.607 1.5421-.2794 1.8396-.3157.8318.3886.091.3946-.3278.8075-1.967.4857-2.3072.4614-3.4364.8136-.0425.0304.0486.0607 1.5482.1457.6618.0364h1.621l3.0175.2247.7892.522.4736.6376-.079.4857-1.2142.6193-1.6393-.3886-3.825-.9107-1.3113-.3279h-.1822v.1093l1.0929 1.0686 2.0035 1.8092 2.5075 2.3314.1275.5768-.3218.4554-.34-.0486-2.2039-1.6575-.85-.7468-1.9246-1.621h-.1275v.17l.4432.6496 2.3436 3.5214.1214 1.0807-.17.3521-.6071.2125-.6679-.1214-1.3721-1.9246L14.38 17.959l-1.1414-1.9428-.1397.079-.674 7.2552-.3156.3703-.7286.2793-.6071-.4614-.3218-.7468.3218-1.4753.3886-1.9246.3157-1.53.2853-1.9004.17-.6314-.0121-.0425-.1397.0182-1.4328 1.9672-2.1796 2.9446-1.7243 1.8456-.4128.164-.7164-.3704.0667-.6618.4008-.5889 2.386-3.0357 1.4389-1.882.929-1.0868-.0062-.1579h-.0546l-6.3385 4.1164-1.1293.1457-.4857-.4554.0608-.7467.2307-.2429 1.9064-1.3114Z"/></svg>',
  cursor:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#F54E00" d="M11.503.131 1.891 5.678a.84.84 0 0 0-.42.726v11.188c0 .3.162.575.42.724l9.609 5.55a1 1 0 0 0 .998 0l9.61-5.55a.84.84 0 0 0 .42-.724V6.404a.84.84 0 0 0-.42-.726L12.497.131a1.01 1.01 0 0 0-.996 0M2.657 6.338h18.55c.263 0 .43.287.297.515L12.23 22.918c-.062.107-.229.064-.229-.06V12.335a.59.59 0 0 0-.295-.51l-9.11-5.257c-.109-.063-.064-.23.061-.23"/></svg>',
  gemini:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#8E75B2" d="M11.04 19.32Q12 21.51 12 24q0-2.49.93-4.68.96-2.19 2.58-3.81t3.81-2.55Q21.51 12 24 12q-2.49 0-4.68-.93a12.3 12.3 0 0 1-3.81-2.58 12.3 12.3 0 0 1-2.58-3.81Q12 2.49 12 0q0 2.49-.96 4.68-.93 2.19-2.55 3.81a12.3 12.3 0 0 1-3.81 2.58Q2.49 12 0 12q2.49 0 4.68.96 2.19.93 3.81 2.55t2.55 3.81"/></svg>',
  antigravity:
    '<svg viewBox="10 16 92 82" xmlns="http://www.w3.org/2000/svg"><path fill="#3186FF" d="M89.7 93.7C94.4 97.2 101.4 94.9 94.9 88.4C75.7 69.8 79.8 18.4 55.9 18.4C31.9 18.4 36 69.8 16.8 88.4C9.8 95.4 17.4 97.2 22 93.7C40.1 81.4 39 59.9 55.9 59.9C72.8 59.9 71.6 81.4 89.7 93.7Z"/></svg>',
  robot:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2M7.5 13A2.5 2.5 0 0 0 5 15.5 2.5 2.5 0 0 0 7.5 18a2.5 2.5 0 0 0 2.5-2.5A2.5 2.5 0 0 0 7.5 13m9 0a2.5 2.5 0 0 0-2.5 2.5 2.5 2.5 0 0 0 2.5 2.5 2.5 2.5 0 0 0 2.5-2.5 2.5 2.5 0 0 0-2.5-2.5z"/></svg>',
  terminal:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M20 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2zm0 14H4V8h16v10zm-9-2h6v-2h-6v2zM7.5 12.5 5.4 10.4l1.4-1.4 3.5 3.5-3.5 3.5-1.4-1.4 2.1-2.1z"/></svg>',
  safari:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#006CFF" d="M12 24C5.373 24 0 18.627 0 12S5.373 0 12 0s12 5.373 12 12-5.373 12-12 12zm0-2c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10zm3.14-14.608l-5.476 2.468-2.832 5.672 5.476-2.468 2.832-5.672zM12 13a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/></svg>',
  xcode:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#147EFB" d="M4.256 18.11L2.61 19.756l1.636 1.635 1.645-1.645zM8.97 13.4l-3.4 3.395 1.636 1.636 3.4-3.395zM12 2.002l-1.6 1.6 8 8 1.6-1.6zm6.746 12.488l-3.4-3.395-1.635 1.636 3.4 3.395zm3.645 3.643l-1.646-1.645-1.635 1.636 1.645 1.645z"/></svg>',
};

export const RULE_ICON_KEYWORDS: { keys: string[]; icon: string }[] = [
  { keys: ['claude code'], icon: 'claude' },
  { keys: ['claude'], icon: 'claude' },
  { keys: ['gemini cli'], icon: 'gemini' },
  { keys: ['gemini'], icon: 'gemini' },
  { keys: ['codex cli'], icon: 'openai' },
  { keys: ['codex'], icon: 'openai' },
  { keys: ['aider'], icon: 'robot' },
  { keys: ['neovim', 'nvim'], icon: 'vscode' },
  { keys: ['vim'], icon: 'vscode' },
  { keys: ['emacs'], icon: 'vscode' },
  { keys: ['helix', 'hx'], icon: 'vscode' },
  { keys: ['gmail', 'mail.google'], icon: 'gmail' },
  { keys: ['notion'], icon: 'notion' },
  { keys: ['chrome', 'google chrome', 'googlechrome'], icon: 'chrome' },
  { keys: ['slack'], icon: 'slack' },
  { keys: ['discord'], icon: 'discord' },
  { keys: ['vscode', 'visual studio code', 'code.app'], icon: 'vscode' },
  { keys: ['cursor'], icon: 'cursor' },
  { keys: ['antigravity'], icon: 'antigravity' },
  { keys: ['firefox'], icon: 'firefox' },
  { keys: ['github'], icon: 'github' },
  { keys: ['twitter', 'x.com', 'x (twitter)', 'tweetdeck'], icon: 'twitter' },
  { keys: ['youtube'], icon: 'youtube' },
  { keys: ['telegram'], icon: 'telegram' },
  { keys: ['whatsapp'], icon: 'whatsapp' },
  { keys: ['wechat', 'weixin', '微信'], icon: 'wechat' },
  { keys: ['line'], icon: 'line' },
  { keys: ['figma'], icon: 'figma' },
  { keys: ['terminal', 'iterm'], icon: 'terminal' },
  { keys: ['safari'], icon: 'safari' },
  { keys: ['xcode'], icon: 'xcode' },
  {
    keys: ['notes', 'pages', 'keynote', 'finder'],
    icon: 'apple',
  },
  { keys: ['word', 'winword'], icon: 'word' },
  { keys: ['excel'], icon: 'excel' },
  { keys: ['teams'], icon: 'teams' },
  { keys: ['linear'], icon: 'linear' },
];

export const FALLBACK_ICON_SVG: Record<MatchType | string, string> = {
  url: '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M3.9 12c0-1.71 1.39-3.1 3.1-3.1h4V7H7c-2.76 0-5 2.24-5 5s2.24 5 5 5h4v-1.9H7c-1.71 0-3.1-1.39-3.1-3.1zM8 13h8v-2H8v2zm9-6h-4v1.9h4c1.71 0 3.1 1.39 3.1 3.1s-1.39 3.1-3.1 3.1h-4V17h4c2.76 0 5-2.24 5-5s-2.24-5-5-5z"/></svg>',
  app_name:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M4 4h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zm0 2v12h16V6H4zm1 1h5v2H5V7z"/></svg>',
  bundle_id:
    '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="currentColor" d="M21 16.5c0 .38-.21.71-.53.88l-7.9 4.44c-.16.12-.36.18-.57.18-.21 0-.41-.06-.57-.18l-7.9-4.44A.991.991 0 0 1 3 16.5v-9c0-.38.21-.71.53-.88l7.9-4.44c.16-.12.36-.18.57-.18.21 0 .41.06.57.18l7.9 4.44c.32.17.53.5.53.88v9zM12 4.15L6.04 7.5 12 10.85l5.96-3.35L12 4.15zM5 15.91l6 3.38v-6.71L5 9.21v6.7zm14 0v-6.7l-6 3.37v6.71l6-3.38z"/></svg>',
};

export function getRuleIcon(rule: { name?: string; match_value?: string; match_type?: string; icon?: string; alt_matches?: { match_value: string }[] }): string {
  // Use explicit icon if set
  if (rule.icon && RULE_ICON_SVG[rule.icon]) {
    return RULE_ICON_SVG[rule.icon];
  }
  // Auto-detect from name/match_value/alt_matches
  const altValues = (rule.alt_matches || []).map((a) => a.match_value || '').join(' ');
  const searchText = ((rule.name || '') + ' ' + (rule.match_value || '') + ' ' + altValues).toLowerCase();
  for (const entry of RULE_ICON_KEYWORDS) {
    for (const key of entry.keys) {
      if (searchText.includes(key.toLowerCase())) {
        return RULE_ICON_SVG[entry.icon];
      }
    }
  }
  return FALLBACK_ICON_SVG[rule.match_type || 'app_name'] || FALLBACK_ICON_SVG.app_name;
}

/** Detect icon key from rule name/match_value/alt_matches, returns undefined if no match */
export function detectRuleIconKey(rule: { name?: string; match_value?: string; alt_matches?: { match_value: string }[] }): string | undefined {
  const altValues = (rule.alt_matches || []).map((a) => a.match_value || '').join(' ');
  const searchText = ((rule.name || '') + ' ' + (rule.match_value || '') + ' ' + altValues).toLowerCase();
  for (const entry of RULE_ICON_KEYWORDS) {
    for (const key of entry.keys) {
      if (searchText.includes(key.toLowerCase())) {
        return entry.icon;
      }
    }
  }
  return undefined;
}

/** All available icon keys for the icon picker.
 *  `labelKey` = i18n key (translated at runtime); `label` = static brand name. */
export const ICON_PICKER_LIST: { key: string; label: string; labelKey?: string }[] = [
  { key: 'gmail', label: 'Gmail' },
  { key: 'notion', label: 'Notion' },
  { key: 'chrome', label: 'Chrome' },
  { key: 'safari', label: 'Safari' },
  { key: 'firefox', label: 'Firefox' },
  { key: 'slack', label: 'Slack' },
  { key: 'discord', label: 'Discord' },
  { key: 'vscode', label: 'VS Code' },
  { key: 'terminal', label: 'Terminal', labelKey: 'promptRules.iconTerminal' },
  { key: 'github', label: 'GitHub' },
  { key: 'twitter', label: 'X' },
  { key: 'youtube', label: 'YouTube' },
  { key: 'telegram', label: 'Telegram' },
  { key: 'whatsapp', label: 'WhatsApp' },
  { key: 'wechat', label: 'WeChat' },
  { key: 'line', label: 'LINE' },
  { key: 'figma', label: 'Figma' },
  { key: 'apple', label: 'Apple' },
  { key: 'xcode', label: 'Xcode' },
  { key: 'word', label: 'Word' },
  { key: 'excel', label: 'Excel' },
  { key: 'teams', label: 'Teams' },
  { key: 'linear', label: 'Linear' },
  { key: 'claude', label: 'Claude' },
  { key: 'anthropic', label: 'Anthropic' },
  { key: 'cursor', label: 'Cursor' },
  { key: 'gemini', label: 'Gemini' },
  { key: 'antigravity', label: 'Antigravity' },
  { key: 'google', label: 'Google' },
  { key: 'openai', label: 'OpenAI' },
  { key: 'robot', label: 'Robot', labelKey: 'promptRules.iconRobot' },
];

// ── STT local models ──

import type { WhisperModelId, SystemInfo } from './types';

export interface SttModelMeta {
  id: WhisperModelId;
  nameKey: string;
  descKey: string;
  languagesKey: string;
  sizeBytes: number;
}

export const STT_MODELS: SttModelMeta[] = [
  { id: 'large_v3_turbo', nameKey: 'sttModel.largeV3Turbo.name', descKey: 'sttModel.largeV3Turbo.desc', languagesKey: 'sttModel.largeV3Turbo.languages', sizeBytes: 1_620_000_000 },
  { id: 'large_v3_turbo_q5', nameKey: 'sttModel.largeV3TurboQ5.name', descKey: 'sttModel.largeV3TurboQ5.desc', languagesKey: 'sttModel.largeV3TurboQ5.languages', sizeBytes: 547_000_000 },
  { id: 'base', nameKey: 'sttModel.base.name', descKey: 'sttModel.base.desc', languagesKey: 'sttModel.base.languages', sizeBytes: 148_000_000 },
  { id: 'large_v3_turbo_zh_tw', nameKey: 'sttModel.largeV3TurboZhTw.name', descKey: 'sttModel.largeV3TurboZhTw.desc', languagesKey: 'sttModel.largeV3TurboZhTw.languages', sizeBytes: 1_600_000_000 },
];

export function recommendSttModel(systemInfo: SystemInfo, locale: string): WhisperModelId {
  const diskGb = systemInfo.available_disk_bytes / 1_073_741_824;
  const lower = locale.toLowerCase();
  const prefersZhTw = lower.startsWith('zh-tw') || lower.startsWith('zh_tw') || lower === 'zh-hant';

  // Determine effective memory for model recommendation:
  // - Apple Silicon: use system RAM (unified memory)
  // - CUDA + discrete GPU (>= 2 GB VRAM): use GPU VRAM
  // - Otherwise: use system RAM
  const vramGb = systemInfo.gpu_vram_bytes / 1_073_741_824;
  const ramGb = systemInfo.total_ram_bytes / 1_073_741_824;
  const effectiveGb =
    systemInfo.is_apple_silicon ? ramGb :
    systemInfo.has_cuda && vramGb >= 2 ? vramGb :
    ramGb;

  if (effectiveGb >= 8 && diskGb >= 3) {
    if (prefersZhTw) return 'large_v3_turbo_zh_tw';
    return 'large_v3_turbo';
  }
  if (effectiveGb >= 4 && diskGb >= 1) return 'large_v3_turbo_q5';
  return 'base';
}

// ── STT languages ──

export const STT_LANGUAGES = [
  // ── Common ──
  { value: 'auto', label: 'Auto' },
  { value: 'zh-TW', label: '繁體中文 (Traditional Chinese)' },
  { value: 'zh-CN', label: '简体中文 (Simplified Chinese)' },
  { value: 'en', label: 'English' },
  { value: 'ja', label: '日本語 (Japanese)' },
  { value: 'ko', label: '한국어 (Korean)' },
  { value: 'yue', label: '粵語 (Cantonese)' },
  // ── All Whisper-supported languages (alphabetical) ──
  { value: 'af', label: 'Afrikaans' },
  { value: 'am', label: 'Amharic' },
  { value: 'ar', label: 'العربية (Arabic)' },
  { value: 'as', label: 'অসমীয়া (Assamese)' },
  { value: 'az', label: 'Azərbaycan (Azerbaijani)' },
  { value: 'ba', label: 'Bashkir' },
  { value: 'be', label: 'Беларуская (Belarusian)' },
  { value: 'bg', label: 'Български (Bulgarian)' },
  { value: 'bn', label: 'বাংলা (Bengali)' },
  { value: 'bo', label: 'བོད་སྐད (Tibetan)' },
  { value: 'br', label: 'Brezhoneg (Breton)' },
  { value: 'bs', label: 'Bosanski (Bosnian)' },
  { value: 'ca', label: 'Català (Catalan)' },
  { value: 'cs', label: 'Čeština (Czech)' },
  { value: 'cy', label: 'Cymraeg (Welsh)' },
  { value: 'da', label: 'Dansk (Danish)' },
  { value: 'de', label: 'Deutsch (German)' },
  { value: 'el', label: 'Ελληνικά (Greek)' },
  { value: 'es', label: 'Español (Spanish)' },
  { value: 'et', label: 'Eesti (Estonian)' },
  { value: 'eu', label: 'Euskara (Basque)' },
  { value: 'fa', label: 'فارسی (Persian)' },
  { value: 'fi', label: 'Suomi (Finnish)' },
  { value: 'fo', label: 'Føroyskt (Faroese)' },
  { value: 'fr', label: 'Français (French)' },
  { value: 'gl', label: 'Galego (Galician)' },
  { value: 'gu', label: 'ગુજરાતી (Gujarati)' },
  { value: 'ha', label: 'Hausa' },
  { value: 'haw', label: 'ʻŌlelo Hawaiʻi (Hawaiian)' },
  { value: 'he', label: 'עברית (Hebrew)' },
  { value: 'hi', label: 'हिन्दी (Hindi)' },
  { value: 'hr', label: 'Hrvatski (Croatian)' },
  { value: 'ht', label: 'Kreyòl Ayisyen (Haitian Creole)' },
  { value: 'hu', label: 'Magyar (Hungarian)' },
  { value: 'hy', label: 'Հայերեն (Armenian)' },
  { value: 'id', label: 'Bahasa Indonesia (Indonesian)' },
  { value: 'is', label: 'Íslenska (Icelandic)' },
  { value: 'it', label: 'Italiano (Italian)' },
  { value: 'jw', label: 'Basa Jawa (Javanese)' },
  { value: 'ka', label: 'ქართული (Georgian)' },
  { value: 'kk', label: 'Қазақ (Kazakh)' },
  { value: 'km', label: 'ខ្មែរ (Khmer)' },
  { value: 'kn', label: 'ಕನ್ನಡ (Kannada)' },
  { value: 'lb', label: 'Lëtzebuergesch (Luxembourgish)' },
  { value: 'ln', label: 'Lingála (Lingala)' },
  { value: 'lo', label: 'ລາວ (Lao)' },
  { value: 'lt', label: 'Lietuvių (Lithuanian)' },
  { value: 'lv', label: 'Latviešu (Latvian)' },
  { value: 'mg', label: 'Malagasy' },
  { value: 'mi', label: 'Te Reo Māori (Maori)' },
  { value: 'mk', label: 'Македонски (Macedonian)' },
  { value: 'ml', label: 'മലയാളം (Malayalam)' },
  { value: 'mn', label: 'Монгол (Mongolian)' },
  { value: 'mr', label: 'मराठी (Marathi)' },
  { value: 'ms', label: 'Bahasa Melayu (Malay)' },
  { value: 'mt', label: 'Malti (Maltese)' },
  { value: 'my', label: 'မြန်မာ (Myanmar)' },
  { value: 'ne', label: 'नेपाली (Nepali)' },
  { value: 'nl', label: 'Nederlands (Dutch)' },
  { value: 'nn', label: 'Nynorsk (Norwegian Nynorsk)' },
  { value: 'no', label: 'Norsk (Norwegian)' },
  { value: 'oc', label: 'Occitan' },
  { value: 'pa', label: 'ਪੰਜਾਬੀ (Punjabi)' },
  { value: 'pl', label: 'Polski (Polish)' },
  { value: 'ps', label: 'پښتو (Pashto)' },
  { value: 'pt', label: 'Português (Portuguese)' },
  { value: 'ro', label: 'Română (Romanian)' },
  { value: 'ru', label: 'Русский (Russian)' },
  { value: 'sa', label: 'संस्कृत (Sanskrit)' },
  { value: 'sd', label: 'سنڌي (Sindhi)' },
  { value: 'si', label: 'සිංහල (Sinhala)' },
  { value: 'sk', label: 'Slovenčina (Slovak)' },
  { value: 'sl', label: 'Slovenščina (Slovenian)' },
  { value: 'sn', label: 'ChiShona (Shona)' },
  { value: 'so', label: 'Soomaali (Somali)' },
  { value: 'sq', label: 'Shqip (Albanian)' },
  { value: 'sr', label: 'Српски (Serbian)' },
  { value: 'su', label: 'Basa Sunda (Sundanese)' },
  { value: 'sv', label: 'Svenska (Swedish)' },
  { value: 'sw', label: 'Kiswahili (Swahili)' },
  { value: 'ta', label: 'தமிழ் (Tamil)' },
  { value: 'te', label: 'తెలుగు (Telugu)' },
  { value: 'tg', label: 'Тоҷикӣ (Tajik)' },
  { value: 'th', label: 'ไทย (Thai)' },
  { value: 'tk', label: 'Türkmen (Turkmen)' },
  { value: 'tl', label: 'Tagalog' },
  { value: 'tr', label: 'Türkçe (Turkish)' },
  { value: 'tt', label: 'Татар (Tatar)' },
  { value: 'uk', label: 'Українська (Ukrainian)' },
  { value: 'ur', label: 'اردو (Urdu)' },
  { value: 'uz', label: 'Oʻzbek (Uzbek)' },
  { value: 'vi', label: 'Tiếng Việt (Vietnamese)' },
  { value: 'yi', label: 'ייִדיש (Yiddish)' },
  { value: 'yo', label: 'Yorùbá (Yoruba)' },
];

// ── History retention options ──

export const RETENTION_OPTIONS = [
  { value: 0, labelKey: 'history.retentionForever' },
  { value: 7, labelKey: 'history.retention7' },
  { value: 30, labelKey: 'history.retention30' },
  { value: 90, labelKey: 'history.retention90' },
  { value: 180, labelKey: 'history.retention180' },
  { value: 365, labelKey: 'history.retention365' },
];
