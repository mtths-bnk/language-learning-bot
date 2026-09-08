# Fala — on-device iPhone rewrite

Native SwiftUI iPhone app that keeps the language-learning-bot idea fully on-device: speak European Portuguese, get corrections, hear a reply, track progress. Nothing is sent to a cloud LLM API.

## Stack

| Piece | On-device technology |
| --- | --- |
| Speech → text | `SFSpeechRecognizer` with `requiresOnDeviceRecognition = true` (pt-PT) |
| Tutor + analysis | Apple **Foundation Models** (`LanguageModelSession`, ~3B on-device) on Apple Intelligence devices |
| Text → speech | `AVSpeechSynthesizer` with a Portuguese (`pt-PT`) voice |
| Progress | Local JSON in the app Documents directory |

If Apple Intelligence is unavailable, a small heuristic tutor still lets you practice (typed or spoken) offline.

## Requirements

- Mac with **Xcode 26+**
- **iPhone 15 Pro or later** (or another Apple Intelligence–capable device) for the full on-device LLM
- iOS 26+
- Microphone permission
- On-device Portuguese speech recognition language pack (Settings › General › Keyboard / Dictation)

## Open & run

1. Open `ios/Fala/Fala.xcodeproj` in Xcode.
2. Select your Team under Signing & Capabilities.
3. Choose your iPhone (device recommended; Foundation Models need Apple Intelligence).
4. Run (⌘R).
5. Allow microphone + speech recognition when prompted.

## How it maps from the Python app

| Python (`main.py`) | Fala iOS |
| --- | --- |
| Whisper STT | On-device Speech framework (pt-PT) |
| Ollama + Mistral dialog | Foundation Models session with the same tutor prompt shape |
| JSON error analysis | `@Generable` structured analysis (with heuristic fallback) |
| MMS / VITS TTS | System Portuguese TTS |
| `learning_progress.json` | Same concepts, stored in app Documents |
| Weakness / trend logic | Ported in `WeaknessAnalyzer.swift` |

## App flow

1. **Home** — brand-first entry, one CTA to start speaking.
2. **Practice** — hold the mic, speak Portuguese, release; Fala analyzes, replies, and speaks back.
3. **Progress** — utterance count, level, error rate, vocabulary, recent focus.

## Notes

- Deployment target is **iOS 26** because Foundation Models ships there.
- Bundle ID default: `com.fala.languagelearning` (change as needed).
- Fonts: Fraunces + Manrope (OFL), bundled under `Fala/Resources`.
