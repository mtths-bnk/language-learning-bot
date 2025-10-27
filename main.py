import time, os, json
from datetime import datetime, timedelta
from collections import Counter
t = time.process_time()
import warnings
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from rich.console import Console
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.llms import Ollama
from TextToSpeechServiceMMS import TextToSpeechService
from string import Template
print("Time elapsed to load packages: " + str(time.process_time() - t))

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

PROGRESS_FILE = "learning_progress.json"
MODEL_NAME_DIALOG = "mistral"
MODEL_NAME_ANALYSIS = "mistral"

console = Console()
t = time.process_time()
stt = whisper.load_model("base")
print("Time elapsed to load Whisper: " + str(time.process_time() - t))
t = time.process_time()
tts = TextToSpeechService()
print("Time elapsed to load TextToSpeechService: " + str(time.process_time() - t))

template = """
You are a friendly AI assistant that helps me learning Portuguese. You are precise in grammar and vocabulary. If I make a mistake, you correct me and provide an explanation and translation if needed. Keep it short and crisp. Always respond in Portuguese! 

{weakness_context}

Use the following format:
---
<A response correcting my mistakes with a quick explanation in Portuguese (max. 2 sentences).>
<Your quick response to my statement containing a follow-up question in Portuguese (max. 2 sentences). Focus on areas where I need improvement based on my learning progress.>
---
The conversation transcript is as follows:
{history}
And here is the user's statement in Portuguese: {input}
Your response:
"""

base_template = """
You are a friendly AI assistant that helps me learning Portuguese. You are precise in grammar and vocabulary. If I make a mistake, you correct me and provide an explanation and translation if needed. Keep it short and crisp. Always respond in Portuguese! 
Use the following format:
---
<A response correcting my mistakes with a quick explanation in Portuguese (max. 2 sentences).>
<Your quick response to my statement containing a follow-up question in Portuguese (max. 2 sentences).>
---
The conversation transcript is as follows:
{history}
And here is the user's statement in Portuguese: {input}
Your response:
"""

t = time.process_time()
PROMPT = PromptTemplate(input_variables=["history", "input"], template=base_template)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=Ollama(model=MODEL_NAME_DIALOG),
)
print("Time elapsed to define ConversationChain: " + str(time.process_time() - t))

analysis_system_preamble = (
    "You are a precise Portuguese language tutor. Analyze the user's Portuguese utterance and return ONLY valid JSON."
)

ANALYSIS_TEMPLATE = Template(
    """
Analyze the following Portuguese utterance from a learner and return ONLY a JSON object with this exact schema:
{
  "grammar": "string | null",
  "vocabulary": "string | null",
  "pronunciation": "string | null",
  "overall_level": "A1|A2|B1|B2|C1|C2",
  "error_count": number,
  "new_words": ["..."],
  "suggested_corrected_sentence": "string"
}
Utterance: "$text"
Return ONLY JSON. No explanations.
"""
)

def _default_progress():
    return {
        "sessions": [],
        "totals": {"utterances": 0, "errors": 0},
        "last_level": None,
        "vocabulary": []
    }

def load_progress(path: str = PROGRESS_FILE) -> dict:
    if not os.path.exists(path):
        return _default_progress()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return _default_progress()

def save_progress(progress: dict, path: str = PROGRESS_FILE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def _parse_json_loose(s: str) -> dict:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1 and end > start:
        chunk = s[start:end+1]
        try:
            return json.loads(chunk)
        except Exception:
            return {}
    return {}

def analyze_errors(text: str, llm: Ollama) -> dict:
    prompt = analysis_system_preamble + "\n\n" + ANALYSIS_TEMPLATE.substitute(text=text)
    try:
        raw = llm.invoke(prompt)
    except Exception as e:
        console.print(f"[red]Analysis LLM error: {e}")
        raw = "{}"
    parsed = _parse_json_loose(raw)

    result = {
        "grammar": parsed.get("grammar"),
        "vocabulary": parsed.get("vocabulary"),
        "pronunciation": parsed.get("pronunciation"),
        "overall_level": parsed.get("overall_level") or "B1",
        "error_count": int(parsed.get("error_count") or 0),
        "new_words": parsed.get("new_words") or [],
        "suggested_corrected_sentence": parsed.get("suggested_corrected_sentence") or "",
        "_raw": raw,
    }
    if not isinstance(result["new_words"], list):
        result["new_words"] = []
    result["new_words"] = [str(w).strip() for w in result["new_words"] if str(w).strip()]
    return result

def update_learning_progress(progress: dict, analysis: dict, user_text: str) -> dict:
    ts = datetime.utcnow().isoformat() + "Z"

    session_entry = {
        "timestamp": ts,
        "utterance": user_text,
        "analysis": {
            "error_count": analysis.get("error_count", 0),
            "overall_level": analysis.get("overall_level"),
            "grammar": analysis.get("grammar"),
            "vocabulary": analysis.get("vocabulary"),
            "pronunciation": analysis.get("pronunciation"),
            "suggested_corrected_sentence": analysis.get("suggested_corrected_sentence", "")
        }
    }
    progress.setdefault("sessions", []).append(session_entry)

    progress.setdefault("totals", {"utterances": 0, "errors": 0})
    progress["totals"]["utterances"] += 1
    progress["totals"]["errors"] += int(analysis.get("error_count", 0))

    progress["last_level"] = analysis.get("overall_level") or progress.get("last_level")

    vocab = set(w.strip().lower() for w in progress.get("vocabulary", []))
    for w in analysis.get("new_words", []):
        lw = w.strip().lower()
        if lw and lw not in vocab:
            vocab.add(lw)
    progress["vocabulary"] = sorted(vocab)

    return progress

def analyze_learning_weaknesses(progress: dict) -> dict:
    sessions = progress.get("sessions", [])
    if len(sessions) < 3:  # minimum data req
        return {
            "insufficient_data": True,
            "message": "Preciso de mais dados para analisar suas fraquezas. Continue praticando!"
        }
    
    recent_cutoff = datetime.utcnow() - timedelta(days=7)
    recent_sessions = []
    all_sessions = []
    
    for session in sessions[-10:]:
        try:
            timestamp = datetime.fromisoformat(session["timestamp"].replace("Z", "+00:00"))
            session_data = {
                "timestamp": timestamp,
                "error_count": session["analysis"].get("error_count", 0),
                "level": session["analysis"].get("overall_level", "B1"),
                "grammar": session["analysis"].get("grammar"),
                "vocabulary": session["analysis"].get("vocabulary"),
                "pronunciation": session["analysis"].get("pronunciation"),
                "utterance": session.get("utterance", ""),
                "suggested_correction": session["analysis"].get("suggested_corrected_sentence", "")
            }
            all_sessions.append(session_data)
            if timestamp >= recent_cutoff:
                recent_sessions.append(session_data)
        except (ValueError, TypeError, KeyError):
            continue
    
    if not recent_sessions:
        recent_sessions = all_sessions[-10:]
    
    analysis = {
        "insufficient_data": False,
        "recent_error_rate": 0.0,
        "trend": "stable",
        "main_weaknesses": [],
        "grammar_issues": [],
        "vocabulary_gaps": [],
        "level_progression": {},
        "recommendations": []
    }
    
    # Error rate
    if recent_sessions:
        total_errors = sum(s["error_count"] for s in recent_sessions)
        total_sessions = len(recent_sessions)
        analysis["recent_error_rate"] = total_errors / total_sessions if total_sessions > 0 else 0.0
        
        # Trend analysis
        if len(recent_sessions) >= 4:
            mid_point = len(recent_sessions) // 2
            first_half_errors = sum(s["error_count"] for s in recent_sessions[:mid_point]) / mid_point
            second_half_errors = sum(s["error_count"] for s in recent_sessions[mid_point:]) / (len(recent_sessions) - mid_point)
            
            if second_half_errors < first_half_errors * 0.8:
                analysis["trend"] = "improving"
            elif second_half_errors > first_half_errors * 1.2:
                analysis["trend"] = "declining"
    
    # Grammar issues
    grammar_issues = []
    vocab_mentions = []
    for session in recent_sessions:
        if session["grammar"] and session["grammar"].lower() not in ["correct", "none", "null"]:
            grammar_issues.append(session["grammar"])
        if session["vocabulary"] and session["vocabulary"].lower() not in ["correct", "none", "null"]:
            vocab_mentions.append(session["vocabulary"])
    
    # grammar patterns
    if grammar_issues:
        grammar_counter = Counter()
        for issue in grammar_issues:
            # keyword extraction
            issue_lower = issue.lower()
            if any(word in issue_lower for word in ["verbo", "verb", "conjugação", "tempo"]):
                grammar_counter["verb_conjugation"] += 1
            if any(word in issue_lower for word in ["gênero", "gender", "masculino", "feminino"]):
                grammar_counter["gender_agreement"] += 1
            if any(word in issue_lower for word in ["preposição", "preposition", "de", "em", "para"]):
                grammar_counter["prepositions"] += 1
            if any(word in issue_lower for word in ["artigo", "article", "definido", "indefinido"]):
                grammar_counter["articles"] += 1
            if any(word in issue_lower for word in ["plural", "singular", "concordância"]):
                grammar_counter["number_agreement"] += 1
        
        analysis["grammar_issues"] = [{"type": k, "frequency": v} for k, v in grammar_counter.most_common(3)]
    
    # Level progression
    levels = [s["level"] for s in all_sessions if s["level"]]
    if levels:
        level_counts = Counter(levels)
        analysis["level_progression"] = dict(level_counts)
        
        recent_levels = [s["level"] for s in recent_sessions[-10:] if s["level"]]
        if len(set(recent_levels)) > 2:
            analysis["main_weaknesses"].append("inconsistent_level")
    
    # Recommendations
    if analysis["recent_error_rate"] > 2.0:
        analysis["recommendations"].append("focus_on_accuracy")
    if analysis["trend"] == "declining":
        analysis["recommendations"].append("review_fundamentals")
    if any(issue["type"] == "verb_conjugation" for issue in analysis["grammar_issues"]):
        analysis["recommendations"].append("practice_verb_conjugation")
    if any(issue["type"] == "gender_agreement" for issue in analysis["grammar_issues"]):
        analysis["recommendations"].append("study_noun_genders")
    
    if analysis["recent_error_rate"] > 1.5:
        analysis["main_weaknesses"].append("high_error_rate")
    if grammar_issues:
        analysis["main_weaknesses"].extend([issue["type"] for issue in analysis["grammar_issues"][:2]])
    
    return analysis

def generate_weakness_context(weaknesses: dict) -> str:
    if weaknesses.get("insufficient_data"):
        return ""
    
    context_parts = []
    
    # Error rate context
    error_rate = weaknesses.get("recent_error_rate", 0)
    if error_rate > 2.0:
        context_parts.append(f"O usuário tem uma taxa de erro alta ({error_rate:.1f} erros por frase em média)")
    elif error_rate > 1.0:
        context_parts.append(f"O usuário comete alguns erros ({error_rate:.1f} erros por frase em média)")
    
    # Trend context
    trend = weaknesses.get("trend", "stable")
    if trend == "declining":
        context_parts.append("O progresso tem diminuído recentemente")
    elif trend == "improving":
        context_parts.append("O usuário está melhorando")
    
    # Grammar weaknesses
    grammar_issues = weaknesses.get("grammar_issues", [])
    if grammar_issues:
        main_issue = grammar_issues[0]["type"]
        issue_map = {
            "verb_conjugation": "conjugação de verbos",
            "gender_agreement": "concordância de gênero",
            "prepositions": "uso de preposições",
            "articles": "uso de artigos",
            "number_agreement": "concordância de número"
        }
        if main_issue in issue_map:
            context_parts.append(f"Dificuldade principal: {issue_map[main_issue]}")
    
    # Recommendations
    recommendations = weaknesses.get("recommendations", [])
    rec_map = {
        "focus_on_accuracy": "focar na precisão",
        "review_fundamentals": "revisar os fundamentos",
        "practice_verb_conjugation": "praticar conjugação de verbos",
        "study_noun_genders": "estudar gêneros dos substantivos"
    }
    
    if recommendations:
        rec_text = ", ".join([rec_map.get(rec, rec) for rec in recommendations[:2]])
        context_parts.append(f"Recomendações: {rec_text}")
    
    if not context_parts:
        return ""
    
    return f"\n\nCONTEXTO DO PROGRESSO: {' | '.join(context_parts)}. Use essas informações para direcionar a conversa e ajudar com as dificuldades específicas."

def record_audio(stop_event, data_queue):

    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe(audio_np: np.ndarray) -> str:
    result = stt.transcribe(audio_np, language="pt", fp16=False)
    text = result["text"].strip()
    return text


def get_llm_response(text: str, progress: dict) -> str:
    weaknesses = analyze_learning_weaknesses(progress)
    weakness_context = generate_weakness_context(weaknesses)
    
    history = chain.memory.buffer
    
    if weakness_context:
        enhanced_template = template.format(
            weakness_context=weakness_context,
            history=history,
            input=text
        )
        response = chain.llm.invoke(enhanced_template)
        chain.memory.save_context({"input": text}, {"output": response})
    else:
        response = chain.predict(input=text)
        if response.startswith("Assistant:"):
            response = response[len("Assistant:") :].strip()
    
    return response


def display_learning_summary(progress: dict, console: Console) -> None:
    total_sessions = len(progress.get("sessions", []))
    total_utterances = progress.get("totals", {}).get("utterances", 0)
    total_errors = progress.get("totals", {}).get("errors", 0)
    last_level = progress.get("last_level", "Unknown")
    vocabulary_size = len(progress.get("vocabulary", []))
    
    if total_sessions == 0:
        console.print("[cyan]First session! Let's start learning Portuguese!")
        return
    
    error_rate = total_errors / total_utterances if total_utterances > 0 else 0
    
    console.print(f"[cyan]Learning Progress Summary:")
    console.print(f"[cyan]  • Total sessions: {total_sessions}")
    console.print(f"[cyan]  • Total sentences: {total_utterances}")
    console.print(f"[cyan]  • Overall error rate: {error_rate:.1f} errors/sentence")
    console.print(f"[cyan]  • Current level: {last_level}")
    console.print(f"[cyan]  • Vocabulary learned: {vocabulary_size} words")
    
    weaknesses = analyze_learning_weaknesses(progress)
    if not weaknesses.get("insufficient_data"):
        recent_error_rate = weaknesses.get("recent_error_rate", 0)
        trend = weaknesses.get("trend", "stable")
        
        console.print(f"[cyan]  • Recent progress: {trend}")
        console.print(f"[cyan]  • Recent error rate: {recent_error_rate:.1f} errors/sentence")
        
        if weaknesses.get("recommendations"):
            recommendations = weaknesses["recommendations"][:2]
            rec_text = ", ".join(recommendations).replace("_", " ")
            console.print(f"[cyan]  • Recommended focus: {rec_text}")
    
    console.print()

def play_audio(sample_rate, audio_array):
    sd.play(audio_array, sample_rate)
    sd.wait()


if __name__ == "__main__":

    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    analysis_llm = Ollama(model=MODEL_NAME_ANALYSIS)
    progress = load_progress()
    console.print("[green]Loaded learning progress.")
    
    display_learning_summary(progress, console)

    try:
        while True:
            console.input("Press Enter to start recording, then press Enter again to stop recording")

            data_queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue)
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            chunks = []
            while not data_queue.empty():
                chunks.append(data_queue.get())
            audio_data = b"".join(chunks)

            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                with console.status("Analyzing errors...", spinner="earth"):
                    t = time.process_time()
                    analysis = analyze_errors(text, analysis_llm)
                    console.print(
                        "[magenta]\nAnalysis (summary):\n"
                        f"  • Level: {analysis.get('overall_level')}\n"
                        f"  • Errors: {analysis.get('error_count')}\n"
                        f"  • Grammar: {analysis.get('grammar')}\n"
                        f"  • Vocabulary: {analysis.get('vocabulary')}\n"
                        f"  • Pronunciation: {analysis.get('pronunciation')}\n"
                        f"  • Suggested correction: {analysis.get('suggested_corrected_sentence')}\n"
                        f"  • New words: {', '.join(analysis.get('new_words', []))}\n"
                    )
                    progress = update_learning_progress(progress, analysis, text)
                    
                    weaknesses = analyze_learning_weaknesses(progress)
                    if not weaknesses.get("insufficient_data"):
                        console.print(
                            f"[blue]\nLearning Progress Analysis:\n"
                            f"  • Recent error rate: {weaknesses.get('recent_error_rate', 0):.1f} errors/sentence\n"
                            f"  • Trend: {weaknesses.get('trend', 'stable')}\n"
                            f"  • Main weaknesses: {', '.join(weaknesses.get('main_weaknesses', ['none']))}\n"
                            f"  • Grammar focus areas: {', '.join([g['type'] for g in weaknesses.get('grammar_issues', [])])}\n"
                            f"  • Recommendations: {', '.join(weaknesses.get('recommendations', ['keep practicing']))}\n"
                        )
                    
                    save_progress(progress)
                    console.print("[green]Progress saved.")
                    print("Time elapsed to analyze+save: " + str(time.process_time() - t))

                with console.status("Generating response...", spinner="earth"):
                    t = time.process_time()
                    response = get_llm_response(text, progress)
                    print("Time elapsed to get response from LLM: " + str(time.process_time() - t))
                    # espeak TTS alternative: espeak_ng(response)
                    t = time.process_time()
                    sample_rate, audio_array = tts.long_form_synthesize(response)
                    print("Time elapsed to synthesize response: " + str(time.process_time() - t))

                console.print(f"[cyan]Assistant: {response}")
                play_audio(sample_rate, audio_array)
            else:
                console.print("[red]No audio recorded. Please check audio input.")

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
            