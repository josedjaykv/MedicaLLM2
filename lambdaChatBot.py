# lambda_function.py
import json
import os
import urllib.request

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
PREDICT_URL = "http://*********/predict"

# Prompt para extracción (para el endpoint de predicción)
EXTRACTOR_SYSTEM_PROMPT = (
    "You are a clinical scribe. Extract only the patient's symptom narrative from a multi-turn conversation "
    "(often in Spanish). Write ONE concise paragraph in ENGLISH suitable for a symptom-based disease classifier. "
    "Include: chief complaint, onset and duration, severity, location and radiation, timing and progression, "
    "triggers and relieving/aggravating factors, relevant associated symptoms, and important negatives explicitly mentioned. "
    "Do NOT include diagnoses, test plans, clinician advice, or meta commentary. "
    "Avoid filler words. Use proper medical terminology where possible."
    "Output ONLY the paragraph."
)

# Prompt de comportamiento del chat (anamnesis en español, sin recomendaciones)
CHAT_SYSTEM_PROMPT_ES = (
    "Primero preséntate brevemente como asistente médico de IA. "
    "Eres un asistente médico en un proceso de anamnesis. "
    "Tu objetivo es obtener información sobre síntomas, antecedentes y condiciones relevantes del paciente. "
    "No des recomendaciones médicas, diagnósticos ni tratamientos. "
    "Haz preguntas claras, una por turno, para profundizar. "
    "Sé empático y profesional. "
    "Responde SIEMPRE en español."
)

def _chat(messages, model="gpt-4o-mini", max_tokens=300, temperature=0.3):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENAI_URL, data=data, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _predict(symptoms_text: str):
    data = json.dumps({"text": symptoms_text}).encode("utf-8")
    req = urllib.request.Request(
        PREDICT_URL, data=data, method="POST",
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))

def cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "content-type,authorization",
        "Access-Control-Allow-Methods": "POST,OPTIONS",
    }

def transcript_from(messages):
    lines = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        prefix = "👤 Usuario" if role == "user" else ("🤖 Asistente" if role == "assistant" else "🛠️ Sistema")
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)

def lambda_handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 204, "headers": cors_headers(), "body": ""}

    try:
        body = json.loads(event.get("body", "{}"))
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return {
                "statusCode": 400,
                "headers": cors_headers(),
                "body": json.dumps({"error": "Debes enviar 'messages' como lista no vacía"})
            }

        # Último mensaje del usuario
        last = messages[-1]
        last_content = (last.get("content") or "").strip().lower()

        # --- Flujo: "recibir resultados" ---
        if last.get("role") == "user" and last_content == "recibir resultados":
            full_transcript = transcript_from(messages)

            extractor_messages = [
                {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": f"Conversation transcript:\n{full_transcript}"}
            ]
            extract_resp = _chat(extractor_messages, model="gpt-4o-mini", max_tokens=250, temperature=0.2)
            symptoms_text = extract_resp["choices"][0]["message"]["content"].strip()

            extractor_messages = [
                {"role": "system", "content": "Translate the following text to spanish and clarify to the user that this result is from an AI agent and recomend him to consult a medical professional. Only return the translated text and recommendations, do not include any additional commentary, or title."},
                {"role": "user", "content": f"Conversation transcript:\n{symptoms_text}"}
            ]
            extract_resp = _chat(extractor_messages, model="gpt-4o-mini", max_tokens=250, temperature=0.2)

            pred = _predict(symptoms_text)
            predictions = pred.get("predictions", [])

            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json", **cors_headers()},
                "body": json.dumps({
                    "mode": "results",
                    "symptoms_text": extract_resp["choices"][0]["message"]["content"].strip(),
                    "predictions": predictions,
                    "turns": len(messages)
                })
            }

        # --- Flujo normal de chat (inyectamos system de anamnesis SIEMPRE) ---
        chat_messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT_ES}] + messages
        resp = _chat(chat_messages)
        answer = resp["choices"][0]["message"]["content"]

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", **cors_headers()},
            "body": json.dumps({"mode": "chat", "answer": answer})
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", **cors_headers()},
            "body": json.dumps({"error": str(e)})
        }
