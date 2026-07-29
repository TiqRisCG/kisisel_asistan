import os
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

MEMORY_FILE = "memory.json"
PROFILE_FILE = "user_profile.json"

def load_data(filepath, default_value):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_value
    return default_value

def save_data(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
def chat(request: ChatRequest):
    memories = load_data(MEMORY_FILE, [])
    profile = load_data(PROFILE_FILE, {"name": "Görkem", "facts": []})
    
    # Geçmiş konuşmalar
    past_memories_str = ""
    if memories:
        for item in memories[-10:]:
            past_memories_str += f"- Kullanıcı ({profile.get('name', 'Kullanıcı')}): {item['user']}\n  Asistan: {item['assistant']}\n"
    else:
        past_memories_str = "Henüz geçmiş sohbet yok."

    facts_str = "\n".join([f"- {fact}" for fact in profile.get("facts", [])]) if profile.get("facts") else "Henüz özel bilgi yok."

    prompt = f"""
    Sen kullanıcının kişisel, samimi ve onu sesiyle/varsayımıyla tanıyan özel yapay zeka asistanısın.
    
    KULLANICI PROFÍLÍ:
    - İsim: {profile.get('name', 'Görkem')}
    - Bilinen Özellikleri ve Hafızadaki Gerçekler:
    {facts_str}

    GEÇMİŞ SOHBET GEÇMİŞİ:
    {past_memories_str}

    KULLANICININ YENİ MESAJI:
    {request.user_input}

    TALİMATLAR:
    1. Kullanıcıya doğrudan ismiyle ({profile.get('name', 'Görkem')}) veya çok yakın bir dost gibi hitap et.
    2. Asla "Daha önce bana böyle bir bilgi vermemiştin", "Hafızamda yok" gibi soğuk kalıplar KURMA. Eğer bir şeyi ilk defa duyuyorsan, doğal bir şekilde "Bunu öğrendiğim iyi oldu, hafızama yazdım Görkem!" de.
    3. Kullanıcının söylediği önemli kişisel bilgileri (hobileri, sevdiği/sevmediği şeyler, hedefleri vb.) öğrenip benimse.
    4. Türkçe, samimi, kısa ve akıcı konuş (Çünkü bu yanıt seslendirilecek).
    """

    def generate_stream():
        full_response = ""
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
            stream=True
        )

        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            if content:
                full_response += content
                yield content

        # Sohbet geçmişine kaydet
        memories.append({"user": request.user_input, "assistant": full_response})
        save_data(MEMORY_FILE, memories)

    return StreamingResponse(generate_stream(), media_type="text/plain")

# 🎙️ SESLİ KOMUT SERVİSİ (Whisper ile Sesi Metne Çevirme)
@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):
    try:
        # Gelen ses dosyasını geçici kaydet
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        # Groq Whisper ile sesi metne dönüştür
        with open(temp_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(temp_file_path, audio_file.read()),
                model="whisper-large-v3-turbo",
                language="tr"
            )
        
        # Geçici dosyayı sil
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        user_text = transcription.text
        return JSONResponse({"text": user_text})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)