import os
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

# 🔑 Groq'tan aldığın gsk_... ile başlayan API Key'ini buraya yapıştır:
GROQ_API_KEY = "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"

client = Groq(api_key=GROQ_API_KEY)
MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_memory(memories):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
def chat(request: ChatRequest):
    memories = load_memory()
    
    past_memories_str = ""
    if memories:
        for item in memories[-5:]:
            past_memories_str += f"- Kullanıcı: {item['user']}\n  Asistan: {item['assistant']}\n"
    else:
        past_memories_str = "Henüz geçmiş kayıt yok."

    prompt = f"""
    Sen kullanıcının kişisel, uzun süreli hafızaya sahip zeki asistanısın.
    
    GEÇMİŞ HAFIZA:
    {past_memories_str}

    KULLANICININ YENİ MESAJI:
    {request.user_input}

    TALİMAT:
    Eğer geçmiş hafızada bu konuyla ilgili bilgi varsa samimi bir dille bunu hatırladığını belirterek Türkçe yanıtla.
    """

    def generate_stream():
        full_response = ""
        # Groq Llama 3.3 70B - Dünyanın en hızlı yayın yapan modellerinden biri
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            stream=True
        )

        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            if content:
                full_response += content
                yield content

        # Yanıt tamamlandığında hafızaya kaydet
        memories.append({"user": request.user_input, "assistant": full_response})
        save_memory(memories)

    return StreamingResponse(generate_stream(), media_type="text/plain")