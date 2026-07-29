import os
import json
from huggingface_hub import InferenceClient

# 🔑 BURAYA KOPYALADIĞIN hf_... İLE BAŞLAYAN TOKEN'I YAPIŞTIR
HF_TOKEN = "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" 

class SmartAssistant:
    def __init__(self, memory_file="memory.json"):
        print("🧠 Asistan başlatılıyor ve hafıza yükleniyor...")
        self.memory_file = memory_file
        self.memories = self._load_memory()
        # Hugging Face istemcisini başlatıyoruz
        self.client = InferenceClient(api_key=HF_TOKEN)

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def save_to_memory(self, user_input: str, ai_response: str):
        entry = {
            "user": user_input,
            "assistant": ai_response
        }
        self.memories.append(entry)
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    def ask(self, user_input: str):
        past_memories_str = ""
        if self.memories:
            for item in self.memories[-5:]:
                past_memories_str += f"- Kullanıcı: {item['user']}\n  Asistan: {item['assistant']}\n"
        else:
            past_memories_str = "Henüz geçmiş kayıt yok."

        prompt = f"""
        Sen kullanıcının kişisel, uzun süreli hafızaya sahip asistanısın.
        Aşağıda kullanıcının geçmiş konuşma kayıtları yer alıyor.

        GEÇMİŞ HAFIZA:
        {past_memories_str}

        KULLANICININ YENİ MESAJI:
        {user_input}

        TALİMAT:
        Eğer geçmiş hafızada bu konuyla ilgili bilgi varsa samimi bir dille bunu hatırladığını belirterek Türkçe yanıtla.
        """

        try:
            # Hugging Face üzerindeki çok güçlü Türkçe destekli ücretsiz model
            completion = self.client.chat.completions.create(
                model="Qwen/Qwen2.5-Coder-32B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            ai_text = completion.choices[0].message.content
            self.save_to_memory(user_input, ai_text)
            return ai_text

        except Exception as e:
            return f"\n❌ HATA DETAYI: {e}"

if __name__ == "__main__":
    bot = SmartAssistant()

    print("\n-------------------------------------------")
    print("--- 1. OTURUM: ASİSTANA ÖĞRETME AŞAMASI ---")
    print("-------------------------------------------")
    soru1 = "Benim adım Görkem. En sevdiğim kahve türü Oat Milk Latte, bunu sakın unutma."
    print(f"Siz: {soru1}")
    print(f"Asistan: {bot.ask(soru1)}")

    print("\n-------------------------------------------")
    print("--- 2. OTURUM: HAFIZA / HATIRLAMA TESTİ ---")
    print("-------------------------------------------")
    soru2 = "Bana güzel bir kahve önerir misin? Bir de adım neydi hatırlıyor musun?"
    print(f"Siz: {soru2}")
    print(f"Asistan: {bot.ask(soru2)}")