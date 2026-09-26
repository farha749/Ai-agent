import os
import time
import ollama
from google import genai
from google.genai import types


# -----------------------------
# Ollama - SLM (Phi 3.5)
# -----------------------------
def run_slm(prompt):
    messages = []

    for item in chat_history:
        messages.append({"role": "user", "content": item["user"]})
        messages.append({"role": "assistant", "content": item["ai"]})

    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model="phi3.5",
        messages=messages
    )

    return response["message"]["content"]

# -----------------------------
# Ollama - GPT-2
# -----------------------------
def run_gpt2(prompt):
    import subprocess

    history_text = ""

    for item in chat_history[-2:]:
        history_text += f"User: {item['user']}\n"
        history_text += f"AI: {item['ai']}\n"

    full_prompt = history_text + f"User: {prompt}\nAI:"

    result = subprocess.run(
        ["ollama", "run", "atel3134/gpt2:124m", full_prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120
    )

    if result.returncode != 0:
        return "GPT-2 Error: " + (result.stderr or "Unknown error")

    return result.stdout.strip()


# -----------------------------
# Gemini API
# -----------------------------
api_key = os.environ.get("GEMINI_API_KEY")

if api_key:
    gemini = genai.Client(api_key=api_key)
else:
    gemini = None


# -----------------------------
# Gemini 3 Flash
# -----------------------------
def run_gemini3(prompt):
    if gemini is None:
        return "GEMINI_API_KEY not found."

    messages = []

    for item in chat_history:
        messages.append({
            "role": "user",
            "parts": [{"text": item["user"]}]
        })
        messages.append({
            "role": "model",
            "parts": [{"text": item["ai"]}]
        })

    messages.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    response = gemini.models.generate_content(
        model="gemini-3-flash-preview",
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful AI assistant. Answer clearly and politely."
        )
    )

    return response.text


# -----------------------------
# Gemini 3.5 Flash
# -----------------------------
def run_gemini35(prompt):
    if gemini is None:
        return "GEMINI_API_KEY not found."

    for attempt in range(3):
        try:
            response = gemini.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a helpful AI assistant. Answer clearly and politely."
                )
            )
            return response.text

        except Exception as e:
            if "503" in str(e) and attempt < 2:
                wait = 5 * (2 ** attempt)
                print(f"Gemini 3.5 busy. Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                return "Gemini 3.5 is temporarily unavailable. Please try again later."


# -----------------------------
# Chat History
# -----------------------------
chat_history = []

# -----------------------------
# AI AGENT
# -----------------------------

print("\n==============================")
print("          AI AGENT")
print("==============================")
print("1. SLM - Phi 3.5 (Ollama)")
print("2. GPT-2 (Ollama)")
print("3. Gemini 3 Flash")
print("4. Gemini 3.5 Flash")
print("5. Exit")
print("================")

while True:
    choice = input("\nChoose model (1-5): ")

    if choice == "5":
        print("AI Agent stopped.")
        break

    user_input = input("You: ")

    try:
        if choice == "1":
            answer = run_slm(user_input)
        elif choice == "2":
            answer = run_gpt2(user_input)
        elif choice == "3":
            answer = run_gemini3(user_input)
        elif choice == "4":
            answer = run_gemini35(user_input)

        else:
            print("Please choose 1, 2, 3, 4 or 5.")
            continue
        chat_history.append({"user": user_input, "ai": answer})

        print("\nAI:", answer)

    except Exception as e:
        print("\nError:", e)