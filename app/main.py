import ollama

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = ollama.chat(
        model="phi3.5",
        messages=[
            {"role": "user", "content": user_input}
        ]
    )

    print("AI:", response["message"]["content"])