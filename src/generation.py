from openai import OpenAI

SYSTEM_PROMPT = {
    "strict": "Sei un assistente personale che deve usare solamente il contesto fornito dall'utente delimitato da <context> </context> per rispondere alla domanda. Se ti mancano informazioni e non è presente nel context, devi rispondere sempre che non lo sai",
    "hybrid": ""
}

def createClient(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

def generateAnswer(client: OpenAI, chunks: list[tuple], question: str, model_name = "deepseek-v4-flash", mode: str = "strict") -> str | None:
    system_content = SYSTEM_PROMPT[mode]

    text = " --- ".join(a[0] for a in chunks)
    user_content = f"<context> {text} </context> {question}"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    )

    return response.choices[0].message.content
