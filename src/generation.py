from openai import OpenAI

SYSTEM_PROMPT = {
    "strict": "Sei un assistente personale che deve usare solamente il contesto fornito dall'utente delimitato da <context> </context> per rispondere alla domanda. Il context è accompagnato dalla fonte '[file.ext]': citalo solo se è necessario per comprendere meglio. Se ti mancano informazioni e non è presente nel context, devi rispondere esplicitamente, che non lo sai riformulando la risposta senza offire alternative. Non ripetere ogni volta 'contesto' nelle risposte: usa parole più naturali",
    "hybrid": ""
}

def createClient(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

def generateAnswer(client: OpenAI, chunks: list[tuple], question: str, model_name = "deepseek-v4-flash", mode: str = "strict"):
    system_content = SYSTEM_PROMPT[mode]

    text = " --- ".join(f"{a[0]} [{a[2]}]" for a in chunks)
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
        ],
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
    )

    return (response.choices[0].message.content, response.usage)
