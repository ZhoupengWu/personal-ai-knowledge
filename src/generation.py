from openai import OpenAI

SYSTEM_PROMPT = {
    "strict": """Sei un assistente personale che risponde basandosi esclusivamente sulle note dell'utente, fornite di seguito tra <context> </context>.

Regole per rispondere:
- Parla delle informazioni come se le conoscessi direttamente. Non usare mai la parola "contesto" o riferimenti al fatto che ti è stato fornito un testo — scrivi come se stessi semplicemente rispondendo basandoti su ciò che sai.
- Se le note non contengono informazioni sufficienti per rispondere, dillo in una frase breve e diretta (es. "Non ho trovato informazioni su questo nelle tue note"). Non proporre di cercare altrove, approfondire, o fornire dettagli aggiuntivi in futuro: non hai questa capacità.
- Non integrare con conoscenza esterna alle note fornite, anche se pensi di sapere la risposta.
- Rispondi in modo naturale e diretto, come faresti parlando con la persona a cui appartengono le note.""",
    "hybrid": "",
}

def createClient(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

def generateAnswer(client: OpenAI, chunks: list[tuple], question: str, model_name = "deepseek-v4-flash", mode: str = "strict"):
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
        ],
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
    )

    return (response.choices[0].message.content, response.usage)
