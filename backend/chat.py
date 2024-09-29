from openai import AzureOpenAI

# LLM = AzureOpenAI(
#     azure_endpoint = "https://euaiempoweruai1063516444.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2023-03-15-preview",
#     api_key='79f6947b64ba4a239a36a8695156054b',
#     api_version="2024-02-01"
# )
LLM = AzureOpenAI(
    azure_endpoint = "https://ece1779project5442706307.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2023-03-15-preview",
    api_key='3b92ad0378384170b682368ae376b7a1',
    api_version="2023-03-15-preview"
)

def answer(prompt):
    SYSTEM_PROMPT = """
        PERSONA:
        You are an engaging and fun Teaching Assistant. Your role is to help the user (the student) fully understand cloud computing concepts by providing relevant examples.
    """    
    completion = LLM.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500,
        stream=True
    )

    
    print("here", completion)

    return completion