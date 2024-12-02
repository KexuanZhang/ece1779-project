from openai_messages_token_helper import build_messages, get_token_limit
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from typing import Any, Coroutine, List, Literal, Optional, Union, overload, TypedDict, cast
from openai import AzureOpenAI
import json
import os

from azure.search.documents.models import (
    QueryCaptionResult,
    QueryType,
    VectorizedQuery,
    VectorQuery,
)
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from dataclasses import dataclass
# LLM = AzureOpenAI(
#     azure_endpoint = "https://euaiempoweruai1063516444.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2023-03-15-preview",
#     api_key='79f6947b64ba4a239a36a8695156054b',
#     api_version="2024-02-01"
# )

search_prompt_template = """Below is a new question asked by the user that needs to be answered by searching in a knowledge base.
    You have access to Azure AI Search index with 100's of documents.
    Generate a search query based on the new question.
    Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
    Do not include any text inside [] or <<>> in the search query terms.
    Do not include any special characters like '+'.
    If the question is not in English, translate the question to English before generating the search query.
    If you cannot generate a search query, return just the number 0.
    """

query_response_token_limit = 100

openai_client = AzureOpenAI(
    azure_endpoint = "https://ece1779-openai.openai.azure.com/",
    api_key='C9aJlBioJzh6Rq6xdv7CtoB6UbxkPggQl3BiNc72xT2oMi998pWfJQQJ99AKACYeBjFXJ3w3AAABACOGAeR0',
    api_version="2023-07-01-preview"
)


embedding_client = AzureOpenAI(
    azure_endpoint = "https://ece1779-openai.openai.azure.com/",
    api_key='C9aJlBioJzh6Rq6xdv7CtoB6UbxkPggQl3BiNc72xT2oMi998pWfJQQJ99AKACYeBjFXJ3w3AAABACOGAeR0',
    api_version="2024-02-01"
)

MOCK_EMBEDDING_DIMENSIONS = 1536
MOCK_EMBEDDING_MODEL_NAME = "text-embedding-3-small"

search_client = SearchClient(
    endpoint="https://ece1779-search.search.windows.net",
    index_name="vector-1732830989390",
    credential=AzureKeyCredential("X2l2kuJQUA77O1eXQmq870XDrWENBcUPO7vm9NTOAwAzSeChpgwe")
    )


@dataclass
class Document:
    id: Optional[str]
    content: Optional[str]
    embedding: Optional[List[float]]
    image_embedding: Optional[List[float]]
    category: Optional[str]
    sourcepage: Optional[str]
    sourcefile: Optional[str]
    oids: Optional[List[str]]
    groups: Optional[List[str]]
    captions: List[QueryCaptionResult]
    score: Optional[float] = None
    reranker_score: Optional[float] = None

    def serialize_for_results(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": Document.trim_embedding(self.embedding),
            "imageEmbedding": Document.trim_embedding(self.image_embedding),
            "category": self.category,
            "sourcepage": self.sourcepage,
            "sourcefile": self.sourcefile,
            "oids": self.oids,
            "groups": self.groups,
            "captions": (
                [
                    {
                        "additional_properties": caption.additional_properties,
                        "text": caption.text,
                        "highlights": caption.highlights,
                    }
                    for caption in self.captions
                ]
                if self.captions
                else []
            ),
            "score": self.score,
            "reranker_score": self.reranker_score,
        }

    @classmethod
    def trim_embedding(cls, embedding: Optional[List[float]]) -> Optional[str]:
        """Returns a trimmed list of floats from the vector embedding."""
        if embedding:
            if len(embedding) > 2:
                # Format the embedding list to show the first 2 items followed by the count of the remaining items."""
                return f"[{embedding[0]}, {embedding[1]} ...+{len(embedding) - 2} more]"
            else:
                return str(embedding)

        return None


@dataclass
class ThoughtStep:
    title: str
    description: Optional[Any]
    props: Optional[dict[str, Any]] = None


def get_search_query(chat_completion: ChatCompletion, user_query: str):
        response_message = chat_completion.choices[0].message

        if response_message.tool_calls:
            for tool in response_message.tool_calls:
                if tool.type != "function":
                    continue
                function = tool.function
                if function.name == "search_sources":
                    arg = json.loads(function.arguments)
                    search_query = arg.get("search_query", "0")
                    if search_query != "0":
                        return search_query
        elif query_text := response_message.content:
            if query_text.strip() != "0":
                return query_text
        return user_query


def compute_text_embedding(q: str):
        SUPPORTED_DIMENSIONS_MODEL = {
            "text-embedding-ada-002": False,
            "text-embedding-3-small": True,
            "text-embedding-3-large": True,
        }


        class ExtraArgs(TypedDict, total=False):
            dimensions: int

        dimensions_args: ExtraArgs = (
            {"dimensions": MOCK_EMBEDDING_DIMENSIONS} if SUPPORTED_DIMENSIONS_MODEL[MOCK_EMBEDDING_MODEL_NAME] else {}
        )
        embedding = embedding_client.embeddings.create(
            # Azure OpenAI takes the deployment name as the model name
            model=MOCK_EMBEDDING_MODEL_NAME,
            input=q,
            **dimensions_args,
        )
        query_vector = embedding.data[0].embedding
        return VectorizedQuery(vector=query_vector, k_nearest_neighbors=50, fields="text_vector")


def search(
        search_client: SearchClient,
        top: int,
        query_text: Optional[str],
        filter: Optional[str],
        vectors: List[VectorQuery],
        use_text_search: bool,
        use_vector_search: bool,
        use_semantic_ranker: bool,
        use_semantic_captions: bool,
        minimum_search_score: Optional[float],
        minimum_reranker_score: Optional[float],
    ) -> List[Document]:
        search_text = query_text if use_text_search else ""
        search_vectors = vectors if use_vector_search else []
        if use_semantic_ranker:
            results = search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                query_caption="extractive|highlight-false" if use_semantic_captions else None,
                vector_queries=search_vectors,
                query_type=QueryType.SEMANTIC,
                # semantic_configuration_name="default",
                semantic_query=query_text,
            )
        else:
            results = search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                vector_queries=search_vectors,
            )

        documents = []
        for page in results.by_page():
            for document in page:
              documents.append(
                    Document(
                        id=document.get("id"),
                        content=document.get("content"),
                        embedding=document.get("embedding"),
                        image_embedding=document.get("imageEmbedding"),
                        category=document.get("category"),
                        sourcepage=document.get("sourcepage"),
                        sourcefile=document.get("sourcefile"),
                        oids=document.get("oids"),
                        groups=document.get("groups"),
                        captions=cast(List[QueryCaptionResult], document.get("@search.captions")),
                        score=document.get("@search.score"),
                        reranker_score=document.get("@search.reranker_score"),
                    )
                )

            qualified_documents = [
                doc
                for doc in documents
                if (
                    (doc.score or 0) >= (minimum_search_score or 0)
                    and (doc.reranker_score or 0) >= (minimum_reranker_score or 0)
                )
            ]

        return qualified_documents


def nonewlines(s: str) -> str:
    return s.replace("\n", " ").replace("\r", " ")

def get_sources_content(
        results: List[Document], use_semantic_captions: bool, use_image_citation: bool
    ) -> list[str]:
        if use_semantic_captions:
            return [
                (get_citation((doc.sourcepage or ""), use_image_citation))
                + ": "
                + nonewlines(" . ".join([cast(str, c.text) for c in (doc.captions or [])]))
                for doc in results
            ]
        else:
            return [
                (get_citation((doc.sourcepage or ""), use_image_citation)) + ": " + nonewlines(doc.content or "")
                for doc in results
            ]

def get_citation(sourcepage: str, use_image_citation: bool) -> str:
        if use_image_citation:
            return sourcepage
        else:
            path, ext = os.path.splitext(sourcepage)
            if ext.lower() == ".png":
                page_idx = path.rfind("-")
                page_number = int(path[page_idx + 1 :])
                return f"{path[:page_idx]}.pdf#page={page_number}"

            return sourcepage


def answer(prompt):
    user_query_request = "Generate search query for: " + prompt

    tools: List[ChatCompletionToolParam] = [
            {
                "type": "function",
                "function": {
                    "name": "search_sources",
                    "description": "Retrieve sources from the Azure AI Search index",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "Query string to retrieve documents from azure search",
                            }
                        },
                        "required": ["search_query"],
                    },
                },
            }
        ]
    
    
    query_messages = build_messages(
            model='gpt-4o-mini',
            system_prompt=search_prompt_template,
            tools=tools,
            new_user_content=user_query_request
        )
    
    chat_completion = openai_client.chat.completions.create(
            messages=query_messages,
            model='gpt-4o-mini',
            temperature=0.0,
            max_tokens=query_response_token_limit,
            n=1,
            tools=tools,
        )

    
    query_text = get_search_query(chat_completion, prompt)

    vectors: list[VectorQuery] = []
    a = compute_text_embedding(query_text)
    vectors.append(a)
    

    results = search(
        search_client=search_client,
        top=3,
        filter=None,
        query_text=query_text,
        vectors=vectors,
        use_text_search=True,
        use_vector_search=True,
        use_semantic_ranker=True,
        use_semantic_captions=True,
        minimum_search_score=0,
        minimum_reranker_score=0,
    )

    sources_content = get_sources_content(results=results, use_semantic_captions=True, use_image_citation=False)
    content = "\n".join(sources_content)

    system_message = "You are a highly knowledgeable and specialized AI tutor with expertise in cloud computing. Your purpose is to teach and clarify cloud computing concepts based on the most relevant and reliable information retrieved from a Retrieval-Augmented Generation (RAG) system. The retrieved documents provide up-to-date and contextually accurate knowledge to guide your explanations. Your role is to: Provide clear, detailed, and easy-to-understand explanations tailored to the user's level of expertise (beginner, intermediate, or advanced). \
                    Focus on teaching concepts, answering questions, and providing examples related to topics such as: Cloud service models (IaaS, PaaS, SaaS). \
                    Cloud providers (e.g., AWS, Azure, Google Cloud). \
                    Security, scalability, and high availability in cloud computing. \
                    Infrastructure components (e.g., virtual machines, containers, storage). \
                    DevOps, CI/CD pipelines, and cloud automation. \
                    Emerging cloud trends (e.g., serverless, edge computing). \
                    Reference the retrieved documents explicitly, ensuring accuracy and traceability of the information. \
                    Supplement retrieved knowledge with examples, analogies, and diagrams (if applicable) to enhance understanding. \
                    Provide actionable advice or next steps, such as resources for further learning or troubleshooting approaches. \
                    When responding: \
                    Always explain terms and concepts clearly, especially for beginners. \
                    If the user is more advanced, delve deeper into technical details and provide links between concepts. \
                    Use concise language but ensure completeness of the explanation. \
                    Encourage user engagement by asking follow-up questions or suggesting practical applications. \
                    You must only provide responses grounded in the retrieved documents and avoid speculation or assumptions. If the retrieved data is insufficient, clarify this to the user and guide them on how to refine their query for better results."

    response_token_limit = 1024
    messages = build_messages(
            model='gpt-4o-mini',
            system_prompt=system_message,
            new_user_content=prompt + "\n\nSources:\n" + content,
            max_tokens=response_token_limit,
        )
    
    data_points = {"text": sources_content}

    extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Prompt to generate search query",
                    [str(message) for message in query_messages],
                    (
                        {"model": "gpt-3.5-turbo", "deployment": "test-chatgpt"}
                        if "test-chatgpt"
                        else {"model": "gpt-3.5-turbo"}
                    ),
                ),
                ThoughtStep(
                    "Search using generated search query",
                    query_text,
                    {
                        "use_semantic_captions": True,
                        "use_semantic_ranker": True,
                        "top": 3,
                        "filter": filter,
                        "use_vector_search": True,
                        "use_text_search": True,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
                ThoughtStep(
                    "Prompt to generate answer",
                    [str(message) for message in messages],
                    (
                        {"model": 'gpt-3.5-turbo', "deployment": "test-chatgpt"}
                        if "test-chatgpt"
                        else {"model": 'gpt-3.5-turbo'}
                    ),
                ),
            ],
        }
    
    chat = openai_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        temperature=0.3,
        max_tokens=response_token_limit,
        n=1,
    )

    return chat.choices[0].message.content, data_points['text'], extra_info
