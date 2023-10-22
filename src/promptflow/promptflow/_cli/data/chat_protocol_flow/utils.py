import os
from typing import List
from promptflow.connections import AzureOpenAIConnection, CognitiveSearchConnection

def create_cp_user_content(question: str):
    cp_message = {}
    cp_message["content"] = question
    cp_message["role"] = "user"
    return cp_message

def create_cp_assistant_content(answer: str):
    cp_message = {}
    cp_message["content"] = answer
    cp_message["role"] = "assistant"
    return cp_message

def convert_chat_history_pf_to_cp(pf_chat_history: List[dict], cp_messages: List[dict]):
    for pf_item in pf_chat_history:
        cp_messages.append(create_cp_user_content(pf_item["inputs"]["question"]))
        cp_messages.append(create_cp_assistant_content(pf_item["outputs"]["output"]))

def setup_aoai_credentials(aoai_connection: AzureOpenAIConnection):
    import openai
    openai.api_type = os.environ["OPENAI_API_TYPE"] = aoai_connection.api_type
    openai.api_key = os.environ["OPENAI_API_KEY"] = aoai_connection.api_key
    openai.api_version = os.environ["OPENAI_API_VERSION"] = aoai_connection.api_version
    openai.api_base = os.environ["OPENAI_API_BASE"] = aoai_connection.api_base

def setup_acs_credentials(acs_connection: CognitiveSearchConnection):
    os.environ["AZURE_COGNITIVE_SEARCH_TARGET"] = acs_connection.api_base
    os.environ["AZURE_COGNITIVE_SEARCH_KEY"] = acs_connection.api_key
