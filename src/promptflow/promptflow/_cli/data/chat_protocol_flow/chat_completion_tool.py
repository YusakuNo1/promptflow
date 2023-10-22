from promptflow import tool
from promptflow.connections import AzureOpenAIConnection, CognitiveSearchConnection

from chat_demo import chat_completion
from utils import convert_chat_history_pf_to_cp, create_cp_user_content, setup_aoai_credentials, setup_acs_credentials

@tool
def chat_completion_tool(
    question: str,
    chat_history: list,
    aoai_connection: AzureOpenAIConnection,
    acs_connection: CognitiveSearchConnection,
) -> str:
    messages = []
    extra_args = {
        "aoai_connection": aoai_connection,
        "acs_connection": acs_connection,
    }

    # Convert PromptFlow chat history into Chat Protocol and append the question
    convert_chat_history_pf_to_cp(chat_history, messages)
    messages.append(create_cp_user_content(question))

    # Setup credentials
    setup_aoai_credentials(aoai_connection)
    setup_acs_credentials(acs_connection)

    # call the entry function
    return chat_completion(
        messages=messages,
        extra_args=extra_args,
    )