from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union, Any
from pydantic import BaseModel, validate_arguments

from langchain.chat_models import ChatOpenAI

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain import LLMChain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.embeddings import OpenAIEmbeddings

from .llm_langchain import LLMLangChain
from .utils import *

ROLE_CLASS_MAP = {
    "assistant": AIMessage,
    "user": HumanMessage,
    "system": SystemMessage
}


class Message(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    conversation: List[Message]

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


class LLMLangChainChat(LLMLangChain):
    """

    """
    def __init__(self, config: Dict[str, Any],
                 **kwargs) -> None:
        super().__init__(config, **kwargs)
        
        return

    def chat_open_ai(self, **kwargs) -> Optional[ChatOpenAI]:  # ChatOpenAI | None:
        if self.openai_api_key is None or self.openai_api_key == "":
            print("OPENAI_API_KEY is not set")
            return None
        # print(
        #    f"ChatOpenAI OPENAI_API_KEY is set: {self.openai_api_key[0:3]}...{self.openai_api_key[-4:]}")

        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.openai_api_key,
            **kwargs)
        return self.llm

    # def chain(self, prompt: str) -> LLMChain:
    #    return LLMChain(llm=self.llm, prompt=prompt)

    # staticmethod
    def system_message(self, content: str = "You are a helpful assistant.") -> Any:
        # return content
        return SystemMessage(content=content)

    # staticmethod
    def human_message(self, content: str) -> Any:
        # return content
        return HumanMessage(content=content)

    # staticmethod
    def ai_message(self, content: str) -> Any:
        # return content
        return AIMessage(content=content)

    def system_message_prompt_templay(self, template: str) -> str:

        return SystemMessagePromptTemplate.from_template(template)

    def human_message_prompt_templay(self, template: str) -> str:

        return HumanMessagePromptTemplate.from_template(template)

    def chat_prompt_templay(self,
                            system_message_prompt: str,
                            human_message_prompt: str) -> str:

        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
