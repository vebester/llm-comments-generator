from dotenv import dotenv_values
import streamlit as st
from streamlit_chat import message

from llm_langchain.llm_langchain_chat import LLMLangChainChat
from llm_langchain.prompt_builder.prompt_builder import PromptBuilder
import json
import logging

# from llm_langchain.utils import pull_from_website

_logger = logging.getLogger(__name__)

# from app.core.config import config
config = dotenv_values(".env")

debug: bool = bool(config["DEBUG"])
env_mode: str = config["ENV_MODE"]

MODEL_NAMES = ("gpt-3.5-turbo", "gpt-4")

config["model_name"] = MODEL_NAMES[0]

llm_chat = LLMLangChainChat(config)

# setup streamlit page
st.set_page_config(
    page_title="ChatGPT",
    page_icon="🤖"
)

# Вы являетесь опытным комментатором тем для обсуждения в Twitter, Instagram или любых других сообществах.
system_template = "You are an experienced discussion topics commenter on Twitter, Instagram or any other communities."

# initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        # llm_chat.system_message(content=system_template)
    ]

st.title("ChatGPT")
# st.subheader("AI Tutor:")


SENTIMENT = {
    "negative",
    "positive"
    #   "neutral"
}

CATEGORIES = (
    "Select",
    "Gun Control",
    "Abortion",
    "Religious Freedom",
    "Animal Rights",
    "Vaccines",
    "Privacy Rights",
    "Free-Market Capitalism",
    "Global Climate Change",
    "Evolution",
    "Marijuana Legalization",
    "Capital Punishment",
    "Marriage Equality",
    "vector",
    "Immigration Reform",
    "Trump Presidency",
    "Opioid Crisis",
    "Transgender Rights",
    "Federal Livable Wage",
    "White Supremacy",
    "Electoral College",
    "Black Lives Matter",
    "Cancel culture",
    "Student Debt Crisis",
    "Israeli-Palestinian Conflict",
    "Universal healthcare",
    "Consensual sex",
    "Gene Editing",
    "Age of marriage",
    "Uniform Civil Code",
    "Fiat or Crypto",
    "Cattle slaughter",
    "Rastra Basha",
    "Sabarimala issue",
    "News Ethics",
    "Beauty pageants",
    "Driverless cars",
    "Rohingya refugees",
    "Mass migration",
    "Whistleblowing",
    "Bullet for bullet",
    "Suicide legislation",
    "Gambling Regulation",
    "Parenting style",
    "Alcohol Ban",
    "Technological unemployment",
    "Online Piracy",
    "Boycott of Chinese goods",
    "Homeschooling",
    "School uniform",
    "Gender equality",
    "Media censorship",
    "Modern Slavery",
    "Conscription",
    "Organ Donation",
    "Caste Reservations",
    "Ban of Bandhs",
    "Russo-Ukrainian War",
    "Syrian civil war",
    "Energy War"
)

text_category: str = ""

n_negative: int = 0
n_positive: int = 0
n_comments: int = n_negative + n_positive

output_language: str = "English"

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0
}

st.header("Comments Generator 🤖")

# sidebar with user input
with st.sidebar:
    model_name = st.selectbox(
        "Select a model",
        MODEL_NAMES
    )

    if model_name:
        config["model_name"] = model_name
        llm_chat = LLMLangChainChat(config)

    n_negative = st.slider("Num of negative comments:", 0, 10, value=1)
    n_positive = st.slider("Num of positive comments:", 0, 10, value=1)

    # max_sentences = st.slider("Max sentences in comment:", 1, 5)
    # max_arguments = st.slider("Max arguments for or against:", 1, 3)

    max_sentences = st.selectbox(
        "Max num of sentences in comment:", ('1', '2', '3', '4', '5'), index=4)
    max_arguments = st.selectbox(
        "Max num of arguments for or against:", ('0', '1', '2', '3'), index=0)
    max_chars = st.number_input(
        "Max num of comment chars:", min_value=1, max_value=300, value=300)

    output_language = st.selectbox(
        "Output Language:", ('English', 'Russian'))

    # def _send_data():
    #    st.session_state.counter += 1

    # form = st.form(key="my_form")
    # n_negative = form.slider("Num negative comments", 0, 10)
    # n_positive = form.slider("Num positive comments", 0, 10)
    # submit_button_click = st.form_submit_button(
    #    label="Rerun",
    #    on_click=_send_data
    # )
    # if submit_button_click:

input_data = ""

# url = st.text_input(
#    "Web Page Url: ", placeholder="https://", key="user_input_url")

# if st.button("Get from url"):
# handle user input
#    if url:
#        st.write("Getting webpages...")
#        input_data = pull_from_website(url)

# user_input = st.text_input("Your message: ", key="user_input")
user_input = st.text_area(
    "Text to comment: ", value=input_data, key="user_input")

text_category = st.selectbox(
    "Text category: ", CATEGORIES, index=0, key="user_category_list")
# text_category = st.text_input("Text category: ", key="user_category_input")

if st.button("Generate"):
    # handle user input
    if user_input or text_category != "Select":
        if text_category == "Select":
            text_category = ""
        n_comments: int = n_negative + n_positive
        # st.write(f"Num negative comments {n_negative}")
        # st.write(f"Num positive comments {n_positive}")
        min_arguments: int = min(int(max_arguments), int(max_sentences))

        pb = PromptBuilder(
            text=user_input, text_category=text_category,
            n_negative=n_negative, n_positive=n_positive,
            max_sentences=int(max_sentences), max_chars=max_chars,
            max_arguments=min_arguments,
            output_language=output_language)
        # input_language="Russian", output_language="Russian")

        st.session_state.messages = []

        category_tag: str = ""

        if user_input:
            if text_category != "":
                category_tag = f"\n\n#{text_category}"
            st.session_state.messages.append(user_input + category_tag)
        else:
            category_tag = f"#{text_category}"
            st.session_state.messages.append(category_tag)

        # llm_chat.human_message(content=user_input))

        with st.spinner("Thinking..."):
            system_message_prompt = llm_chat.system_message_prompt_templay(
                system_template)

            human_template = pb.get_human_template()

            if debug:
                st.write(human_template)

            human_message_prompt = llm_chat.human_message_prompt_templay(
                human_template)

            chat_prompt = llm_chat.chat_prompt_templay(
                system_message_prompt, human_message_prompt)

            chat = llm_chat.chat_open_ai(temperature=0.8)

            chain = llm_chat.chain(prompt=chat_prompt)

            output = chain.run(
                text=user_input, text_category=text_category,
                n_negative=n_negative, n_positive=n_positive,
                max_sentences=int(max_sentences), max_chars=max_chars,
                max_arguments=min_arguments,
                output_language=output_language)
            # input_language="Russian", output_language=output_language)

            # st.write(output)
            # st.write(type(output))

            response: dict[str, list] = json.loads(output)
            comments: list = response["comments"]

            # print(comments, type(comments), type(response))

            for obj in comments:
                comment: str = obj["comment"]
                sentiment: str = obj["label"]
                out: str = f"{comment}\n\n{sentiment}"
                st.session_state.messages.append(out)

            # st.session_state.messages.append(output)

            # for i in range(1, n_comments + 1):
            #    st.session_state.messages.append(
            #        str(i))

                # llm_chat.ai_message(content=response.content))
    else:
        st.write('Enter the text or select the category.')


# handle user input
# if user_input:
#   st.session_state.messages.append(llm_chat.human_message(content=user_input))
#   with st.spinner("Generating..."):
#       response = llm_chat.chat_open_ai(st.session_state.messages)
#   st.session_state.messages.append(
#       llm_chat.ai_message(content=response.content))

# display message history
messages = st.session_state.get('messages', [])

for i, msg in enumerate(messages):
    if i == 0:
        message(msg, is_user=True, key=str(i) + '_user')
    else:
        message(msg, is_user=False, key=str(i) + '_ai')

    # for i, msg in enumerate(messages[1:]):
    # if i % 2 == 0:
    #     message(msg.content, is_user=True, key=str(i) + '_user')
    # else:
#     message(msg.content, is_user=False, key=str(i) + '_ai')
