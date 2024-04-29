from openai import OpenAI
import streamlit as st
from io import BytesIO
import numpy as np
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import measurements
import warnings


avatar_ai = "📀"
PROMPT_PATH = "./prompts/data_prompt5.txt"

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache_resource
def get_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


@st.cache_data
def get_sys_prompt(path = PROMPT_PATH):
    with open(PROMPT_PATH, 'r') as f:
        data = f.read()
    return data


def get_response(sys_prompt, prompt):
    client = get_client()
    res = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {
        "role": "system",
        "content": sys_prompt
        },
        {
        "role": "user",
        "content": prompt
        }
    ],
    temperature=0.6,
    seed=999,
    #max_tokens=64,
    #top_p=1
    )

    return res


@dataclass
class Message:
    actor: str
    payload: str


st.title("Fab-X 📀")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "wafermap" not in st.session_state:
    st.session_state["wafermap"] = "false"

for msg in st.session_state["messages"]:
    if isinstance(msg.payload, str):
        st.chat_message(msg.actor).write(msg.payload)
    if isinstance(msg.payload, dict):
        container_msg = msg.payload

        '''
        with st.chat_message("system", avatar=avatar_ai):
            st.write(container_msg['intro'])
            uploaded_file = st.file_uploader('Add data')
        '''


prompt: str = st.chat_input("Enter a command")

if prompt:
    st.session_state["messages"].append(Message(actor="user", payload=prompt))
    st.chat_message("user").write(prompt)

    if "wafermap" in prompt:
        st.session_state["wafermap"] = "true"

if st.session_state["wafermap"] == "true":
    with st.chat_message("system", avatar=avatar_ai):
        container_msg = {}
        container_msg['intro'] = 'Upload data for analysis.'
        st.write(container_msg['intro'])

        uploaded_file = st.file_uploader('Add data')
        if uploaded_file is not None:
            wafermaps = np.load(BytesIO(uploaded_file.getvalue()))
            wafermap = wafermaps['arr_0']
            wafermap_str = str(wafermap.tolist())

            with st.spinner("Analyzing data.."):
                #fig = plot_wafer_map(wafermap)
                res = get_response(get_sys_prompt(), wafermap_str)
                clean_res = res.choices[0].message.content.strip("'")
                code = clean_res.strip("```")
                code = code.strip("python")

                st.header('Generated Code:')
                st.code(code, language='python')

                #exec(code)

                

                def run_gen_code(wafermap):
                    fig = None
                    exec(code)
                    return fig
                
                fig = run_gen_code(wafermap)
                st.pyplot(fig)

                
