from openai import OpenAI
import streamlit as st
from io import BytesIO
import numpy as np
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import measurements

avatar_ai = "📀"
PROMPT_PATH = "./prompts/data_prompt5.txt"

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


@st.cache_data
def plot_wafer_map(wafer_map):
    # Identify the circle to mask out non-wafer areas
    y, x = np.ogrid[:wafer_map.shape[0], :wafer_map.shape[1]]
    center = (np.array(wafer_map.shape)-1)/2
    distance_to_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    mask = distance_to_center > center[0]

    # Apply mask
    wafer_map[mask] = 0
    
    # Create colormap
    cmap = plt.cm.viridis
    cmap.set_under(color='white')  # Set the background color

    # Define plotting
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(wafer_map, cmap=cmap, vmin=0.1)  # vmin=0.1 to use 'under' in cmap

    # Outline defects
    labeled_array, num_features = measurements.label(wafer_map==2)
    areas = measurements.sum(wafer_map==2, labeled_array, index=np.arange(labeled_array.max() + 1))
    area_map = areas[labeled_array]
    contours = np.logical_and(wafer_map == 2, area_map > 5)  # Highlight large clusters

    plt.contour(contours, colors='red', linewidths=1.5)  # Draw contour around defects clusters
    
    #plt.colorbar()
    plt.title('Wafer Map with Defective Chips Highlighted')
    plt.axis('off')  # Turn off axis numbers and ticks
    #plt.show()

    return fig


@dataclass
class Message:
    actor: str
    payload: str


st.title("Fab-X")

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

    if "w" in prompt:
        st.session_state["wafermap"] = "true"

if st.session_state["wafermap"] == "true":
    with st.chat_message("system", avatar=avatar_ai):
        container_msg = {}
        container_msg['intro'] = 'Upload data for analysis.'
        st.write(container_msg['intro'])

        uploaded_file = st.file_uploader('Add data')
        if uploaded_file is not None:
            wafermaps = np.load(BytesIO(uploaded_file.getvalue()))
            wafermap = wafermaps['arr_0'][25]
            wafermap_str = str(wafermap.tolist())

            with st.spinner("Analyzing data.."):
                #fig = plot_wafer_map(wafermap)
                res = get_response(get_sys_prompt(), wafermap_str)
                clean_res = res.choices[0].message.content.strip("'")
                code = clean_res.strip("```")
                code = code.strip("python")

                st.write('Generated Code:')
                st.code(code, language='python')

                def run_gen_code(wafermap):
                    fig = None
                    exec(code)
                    return fig
                
                st.pyplot(run_gen_code(wafermap))


