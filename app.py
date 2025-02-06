import os
import time
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from graph import invoke_our_graph
from st_callable_util import get_streamlit_cb
from realsensestream import ColorRealSenseStream, ColorFrame

# Create two columns - one for chat, one for camera
st.set_page_config(layout="wide")
col2, col1 = st.columns([3, 3])

with col1:
    
    # Initialize the chat session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state["messages"] = [AIMessage(content="Hello!")]

    # Loop through all messages and display them
    for msg in st.session_state.messages:
        if isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

    # Handle new user input
    if prompt := st.chat_input():
        # Add user message to session state and display it
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)

        # Process AI response
        with st.chat_message("assistant"):
            # Create container for streaming
            response_container = st.container()
            
            # Get AI response
            with response_container:
                st_callback = get_streamlit_cb(response_container)
                response = invoke_our_graph(st.session_state.messages, [st_callback])
                
                # Display the response immediately
                latest_response = response["messages"][-1].content
                st.write(latest_response)
                
                # Add to session state after displaying
                st.session_state.messages.append(AIMessage(content=latest_response))

with col2:
    # Camera stream container
    frame_placeholder = st.empty()

    # Initialize camera feed stream if not already present
    if 'camera_stream' not in st.session_state:
        st.session_state.camera_stream = ColorRealSenseStream()

    # Initialize last_frame in session_state
    if 'last_frame' not in st.session_state:
        st.session_state.last_frame = None

    try:
        while True:
            # Get frame from camera
            frame_data: ColorFrame = st.session_state.camera_stream.streaming_color_frame()
            
            # Store current frame
            st.session_state.last_frame = frame_data.image
            
            # Display frame from the camera
            frame_placeholder.image(
                frame_data.image,
                channels="RGB",
                use_container_width=True
            )
            
            time.sleep(0.01)

    except Exception as e:
        st.error(f"Camera stream error: {str(e)}")
        
    finally:
        # This only runs when the session ends or on error
        if 'camera_stream' in st.session_state:
            st.session_state.camera_stream.stop()
            # del st.session_state.camera_stream