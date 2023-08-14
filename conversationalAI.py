import streamlit as st
import openai
import PyPDF2
import pytesseract
from PIL import Image

# Custom CSS to style the input bar and code box
st.markdown("""
    <style>
        .stTextInput input {
            background-color: #f0f0f0;
            color: #000000;  # Font color for input bar
        }
        .stCodeBlock {
            background-color: #333333;  # Dark background for code box
            color: #f0f0f0;  # Light font color for code box
        }
    </style>
""", unsafe_allow_html=True)

# Set your OpenAI API key
openai_api_key = ""
openai.api_key = openai_api_key

# Function to read PDF with error handling
def read_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page_num in range(pdf_reader.numPages):
            text += pdf_reader.getPage(page_num).extractText()
        return text
    except Exception as e:
        st.warning(f"An error occurred while reading the PDF: {str(e)}")
        return ""

# Function to perform OCR on an image with error handling
def ocr_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.warning(f"An error occurred while performing OCR: {str(e)}")
        return ""

# Streamlit app
def app():
    st.title("Conversational AI Chatbot")
    
    # PDF reading
    pdf_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    if pdf_file:
        pdf_text = read_pdf(pdf_file)
        st.text(pdf_text)
    
    # OCR
    image_file = st.file_uploader("Upload an image for OCR", type=['png', 'jpg', 'jpeg'])
    if image_file:
        ocr_text = ocr_image(image_file)
        st.text(ocr_text)

    # Use session state to maintain conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    user_input = st.text_input("You:", "")
    if user_input:
        st.session_state.conversation_history.append(f"You: {user_input}")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
        )
        bot_response = response["choices"][0]["message"]["content"]
        st.session_state.conversation_history.append(f"Bot: {bot_response}")
        
        # Display conversation history
        for message in st.session_state.conversation_history:
            if "Bot:" in message and "```" in message:  # Check if the message contains code
                st.code(message.replace("Bot:", "").strip())  # Display code in a code box
            else:
                st.text(message)
    
    # Button to clear chat history
    if st.button('Clear Chat'):
        st.session_state.conversation_history = []

if __name__ == "__main__":
    app()