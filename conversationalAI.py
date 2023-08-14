import streamlit as st
import openai
import PyPDF2
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

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
openai_api_key = "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"
openai.api_key = openai_api_key

# Function to read PDF with error handling
def read_pdf(file):
    try:
        # PDF多页显示
        pdf_reader = PyPDF2.PdfReader(file)
        page_nums = range(len(pdf_reader.pages))
        selected_pages = st.multiselect('Select pages:', page_nums, default = page_nums)

        text = ""
        for page_num in selected_pages:
            text += pdf_reader.pages[page_num].extract_text()
        return text
    
    except Exception as e:
        st.warning(f"An error occurred while reading the PDF: {str(e)}")
        return ""

# Function to perform OCR on an image with error handling
def ocr_image(image, lang="eng"):
    try:
        # Convert Image to PIL Format
        image = Image.open(image).convert("RGB")

        # Convert to graysacle
        image = image.convert('L')

        # Binarization using a threshold
        threshold = 128
        image = image.point(lambda p: p > threshold and 255)

        # Denoise using a median filter
        image = image.filter(ImageFilter.MedianFilter(3))

        # Image enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        text = pytesseract.image_to_string(image, lang=lang)
        return text
    
    except Exception as e:
        st.warning(f"An error occurred while performing OCR: {str(e)}")
        return ""

# 初始化数据库
def init_db():
    conn = sqlite3.connect('data_stats.db')
    c = conn.cursor()

    # 检查表是否已存在
    c.execute('''SELECT name FROM sqlite_master WHERE type='table' AND name='stats';''')
    result = c.fetchall()

    # 如果表不存在，创建
    if len(result) == 0:
        c.execute('''CREATE TABLE IF NOT EXISTS stats
                    (timestamp TEXT, action TEXT, details TEXT)''')
        conn.commit()
    conn.close()

# 记录数据
def log_data(action, details=""):
    conn = sqlite3.connect('data_stats.db')
    c = conn.cursor()
    c.execute("INSERT INTO stats (timestamp, action, details) VALUES (datetime('now'), ?, ?)", (action, details))
    conn.commit()
    conn.close()

#获取统计数据
def get_stats():
    conn = sqlite3.connect('data_stats.db')
    c = conn.cursor()
    c.execute("SELECT action, COUNT(*) FROM stats GROUP BY action")
    data = c.fetchall()
    conn.close()
    return data

# Streamlit app
def app():
    st.title("Code Generator")
    
    # Sidebar with PDF and Image upload buttons
    with st.sidebar:
        st.subheader("File Uploads")
        pdf_file = st.file_uploader("Upload a PDF file", type=['pdf'])
        image_file = st.file_uploader("Upload an image for OCR", type=['png', 'jpg', 'jpeg'])

    # Use session state to maintain conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    user_input = st.text_input("You:", "")
    if user_input:
        log_data("Chatbot Interaction", user_input)
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

    # -----------PDF Reading------------
    if pdf_file:
        log_data("Uploaded PDF", f"Size: {pdf_file.size}")
        pdf_text = read_pdf(pdf_file)
        st.text(pdf_text)
    
    # -----------OCR------------
    # 图片预览和处理
    if image_file:
        log_data("Uploaded Image for OCR", f"Size: {image_file.size}")
    
        # 处理OCR识别的语言包
        languages = ["English", "简体中文"]
        selected_language = st.selectbox("Choose your language:", languages)
        language_map = {
            "English": "eng",
            "简体中文": "chi_sim"
        }
        lang_code = language_map[selected_language]

        image = Image.open(image_file)
        st.image(image, caption='Uploaded Image.', use_column_width = True)

        # Rotation Effect
        rotate_deg = st.slider('Rotate Degree:', min_value = 0, max_value = 360, value = 0)
        image = image.rotate(rotate_deg)

        # Crop Effect
        if st.button("Crop Image"):
            crop_options = st.columns(4)
            left = crop_options[0].number_input('Left', min_value=0, max_value=image.width, value=0)
            top = crop_options[1].number_input('Top', min_value=0, max_value=image.height, value=0)
            right = crop_options[2].number_input('Right', min_value=0, max_value=image.width, value=image.width)
            bottom = crop_options[3].number_input('Bottom', min_value=0, max_value=image.height, value=image.height)
            image = image.crop((left, top, right, bottom))
        
        # Zoom Effect
        zoom_level = st.slider('Zoom Level:', min_value = 0.5, max_value = 4.0, value = 1.0, step = 0.1)
        image = image.resize((int(image.width * zoom_level), int(image.height * zoom_level)))

        st.image(image, caption='Processed Image.', use_column_width = True)
        ocr_text = ocr_image(image_file, lang=lang_code)
        st.text(ocr_text)

    # 显示统计信息
    if st.checkbox('Show Statistics'):
        data = get_stats()
        actions, counts = zip(*data)

        actions = list(actions)
        counts = list(counts)

        # 设置Seaborn风格
        sns.set_theme(style="whitegrid")

        # 使用Seaborn绘制条形图
        plt.figure(figsize=(10, 5))
        print(actions)
        print(counts)
        ax = sns.barplot(x=actions, y=counts, palette="viridis")
        ax.set_title("Action Counts", fontsize=16)
        ax.set_xlabel("Actions", fontsize=14)
        ax.set_ylabel("Counts", fontsize=14)

        st.pyplot(plt.gcf())
        # st.bar_chart({"Action Count": counts}, x_axis_label="Actions", y_axis_label="Counts")

if __name__ == "__main__":
    init_db()
    app()
