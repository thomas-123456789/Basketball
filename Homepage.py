import streamlit as st
import base64
#import st_pages
from streamlit import switch_page

st.set_page_config(layout="wide")

if "chatbot" not in st.session_state:
    st.session_state["chatbot"] = {}

def chat_bot():
    # Load custom CSS from file
    def load_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    load_css('styles.css')

    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"

    # Header
    st.markdown(f"""
    <div class="header">
        <div class="animated-bg"></div>
        <div class="header-content">
            <h1 class="header-title">Ollama Chatbot Multi-Model Interface</h1> 
            <p class="header-subtitle">Advanced Language Models & Intelligent Conversations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced pages definition
    PAGES = {
        "Home": {
            "icon": "house-door",
            #"func": st_pages.page_home,
            "description": "Guidelines & Overview",
            "badge": "Informative",
            "color": "var(--primary-color)"
        },
        "Language Models Management": {
            "icon": "gear",
            #"func": st_pages.model_management,
            "description": "Download Models",
            "badge": "Configurations",
            "color": "var(--secondary-color)"
        },
        "AI Conversation": {
            "icon": "chat-dots",
            #"func": st_pages.ai_chatbot,
            "description": "Interactive AI Chat",
            "badge": "Application",
            "color": "var(--highlight-color)"
        },
        "RAG Conversation": {
            "icon": "chat-dots",
            #"func": st_pages.rag_chat,
            "description": "PDF AI Chat Assistant",
            "badge": "Application",
            "color": "var(--highlight-color)"
        }
    }

    st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
    """, unsafe_allow_html=True)

    def navigate():
        with st.sidebar:
            st.markdown('''
            <a href="https://github.com/TsLu1s/talknexus" target="_blank" style="text-decoration: none; color: inherit; display: block;">
                <div class="header-container" style="cursor: pointer;">
                    <div class="profile-section">
                        <div class="profile-info">
                            <h1 style="font-size: 38px;">TalkNexus</h1>
                            <span class="active-badge" style="font-size: 20px;">AI Chatbot Multi-Model Application</span>
                        </div>
                    </div>
                </div>
            </a>
            ''', unsafe_allow_html=True)

            st.markdown('---')

            # Create menu items
            for page, info in PAGES.items():
                selected = st.session_state.current_page == page

                # Create the button (invisible but clickable)
                if st.button(
                        f"{page}",
                        key=f"nav_{page}",
                        use_container_width=True,
                        type="secondary" if selected else "primary"
                ):
                    st.session_state.current_page = page
                    st.rerun()

                # Visual menu item
                st.markdown(f"""
                    <div class="menu-item {'selected' if selected else ''}">
                        <div class="menu-icon">
                            <i class="bi bi-{info['icon']}"></i>
                        </div>
                        <div class="menu-content">
                            <div class="menu-title">{page}</div>
                            <div class="menu-description">{info['description']}</div>
                        </div>
                        <div class="menu-badge">{info['badge']}</div>
                    </div>
                """, unsafe_allow_html=True)

            # Close navigation container
            st.markdown('</div>', unsafe_allow_html=True)

            if page == "Home":
                st.switch_page("pages/page_home.py")
            else:
                st.switch_page("pages/Chat With AI.py")

            return st.session_state.current_page

    # Get selected page and run its function
    try:
        selected_page = navigate()
        # Update session state
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            st.rerun()

        # Run the selected function
        page_function = PAGES[selected_page]["func"]
        page_function()
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        #st_pages.home.run()

    # Display the footer
    st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <p>© 2024 Powered by <a href="https://github.com/TsLu1s" target="_blank">TsLu1s </a>. 
            Advanced Language Models & Intelligent Conversations
            | Project Source: <a href="https://github.com/TsLu1s/talknexus" target="_blank"> TalkNexus</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
    }}
    .title {{
        font-size: 80px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-top: 150px;
    }}
    .subtitle {{
        font-size: 20px;
        color: white;
        text-align: center;
        margin-top: -40px;
    }}
    .button-row {{
        display: flex;
        justify-content: center;
        gap: 50px;
        margin-top: 50px;
    }}
    .stButton>button {{
        font-size: 24px;
        font-weight: bold;
        color: white;
        background-color: black;
        border-radius: 30px;
        width: 200px;
        height: 60px;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


background_image_path = "BackgroundHomepage.jpg"
set_png_as_page_bg(background_image_path)


st.markdown("""
<div class="subtitle">WELCOME TO</div>
<div class="title">BASKETBALL</div>
""", unsafe_allow_html=True)


st.markdown('<div class="button-row">', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col1:
    st.link_button("Athletes", "https://basketball-website.streamlit.app/Player")
with col3:
    st.link_button("Coaching", "https://basketball-website.streamlit.app/Coaching")

