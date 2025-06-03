import streamlit as st
import os

st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    .centered-heading {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: white;
        margin-bottom: 40px;
    }
    .content-column {
        padding: 20px;
        margin: 0 15px;
    }
    .content-column > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
    }
    .video-section {
        margin-top: 50px;
        text-align: center;
    }
    .video-title {
        font-size: 2em;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
    }
    .video-description {
        font-size: 1em;
        color: white;
        text-align: left;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True)

st.markdown('<h1 class="centered-heading">HOW TO BECOME A COACH</h1>', unsafe_allow_html=True)


col1, col2, col3 = st.columns([1, 1, 1])

def load_image(image_path):
    if os.path.exists(image_path):
        return st.image(image_path, use_container_width=True)
    else:
        st.warning(f"Image not found: {image_path}")
        return None

with col1:
    st.markdown('<div class="content-column">', unsafe_allow_html=True)
    load_image("images/Duke1.png")
    st.markdown("### 1. Plan & Study The Game")
    st.markdown("""
    - Start by playing basketball (high school and/or AAU) and studying coaching techniques.
    - **Earn playing experience** in high school and ideally in college (NCAA/NJCAA/NAIA).
    - **Study coaching** through books, online courses, and by watching the great coaches.
    - Build a deep understanding of the game (tactics, player development, team management).
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="content-column">', unsafe_allow_html=True)
    load_image("images/NBAColleges.png")
    st.markdown("### 2. Get Educated & Build Experience")
    st.markdown("""
    - Pursue a degree (usually in Sports Science, Physical Education, or a related field).
    - **Volunteer or become a graduate assistant** with a college basketball program.
    - **Start coaching** at youth levels, high school, or as an assistant at a college.
    - Consider getting certified through programs like USA Basketball's Coach License.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="content-column">', unsafe_allow_html=True)
    load_image("images/Duke.png")
    st.markdown("### 3. Climb the Coaching Ladder")
    st.markdown("""
    - From here, it's about networking and climbing up from assistant roles to head coach jobs.
    - **Become an assistant coach** at a college, G-League, or even international level.
    - **Get noticed** through strong performance, networking, and player development success.
    - **Top-level coaches often get interviewed** for NBA assistant or head coaching jobs through these channels.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Add video section
st.markdown('<div class="video-section">', unsafe_allow_html=True)
st.markdown('<h2 class="video-title">Recommended Videos</h2>', unsafe_allow_html=True)

# Two columns for videos
video_col1, video_col2 = st.columns(2)

with video_col1:
    VIDEO_URL = "https://www.youtube.com/watch?v=J0TfAic2Jfk"
    try:
        st.video(VIDEO_URL)
    except Exception as e:
        st.error(f"Failed to load video: {VIDEO_URL}. Error: {str(e)}")
    st.markdown("""
    <div class="video-description">
    <strong>How to Become a Basketball Coach - Basketball Coaching Tips</strong><br>
    This video provides practical advice on starting a coaching career, including tips on gaining experience, building a coaching philosophy, and networking within the basketball community. More videos from this series: https://www.youtube.com/playlist?list=PLCXERy73Oiz-9QWdPGI-Z1AFXGLTU8ThI
    </div>
    """, unsafe_allow_html=True)

with video_col2:
    VIDEO_URL = "https://www.youtube.com/watch?v=7WlwWhegq-s"
    try:
        st.video(VIDEO_URL)
    except Exception as e:
        st.error(f"Failed to load video: {VIDEO_URL}. Error: {str(e)}")
    st.markdown("""
    <div class="video-description">
    <strong>Basketball Coaching: Tips From a Pro</strong><br>
    A professional coach shares insights on effective coaching techniques, player development strategies, and how to advance from assistant to head coaching roles. More videos from this series: https://www.youtube.com/playlist?list=PL4SqFkuN1MD4VaEyh-ksxvxuOULnwpShc
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)