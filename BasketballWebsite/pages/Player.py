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
        text-transform: uppercase;
    }
    .content-column {
        padding: 20px;
        margin: 0 15px;
    }
    .content-column > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .content-column img {
        max-width: 100%;
        height: auto;
        margin-bottom: 15px;
    }
    .content-column ul {
        text-align: left;
        padding-left: 20px;
        margin-top: 10px;
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

st.markdown('<h1 class="centered-heading">HOW TO BECOME A PLAYER</h1>', unsafe_allow_html=True)

# Three equal-sized columns with added spacing
col1, col2, col3 = st.columns([1, 1, 1])

# Function to safely load images
def load_image(image_path):
    if os.path.exists(image_path):
        return st.image(image_path, use_container_width=False)
    else:
        st.warning(f"Image not found: {image_path}")
        return None

with col1:
    st.markdown('<div class="content-column">', unsafe_allow_html=True)
    load_image("images/Training.png")  # Placeholder for training image
    st.markdown("### 1. Master the Fundamentals & Play Competitively")
    st.markdown("""
    <ul>
        <li>Start young and develop elite skills: shooting, ball-handling, defense, court vision.</li>
        <li>Join competitive youth leagues (AAU, club teams) to get serious reps and face high-level competition.</li>
        <li>Play for your middle and high school teams to gain visibility and build a strong foundation.</li>
        <li>Work with a coach or trainer to refine techniques and create personalized drills.</li>
        <li>Practice consistently, focusing on agility, endurance, and game IQ to stand out early.</li>
        <li>Participate in local tournaments to showcase your skills to scouts and recruiters.</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="content-column">', unsafe_allow_html=True)
    load_image("images/NCAA.png")
    st.markdown("### 2. Get Noticed & Play College Basketball")
    st.markdown("""
    <ul>
        <li>Attend exposure camps and tournaments (AAU circuit is key) to catch the eye of college coaches.</li>
        <li>Get recruited to play at the college level - D1, D2, D3, NAIA, or JUCO, depending on your skill level.</li>
        <li>Excel on and off the court - scouts and coaches look for talent, character, academics, and leadership.</li>
        <li>Build a highlight reel showcasing your best plays and send it to college programs.</li>
        <li>Network with coaches and attend college showcases to increase your exposure.</li>
        <li>Maintain good grades, as academic eligibility is crucial for college sports scholarships.</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="content-column">', unsafe_allow_html=True)
    load_image("images/NBA.png")  # Placeholder for NBA logo
    st.markdown("### 3. Go Pro (NBA, G-League, or Overseas)")
    st.markdown("""
    <ul>
        <li>If you're elite, declare for the NBA Draft or get into the G-League after college or high school.</li>
        <li>Not drafted? Play overseas or in the G-League to build your résumé and gain professional experience.</li>
        <li>Keep improving: Many players work their way up through summer leagues, international ball, and tryouts.</li>
        <li>Work with agents or mentors to navigate contract negotiations and career opportunities.</li>
        <li>Stay in top physical condition with year-round training and nutrition plans.</li>
        <li>Study the game through film analysis to adapt to professional-level strategies.</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Add video section
st.markdown('<div class="video-section">', unsafe_allow_html=True)
st.markdown('<h2 class="video-title">Recommended Videos</h2>', unsafe_allow_html=True)

# Two columns for videos
video_col1, video_col2 = st.columns(2)

with video_col1:
    VIDEO_URL = "https://www.youtube.com/watch?v=aDwC_m8wUs0"
    try:
        st.video(VIDEO_URL)
    except Exception as e:
        st.error(f"Failed to load video: {VIDEO_URL}. Error: {str(e)}")
    st.markdown("""
    <div class="video-description">
    <strong>LeBron James Documentary</strong><br>
    This documentary explores LeBron James' journey to becoming an NBA superstar, highlighting his skills and work ethic.
    </div>
    """, unsafe_allow_html=True)

with video_col2:
    VIDEO_URL = "https://www.youtube.com/watch?v=ZXyIHe-iY7Q"
    try:
        st.video(VIDEO_URL)
    except Exception as e:
        st.error(f"Failed to load video: {VIDEO_URL}. Error: {str(e)}")
    st.markdown("""
    <div class="video-description">
    <strong>Stephen Curry Documentary</strong><br>
    This documentary follows Stephen Curry's rise to NBA stardom, focusing on his shooting prowess and determination.
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)