import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from deep_translator import GoogleTranslator
from urllib.parse import urlparse, parse_qs

def get_video_id(url):
    """Extract the video ID from various YouTube URL formats."""
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
            return parse_qs(parsed_url.query).get("v", [None])[0]
        elif parsed_url.hostname in ["youtu.be"]:
            return parsed_url.path.lstrip("/")
        return None
    except Exception:
        return None

def get_transcript(video_id):
    """Fetch the Hindi transcript."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi'])
        return " ".join([entry['text'] for entry in transcript])
    except Exception:
        return None

def translate_to_english(hindi_text):
    """Translate Hindi text to English using deep-translator."""
    try:
        translated = GoogleTranslator(source='hi', target='en').translate(hindi_text)
        return translated
    except Exception:
        return "Translation failed. Please try again later."

# Streamlit App
st.title("YouTube Hindi-to-English Transcript Translator")
st.write("Enter a YouTube video link, and this app will fetch the Hindi transcript and translate it into English.")

video_url = st.text_input("Enter YouTube Video URL:")

if st.button("Transcribe & Translate"):
    if video_url:
        video_id = get_video_id(video_url)
        if video_id:
            hindi_transcript = get_transcript(video_id)
            if hindi_transcript:
                english_translation = translate_to_english(hindi_transcript)
                st.subheader("English Translation:")
                st.write(english_translation)
            else:
                st.error("Hindi transcript not available for this video.")
        else:
            st.error("Invalid YouTube URL. Please check the link and try again.")
    else:
        st.error("Please provide a valid YouTube video URL.")
