import os

API_KEY = os.getenv("API_KEY", "dev-key")
PORT = int(os.getenv("PORT", "10000"))
PHONEMIZER_LANGUAGE = "nl"
PHONEMIZER_BACKEND = "espeak"
FAISS_TOP_K = 5
SIMILARITY_THRESHOLD = 0.15

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
