def detect_emotion(text):
    text = text.lower()

    if any(w in text for w in ["sad", "cry", "depressed"]):
        return "Sad"
    elif any(w in text for w in ["happy", "good", "great"]):
        return "Happy"
    elif any(w in text for w in ["angry", "mad"]):
        return "Angry"
    elif any(w in text for w in ["anxious", "worried"]):
        return "Anxious"
    else:
        return "Neutral"