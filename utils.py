from emoji import replace_emoji
import re

def clean_text(text):
    if text is None: return ""

    text = replace_emoji(text, "")
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)

    return text.lower().strip()