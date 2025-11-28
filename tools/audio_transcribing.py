from langchain.tools import tool
import whisper
import os
import re

model = whisper.load_model("base")

@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio AND return the SUM of all numbers spoken.
    Returns a stringified integer ready for submission.
    """
    try:
        fp = os.path.join("LLMFiles", file_path)

        # run whisper
        result = model.transcribe(fp)
        text = result["text"]

        # extract all numbers
        nums = re.findall(r"\d+", text)
        nums_int = [int(n) for n in nums]

        if not nums_int:
            return "Error: No numbers detected"

        # sum them
        total = sum(nums_int)
        return str(total)

    except Exception as e:
        return f"Error: {e}"
