from utils.speech_utils import recognize_speech

def convert_speech_to_text(save_file="output.txt", language="en-US"):
    """
    Convert speech to text and save it to a file.
    """
    text = recognize_speech(language)
    if text:
        with open(save_file, "a") as f:
            f.write(text + "\n")
        print(f"Saved to {save_file}")

if __name__ == "__main__":
    convert_speech_to_text()