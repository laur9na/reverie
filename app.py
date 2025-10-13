import gradio as gr
import os
import urllib.request
import tarfile
from spleeter.separator import Separator
from basic_pitch.inference import predict_and_save


def download_model():
    os.makedirs("models", exist_ok=True)
    model_dir = "models/2stems"
    model_tar_path = "models/2stems.tar.gz"

    # Skip download if model already exists
    if os.path.exists(os.path.join(model_dir, "pretrained_model")):
        print("Model already exists, skipping download.")
        return model_dir

    print("Downloading and extracting Spleeter model...")

    # Stable direct GitHub link to 2stems.tar.gz
    model_url = "https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz"

    # Handle redirects manually to avoid 404
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    urllib.request.urlretrieve(model_url, model_tar_path)
    print("Download complete!")

    # Extract into models/
    with tarfile.open(model_tar_path, "r:gz") as tar:
        tar.extractall("models")

    print("Extraction complete!")
    return model_dir


def process_audio(audio_file):
    try:
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("transcribed_output", exist_ok=True)

        # 1. Ensure model is ready
        model_dir = download_model()

        # 2. Initialize Separator
        separator = Separator(os.path.join(model_dir, "pretrained_model"))

        # 3. Separate instrumental + vocals
        separator.separate_to_file(audio_file, "outputs")

        # 4. Locate instrumental
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        instrumental_path = f"outputs/{base_name}/other.wav"

        # 5. Transcribe instrumental → MIDI
        predict_and_save([instrumental_path], "transcribed_output")

        midi_file = f"transcribed_output/{base_name}_basic_pitch.mid"
        return f"Success! Transcribed and saved as {midi_file}"

    except Exception as e:
        return f"Error: {str(e)}"


# Gradio app
app = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Upload an MP3"),
    outputs=gr.Textbox(label="Status"),
    title="Reverie AI: Song → Sheet Music",
    description="Upload an MP3 to extract instrumentals and generate sheet music using AI.",
    allow_flagging="never"
)

if __name__ == "__main__":
    app.launch()
