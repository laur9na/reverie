import gradio as gr
import os
import urllib.request
import tarfile
from spleeter.separator import Separator
from basic_pitch.inference import predict_and_save


def process_audio(audio_file):
    try:
        # 1. Set up folders
        os.makedirs("models", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("transcribed_output", exist_ok=True)

        # 2. Download + extract Spleeter model if missing
        model_dir = "models/2stems"
        model_tar_path = "models/2stems.tar.gz"

        if not os.path.exists(os.path.join(model_dir, "pretrained_model")):
            print("Downloading and extracting Spleeter model...")
            urllib.request.urlretrieve(
                "https://huggingface.co/spaces/akhaliq/Spleeter/resolve/main/2stems.tar.gz",
                model_tar_path
            )

            with tarfile.open(model_tar_path, "r:gz") as tar:
                tar.extractall("models")

            print("Model downloaded and extracted!")

        # 3. Initialize Separator
        separator = Separator(os.path.join(model_dir, "pretrained_model"))

        # 4. Separate instrumental + vocals
        separator.separate_to_file(audio_file, "outputs")

        # 5. Locate instrumental file
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        instrumental_path = f"outputs/{base_name}/other.wav"

        # 6. Transcribe instrumental to MIDI + sheet music
        predict_and_save(
            [instrumental_path],
            "transcribed_output"
        )

        midi_file = f"transcribed_output/{base_name}_basic_pitch.mid"
        return f"Success! Transcribed and saved as {midi_file}"

    except Exception as e:
        return f"Error: {str(e)}"


# Gradio app setup
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
