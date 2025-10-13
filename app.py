import gradio as gr
import os
import urllib.request
from spleeter.separator import Separator
from basic_pitch.inference import predict_and_save

def process_audio(audio_file):
    try:
        # 1. Set up folders
        os.makedirs("models", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("transcribed_output", exist_ok=True)

        # 2. Ensure Spleeter model is available locally
        model_path = "models/2stems.tar.gz"
        if not os.path.exists(model_path):
            print("Downloading Spleeter 2-stem model...")
            urllib.request.urlretrieve(
                "https://huggingface.co/spaces/akhaliq/Spleeter/resolve/main/2stems.tar.gz",
                model_path
            )
            print("Download complete!")

        # 3. Load Spleeter with local model
        separator = Separator(model_path)

        # 4. Separate instrumental and vocals
        separator.separate_to_file(audio_file, "outputs")

        # 5. Locate instrumental file
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        instrumental_path = f"outputs/{base_name}/other.wav"

        # 6. Run transcription to MIDI and sheet music
        predict_and_save(
            [instrumental_path],
            "transcribed_output"
        )

        midi_file = f"transcribed_output/{base_name}_basic_pitch.mid"

        # Return the MIDI file path for download
        return midi_file, f"Success! Transcribed and saved as {midi_file}"

    except Exception as e:
        return None, f"Error: {str(e)}"


# Define the Gradio app
app = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Upload an MP3"),
    outputs=[
        gr.File(label="Download Transcribed MIDI"),
        gr.Textbox(label="Status")
    ],
    title="Reverie AI: Song to Sheet Music",
    description=(
        "Upload any MP3 file to automatically extract instrumentals "
        "and generate sheet music using AI."
    ),
    allow_flagging="never"
)

# Launch app
if __name__ == "__main__":
    app.launch()
