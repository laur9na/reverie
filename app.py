import gradio as gr
import os
from spleeter.separator import Separator
from basic_pitch.inference import predict_and_save

def process_audio(audio_file):
    try:
        # 1. Set up folders
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("transcribed_output", exist_ok=True)

        # 2. Separate instrumental + vocals
        separator = Separator("spleeter:2stems")
        separator.separate_to_file(audio_file, "outputs")

        # 3. Locate instrumental file
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        instrumental_path = f"outputs/{base_name}/other.wav"

        # 4. Run transcription → MIDI + sheet music
        predict_and_save(
            [instrumental_path],
            "transcribed_output"
        )

        midi_file = f"transcribed_output/{base_name}_basic_pitch.mid"
        note_output = f"Success! Transcribed and saved as {midi_file}"
        return note_output

    except Exception as e:
        return f"Error: {str(e)}"

app = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="🎵 Upload an MP3"),
    outputs=gr.Textbox(label="Status"),
    title="🎶 Reverie AI: Song → Sheet Music",
    description="Upload any MP3 file to automatically extract instrumentals and generate sheet music using AI.",
    allow_flagging="never"
)

if __name__ == "__main__":
    app.launch()
