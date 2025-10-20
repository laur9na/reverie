import gradio as gr
import os
import tempfile
import shutil
from spleeter.separator import Separator
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

def process_audio(audio_file):
    try:
        if audio_file is None:
            return "Error: Please upload an audio file."

        # 1. Create temp + output directories
        tmp_dir = tempfile.mkdtemp()
        outputs_dir = os.path.join(tmp_dir, "outputs")
        transcribed_dir = os.path.join(tmp_dir, "transcribed_output")
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(transcribed_dir, exist_ok=True)

        # 2. Separate into 4 stems (vocals, drums, bass, other)
        os.environ["MODEL_PATH"] = "pretrained_models"
        separator = Separator('spleeter:4stems', stft_backend='librosa', multiprocess=False)
        separator.separate_to_file(audio_file, outputs_dir)

        # 3. Locate the 'other' stem (instrumental)
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        instrumental_path = os.path.join(outputs_dir, base_name, "other.wav")
        if not os.path.exists(instrumental_path):
            raise FileNotFoundError("Instrumental file not found after separation.")

        # 4. Transcribe instrumental to MIDI
        predict_and_save(
            [instrumental_path],
            transcribed_dir,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )

        # 5. Rename and save MIDI as <songname>.mid
        midi_file = os.path.join(
            transcribed_dir,
            f"{os.path.splitext(os.path.basename(instrumental_path))[0]}_basic_pitch.mid"
        )
        if not os.path.exists(midi_file):
            raise FileNotFoundError("MIDI file not found after transcription.")

        final_name = f"{base_name}.mid"
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads", final_name)
        shutil.copy(midi_file, downloads_path)

        return f"Your MIDI file has been saved to your Downloads folder! \n{downloads_path}"

    except Exception as e:
        return f"Error: {str(e)}"


# --- Interface ---
app = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="UPLOAD MP3 HERE"),
    outputs=gr.Textbox(label="STATUS"),
    title="REVERIE AI - TURN ANY SONG INTO SHEET MUSIC",
    description=(
        "<p>1. Upload an MP3 file.</p>"
        "<p>2. The app separates vocals, drums, bass, and instruments, then converts the instrumental stem into a MIDI file.</p>"
        "<p>3. The MIDI file will be automatically saved to your Downloads folder (can take around 1-3 minutes).</p>"
        "<p>4. Download <a href='https://musescore.org' target='_blank'>MuseScore</a>.</p>"
        "<p>5. Open your MIDI file from your Downloads folder with MuseScore.</p>"
    ),
    allow_flagging="never",
    theme = "laur9na/pink"
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
