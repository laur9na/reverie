# splice_song.py
import os
from spleeter.separator import Separator # type: ignore
from tkinter import Tk, filedialog

# ---- CONFIGURATION ----
OUTPUT_FOLDER = "spliced_output"

def splice(song_filename, output_path):
    """Uses Spleeter to separate a song into 4 stems."""
    print(f"\nLoading the Spleeter AI model... (this might take a moment)")
    separator = Separator("spleeter:4stems")

    print(f"Splicing '{song_filename}'... Please wait.")
    separator.separate_to_file(song_filename, output_path)

    print(f"\nSuccess! Your separated audio files are in the '{output_path}' folder.")

if __name__ == "__main__":
    # Hide the Tkinter root window
    Tk().withdraw()

    # Open Finder file picker
    song_path = filedialog.askopenfilename(
        title="Select an MP3 file to splice",
        filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.flac")],
    )

    if not song_path:
        print("No file selected. Exiting.")
    elif not os.path.exists(song_path):
        print(f"File not found: {song_path}")
    else:
        splice(song_path, OUTPUT_FOLDER)
