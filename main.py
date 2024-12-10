from utils import AudioTranscriber, MedicalSummarizer

def main():
    # Specify the audio file directory and file name
    audio_directory = "audio_files"
    audio_file_name = "Test_audio.m4a"

    # Initialize the AudioTranscriber with the desired Whisper model
    transcriber = AudioTranscriber(model_name="small")
    summarizer = MedicalSummarizer(model_name="gpt-3.5-turbo", temperature=0.2)

    try:
        # Transcribe the audio file
        transcription = transcriber.transcribe(audio_directory, audio_file_name)

        # Summarize the transcription into a doctor's note
        doctor_note = summarizer.summarize_transcription(transcription)

        print("Doctor's Note:")
        print(doctor_note)

    except FileNotFoundError as e:
        # Handle the case where the audio file is not found
        print(f"Error: {e}")
    except Exception as e:
        # Handle any other exceptions
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()