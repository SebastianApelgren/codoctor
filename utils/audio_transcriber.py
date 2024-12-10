import os
import whisper

class AudioTranscriber:
    def __init__(self, model_name="tiny"):
        """
        Initialize the AudioTranscriber with a specified Whisper model.

        :param model_name: Whisper model to use (e.g., 'tiny', 'base', 'small', 'medium', 'large').
        """
        self.model = whisper.load_model(model_name)
    
    def transcribe(self, directory, file_name):
        """
        Transcribe an audio file located in the specified directory.

        :param directory: The directory where the audio file is located.
        :param file_name: The name of the audio file to transcribe.
        :return: The transcription as a string.
        :raises FileNotFoundError: If the specified file does not exist.
        """
        # Construct the full file path
        file_path = os.path.join(directory, file_name)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Transcribe the audio file
        transcription_result = self.model.transcribe(file_path)
        
        # Return the transcription
        return transcription_result["text"]