from langchain_community.chat_models import ChatOpenAI

class MedicalSummarizer:
    """
    A class to summarize transcribed audio into a doctor's note.
    """
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.2):
        """
        Initialize the MedicalSummarizer with a language model.
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    def summarize_transcription(self, transcription: str) -> str:
        """
        Summarize the transcribed audio into a doctor's note.

        Parameters:
            transcription (str): The transcribed audio text.

        Returns:
            str: A summarized doctor's note.
        """
        prompt = f"""
        You are an expert medical assistant. Your task is to take detailed transcriptions
        of patient conversations or medical consultations and create a concise doctor's note.
        You need to make sure to get all the clients symptoms and other information that is mentioned 
        and also write what the doctor has recommended, if anything.

        Transcription: {transcription}

        Please provide a professional and concise summary as a doctor's note:
        """
        try:
            response = self.llm(chat_prompt=prompt)
            return response
        except Exception as e:
            return f"An error occurred while summarizing: {e}"
