import os
from groq import Groq
from dotenv import load_dotenv
from logger import create_logger

load_dotenv()
logger = create_logger(__name__)

def transcribe( audio_file: str,
                response_format: str = "text",
                language: str = "en"):
    
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("WHISPER_MODEL")

    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    if not model:
        raise ValueError("WHISPER_MODEL environment variable is not set")

    prompt_template = """
                Accurately transcribe the audio conversation between the patient and the doctor. Ensure the transcription is clear and precise, 
                capturing every detail of the discussion, including all medical terms, diagnoses, symptoms, and treatment plans that are mentioned. 
                Emphasize the correct spelling and usage of medical terminology, 
                and clearly differentiate between questions asked by the doctor and responses or concerns expressed by the patient. 
                The goal is to produce a comprehensive and accurate representation of the conversation that can be used for medical documentation or review.
                """
    try:
        # Validate file existence
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Initialize OpenAI client
        client = Groq(api_key=api_key)

        # Process the audio file
        with open(audio_file, "rb") as file:
            logger.info(f"Starting transcription of {audio_file}")
            response = client.audio.transcriptions.create(
                file=file,
                prompt=prompt_template,
                response_format=response_format,
                model=model,
                language=language
            )
            logger.info("Transcription completed successfully")
            audi_json={"audio_file_path":audio_file,"transcriptions":response}
            logger.info(f"Transcription response: {audi_json}")
            return response

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise Exception(f"Transcription failed: {str(e)}")

# path=r"D:\Gen_AI\Cranial-Nerve-Examination-DeepD.MP3"
# response=transcribe(audio_file=path,language="en")

# paths=[r"D:\Gen_AI\uop-apha-asp-patient-counseling-competition-2023-amandip-chauhan-128-ytshorts.savetube.me.mp3",
#        r"D:\Gen_AI\uop-apha-asp-patient-counseling-competition-2023-an-nguyen-128-ytshorts.savetube.me.mp3",
#        r"D:\Gen_AI\uop-apha-asp-patient-counseling-competition-2023-jasmin-prasad-128-ytshorts.savetube.me.mp3",
#        r"D:\Gen_AI\uop-apha-asp-patient-counseling-competition-2023-maygree-lindor-dee-128-ytshorts.savetube.me.mp3",
#        r"D:\Gen_AI\uop-apha-asp-patient-counseling-competition-2023-patrick-delgado-128-ytshorts.savetube.me.mp3"]
# texts=[]
# for path in paths:
#     print(path)
#     text=transcribe(audio_file=path,language="en")
#     texts.append(text)
