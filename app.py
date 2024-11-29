import os
import uvicorn
import tempfile
from pathlib import Path
from transcribe_data import transcribe
from main import extrcated_information_from_audio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from logger import create_logger
app =FastAPI(debug=True)
logger = create_logger(__name__)

def get_audio_file_extension(file_path):
    extension = Path(file_path).suffix
    return extension.lower()

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...),
                        llm: str = Form(None),
                        ):
    temp_file_path = None
    try:
        
        logger.info(f"Processing audio file: {file.filename}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
            logger.info(f"Audio file saved temporarily at: {temp_file_path}")

        # Transcribe audio
        try:
            transcription = transcribe(temp_file_path, language="en")
            if not transcription:
                raise ValueError("Transcription returned empty result")
            logger.info("Transcription completed successfully")
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Transcription failed")

        # Extract features and summary
        try:
            final_output= extrcated_information_from_audio(transcription,llm)
            logger.info("Feature extraction and summary generation is completed successfully")
            return final_output
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Feature extraction failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
    
    finally:
        # Cleanup temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.error(f"Failed to cleanup temporary file: {str(e)}")


