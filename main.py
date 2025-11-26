import os
from logger import create_logger
from typing import List, Optional
from pydantic import BaseModel
from utility import llm_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,JsonOutputParser, StrOutputParser
logger = create_logger(__name__)

# Section 4: Comprehensive Medical History
class ChronicIllness(BaseModel):
    name: List[str] = []

class PreviousDiagnosis(BaseModel):
    condition: str
    diagnosis_date: Optional[str] = None

class SurgicalHistory(BaseModel):
    date: Optional[str] = None
    procedure: List[str] = []
    outcome: Optional[str] = None

class FamilyHistory(BaseModel):
    genetic_disorders: List[str] = []
    illnesses: List[str] = []

class Allergy(BaseModel):
    drug_allergies: List[str] = []
    food_allergies: List[str] = []
    environmental_allergies: List[str] = []

class Medication(BaseModel):
    name: List[str] = []
    dosage: str 
    frequency: str
    duration: Optional[str] = "N/A"

class ComprehensiveMedicalHistory(BaseModel):
    past_medical_history: List[ChronicIllness] = []
    previous_diagnoses: List[PreviousDiagnosis] = []
    past_surgical_history: List[SurgicalHistory] = []
    family_medical_history: FamilyHistory = FamilyHistory()
    allergies: Allergy = Allergy()
    current_medications: List[Medication] = []
    past_medications: List[Medication] = []

# Section 5: Social History
class SocialHistory(BaseModel):
    smoking: Optional[str] = None
    alcohol: Optional[str] = None
    substance_use: Optional[str] = None
    occupation: Optional[str] = None
    living_conditions: Optional[str] = None
    dietary_habits: Optional[str] = None
    exercise_routine: Optional[str] = None


# Section 6: Current Visit Details
class HistoryOfPresentIllness(BaseModel):
    onset: Optional[str] = None
    location: Optional[str] = None
    duration: Optional[str] = None
    characteristics: Optional[str] = None
    aggravating_factors: Optional[str] = None
    relieving_factors: Optional[str] = None
    associated_symptoms: Optional[List[str]] = None  


# Section 7: Review of Systems
class ReviewOfSystems(BaseModel):
    general: Optional[str] = None
    heent: Optional[str] = None
    cardiovascular: Optional[str] = None
    respiratory: Optional[str] = None
    gastrointestinal: Optional[str] = None
    genitourinary: Optional[str] = None
    musculoskeletal: Optional[str] = None
    neurological: Optional[str] = None
    psychiatric: Optional[str] = None
    skin: Optional[str] = None
    endocrine: Optional[str] = None
    hematologic: Optional[str] = None
    immunologic: Optional[str] = None


# Section 8: Physical Examination
class Vitals(BaseModel):
    temperature: Optional[str] = None
    blood_pressure: Optional[str] = None
    heart_rate: Optional[str] = None
    respiratory_rate: Optional[str] = None
    spO2: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[str] = None
    bmi: Optional[str] = None

class PhysicalExamination(BaseModel):
    vitals: Vitals
    general_appearance: Optional[str] = None
    heent: Optional[str] = None
    cardiovascular: Optional[str] = None
    respiratory: Optional[str] = None
    abdomen: Optional[str] = None
    extremities: Optional[str] = None
    neurological: Optional[str] = None
    skin: Optional[str] = None


# Section 9: Diagnostic Investigations
class LaboratoryTests(BaseModel):
    blood_tests: Optional[List[str]] = None 
    urine_analysis: Optional[str] = None


class DiagnosticInvestigations(BaseModel):
    laboratory_tests: LaboratoryTests
    imaging_studies: Optional[str] = None
    special_tests: Optional[str] = None


# Section 10: Assessment and Diagnosis
class AssessmentAndDiagnosis(BaseModel):
    primary_diagnosis: Optional[str] = None
    secondary_diagnoses: Optional[List[str]] = None
    differential_diagnoses: Optional[List[str]] = None


# Section 11: Treatment Plan
class Medication(BaseModel):
    name: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None

class TreatmentPlan(BaseModel):
    medications: Medication
    procedures_interventions: Optional[List[str]] = None
    therapies_recommended: Optional[List[str]] = None
    lifestyle_modifications: Optional[List[str]] = None
    follow_up_date_and_time: Optional[str] = None
    additional_tests: Optional[List[str]] = None

class PatientCounselingAndEducation(BaseModel):
    key_topics_discussed: Optional[List[str]] = None
    materials_provided: Optional[List[str]] = None
    patient_questions_addressed: Optional[str] = None


# Main Model
class MedicalData(BaseModel):
    comprehensive_medical_history: ComprehensiveMedicalHistory
    social_history: SocialHistory
    current_visit_details: HistoryOfPresentIllness
    review_of_systems: ReviewOfSystems
    physical_examination: PhysicalExamination
    diagnostic_investigations: DiagnosticInvestigations
    assessment_and_diagnosis: AssessmentAndDiagnosis
    treatment_plan: TreatmentPlan
    patient_counseling_and_education: PatientCounselingAndEducation


# Prompt design
def feature_extraction(text, llm):
    
    try:
        model=llm_model(llm)
        logger.info(f"Started  extraction of features from the text")
        template = f"""
            You are an advanced AI doctor's assistant tasked with extracting structured medical information from input text.

            Schema:
            A. comprehensive_medical_history:
                1. Past Medical History:
                - Chronic Illnesses
                - Previous Diagnoses
                2. Past Surgical History:
                - Date
                - Procedure 
                - Outcome
                3. Family Medical History:
                - Genetic Disorders
                - Illnesses in Parents/Siblings/Grandparents
                4. Allergies:
                - Drug
                - Food
                - Environmental
                5. Medications 
                - Current: 
                - Name
                - Dosage
                - Frequency 
                - Duration
                - Past: Last 6 Months
            B. social_history:
                1. Smoking: Yes/No; Frequency
                2. Alcohol: Yes/No; Frequency
                3. Substance Use: Yes/No; Type
                4. Occupation
                5. Living Conditions: (Lives alone, with family, etc.)
                6. Dietary Habits: (Vegetarian, Non-Vegetarian, Vegan)
                7. Exercise Routine
            C. current_visit_details:
                1. Onset: When did it start?
                2. Location: Where is the pain/symptom?
                3. Duration: How long has it lasted?
                4. Characteristics: Describe the pain/symptom
                5. Aggravating Factors: What makes it worse?
                6. Relieving Factors: What makes it better?
                7. Associated Symptoms: e.g., nausea, vomiting, dizziness
            D. review_of_systems:
                1. General: Fever, fatigue, weight loss/gain
                2. HEENT: Vision issues, ear pain, sinus problems
                3. Cardiovascular: Chest pain, palpitations, swelling
                4. Respiratory: Cough, shortness of breath, wheezing
                5. Gastrointestinal: Nausea, vomiting, diarrhea, constipation
                6. Genitourinary: Dysuria, frequency, incontinence
                7. Musculoskeletal: Pain, stiffness, swelling, weakness
                8. Neurological: Dizziness, headache, tingling, numbness
                9. Psychiatric: Depression, anxiety, sleep disturbances
                10. Skin: Rash, itching, lesions
                11. Endocrine: Heat/cold intolerance, increased thirst/hunger
                12. Hematologic: Easy bruising, bleeding
                13. Immunologic: Recurrent infections, known immune disorders
            E. physical_examination:
                1. Vitals:
                - Temperature
                - Blood Pressure
                - Heart Rate
                - Respiratory Rate
                - SpO2
                - Height
                - Weight
                - BMI
                2. General Appearance: (Well-nourished, pale, anxious, etc.)
                3. HEENT:
                - Eyes: Pupil reaction, vision clarity
                - Ears: Tympanic membrane status
                - Nose: Congestion, discharge
                - Throat: Tonsils, redness, swelling
                4. Cardiovascular:
                - Heart Sounds: Regular/Irregular
                - Murmurs: Present/Absent
                - Edema: Yes/No
                5. Respiratory:
                - Breath Sounds: Clear/Wheezing/Rales
                - Chest Shape: Normal/Abnormal
                6. Abdomen:
                - Tenderness: Yes/No
                - Palpable Mass: Yes/No
                - Bowel Sounds: Normal/Absent
                7. Extremities:
                - Swelling: Yes/No
                - Range of Motion: Normal/Restricted
                8. Neurological:
                - Reflexes: Normal/Absent
                9. Skin:
                - Lesions, Rash, Color Changes
            F. diagnostic_investigations:
                1. Laboratory Tests:
                - blood_tests: CBC, CMP, Blood Sugar
                - urine_analysis:
                2. Imaging Studies: X-rays, CT scans, MRI, Ultrasound
                3. Special Tests: ECG, ECHO, PFTs
            G. assessment_and_diagnosis:
                1. Primary Diagnosis
                2. Secondary Diagnoses
                3. Differential Diagnoses
            H. treatment_plan:
                1. Medications:
                - Name
                - Dosage 
                - Frequency 
                - Duration
                2. Procedures/Interventions
                3. Therapies Recommended
                4. Lifestyle Modifications
                5. Follow-up Instructions
                6. Follow-up date and time
                7. Additional tests
            I. patient_counseling_and_education:
                1. Key Topics Discussed
                2. Materials Provided
                3. Patient Questions Addressed


            Text:
            {text}

            ### Instructions:
            1. **Accuracy**: Extract only the information relevant to the specified categories. Do not include unrelated details.
            2. **Completeness**: Ensure that every category and subfield in schema is represented in the output json, even if the value is `N/A`.
            3. **Conciseness**: Avoid adding any explanatory or extra text. 
            4. **Formatting**: Return only the JSON object adhering strictly to the schema. Do not include any additional text or explanation.

            Output:
            Return only the JSON object adhering strictly to the schema. Do not include any additional text or explanation.
            If no information is found, return N/A for those fields.

            """

        output_parser = JsonOutputParser()
        parser = PydanticOutputParser(pydantic_object=MedicalData)
        prompt = PromptTemplate(input_variables=["text"], 
                            template=template,
                            partial_variables={"format_instructions": parser.get_format_instructions()})
        
        chain = prompt | model | output_parser
        final_response=chain.invoke({"text": text})
        logger.info("Feature extraction completed successfully")
        return final_response
    
    except Exception as e:
        logger.error(f"Error during Feature Extraction: {str(e)}")
        raise

def transcription_summary(response, llm):
    try:
        model=llm_model(llm)
        logger.info("Started text summarization")
        template = """
                You are a diligent and attentive doctor's assistant. 
                Your task is to carefully analyze the provided doctor-patient conversation text: {response}. 
                Based on this conversation, generate a concise and accurate summary that includes key details such as:

                Instructions:
                Symptoms mentioned by the patient
                Diagnoses discussed or considered by the doctor
                Relevant medical terms, conditions, or treatments
                Try to fetch all the information from the text
                Any treatment plans or recommendations provided by the doctor
                Any follow-up instructions or advice given to the patient

                Output:
                Ensure your summary captures all essential points from both the doctor's and patient's statements, 
                while remaining clear, precise, and focused on the critical information.
                Do not include any extra text before and after the summary.
                """

        prompt = PromptTemplate(template=template, input_variables=['response'])
        chain = prompt | model | StrOutputParser()

        result = chain.invoke({"response": response})
        logger.info("Text summarization completed successfully")

        return result
    except Exception as e:
        logger.error(f"Error during text summarization: {str(e)}")
        raise

def patient_conversation(response, llm):
    try:
        model=llm_model(llm)
        logger.info("patient conversation only.")
        template = """
                You are a diligent and attentive doctor's assistant. 
                Your task is to carefully analyze the provided doctor-patient conversation text: {response}. 
                Based on this conversation, Rephrase the patient conversation alone as a summary.

                Instructions:
                Output should be only patient's conversation. don't include doctor conversation.

                Output:
                Ensure your captures all essential points from the patient's conversation, 
                while remaining clear, precise, and focused on the critical information.
                Do not include any extra text before and after the summary.
                """

        prompt = PromptTemplate(template=template, input_variables=['response'])
        chain = prompt | model | StrOutputParser()

        result = chain.invoke({"response": response})
        logger.info("Patient conversation completed successfully")

        return result
    except Exception as e:
        logger.error(f"Error during patient conversation: {str(e)}")
        raise

def doctor_conversation(response, llm):
    try:
        model=llm_model(llm)
        logger.info("Doctor conversation only.")
        template = """
                You are a diligent and attentive doctor's assistant. 
                Your task is to carefully analyze the provided doctor-patient conversation text: {response}. 
                Based on this conversation, Rephrase the doctor conversation alone as a summary.

                Instructions:
                Output should be only doctor's conversation. don't include patient conversation.

                Output:
                Ensure your captures all essential points from the doctor's conversation, 
                while remaining clear, precise, and focused on the critical information.
                Do not include any extra text before and after the summary.
                """

        prompt = PromptTemplate(template=template, input_variables=['response'])
        chain = prompt | model | StrOutputParser()

        result = chain.invoke({"response": response})
        logger.info("Doctor conversation completed successfully")

        return result
    except Exception as e:
        logger.error(f"Error during doctor conversation: {str(e)}")
        raise


def extrcated_information_from_audio(response,llm="groq"):
    try:

        features_json = feature_extraction(text=response,llm=llm)
        summary = transcription_summary(response, llm)
        # patient_text = patient_conversation(response,llm)
        # doctor_text = doctor_conversation(response,llm)
        logger.info("Text processing completed successfully")
        # final_responses = {"features": features_json, "summary": summary, "patient_conversation":patient_text, "doctor_conversation":doctor_text}
        final_responses = {"features": features_json, "summary": summary}
        logger.info(f"Final response created successfully: {final_responses}")
        return final_responses
    
    except ValueError as e:
        logger.error(f"Error during Extraction information from audio: {str(e)}")
        raise
    

# text =  """David, today, included with your hygiene visit, with your cleaning, doctors asked me to go ahead and do a head and neck exam for you. Okay. I'm going to give you this graph. This actually will show you all of the lymph nodes that I'm going to be checking, and in addition, I'm going to be checking your jaw joint, the joint that allows your jaw to open and close. Now, this is a complementary service for you today. There's no charge for this, and the reason we're doing it is because the morbidity rate, that is the incidence of head and neck cancers, as well as the mortality rate, the number of deaths that occur from head and neck cancers, has been going up every year. So we're doing this as a service to our patients. Wow, that sounds serious. Well, it's not serious, but it's a really good screening tool. It would be serious if you had a problem. Okay. So we're going to screen to make sure you don't have one. Now, I've just washed my hands, so if you don't mind, I'll do your exam without gloves on. Sure. It'll be more comfortable for you, and I'll be able to actually feel better. I am going to ask you, however, to remove your headset and your glasses. Okay. And I can just put those on the counter for you. Thank you. Thank you. Okay, go ahead and rest your head back against the chair, and I'm going to start right here at your jaw joint. I'm going to ask you to go ahead and open and close. Open nice and wide, big as you can. There you go, and close. And I felt a little clunking when you opened that time. Okay. Where do you feel that? Right side. Right side. Do you feel any pain or discomfort? A little bit on the left. A little bit on the left. Do you know if you clench or grind your teeth at night? I might clench a little. Do you wake up with a tired jaw in the morning? Sometimes. Sometimes you do. Okay, let me check the muscles here. Squeeze your teeth together and release. This is your masseter muscle. Any pain here? No. Okay, let's try this one. Squeeze. This is your anterior temporalis. Just release. Any pain here? On the left side. On the left side, a bit of pain. Okay, and your posterior temporalis muscle. Squeeze, release. How about here? Any pain? Okay, so the right anterior temporalis and the masseter and the jaw joint. We'll have doctor check that when he comes in. Now I'm going to start on the lymph nodes and I'm going to start here around your ears. As you can see on that graph you're holding, there's lymph nodes in front of and behind your ears. I'm going to check the lymph nodes that come out underneath your cheekbones. Most people don't even know you have lymph nodes here. And I'm going to feel underneath the jaw for these lymph nodes as well. I'm going to come right down the front of your neck. Feel your Adam's apple and the thyroid. Do you have any family history of thyroid issues that you're aware of? Not that I know of. Swallow for me. Swallow on command is kind of hard to do, isn't it? And I'm going to feel the lymph nodes that go all the way around your collarbone. So I'm going to be just right inside your collarbone, between your neck and collarbone, and I'm going to press pretty firmly. Again, for any abnormalities again from side to side. Anything that's hard, lumpy, bumpy that doesn't belong there. Turn your head to the right and just let me support it here. I'm going to feel down the back of your neck. And lift your head up. This is your sternocleidomastoid muscle and I'm going to feel right down this muscle also. Now while I have your head turned to the side, I'm going to do a skin cancer check. I'm going to look behind the ear and along the hairline for any suspicious-looking lesions that don't belong there. Do you wear sunscreen? Yes. A good idea. You're very fair-complected, what with your reddish hair and your blue eyes. So it's a good idea to keep sunscreen on. Lift your head up again real quick. There you go. Rest back. Identify that sternocleidomastoid muscle. All right, and now just looking straight ahead and rest your head back into my hands. I'm going to feel the lymph nodes here at the base of the skull. These are the occipital lymph nodes. Okay, good news. I didn't feel anything that was asymmetrical from side to side. No lumps or bumps that you need to be worried about. We'll be doing this exam for you once a year during your hygiene visits. Okay. All right, so you can look forward to that. Thanks. All right, you're welcome."""
# # # # doctor_type = "dentist"
# # # # # response = feature_extraction(doctor_type, text,llm="gpt4")
# # # # # print(response)
# response=extrcated_information_from_audio(text,llm="groq")
# print(response)