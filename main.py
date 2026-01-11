# ==============================================================================
# 1. IMPORTS & CONFIGURATION
# ==============================================================================
import os
import logging
import pickle
from pathlib import Path
from difflib import get_close_matches
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify, session
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect, generate_csrf
from dotenv import load_dotenv

# LangChain / AI Imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__, static_folder='static')
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())
app.config['WTF_CSRF_ENABLED'] = True

# Security & Middleware
csrf = CSRFProtect(app)
CORS(app, supports_credentials=True)


# ==============================================================================
# 2. PATHS & DATA LOADING
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
logger.info(f"Base directory: {BASE_DIR}")

def get_path(relative_path):
    path = BASE_DIR / relative_path
    return path

# Load Datasets
try:
    logger.info("Loading Datasets...")
    sym_des = pd.read_csv(get_path("Datasets/symptoms_df.csv"))
    precautions_df = pd.read_csv(get_path("Datasets/precautions_df.csv"))
    workout_df = pd.read_csv(get_path("Datasets/workout_df.csv"))
    description_df = pd.read_csv(get_path("Datasets/description.csv"))
    medications_df = pd.read_csv(get_path('Datasets/medications.csv'))
    diets_df = pd.read_csv(get_path("Datasets/diets.csv"))
    logger.info("Successfully loaded all Datasets")
except Exception as e:
    logger.error(f"Error loading Datasets: {str(e)}")
    raise

# Load Model
try:
    logger.info("Loading model...")
    with open(get_path('models/svc.pkl'), 'rb') as f:
        svc = pickle.load(f)
    logger.info("Successfully loaded the model")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise


# ==============================================================================
# 3. CONSTANTS & DICTIONARIES
# ==============================================================================

# Symptoms Mapping
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 
    'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 
    'vomiting': 11, 'burning_micturition': 12, 'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 
    'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 
    'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 
    'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 
    'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 
    'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 
    'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 
    'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 
    'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 
    'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 
    'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 
    'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 
    'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 
    'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 
    'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 
    'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 
    'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91, 
    'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 
    'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 
    'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 
    'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 
    'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 
    'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 
    'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 
    'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 
    'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 
    'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 
    'yellow_crust_ooze': 131
}

SYMPTOM_GROUPS = {
    "cold": ["cough", "runny_nose", "continuous_sneezing", "throat_irritation", "congestion", "redness_of_eyes", "sinus_pressure"],
    "flu": ["high_fever", "headache", "fatigue", "muscle_pain", "chills", "nausea"],
    "fever": ["high_fever", "mild_fever", "shivering", "sweating"],
    "stomach": ["stomach_pain", "acidity", "vomiting", "nausea", "abdominal_pain", "diarrhoea", "constipation"],
    "skin": ["itching", "skin_rash", "nodal_skin_eruptions", "red_spots_over_body"],
    "pain": ["joint_pain", "headache", "back_pain", "chest_pain", "muscle_pain", "neck_pain"]
}

# Diseases List Mapping - Kept for reference, but model output is prioritized
diseases_list = {
    0: '(vertigo) Paroymsal  Positional Vertigo', 
    1: 'AIDS', 
    2: 'Acne', 
    3: 'Alcoholic hepatitis', 
    4: 'Allergy', 
    5: 'Arthritis', 
    6: 'Bronchial Asthma', 
    7: 'Cervical spondylosis', 
    8: 'Chicken pox', 
    9: 'Chronic cholestasis', 
    10: 'Common Cold', 
    11: 'Dengue', 
    12: 'Diabetes ', 
    13: 'Dimorphic hemmorhoids(piles)', 
    14: 'Drug Reaction', 
    15: 'Fungal infection', 
    16: 'GERD', 
    17: 'Gastroenteritis', 
    18: 'Heart attack', 
    19: 'Hepatitis B', 
    20: 'Hepatitis C', 
    21: 'Hepatitis D', 
    22: 'Hepatitis E', 
    23: 'Hypertension ', 
    24: 'Hyperthyroidism', 
    25: 'Hypoglycemia', 
    26: 'Hypothyroidism', 
    27: 'Impetigo', 
    28: 'Jaundice', 
    29: 'Malaria', 
    30: 'Migraine', 
    31: 'Osteoarthristis', 
    32: 'Paralysis (brain hemorrhage)', 
    33: 'Peptic ulcer diseae', 
    34: 'Pneumonia', 
    35: 'Psoriasis', 
    36: 'Tuberculosis', 
    37: 'Typhoid', 
    38: 'Urinary tract infection', 
    39: 'Varicose veins', 
    40: 'hepatitis A'
}

SYMPTOM_ALIASES = {
    "fever": "mild_fever",
    "high fever": "high_fever",
    "low fever": "mild_fever",
    "headache": "headache",
    "burning urination": "burning_micturition",
    "painful urination": "burning_micturition",
    "frequent urination": "continuous_feel_of_urine",
    "stomach pain": "abdominal_pain",
    "lower abdominal pain": "abdominal_pain",
    "body pain": "muscle_pain"
}

# Critical symptoms that trigger the "Safety Check"
CRITICAL_SYMPTOMS = {
    "burning_micturition", "continuous_feel_of_urine", "foul_smell_of_urine",
    "abdominal_pain", "bladder_discomfort"
}

GENERAL_SYMPTOMS = {
    "fever", "mild_fever", "high_fever", "fatigue", "headache"
}

# Derived Lists
predictable_diseases = sorted(list(set(diseases_list.values())))
medicine_diseases = medications_df['Disease'].str.strip().str.title().unique()
all_diseases = sorted(list(set(predictable_diseases) | set(medicine_diseases)))
recognizable_symptoms = sorted(list(symptoms_dict.keys()))
normalized_symptoms = {symptom.lower().replace(' ', '_'): symptom for symptom in symptoms_dict}


# ==============================================================================
# 4. LLM & AI SETUP
# ==============================================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

system_instructions = (
    "You are HealixAI, a responsible and cautious medical AI assistant.\n\n"
    "RULES YOU MUST FOLLOW:\n"
    "1. NEVER give medicine dosages or prescriptions.\n"
    "2. NEVER claim certainty.\n"
    "3. ALWAYS mention confidence and severity if provided.\n"
    "4. ALWAYS advise consulting a healthcare professional.\n\n"
    "SEVERITY-AWARE BEHAVIOR:\n"
    "- If severity is HIGH:\n"
    "  • Use urgent and serious tone\n"
    "  • Strongly recommend immediate doctor consultation\n"
    "  • Warn about potential complications\n\n"
    "- If severity is MEDIUM:\n"
    "  • Use cautious but calm tone\n"
    "  • Suggest monitoring symptoms\n"
    "  • Doctor visit if symptoms worsen\n\n"
    "- If severity is LOW:\n"
    "  • Use reassuring tone\n"
    "  • Suggest rest and basic care\n"
    "  • Mention that condition is likely manageable\n\n"
    "FORMAT:\n"
    "• Disease summary\n"
    "• What this means\n"
    "• What to do next\n"
    "• Clear medical disclaimer\n"
)


# ==============================================================================
# 5. HELPER FUNCTIONS (UTILITIES & LOGIC)
# ==============================================================================

def clean_list(items):
    clean = []
    for item in items:
        if pd.isna(item):
            continue
        if isinstance(item, str):
            item = item.replace("[", "").replace("]", "")
            item = item.replace("'", "").replace('"', "")
            parts = [p.strip() for p in item.split(",") if p.strip()]
            clean.extend(parts)
        else:
            clean.append(str(item))
    return clean if clean else ["Not available"]

def helper(dis):
    try:
        desc_series = description_df[description_df['Disease'] == dis]['Description']
        desc = desc_series.iloc[0] if not desc_series.empty else "Description not available"

        pre_df = precautions_df[precautions_df['Disease'] == dis][
            ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
        ]
        pre = clean_list(pre_df.values.flatten().tolist()) if not pre_df.empty else ["No precautions available"]

        med_series = medications_df[medications_df['Disease'] == dis]['Medication']
        med = clean_list(med_series.tolist()) if not med_series.empty else ["No medications recommended"]

        die_series = diets_df[diets_df['Disease'] == dis]['Diet']
        die = clean_list(die_series.tolist()) if not die_series.empty else ["No specific diet recommendations"]

        wrkout_series = workout_df[workout_df['disease'] == dis]['workout']
        wrkout = clean_list(wrkout_series.tolist()) if not wrkout_series.empty else ["No workout recommendations"]

        return desc, pre, med, die, wrkout

    except Exception as e:
        logger.error(f"Error in helper function: {str(e)}")
        return "Description not available", ["No precautions available"], ["No medications recommended"], ["No specific diet recommendations"], ["No workout recommendations"]

def get_disease_details(disease_name: str):
    desc, pre, med, die, wrkout = helper(disease_name)
    return {
        "name": disease_name,
        "description": desc,
        "precautions": pre,
        "medications": med,
        "diets": die,
        "workouts": wrkout
    }

def get_predicted_value(patient_symptoms):
    """
    Returns (Disease Name, Confidence Score)
    FIX: Now returns the class string directly from the model, skipping the int->string lookup failure.
    """
    try:
        input_vector = np.zeros(len(symptoms_dict))
        for item in patient_symptoms:
            if item in symptoms_dict:
                input_vector[symptoms_dict[item]] = 1

        if hasattr(svc, "predict_proba"):
            probabilities = svc.predict_proba([input_vector])[0]
            pred_idx = np.argmax(probabilities)
            confidence = round(probabilities[pred_idx] * 100, 2)
        else:
            pred_idx = svc.predict([input_vector])[0]
            confidence = 75.0 

        # ✅ FIXED: Use the model's class labels (Strings) directly
        if hasattr(svc, "classes_"):
            predicted_label = svc.classes_[pred_idx]
        else:
            # Fallback if classes are integers (rare for this dataset)
            predicted_label = diseases_list.get(pred_idx, "Unknown Condition")
            
        return predicted_label, confidence

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return None, None

def validate_symptoms(user_input):
    try:
        valid = []
        invalid = []
        suggestions = {}

        text = user_input.lower()
        text = text.replace('+', ',').replace(' and ', ',').replace(';', ',')
        raw_symptoms = [s.strip() for s in text.split(',') if s.strip()]

        for symptom in raw_symptoms:
            original = symptom 

            if symptom in SYMPTOM_ALIASES:
                symptom = SYMPTOM_ALIASES[symptom]

            normalized = symptom.replace(' ', '_')

            if normalized in symptoms_dict:
                valid.append(normalized)
                continue
            
            if normalized in SYMPTOM_GROUPS:
                suggestions[original] = SYMPTOM_GROUPS[normalized]
                invalid.append(original)
                continue

            close_matches = get_close_matches(
                normalized,
                symptoms_dict.keys(),
                n=3,
                cutoff=0.6
            )

            if close_matches:
                suggestions[original] = close_matches
                invalid.append(original)
            else:
                invalid.append(original)

        return valid, invalid, suggestions

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return [], [user_input], {}

def extract_text_from_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", str(content))
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content)

def gemini_chat(user_input, history, context_text=""):
    full_system_prompt = system_instructions
    if context_text:
        full_system_prompt += f"\n\n{context_text}"

    messages = [SystemMessage(content=full_system_prompt)]

    if isinstance(history, list) and history:
        messages.extend(history)

    messages.append(HumanMessage(content=str(user_input)))
    response = llm.invoke(messages)
    return extract_text_from_content(response.content)

# --- Medical Logic Helpers ---

def calculate_confidence(symptoms):
    symptoms = set(symptoms)
    has_specific = bool(CRITICAL_SYMPTOMS.intersection(symptoms))
    count = len(symptoms)

    if count == 1:
        return 30.0
    elif count == 2 and not has_specific:
        return 45.0
    elif count >= 3 and has_specific:
        return 70.0
    else:
        return 55.0

def medical_note(confidence):
    if confidence < 40:
        return "Prediction confidence is low. More symptoms are required for a reliable assessment."
    elif confidence < 60:
        return "This assessment is based on the symptoms currently provided and may change with additional information."
    else:
        return "Symptoms show moderate correlation with the predicted condition."

def severity_level(confidence):
    if confidence >= 85:
        return "High"
    elif confidence >= 65:
        return "Medium"
    else:
        return "Low"

def doctor_advice(severity):
    if severity == "High":
        return "Consult a doctor immediately."
    elif severity == "Medium":
        return "Consult a doctor if symptoms persist."
    else:
        return "Home care and monitoring advised."

def explain_prediction(patient_symptoms, top_n=5):
    try:
        readable = []
        for sym in patient_symptoms[:top_n]:
            s = sym.replace('_', ' ').lower()
            if "blister" in s:
                readable.append("Fluid-filled blisters commonly associated with skin infections")
            elif "red" in s or "sore" in s:
                readable.append("Red sores indicating possible bacterial skin involvement")
            elif "itch" in s:
                readable.append("Skin irritation and itching suggest infectious dermatitis")
            elif "crust" in s or "ooze" in s:
                readable.append("Crusting lesions often seen in impetigo cases")
            else:
                readable.append(f"Presence of symptom: {s.title()}")
        return readable
    except Exception as e:
        logger.error(f"Explainability error: {str(e)}")
        return ["Symptom pattern analysis matched known disease profiles"]

def needs_followup(symptoms):
    """
    Decides if we should block prediction and ask for more info.
    TRUE = Stop and ask questions.
    FALSE = Go ahead and predict.
    """
    symptoms = set(symptoms)
    specific = CRITICAL_SYMPTOMS.intersection(symptoms)
    
    # 1. If user entered CRITICAL symptoms (urinary issues), we WANT to ask follow-up questions
    if len(specific) > 0:
        return True
        
    # 2. If user entered very few symptoms and NONE are critical, we need more info
    if len(symptoms) < 3:
        return True
        
    # 3. Otherwise (3+ symptoms, or non-critical cases), just predict
    return False


# ==============================================================================
# 6. FLASK CONTEXT PROCESSORS
# ==============================================================================
@app.context_processor
def inject_global_data():
    return dict(
        normalized_symptoms=normalized_symptoms,
        csrf_token=generate_csrf
    )


# ==============================================================================
# 7. ROUTE HANDLERS
# ==============================================================================

@app.route("/")
def root():
    session.clear()
    return render_template("index.html", 
                           predictable_diseases=predictable_diseases, 
                           all_diseases=all_diseases,
                           recognizable_symptoms=recognizable_symptoms)

@app.route("/index")
def index():
    return root()

@app.route('/predict', methods=['POST'])
@csrf.exempt
def home():
    symptoms = request.form.get('symptoms', '').strip()
    
    if not symptoms or symptoms.lower() == "symptoms":
        return render_template('index.html', 
                               message="Please enter your symptoms. Example: headache, fever, nausea",
                               predictable_diseases=predictable_diseases)
    
    valid, invalid, suggestions = validate_symptoms(symptoms)
    
    if invalid:
        error_message = """
        <div>
            <strong class="d-block mb-1">Invalid Symptoms Detected</strong>
            <ul class="mb-1 ps-3 small">
        """
        for symptom in invalid:
            if symptom in suggestions:
                suggestion_text = ", ".join(suggestions[symptom])
                error_message += f"<li><strong>{symptom}</strong> - <span class='text-white-50'>Did you mean: {suggestion_text}?</span></li>"
            else:
                error_message += f"<li><strong>{symptom}</strong> - <span class='text-white-50'>No similar symptoms found</span></li>"

        error_message += """
            </ul>
            <small>Please correct your input and try again.</small>
        </div>
        """

        return render_template('index.html', 
                               message=error_message,
                               predictable_diseases=predictable_diseases)
    
    if not valid:
        return render_template('index.html', 
                               message="No valid symptoms recognized. Please check your input and try again.",
                               predictable_diseases=predictable_diseases)
    
    try:
        # ---------- MEDICAL SAFETY CHECK (Logic Updated) ----------
        if needs_followup(valid):
            return render_template(
            "index.html",
            message="More information is required for a safe medical assessment.",
            followup_questions=[
            "Do you experience burning sensation while urinating?",
            "Are you urinating more frequently than usual?",
            "Do you have lower abdominal or pelvic pain?",
            "Is your urine cloudy or foul-smelling?"
            ],
            predictable_diseases=predictable_diseases
            )
        
        # ---------- SAFE PREDICTION ----------
        predicted_disease, confidence = get_predicted_value(valid)
        
        if not predicted_disease or confidence is None:
             raise ValueError("Prediction calculation failed")

        severity = severity_level(confidence)
        advice = doctor_advice(severity)
        note = medical_note(confidence)
        explanation = explain_prediction(valid)
        
        logger.info(
            f"PREDICTION | Disease={predicted_disease} | "
            f"Confidence={confidence}% | Severity={severity}"
        )

        session['last_prediction'] = {
            "disease": predicted_disease,
            "confidence": confidence,
            "severity": severity,
            "explanation": explanation
        }
        
        dis_des, precautions_list, medications_list, rec_diet, workout_list = helper(predicted_disease)
        
        return render_template('index.html', 
                               predicted_disease=predicted_disease,
                               confidence=confidence,
                               severity=severity,
                               doctor_advice=advice,
                               medical_note=note,
                               explanation=explanation,
                               dis_des=dis_des,
                               my_precautions=precautions_list,
                               medications=medications_list,
                               my_diet=rec_diet,
                               workout_list=workout_list,
                               predictable_diseases=predictable_diseases
                               )
    except Exception as e:
        logger.error(f"Full prediction error: {str(e)}")
        return render_template('index.html', 
                               message=f"System error: {str(e)}",
                               predictable_diseases=predictable_diseases)
    
@app.route('/api/chat', methods=['POST'])
@csrf.exempt
def chat():
    data = request.get_json(force=True) or {}
    user_input = data.get('message', '').strip()

    if not user_input:
        return jsonify({
            "response": "Please enter a message.",
            "history": session.get("chat_history", [])
        })

    if 'chat_history' not in session:
        session['chat_history'] = []

    chat_history = session['chat_history']

    if user_input.lower() in {"hi", "hello", "hey", "hii"}:
        response_text = (
            "Hello 👋 I’m HealixAI.\n"
            "Please describe your symptoms so I can assist you."
        )
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response_text})
        session.modified = True

        return jsonify({
            "response": response_text,
            "history": chat_history
        })

    if user_input.lower() in SYMPTOM_GROUPS:
        symptoms = ", ".join(SYMPTOM_GROUPS[user_input.lower()])
        response_text = (
            f"When you say **{user_input}**, it may include symptoms like:\n"
            f"{symptoms}.\n\n"
            "Please tell me which ones you are experiencing."
        )

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response_text})
        session.modified = True

        return jsonify({
            "response": response_text,
            "history": chat_history
        })

    try:
        converted_history = []
        for msg in chat_history:
            if msg["role"] == "user":
                converted_history.append(HumanMessage(content=msg["content"]))
            else:
                converted_history.append(AIMessage(content=msg["content"]))

        context_info = ""
        if "last_prediction" in session:
            lp = session["last_prediction"]
            expl_text = ", ".join(lp.get("explanation", [])) if isinstance(lp.get("explanation"), list) else str(lp.get("explanation"))
            
            context_info = (
                f"Disease: {lp.get('disease')}\n"
                f"Confidence: {lp.get('confidence')}%\n"
                f"Severity: {lp.get('severity')}\n"
                f"Key symptoms: {expl_text}\n"
            )

        ai_response = gemini_chat(user_input, converted_history, context_info)

        response_text = (
            ai_response
            + "\n\n*Consult a healthcare professional for medical advice*"
        )

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": ai_response})
        session.modified = True

        return jsonify({
            "response": response_text,
            "history": chat_history
        })

    except Exception as e:
        logger.exception("Chat processing error")
        return jsonify({
            "response": (
                "⚠️ HealixAI is temporarily unavailable.\n"
                "Please try again in a moment."
            ),
            "history": chat_history
        }), 503

@app.route('/api/disease-details', methods=['POST'])
@csrf.exempt
def disease_details():
    data = request.get_json(force=True) or {}
    disease_name = data.get('disease', '').strip()
    
    if not disease_name:
        return jsonify({"error": "Missing disease parameter"}), 400
    
    try:
        details = get_disease_details(disease_name)
        return jsonify(details)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/technology")
def technology():
    return render_template("technology.html")

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

@app.route('/api/suggest')
def suggest_symptoms():
    query = request.args.get('q', '').lower().strip()
    if not query: 
        return jsonify([])
    
    search_term = query.split(',')[-1].strip()
    results = set()
    
    for group_name, symptoms in SYMPTOM_GROUPS.items():
        if search_term in group_name and len(search_term) > 2:
            results.update(symptoms)

    for sym in symptoms_dict.keys():
        readable_sym = sym.replace('_', ' ')
        if search_term in readable_sym:
            results.add(sym)

    return jsonify(sorted(list(results))[:10])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)