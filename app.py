from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz
import spacy
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configuración de Flask
app = Flask(__name__)
CORS(app)  # <-- ¡Esta línea habilita CORS para todos los orígenes!

# Cargar modelos necesarios
model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab)

# Dataset de habilidades clave
try:
    with open("skills200.txt", "r", encoding="utf-8") as file:
        list_of_skills = [line.strip().lower() for line in file]
except FileNotFoundError:
    list_of_skills = ["python", "javascript", "machine learning", "nlp"]

list_of_key_phrases = ["experience in", "worked with", "developed", "skilled in"]
matcher.add("SKILLS", [nlp.make_doc(skill) for skill in list_of_skills])
matcher.add("KEY_PHRASES", [nlp.make_doc(phrase) for phrase in list_of_key_phrases])


# FUNCIÓN: Cargar archivo y procesar
def loadFile(filename):
    _, ext = os.path.splitext(filename)
    if ext == ".txt":
        with open(filename, "r", encoding="utf-8") as file:
            CV = file.read()
    elif ext == ".pdf":
        pdf = fitz.open(filename)
        CV = ""
        for page in pdf:
            CV += page.get_text()
        pdf.close()
    else:
        raise ValueError(f"File format not supported: {ext}")

    # Preprocesar texto
    lines = [line.strip() for line in CV.split("\n") if line.strip()]
    preprocessed_CV = ""
    for line in lines:
        if line.endswith("."):
            preprocessed_CV += line + " "
        else:
            preprocessed_CV += line + ". "
    CV = preprocessed_CV.lower()
    return nlp(CV)


# FUNCIÓN: Busqueda de coincidencias usando PhraseMatcher
def findInfoMatch(doc):
    results = []
    for sent in doc.sents:
        dates = [
            ent for ent in sent.ents
            if ent.label_ == "DATE" or (ent.label_ == "CARDINAL" and ent.text.isdigit() and 1900 < int(ent.text) < 2099)
        ]
        matched_skills = []
        matched_key_phrases = []
        matches_in_sent = matcher(sent)
        for match_id, start, end in matches_in_sent:
            span = doc[start:end]
            label = nlp.vocab.strings[match_id]
            if label == "SKILLS":
                matched_skills.append(span.text)
            elif label == "KEY_PHRASES":
                matched_key_phrases.append(span.text)

        if dates and (matched_skills or matched_key_phrases):
            results.append({
                "dates": [str(date) for date in dates],
                "skills": matched_skills,
                "phrases": matched_key_phrases,
                "context": sent.text.strip().replace("\n", " ")
            })
        elif matched_skills:
            results.append({
                "skills": matched_skills,
                "context": sent.text.strip().replace("\n", " ")
            })
    return results


# FUNCIÓN: Evaluar coincidencia entre el CV y la descripción del puesto
def evaluate_CV_vs_Position(cv_info, jp_info):
    cv_skills = set([skill for entry in cv_info for skill in entry.get("skills", [])])
    jp_skills = set([skill for entry in jp_info for skill in entry.get("skills", [])])

    required_skills_matched = cv_skills & jp_skills
    required_score = len(required_skills_matched) / len(jp_skills) * 100 if jp_skills else 0
    return {
        "score": required_score,
        "cv_skills": list(cv_skills),
        "matched_skills": list(required_skills_matched),
        "required_skills": list(jp_skills)
    }


# FUNCIÓN: Obtener embeddings para las oraciones
def get_embeddings(sentences):
    return model.encode(sentences)


# FUNCIÓN: Comparar embeddings y calcular similitud entre el CV y la descripción del puesto
def findInfoEmbedding(docCV, docJP):
    CV_sentences = [sent.text.strip() for sent in docCV.sents if len(sent.text.strip()) > 0]
    JP_sentences = [sent.text.strip() for sent in docJP.sents if len(sent.text.strip()) > 0]

    CV_embeddings = get_embeddings(CV_sentences)
    JP_embeddings = get_embeddings(JP_sentences)

    similarity_matrix = cosine_similarity(JP_embeddings, CV_embeddings)

    results = []
    for jp_idx, jp_sentence in enumerate(JP_sentences):
        most_similar_idx = np.argmax(similarity_matrix[jp_idx])
        most_similar_score = similarity_matrix[jp_idx][most_similar_idx]
        most_similar_sentence = CV_sentences[most_similar_idx]

        results.append({
            "job_sentence": jp_sentence,
            "most_similar_cv_sentence": most_similar_sentence,
            "similarity_score": float(most_similar_score)
        })

    filtered_results = [result for result in results if result['similarity_score'] > 0.64]
    return filtered_results


# ENDPOINT: Procesar los archivos subidos
@app.route('/api/upload', methods=['POST'])
def upload_files():
    file_cv = request.files.get('cv')
    file_jp = request.files.get('job_position')

    if not file_cv or not file_jp:
        return jsonify({"error": "Both CV and Job Position files are required"}), 400

    # Guardar los archivos temporalmente
    file_cv_path = "temp_cv.pdf"
    file_jp_path = "temp_jp.txt"
    file_cv.save(file_cv_path)
    file_jp.save(file_jp_path)

    # Procesar los documentos
    try:
        docCV = loadFile(file_cv_path)
        docJP = loadFile(file_jp_path)

        # Obtener coincidencias basadas en palabras clave
        cv_info = findInfoMatch(docCV)
        jp_info = findInfoMatch(docJP)

        # Evaluar habilidades y similitudes
        evaluation = evaluate_CV_vs_Position(cv_info, jp_info)
        embedding_results = findInfoEmbedding(docCV, docJP)

        # Respuesta JSON
        response = {
            "keyword_analysis": evaluation,
            "embedding_analysis": embedding_results
        }

    finally:
        # Eliminar archivos temporales
        os.remove(file_cv_path)
        os.remove(file_jp_path)

    return jsonify(response)


# Ejecutar el servidor
if __name__ == '__main__':
    app.run(debug=True)
