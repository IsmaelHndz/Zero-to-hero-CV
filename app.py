from flask import Flask, request, jsonify # use to create the API
from flask_cors import CORS # implements cross origin resource sharing
import os # load documents
import fitz # pdf loader
import spacy # NLP
from spacy.matcher import PhraseMatcher # Mathcer spacy model
from sentence_transformers import SentenceTransformer # Embeddings model
from sklearn.metrics.pairwise import cosine_similarity # calculate cosine similarity. Semantic similarity
import numpy as np

# FLASK CONFIGURATION
app = Flask(__name__)
CORS(app)  # Enable cors

# LOAD MODELS
model = SentenceTransformer('all-MiniLM-L6-v2') # Embeddings model object
nlp = spacy.load("en_core_web_sm") # NLP spacy model
matcher = PhraseMatcher(nlp.vocab) # Creates a PhraseMatcher object

# LOAD SKILLS DATASET
try: # loads skills dataset and transforms it to lower case
    with open("skills200.txt", "r", encoding="utf-8") as file:
        list_of_skills = [line.strip().lower() for line in file] 
except FileNotFoundError:
    list_of_skills = ["python", "javascript", "machine learning", "nlp"]
list_of_key_phrases = ["experience in", "worked with", "developed", "skilled in"] # list of "keywords"
matcher.add("SKILLS", [nlp.make_doc(skill) for skill in list_of_skills]) # loads the skills dataset into the Matcher object
matcher.add("KEY_PHRASES", [nlp.make_doc(phrase) for phrase in list_of_key_phrases])


# FUNCTION: LOAD AND PROCESS
def loadFile(filename):
    _, ext = os.path.splitext(filename) # Opens the file and saves its extention 
    if ext == ".txt":
        with open(filename, "r", encoding="utf-8") as file:
            doc = file.read() # if .txt just open and save the info
    elif ext == ".pdf":
        pdf = fitz.open(filename) # if .pdf open with fitz
        doc = ""
        for page in pdf:
            doc += page.get_text() # saving the information
        pdf.close()
    else:
        raise ValueError(f"File format not supported: {ext}")

    # PREPROCESSING 
    lines = [line.strip() for line in doc.split("\n") if line.strip()] # Removes whitespaces at start and end of each row
    preprocessed_CV = ""
    for line in lines: # transform all the lines into a single string to standarize sentences
        if line.endswith("."):
            preprocessed_CV += line + " " # if the line already has a period just add a whitespace
        else:
            preprocessed_CV += line + ". " # if the line does not end with a period add the period and whitespace
    doc = preprocessed_CV.lower() # transforms the string to lowercase
    return nlp(doc) # returns the document either the Jop position or CV


# FUNCTION: SEARCH SKILLS MATCHES FROM THE JOP POSITION INTO THE CV USING PHRASE MATCHER
def findInfoMatch(doc):
    results = []
    for sent in doc.sents:
        dates = [
            ent for ent in sent.ents # search for DATE or CARDINAL labels using NER to find possible dates.
            if ent.label_ == "DATE" or (ent.label_ == "CARDINAL" and ent.text.isdigit() and 1900 < int(ent.text) < 2099)
        ]
        matched_skills = []
        matched_key_phrases = []
        matches_in_sent = matcher(sent) # gets the "matches" for every sentence in the doc using the matcher object
        for match_id, start, end in matches_in_sent: # for every match, extracts it in span, label and save them in their list
            span = doc[start:end]
            label = nlp.vocab.strings[match_id] 
            if label == "SKILLS":
                matched_skills.append(span.text)
            elif label == "KEY_PHRASES":
                matched_key_phrases.append(span.text)

        if dates and (matched_skills or matched_key_phrases): # if there is a date and a match, save it in dictionary format
            results.append({
                "dates": [str(date) for date in dates],
                "skills": matched_skills,
                "phrases": matched_key_phrases,
                "context": sent.text.strip().replace("\n", " ")
            })
        elif matched_skills: # if there is just a match, save it.
            results.append({
                "skills": matched_skills,
                "context": sent.text.strip().replace("\n", " ")
            })
    return results


# FUNCTION: EVAULUATE WHAT IS FOUND BOTH IN THE CV AND JOB POSITION. USED FOR "SKILLS" KEYWORD MATCHES
def evaluate_CV_vs_Position(cv_info, jp_info):
    cv_skills = set([skill for entry in cv_info for skill in entry.get("skills", [])])
    jp_skills = set([skill for entry in jp_info for skill in entry.get("skills", [])])

    required_skills_matched = cv_skills & jp_skills
    required_score = len(required_skills_matched) / len(jp_skills) * 100 if jp_skills else 0 # gets the % of skills found
    return {
        "score": required_score,
        "cv_skills": list(cv_skills),
        "matched_skills": list(required_skills_matched),
        "required_skills": list(jp_skills)
    }


# FUNCTION: GETS THE EMBEDDINGS OF THE SENTENCES
def get_embeddings(sentences):
    return model.encode(sentences)


# FUNCTION: EVALUATE THE EMBEDDINGS AND COSINE SIMILARITY BETWEEN CV AND JOB DESCRIPTION
def findInfoEmbedding(docCV, docJP):
    CV_sentences = [sent.text.strip() for sent in docCV.sents if len(sent.text.strip()) > 0]
    JP_sentences = [sent.text.strip() for sent in docJP.sents if len(sent.text.strip()) > 0]

    CV_embeddings = get_embeddings(CV_sentences)
    JP_embeddings = get_embeddings(JP_sentences)

    similarity_matrix = cosine_similarity(JP_embeddings, CV_embeddings) # Consine similarity of the embeddings

    results = []
        # Compares the cosine similarity between every sentence in both docs and selects the most similiar for each sentence in Job position with an umbral
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


# ENDPOINT: PROCESS DE DATA SENT FROM THE FRONTEND
@app.route('/api/upload', methods=['POST'])
def upload_files():
    file_cv = request.files.get('cv')
    file_jp = request.files.get('job_position')

    if not file_cv or not file_jp:
        return jsonify({"error": "Both CV and Job Position files are required"}), 400

    # saves the files locally
    file_cv_path = "temp_cv.pdf"
    file_jp_path = "temp_jp.txt"
    file_cv.save(file_cv_path)
    file_jp.save(file_jp_path)

    # process the files
    try:
        docCV = loadFile(file_cv_path)
        docJP = loadFile(file_jp_path)

        # get keyword skills from both docs
        cv_info = findInfoMatch(docCV)
        jp_info = findInfoMatch(docJP)

        # evaluate and match keyword skills / gets the embeddings result
        evaluation = evaluate_CV_vs_Position(cv_info, jp_info)
        embedding_results = findInfoEmbedding(docCV, docJP)

        # JSON response
        response = {
            "keyword_analysis": evaluation,
            "embedding_analysis": embedding_results
        }

    finally:
        # eliminate temporal files
        os.remove(file_cv_path)
        os.remove(file_jp_path)

    return jsonify(response)


# execute the app
if __name__ == '__main__':
    app.run(debug=True)
