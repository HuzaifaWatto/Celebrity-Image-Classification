# Celebrity Image Classification  

🚀 **End-to-end Machine Learning project for classifying celebrity images using Python, scikit-learn, Flask (Backend), and Streamlit (Frontend).**  

![Project Demo](https://img.youtube.com/vi/_ikEeVD5AUY/0.jpg)  
[🔗 Watch Demo on YouTube](https://youtu.be/_ikEeVD5AUY)

---

## 📚 Project Overview  

This project is part of my **Data Science Learning Journey**, following the **Machine Learning Module of Codebasics' Data Science Roadmap**.  

The goal was to build an **end-to-end image classification pipeline**, covering:  

✅ **Model training (on Kaggle)**  
✅ **Backend API (Flask)**  
✅ **Frontend UI (Streamlit)**  
✅ **Model & Encoder Storage for Deployment**  

The model itself classifies images of celebrities into predefined categories using **traditional machine learning techniques**.

---

## 📂 Repository Structure  

```text
📁 backend
    ├── app.py                    # Flask API to serve predictions
    ├── requirements.txt          # Backend dependencies
    📁 model
        ├── class_dict.json       # Dictionary mapping class indices to celebrity names
        ├── label_encoder.pkl     # Label encoder object used during training
    └── (Note: Main model is hosted on Kaggle, not in this repo)

📁 frontend
    ├── streamlit_app.py          # Streamlit app for uploading images & viewing predictions
    ├── requirements.txt          # Frontend dependencies

📄 README.md                       # Project documentation (this file)

```

---

## 🔗 Important Links  

- 📊 **Dataset**: [Celebrity Face Image Dataset on Kaggle](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset)  
- 📈 **Trained Model**: [Celebrity Image Classification Model on Kaggle](https://www.kaggle.com/models/huzaifawatto/celebrity-image-classification)  
- 📓 **Model Training Notebook**: [Kaggle Notebook](https://www.kaggle.com/code/huzaifawatto/celeb-image-classification)  
- 🎥 **Demo Video**: [Watch on YouTube](https://youtu.be/_ikEeVD5AUY)

---

## 💻 Technologies Used  

| Component   | Technology |
|-------------|-------------|
| Language    | Python |
| Model Training | scikit-learn |
| Backend API | Flask |
| Frontend UI | Streamlit |
| Deployment  | Localhost (can be hosted anywhere) |

---

## 🚀 How to Run  

### 1️⃣ Clone the Repository  

```bash
git clone https://github.com/HuzaifaWatto/Celebrity-Image-Classification.git
cd Celebrity-Image-Classification
```

---

### 2️⃣ Setup Backend  

```bash
cd backend
pip install -r requirements.txt
python app.py
```

This will start a **Flask server** at:  
http://localhost:5000

The backend uses the trained **model** (hosted on Kaggle), the **class_dict.json**, and the **label_encoder.pkl** to process predictions.

---

### 3️⃣ Setup Frontend  

Open a new terminal and run:

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

This will start the **Streamlit web app** at:  
http://localhost:8501

You can upload images and get predictions through the interface.

---

## 📷 Example Workflow  

1. Upload a celebrity image via the **Streamlit app**.
2. Image is sent to the **Flask backend**.
3. Flask loads the **trained model (from Kaggle storage)**, applies the **label encoder**, and makes a prediction.
4. Prediction result (celebrity name) is shown in **Streamlit**.

---

## 📬 Contact  

If you have any questions, feel free to connect with me:  

- **Kaggle**: [HuzaifaWatto](https://www.kaggle.com/huzaifawatto)  
- **GitHub**: [HuzaifaWatto](https://github.com/HuzaifaWatto)  

---

Let me know if you also want a **custom project banner image prompt** for your GitHub repo! 🚀
