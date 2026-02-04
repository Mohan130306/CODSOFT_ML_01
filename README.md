# CODSOFT_ML_01
#  Movie Genre Prediction using Machine Learning  
### CodSoft Internship – Task 1

##  Project Overview
This project was completed as **Task 1 of the CodSoft Machine Learning Internship**.  
The goal of the task is to build a machine learning model that can **predict the genre of a movie** based on its **plot summary or textual description** using Natural Language Processing (NLP) techniques.

---

##  Objective
To classify movies into their respective genres by analyzing plot summaries using supervised machine learning algorithms and text feature extraction methods.

---

##  Tools & Technologies Used
- **Programming Language:** Python  
- **Development Environment:** VS Code  
- **Libraries:**
  - `pandas` – data loading and preprocessing  
  - `scikit-learn` – machine learning models and evaluation  
  - `TfidfVectorizer` – text feature extraction  
  - `Multinomial Naive Bayes` – classification algorithm  

---

##  Dataset Description
The dataset consists of movie plot summaries along with their corresponding genres.
- `train_data.txt` – training data containing movie plots and genres  
- `test_data.txt` – test data containing movie plots  
- `test_data_solution.txt` – genre labels for test data  
- `description.txt` – dataset information  

During experimentation, inconsistencies were identified between training and test labels. To ensure reliable evaluation, a **train–validation split** was applied on the training dataset.

---

##  Project Workflow

1. **Data Loading**
   - Loaded the training dataset using pandas.
   - Analyzed the structure of plot summaries and genre labels.

2. **Data Preprocessing**
   - Cleaned genre labels by removing extra spaces and converting text to lowercase.
   - Converted plot summaries into string format for text processing.

3. **Feature Extraction**
   - Applied **TF-IDF Vectorization** to transform textual movie plots into numerical features.
   - Limited feature size to improve performance and avoid overfitting.

4. **Model Training**
   - Used a **Multinomial Naive Bayes** classifier, which is well-suited for text classification problems.
   - Trained the model on the TF-IDF features.

5. **Model Evaluation**
   - Performed a **stratified train–validation split** on the training data.
   - Evaluated the model using accuracy and classification metrics.

---

##  Results
- **Validation Accuracy:** ~51%
- The model demonstrated reasonable performance for a multi-class text classification problem.
- Results show that TF-IDF combined with Naive Bayes is effective for genre prediction from text.

---

##  Conclusion
This project provided hands-on experience with **Natural Language Processing, feature engineering, supervised learning, and model evaluation**. It also highlighted the importance of handling real-world dataset challenges such as label inconsistencies and proper evaluation strategies.

---

##  Future Improvements
- Experiment with **Logistic Regression** or **Support Vector Machines (SVM)**
- Apply **hyperparameter tuning**
- Use **word embeddings** such as Word2Vec or GloVe
- Handle class imbalance using resampling techniques

---

##  Internship Details
- **Internship Provider:** CodSoft  
- **Task:** Task 1 – Movie Genre Prediction  
- **Domain:** Machine Learning / NLP  

