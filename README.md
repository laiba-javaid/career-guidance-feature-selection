# 🎓 Career Guidance System

This project is a **machine learning-based Career Guidance System** that recommends suitable career paths based on user interests and strengths.  

It combines **feature selection (Information Gain)** with multiple classifiers (**Decision Tree, k-NN, Naive Bayes**) to evaluate performance and select the most accurate model.  
The final system is deployed using **Streamlit** for an interactive UI.

---

## 🧠 Project Workflow

### 1. Dataset
- The dataset contains student responses to various career-related questions.
- Features include:  
  - **Interest Area**  
  - **Problem-Solving Skills**  
  - **Math Ability**  
  - **Creativity Level**  
  - **Leadership Aspiration**  
  - **Risk Taking**  
  - **Interest in Helping Others**  
  - and more.  
- Target: **Recommended Career**

### 2. Preprocessing
- Missing values removed.  
- **Label Encoding** applied to categorical columns.  
- Data split into **70% training** and **30% testing**.  

### 3. Feature Selection
- **Information Gain (Mutual Information)** used to select the most important features.  
- Models were evaluated at multiple thresholds (`0.0`, `0.01`, `0.05`, `0.1`) and compared against the **baseline (all features)**.  

### 4. Algorithms Used
- **Decision Tree Classifier**
  - Works well for categorical data.
  - Best performing model in this project.  
- **Naive Bayes (GaussianNB)**
  - Probabilistic model based on Bayes’ theorem.
  - Good for text-like categorical data.  
- **k-Nearest Neighbors (k-NN)**
  - Distance-based classifier.
  - Performance was lower compared to other models.  

### 5. Model Selection
- Metrics evaluated: **Accuracy, Precision, Recall**.  
- ✅ **Best Model:** **Decision Tree Classifier** at threshold `0.0` with **~50% accuracy**.  

---

## 📊 Evaluation Results

| Threshold | Model          | Accuracy | Precision | Recall |
|-----------|---------------|----------|-----------|--------|
| 0.0       | Decision Tree | 0.5033   | 0.6540    | 0.6547 |
| 0.0       | k-NN          | 0.3467   | 0.4373    | 0.4171 |
| 0.0       | Naive Bayes   | 0.5000   | 0.6524    | 0.6536 |
| 0.01      | Decision Tree | 0.4700   | 0.5750    | 0.5708 |
| 0.05      | Decision Tree | 0.4733   | 0.5930    | 0.5818 |
| 0.1       | Decision Tree | 0.3467   | 0.2937    | 0.4779 |
| All Feat. | Decision Tree | 0.5100   | 0.6595    | 0.6597 |

- Decision Tree consistently outperformed other models.  
- Feature selection improved interpretability but not accuracy beyond baseline.  

---

## 📈 Visualizations

Performance graphs (Accuracy, Precision, Recall vs. Threshold) are generated and saved in the `plots/` folder.

- `accuracy_vs_threshold.png`  
- `precision_vs_threshold.png`  
- `recall_vs_threshold.png`  

---

## 🚀 Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/career-guidance-system.git
cd career-guidance-system
