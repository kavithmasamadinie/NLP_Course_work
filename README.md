# NLP-Based Reddit Analysis

## 📌 Project Overview
This project focuses on **Natural Language Processing (NLP) techniques** to analyze Reddit comments and posts related to COVID-19. It utilizes various NLP methodologies, including **text preprocessing, sentiment analysis, topic modeling, POS tagging, Named Entity Recognition (NER), dependency parsing, and document clustering** to extract meaningful insights from unstructured text data.

## 🛠️ Technologies & Libraries Used
- **Programming Language:** Python
- **Libraries:**
  - NLP: `spaCy`, `NLTK`, `TextBlob`, `VADER`, `BERTopic`
  - Machine Learning: `scikit-learn`, `Gensim`, `Doc2Vec`, `Word2Vec`
  - Data Processing: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`, `wordcloud`, `Plotly`
  - Parallel Processing: `multiprocessing`

## 📂 Project Structure
```
📦 NLP-Reddit-Analysis
├── 📄 Course_Work.ipynb  # Jupyter Notebook with all NLP tasks
├── 📄 README.md          # Project documentation
├── 📂 data               # Raw and processed datasets
│   ├── coronavirus_reddit_raw_comments.csv
│   ├── coronavirus_reddit_posts.csv
│   ├── final_processed_reddit_comments.csv
│   ├── final_processed_reddit_posts.csv
├── 📂 results            # Output files and visualizations
│   ├── task3_pos_ner_results.csv
│   ├── task4_sentiment_results.csv
│   ├── topic_modeling_output.csv
│   ├── clustering_results.csv
│   ├── dependency_parsing_results.csv
└── 📂 models             # Saved models (Doc2Vec, Word2Vec, etc.)
```

## 🚀 Installation & Setup
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Step 1: Clone the Repository
```sh
git clone https://github.com/yourusername/NLP-Reddit-Analysis.git
cd NLP-Reddit-Analysis
```

### Step 2: Install Dependencies
Create a virtual environment (recommended) and install the required libraries:
```sh
python -m venv nlp_env
source nlp_env/bin/activate  # On Windows: nlp_env\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Run the Jupyter Notebook
```sh
jupyter notebook Course_Work.ipynb
```

## 🔍 Key Features & Tasks
### **1. Data Preprocessing & Cleaning**
- Lowercasing, stopword removal, tokenization, and lemmatization
- Named Entity Recognition (NER) for extracting key entities
- POS tagging to analyze linguistic structures

### **2. Sentiment Analysis**
- Used **VADER and TextBlob** to classify sentiments as Positive, Negative, or Neutral
- Analyzed sentiment trends over time and correlation with topics

### **3. Topic Modeling**
- Implemented **BERTopic** for advanced topic extraction
- Visualized key discussion topics from Reddit data

### **4. Document Clustering**
- Used **Word2Vec & Doc2Vec** for text vectorization
- Applied **K-Means clustering** to group similar discussions

### **5. Dependency Parsing & Syntactic Analysis**
- Parsed sentence structures to analyze sentence dependencies
- Visualized relationships between words in complex sentences

## 📊 Results & Insights
- **Trending topics and key concerns** extracted from Reddit discussions
- **Sentiment distribution** across different phases of the pandemic
- **Clustering patterns** highlighting distinct discussion groups
- **Dependency parsing insights** on sentence structures and complexity

## 🏆 Future Improvements
- Implement **transformer-based models (BERT, GPT-4)** for better contextual analysis
- Develop **real-time sentiment tracking** using streaming APIs
- Extend the analysis to **multilingual datasets** for global sentiment tracking

## 📜 License
This project is licensed under the **MIT License**.

## 🙏 Acknowledgments
Special thanks to **[Your Name]** and [Institution/University] for supporting this project.

## 📬 Contact
For queries or collaborations, reach out via **[your.email@example.com](mailto:your.email@example.com)** or visit **[your GitHub profile](https://github.com/yourusername)**.

---
🎯 *"Unlocking insights from text, one dataset at a time!"*

