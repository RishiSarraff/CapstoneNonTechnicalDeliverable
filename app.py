import streamlit as st
import pandas as pd
import pickle
import numpy as np
import asyncio
import os
from typing import Dict, List, Optional
import json
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import plotly.express as px ## for the plots made 

load_dotenv()

st.set_page_config(
    page_title="Software Engineering Interview Analysis",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Software Engineering Interview Analysis System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Capstone Project: Interview Effectiveness & Job Market Analysis</div>', unsafe_allow_html=True)

if 'classifier' not in st.session_state:
    st.session_state.classifier = None # Coding Problem classifier
if 'similarity_df' not in st.session_state:
    st.session_state.similarity_df = None  # Job Similarity dataset
if 'job_descriptions_df' not in st.session_state:
    st.session_state.job_descriptions_df = None  # Job Descriptions dataset

class SimpleClassifier:
    # We will instantiate and use a simplified version of the coding problem classifier
    # mainly so that our streamlit frontend can actually run without running for a long period of time
    
    def __init__(self, model_path: str):
        with open(model_path, 'rb') as f:
            system_data = pickle.load(f)
            # use .pkl file (in this case ai_difficulty_system.pkl) to get the question embessings
        
        self.training_data = system_data["training_data"]
        self.question_embeddings = system_data["question_embeddings"]
        # these will be the training data and question embedding from the .pkl file 
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        # located in our env file and need for the query embeddings
        if not api_key:
            st.error("DEEPSEEK_API_KEY environment variable not set!")
            st.stop()
            
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.1,
            max_tokens=1000
        )
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        # using the model all-MiniLM-L6-v2 we have a simple Sentence Transofmer as an LLM whic will be used to create the chat and response portion of the first tab on th epage
    
    # This function will be used to generate similar examples of question using keyword similairty as an added UI/display bonus
    # Using the question embedding we get the top k documents(questions) that are similar to our query from the dataset
    def _find_similar_examples(self, question: str, top_k: int = 5) -> List[Dict]:
        question_embedding = self.sentence_model.encode([question])[0]
        # returns query embedding
        similarities = cosine_similarity([question_embedding], self.question_embeddings)[0]
        # gets similarities
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        similar_examples = []
        
        for idx in top_indices:
            if similarities[idx] > 0.3:
                similar_examples.append({
                    "question": self.training_data[idx]["question"],
                    "difficulty": self.training_data[idx]["difficulty"],
                    "similarity": float(similarities[idx]),
                    "question_number": idx + 1
                })
        # for each of the cosine similarities append onto the similar examples array which we return back to display
        
        return similar_examples
    
    async def _get_intent_analysis(self, question: str, similar_examples: List[Dict]) -> Dict:
        examples_text = "\n".join([
            f"Q: {ex['question'][:100]}... | Difficulty: {ex['difficulty']}"
            for ex in similar_examples[:5]
        ])
        
        prompt = f"""
            Analyze this competitive programming question for difficulty classification:

            Question: "{question}"

            Similar examples from training data:
            {examples_text}

            Return JSON with detailed analysis:
            {{
                "intent_classification": "Easy|Medium|Hard",
                "confidence": 0.0-1.0,
                "key_indicators": ["keyword1", "keyword2", "keyword3"],
                "question_type": "algorithm|data_structure|dynamic_programming|graph|math|etc",
                "complexity_signals": ["signal1", "signal2"],
                "reasoning": "brief explanation of why this difficulty was chosen"
            }}
        """
        # ^ Prompt used for the intent analysis and also the reponse formatted in a JSON format with:
        ##### 1) Classification --> Easy, Medium, or Hard
        ##### 2) Confidence --> which will be dsiplayed to the user
        ##### 3) Key indicators --> Array of indicators that lean towards a specific classifications
        ##### 4) Question Type --> Specific kind of coding problem
        ##### 5) Complexity Signals --> Array of signals of complexity.
        ##### 6) Reasoning --> Reasons
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            # Send over the prompt aas a human message which contains our question and also the examples as well joined together
            return self._parse_json_response(response.content)
        except Exception as e:
            return {"error": str(e)}
    
    async def _apply_learned_patterns(self, question: str, intent_analysis: Dict, similar_examples: List[Dict]) -> Dict:
        examples_text = "\n".join([
            f"Q: {ex['question'][:60]}... | Difficulty: {ex['difficulty']}"
            for ex in similar_examples[:3]
        ])
        
        prompt = f"""
            Apply learned patterns to classify this question:

            Question: "{question}"
            Intent Analysis: {json.dumps(intent_analysis, indent=2)}
            Similar Training Examples: {examples_text}

            Return JSON with detailed analysis:
            {{
                "predicted_difficulty": "Easy|Medium|Hard",
                "learning_confidence": 0.0-1.0,
                "key_patterns": ["pattern1", "pattern2"],
                "matched_concepts": ["concept1", "concept2"],
                "reasoning": "explanation of patterns from training data"
            }}
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return self._parse_json_response(response.content)
        except Exception as e:
            return {"error": str(e)}
    
    def _parse_json_response(self, response: str) -> Dict:
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                return json.loads(json_str)

            ## Return back our json object as a json formatted str
        except:
            pass
        return {"error": "Failed to parse response"}
    
    async def classify_question(self, question: str) -> Optional[Dict]:
        try:
            similar_examples = self._find_similar_examples(question)
            intent_analysis = await self._get_intent_analysis(question, similar_examples)
            
            if "error" in intent_analysis:
                return None
            
            learning_result = await self._apply_learned_patterns(question, intent_analysis, similar_examples)

            # Based on the json response form the intent analysis we get the result of the learning result function

            if "error" in learning_result:
                return None

            # if a result from the learning result is produced, we cna make a prediction
            
            # Make final prediction
            intent_pred = intent_analysis.get("intent_classification", "Medium")
            learning_pred = learning_result.get("predicted_difficulty", "Medium")
            
            intent_conf = intent_analysis.get("confidence", 0.5)
            learning_conf = learning_result.get("learning_confidence", 0.5)
            
            similar_difficulties = [ex["difficulty"] for ex in similar_examples[:3]]
            similarity_pred = max(set(similar_difficulties), key=similar_difficulties.count) if similar_difficulties else "Medium"
            
            predictions = [intent_pred, learning_pred, similarity_pred]
            final_pred = max(set(predictions), key=predictions.count)
            
            agreement_score = predictions.count(final_pred) / len(predictions)
            avg_confidence = (intent_conf + learning_conf) / 2
            final_confidence = avg_confidence * agreement_score
            
            return {
                "difficulty": final_pred,
                "confidence": final_confidence,
                "intent_analysis": intent_analysis,
                "learning_result": learning_result,
                "similar_examples": similar_examples[:3],
                "predictions": {
                    "intent": intent_pred,
                    "learning": learning_pred,
                    "similarity": similarity_pred
                }
            }

            # return all of the JSON results from the intent analysis, learning patterns, and finally predictions in terms of similarity, intent, and learning
            
        except Exception as e:
            st.error(f"Classification error: {str(e)}")
            return None

@st.cache_resource # Helps reduce size of our ML/AI models and their loading
def load_classifier():
    try:
        with st.spinner("Loading the AI classifier model"):
            classifier = SimpleClassifier("AIDifficultyClassifier/ai_difficulty_system.pkl")
            # Instantiate SimpleClassifier object and return back out the classifier
        return classifier
    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        return None

@st.cache_data # similar to cache_resource helps with loading large csv data
def load_similarity_data():
    try:
        valid_problem_difficulties = ["easy", "medium", "hard"]
        # predefine our 3 fields to filter out only the successfully classified coding questions based on difficulties.

        df = pd.read_csv("Datasets/job_problem_similarity.csv")
        df = df[df["difficulty_level"].str.lower().isin(valid_problem_difficulties)]
        # filter df based on difficulties.
        return df
    except Exception as e:
        st.error(f"Error loading similarity data: {e}")
        return None

@st.cache_data
def load_job_descriptions():
    try:
        df = pd.read_pickle("SimilarityScorePredictor/job_descriptions_with_embeddings.pkl")
        # Get the job description data and the embeddings
        return df
    except Exception as e:
        st.error(f"Error loading job descriptions: {e}")
        return None

tab1, tab2, tab3 = st.tabs(["1) Question Difficulty Classifier", "2) Job-Problem Similarity Explorer", "3) Dataset Overview"])
# 3 tabs: coding problem classifier, job problem similarity (stats explorer), and General dataset overview

# TAB 1: Question Classifier
with tab1:
    st.header("Question Difficulty Classifier")
    st.markdown("Enter a coding question to predict its difficulty level.")
    
    # Check for API key from env
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("DEEPSEEK_API_KEY not found!")
        st.info("Please set your DeepSeek API key as an environment variable:\n```\nexport DEEPSEEK_API_KEY='your-key-here'\n```")
        st.stop()
    else:
        st.success("API Key detected")
    
    # Load classifier
    if st.session_state.classifier is None:
        st.session_state.classifier = load_classifier() # Simple Classifier loading
    
    if st.session_state.classifier:
        question_input = st.text_area(
            "Enter your coding question:",
            height=150,
            placeholder="Example: Given an array of integers, find two numbers that add up to a target sum..."
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            classify_button = st.button("Classify Question", type="primary", use_container_width=True)
        
        if classify_button and question_input:
            # if the classify button is pressed and the result question input exists, we can send over 
            # the question input into the classifier.
            with st.spinner("AI agents analyzing question..."):
                result = asyncio.run(st.session_state.classifier.classify_question(question_input))
                
                # if the result comes back we can respond back out with the prediction statistics
                if result:
                    st.success("Analysis Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Predicted Difficulty",
                            value=result['difficulty'],
                            delta=None
                        )
                    # prediction difficulty
                    
                    with col2:
                        confidence_pct = result['confidence'] * 100
                        st.metric(
                            label="Confidence Score",
                            value=f"{confidence_pct:.1f}%",
                            delta=None
                        )
                    # confidence score of prediction
                    
                    with col3:
                        conf_level = "HIGH" if result['confidence'] > 0.7 else "MODERATE" if result['confidence'] > 0.5 else "LOW"
                        st.metric(
                            label="Confidence Level",
                            value=conf_level,
                            delta=None
                        )
                    # qualitative classification of the quantitative confidence score
                    
                    st.markdown("---")
                    
                    st.subheader("Detailed Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("Intent Agent:")
                        st.info(result['predictions']['intent'])
                    with col2:
                        st.markdown("Learning Agent:")
                        st.info(result['predictions']['learning'])
                    with col3:
                        st.markdown("Similarity Agent:")
                        st.info(result['predictions']['similarity'])
                    # more information about the intent, learning, similarity agents
                    
                    with st.expander("Intent Analysis Details", expanded=True):
                        intent = result['intent_analysis']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"Question Type: {intent.get('question_type', 'N/A')}")
                            st.markdown(f"Key Indicators: {', '.join(intent.get('key_indicators', []))}")
                        with col2:
                            st.markdown(f"Complexity Signals: {', '.join(intent.get('complexity_signals', []))}")
                        
                        st.markdown("Reasoning:")
                        st.write(intent.get('reasoning', 'N/A'))
                    
                    with st.expander("Learning Agent Analysis"):
                        learning = result['learning_result']
                        
                        st.markdown(f"Key Patterns: {', '.join(learning.get('key_patterns', []))}")
                        st.markdown(f"Matched Concepts: {', '.join(learning.get('matched_concepts', []))}")
                        st.markdown("Reasoning:")
                        st.write(learning.get('reasoning', 'N/A'))
                    
                    with st.expander("Similar Training Examples"):
                        if result['similar_examples']:
                            for i, ex in enumerate(result['similar_examples'], 1):
                                st.markdown(f"{i}. [{ex['difficulty']}] (Similarity: {ex['similarity']:.3f})")
                                st.text(ex['question'][:200] + "...")
                                st.markdown("---")
                        else:
                            st.info("No similar examples found above threshold")
                    
                else:
                    st.error("Classification failed. Please try again.")
        
        elif classify_button:
            st.warning("Please enter a question first!")

# TAB 2: Similarity Explorer 
with tab2:
    st.header("Job-Problem Similarity Explorer")
    st.markdown("Explore how coding problems align with real job descriptions.")
    
    # similarity dataset
    if st.session_state.similarity_df is None:
        st.session_state.similarity_df = load_similarity_data()
    
    # job description dataset
    if st.session_state.job_descriptions_df is None:
        st.session_state.job_descriptions_df = load_job_descriptions()
    
    if st.session_state.similarity_df is not None:
        df = st.session_state.similarity_df
        
        with st.sidebar:
            st.header("Filters")
        
            # Filter field for the categories field
            ## Contains:
            ### --> [All, Architect, Backend, Data Engineering, Data Science, Database, DevOps, Frontend, Manager, Mobile, Other, QA, Security]
            categories = ['All'] + sorted(df['category'].unique().tolist())
            selected_category = st.sidebar.selectbox("Job Category", categories)
        
            # Filter field for the difficulties field
            ## Contains:
            ### --> [All, Easy, Medium, Hard]
            difficulties = ['All'] + sorted(df['difficulty_level'].unique().tolist())
            selected_difficulty = st.sidebar.selectbox("Problem Difficulty", difficulties)
        
            # Filter field for the platforms field
            ## Contains:
            ### --> [All, Codeforces, Hackerrank, LeetCode]
            platforms = ['All'] + sorted(df['platform'].unique().tolist())
            selected_platform = st.sidebar.selectbox("Platform", platforms)
        
        filtered_df = df.copy()
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        if selected_difficulty != 'All':
            filtered_df = filtered_df[filtered_df['difficulty_level'] == selected_difficulty]
        if selected_platform != 'All':
            filtered_df = filtered_df[filtered_df['platform'] == selected_platform]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Job-Problem Pairs", len(filtered_df))
        with col2:
            st.metric("Unique Jobs", filtered_df['job_title'].nunique())
        with col3:
            st.metric("Unique Problems", filtered_df['question_text'].nunique())
        with col4:
            st.metric("Avg Similarity", f"{filtered_df['similarity'].mean():.3f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Similarity Distribution by Category")
            fig1 = px.box(
                filtered_df,
                x='category',
                y='similarity',
                color='category',
                title='Similarity Scores by Job Category'
            )
            fig1.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig1, use_container_width=True)
            # Box plot for similarity scores by the job category
        
        with col2:
            st.subheader("Difficulty Distribution")
            difficulty_counts = filtered_df['difficulty_level'].value_counts()
            fig2 = px.pie(
                values=difficulty_counts.values,
                names=difficulty_counts.index,
                title='Problem Difficulty Distribution'
            )
            st.plotly_chart(fig2, use_container_width=True)
            # Circle/Pie chat for the difficulty distirbution with frequency
        
        st.subheader("Platform Comparison")
        platform_stats = filtered_df.groupby('platform').agg({
            'similarity': ['mean', 'std', 'count']
        }).round(3)
        platform_stats.columns = ['Mean Similarity', 'Std Dev', 'Count']
        st.dataframe(platform_stats, use_container_width=True)
        # Stats column regarding mean standard deviation and the frequency for the codeforces, hackerrank, and leetcode platforms
        
        st.subheader("Top Job-Problem Matches")
        
        top_n = st.slider("Number of top matches to show:", 5, 50, 10)
        top_matches = filtered_df.nlargest(top_n, 'similarity')
        
        for idx, row in top_matches.iterrows():
            with st.expander(f"Match #{idx+1}: {row['job_title']} ↔ {row['platform']} Problem (Similarity: {row['similarity']:.3f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("Job Information:")
                    st.markdown(f"- Title: {row['job_title']}")
                    st.markdown(f"- Level: {row['job_level']}")
                    st.markdown(f"- Category: {row['category']}")
                    st.markdown(f"- Type: {row['job_type']}")
                
                with col2:
                    st.markdown("Problem Information:")
                    st.markdown(f"- Platform: {row['platform']}")
                    st.markdown(f"- Difficulty: {row['difficulty_level']}")
                    st.markdown(f"- Similarity Score: {row['similarity']:.4f}")
                    st.markdown(f"- Normalized (Job): {row['similarity_norm_job']:.4f}")
                
                st.markdown("Job Description (excerpt):")
                st.text(row['description'][:300] + "...")
                
                st.markdown("Problem Statement (excerpt):")
                st.text(row['question_text'][:300] + "...")

# TAB 3: Dataset Overview
with tab3:
    st.header("Dataset Overview")
    st.markdown("Summary statistics and information about the datasets used in this analysis.")
    
    if st.session_state.similarity_df is not None:
        df = st.session_state.similarity_df
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Category Distribution")
            category_counts = df.groupby('category')['job_title'].nunique()
            fig = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                labels={'x': 'Category', 'y': 'Number of Unique Jobs'},
                title='Unique Jobs per Category'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            ## Bar graph to show unique jobs per category
        
        with col2:
            st.subheader("Similarity Score Distribution")
            fig = px.histogram(
                df,
                x='similarity',
                nbins=50,
                title='Overall Similarity Score Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
            ## Histogram for the score distribution 
        
        st.subheader("Sample Data Preview")
        st.dataframe(
            df[['job_title', 'category', 'difficulty_level', 'platform', 'similarity']].head(20),
            use_container_width=True
        )
        
        st.subheader("Statistics by Difficulty Level")
        difficulty_stats = df.groupby('difficulty_level').agg({
            'similarity': ['mean', 'median', 'std', 'min', 'max'],
            'job_title': 'count'
        }).round(4)
        difficulty_stats.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Count']
        st.dataframe(difficulty_stats, use_container_width=True)