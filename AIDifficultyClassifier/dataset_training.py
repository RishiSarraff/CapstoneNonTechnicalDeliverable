import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, TypedDict
import json
import pickle
from pathlib import Path
import os
from collections import defaultdict, Counter
import asyncio

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ML for embeddings and similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Configuration
import warnings
warnings.filterwarnings('ignore')

# State definition for LangGraph multi-agent system
class DifficultyAnalysisState(TypedDict):
    # Input
    question: str
    training_examples: List[Dict]
    
    # Intent Mapping Agent outputs
    intent_analysis: Optional[Dict]
    question_patterns: Optional[Dict]
    semantic_features: Optional[Dict]
    
    # Learning Agent outputs
    learned_mappings: Optional[Dict]
    pattern_confidence: Optional[float]
    similar_examples: List[Dict]
    
    # Final outputs
    difficulty_prediction: Optional[str]
    confidence_score: float
    reasoning_chain: List[str]
    improvement_feedback: Optional[Dict]

class IntentMappingAgent:
    """Agent that maps questions to difficulty levels through intent recognition"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            model=model,
            temperature=0.1
        )
        
        # Semantic understanding
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Intent mapping prompts
        self.intent_analysis_prompt = PromptTemplate(
            input_variables=["question", "examples"],
            template="""
            You are an expert at analyzing software engineering questions to determine their difficulty intent.
            
            Analyze this question: "{question}"
            
            Based on these training examples:
            {examples}
            
            Extract the following intent features and return as JSON:
            {{
                "question_type": "algorithm|data_structure|system_design|optimization|implementation",
                "complexity_indicators": ["list", "of", "complexity", "signals"],
                "required_knowledge": ["concepts", "needed", "to", "solve"],
                "problem_scope": "narrow|medium|broad",
                "abstraction_level": "concrete|moderate|abstract",
                "solution_difficulty": "straightforward|requires_thinking|complex_reasoning",
                "intent_classification": "easy|medium|hard",
                "confidence": 0.0-1.0,
                "reasoning": "explanation of why this intent maps to this difficulty"
            }}
            """
        )
        
        self.pattern_extraction_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            Analyze this software engineering question for structural and semantic patterns:
            
            Question: "{question}"
            
            Extract patterns and return as JSON:
            {{
                "linguistic_patterns": ["key", "phrases", "and", "terms"],
                "question_structure": "how the question is formulated",
                "domain_indicators": ["technical", "domain", "signals"],
                "constraint_complexity": "none|simple|moderate|complex",
                "expected_solution_type": "direct|algorithmic|design|optimization",
                "technical_depth": 1-10,
                "cognitive_load": "low|medium|high"
            }}
            """
        )
    
    async def analyze_intent(self, question: str, training_examples: List[Dict]) -> Dict:
        """Analyze the intent of a question using AI"""
        
        # Prepare examples for context
        example_text = self._format_examples(training_examples[:10])  # Use top 10 similar
        
        # Get intent analysis
        intent_prompt = self.intent_analysis_prompt.format(
            question=question,
            examples=example_text
        )
        
        intent_response = await self.llm.ainvoke([HumanMessage(content=intent_prompt)])
        intent_analysis = self._parse_json_response(intent_response.content)
        
        # Get pattern analysis
        pattern_prompt = self.pattern_extraction_prompt.format(question=question)
        pattern_response = await self.llm.ainvoke([HumanMessage(content=pattern_prompt)])
        pattern_analysis = self._parse_json_response(pattern_response.content)
        
        # Get semantic features
        semantic_features = self._extract_semantic_features(question)
        
        return {
            "intent_analysis": intent_analysis,
            "pattern_analysis": pattern_analysis,
            "semantic_features": semantic_features
        }
    
    def _format_examples(self, examples: List[Dict]) -> str:
        """Format training examples for prompt context"""
        formatted = []
        for ex in examples:
            formatted.append(f"Q: {ex['question'][:100]}... | Difficulty: {ex['difficulty']}")
        return "\n".join(formatted)
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response"""
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass
        return {"error": "Failed to parse response"}
    
    def _extract_semantic_features(self, question: str) -> Dict:
        """Extract semantic features using sentence transformers"""
        embedding = self.sentence_model.encode([question])[0]
        
        return {
            "embedding_norm": float(np.linalg.norm(embedding)),
            "embedding_mean": float(np.mean(embedding)),
            "embedding_std": float(np.std(embedding)),
            "question_length": len(question.split()),
            "semantic_density": len(question.split()) / len(question) if question else 0
        }

class LearningAgent:
    """Agent that learns and improves intent mappings over time"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            model=model,
            temperature=0.2
        )
        
        # Learning prompts
        self.pattern_learning_prompt = PromptTemplate(
            input_variables=["question", "intent_analysis", "similar_examples", "historical_patterns"],
            template="""
            You are a learning agent that improves difficulty classification by analyzing patterns.
            
            Current Question: "{question}"
            Intent Analysis: {intent_analysis}
            Similar Examples from Training: {similar_examples}
            Historical Patterns: {historical_patterns}
            
            Learn from this data and return insights as JSON:
            {{
                "pattern_strength": 0.0-1.0,
                "difficulty_mapping_confidence": 0.0-1.0,
                "learned_rules": ["rule1", "rule2", "rule3"],
                "pattern_exceptions": ["exception1", "exception2"],
                "improvement_suggestions": ["suggestion1", "suggestion2"],
                "predicted_difficulty": "easy|medium|hard",
                "learning_confidence": 0.0-1.0,
                "reasoning": "detailed explanation of learning process"
            }}
            """
        )
        
        self.feedback_learning_prompt = PromptTemplate(
            input_variables=["prediction", "actual", "question", "reasoning"],
            template="""
            Learn from this prediction feedback:
            
            Question: "{question}"
            Predicted: {prediction}
            Actual: {actual}
            Reasoning: {reasoning}
            
            Analyze what went wrong/right and how to improve:
            {{
                "prediction_accuracy": "correct|incorrect",
                "error_type": "overestimation|underestimation|correct",
                "learning_points": ["point1", "point2"],
                "pattern_updates": ["update1", "update2"],
                "confidence_adjustment": -1.0 to 1.0,
                "improved_reasoning": "better reasoning approach"
            }}
            """
        )
        
        # Learning storage
        self.learned_patterns = defaultdict(list)
        self.pattern_confidence = defaultdict(float)
        self.feedback_history = []
    
    async def learn_from_examples(self, question: str, intent_analysis: Dict, 
                                similar_examples: List[Dict]) -> Dict:
        """Learn patterns from similar examples and improve mappings"""
        
        # Format similar examples
        examples_text = self._format_learning_examples(similar_examples)
        
        # Get historical patterns
        historical_patterns = self._get_historical_patterns()
        
        # Learn patterns
        learning_prompt = self.pattern_learning_prompt.format(
            question=question,
            intent_analysis=json.dumps(intent_analysis, indent=2),
            similar_examples=examples_text,
            historical_patterns=json.dumps(historical_patterns, indent=2)
        )
        
        learning_response = await self.llm.ainvoke([HumanMessage(content=learning_prompt)])
        learning_result = self._parse_json_response(learning_response.content)
        
        # Update learned patterns
        self._update_learned_patterns(learning_result)
        
        return learning_result
    
    async def learn_from_feedback(self, prediction: str, actual: str, 
                                question: str, reasoning: str) -> Dict:
        """Learn from prediction feedback to improve future classifications"""
        
        feedback_prompt = self.feedback_learning_prompt.format(
            prediction=prediction,
            actual=actual,
            question=question,
            reasoning=reasoning
        )
        
        feedback_response = await self.llm.ainvoke([HumanMessage(content=feedback_prompt)])
        feedback_result = self._parse_json_response(feedback_response.content)
        
        # Store feedback for learning
        self.feedback_history.append({
            "question": question,
            "prediction": prediction,
            "actual": actual,
            "feedback": feedback_result
        })
        
        return feedback_result
    
    def _format_learning_examples(self, examples: List[Dict]) -> str:
        """Format examples for learning context"""
        formatted = []
        for ex in examples:
            formatted.append(
                f"Q: {ex['question'][:80]}... | "
                f"Difficulty: {ex['difficulty']} | "
                f"Similarity: {ex.get('similarity', 0):.3f}"
            )
        return "\n".join(formatted)
    
    def _get_historical_patterns(self) -> Dict:
        """Get previously learned patterns"""
        return {
            "learned_rules": dict(self.learned_patterns),
            "pattern_confidences": dict(self.pattern_confidence),
            "feedback_count": len(self.feedback_history)
        }
    
    def _update_learned_patterns(self, learning_result: Dict):
        """Update learned patterns based on new learning"""
        if "learned_rules" in learning_result:
            for rule in learning_result["learned_rules"]:
                self.learned_patterns["rules"].append(rule)
        
        if "pattern_strength" in learning_result:
            self.pattern_confidence["overall"] = learning_result["pattern_strength"]
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass
        return {"error": "Failed to parse response"}

class DifficultyClassificationSystem:
    """Main system orchestrating Intent Mapping and Learning agents with LangGraph"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        # Initialize agents
        self.intent_agent = IntentMappingAgent(api_key, model)
        self.learning_agent = LearningAgent(api_key, model)
        
        # Dataset storage
        self.training_data = []
        self.question_embeddings = None
        
        # Build LangGraph workflow
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph multi-agent workflow"""
        workflow = StateGraph(DifficultyAnalysisState)
        
        # Add nodes for each agent and process
        workflow.add_node("load_similar_examples", self._load_similar_examples)
        workflow.add_node("intent_mapping", self._intent_mapping_node)
        workflow.add_node("learning_analysis", self._learning_analysis_node)
        workflow.add_node("difficulty_prediction", self._difficulty_prediction_node)
        workflow.add_node("confidence_validation", self._confidence_validation_node)
        
        # Define the workflow edges
        workflow.add_edge("load_similar_examples", "intent_mapping")
        workflow.add_edge("intent_mapping", "learning_analysis")
        workflow.add_edge("learning_analysis", "difficulty_prediction")
        workflow.add_edge("difficulty_prediction", "confidence_validation")
        workflow.add_edge("confidence_validation", END)
        
        # Set entry point
        workflow.set_entry_point("load_similar_examples")
        
        return workflow.compile()
    
    async def _load_similar_examples(self, state: DifficultyAnalysisState) -> DifficultyAnalysisState:
        """Load similar examples from training data"""
        question = state["question"]
        
        if not self.training_data:
            state["similar_examples"] = []
            state["reasoning_chain"].append("No training data available")
            return state
        
        # Get semantic similarity
        question_embedding = self.intent_agent.sentence_model.encode([question])[0]
        similarities = cosine_similarity([question_embedding], self.question_embeddings)[0]
        
        # Get top similar examples
        top_indices = np.argsort(similarities)[-10:][::-1]
        similar_examples = []
        
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Similarity threshold
                similar_examples.append({
                    "question": self.training_data[idx]["question"],
                    "difficulty": self.training_data[idx]["difficulty"],
                    "similarity": float(similarities[idx])
                })
        
        state["similar_examples"] = similar_examples
        state["reasoning_chain"].append(f"Found {len(similar_examples)} similar examples")
        
        return state
    
    async def _intent_mapping_node(self, state: DifficultyAnalysisState) -> DifficultyAnalysisState:
        """Intent Mapping Agent node"""
        question = state["question"]
        similar_examples = state["similar_examples"]
        
        # Use Intent Mapping Agent
        intent_result = await self.intent_agent.analyze_intent(question, similar_examples)
        
        state["intent_analysis"] = intent_result["intent_analysis"]
        state["question_patterns"] = intent_result["pattern_analysis"]
        state["semantic_features"] = intent_result["semantic_features"]
        
        state["reasoning_chain"].append("Intent mapping completed")
        
        return state
    
    async def _learning_analysis_node(self, state: DifficultyAnalysisState) -> DifficultyAnalysisState:
        """Learning Agent node"""
        question = state["question"]
        intent_analysis = state["intent_analysis"]
        similar_examples = state["similar_examples"]
        
        # Use Learning Agent
        learning_result = await self.learning_agent.learn_from_examples(
            question, intent_analysis, similar_examples
        )
        
        state["learned_mappings"] = learning_result
        state["pattern_confidence"] = learning_result.get("pattern_strength", 0.5)
        
        state["reasoning_chain"].append("Learning analysis completed")
        
        return state
    
    async def _difficulty_prediction_node(self, state: DifficultyAnalysisState) -> DifficultyAnalysisState:
        """Combine agent outputs for final prediction"""
        intent_prediction = state["intent_analysis"].get("intent_classification", "medium")
        learning_prediction = state["learned_mappings"].get("predicted_difficulty", "medium")
        
        # Combine predictions with confidence weighting
        intent_confidence = state["intent_analysis"].get("confidence", 0.5)
        learning_confidence = state["learned_mappings"].get("learning_confidence", 0.5)
        
        # Weighted prediction logic
        if intent_prediction == learning_prediction:
            final_prediction = intent_prediction
            confidence = (intent_confidence + learning_confidence) / 2
        else:
            # Choose prediction with higher confidence
            if intent_confidence > learning_confidence:
                final_prediction = intent_prediction
                confidence = intent_confidence * 0.8  # Reduce due to disagreement
            else:
                final_prediction = learning_prediction
                confidence = learning_confidence * 0.8
        
        state["difficulty_prediction"] = final_prediction
        state["confidence_score"] = confidence
        state["reasoning_chain"].append(f"Predicted {final_prediction} with {confidence:.3f} confidence")
        
        return state
    
    async def _confidence_validation_node(self, state: DifficultyAnalysisState) -> DifficultyAnalysisState:
        """Validate and adjust final confidence"""
        similar_examples = state["similar_examples"]
        prediction = state["difficulty_prediction"]
        
        # Check agreement with similar examples
        if similar_examples:
            similar_difficulties = [ex["difficulty"] for ex in similar_examples[:5]]
            agreement_score = similar_difficulties.count(prediction) / len(similar_difficulties)
            
            # Adjust confidence based on agreement
            original_confidence = state["confidence_score"]
            adjusted_confidence = original_confidence * (0.5 + 0.5 * agreement_score)
            
            state["confidence_score"] = adjusted_confidence
            state["reasoning_chain"].append(f"Confidence adjusted to {adjusted_confidence:.3f} based on similar examples")
        
        return state
    
    def load_training_data(self, dataset_path: str, question_col: str = "description", 
                          difficulty_col: str = "difficulty"):
        """Load and process training dataset"""
        print(f"Loading training data from {dataset_path}...")
        
        # Load dataset
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            df = pd.read_json(dataset_path)
        else:
            raise ValueError("Supported formats: CSV, JSON")
        
        # Validate columns
        if question_col not in df.columns or difficulty_col not in df.columns:
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Columns '{question_col}' or '{difficulty_col}' not found")
        
        # Process data
        df = df.dropna(subset=[question_col, difficulty_col])
        self.training_data = [
            {"question": row[question_col], "difficulty": row[difficulty_col]}
            for _, row in df.iterrows()
        ]
        
        # Generate embeddings for similarity search
        questions = [item["question"] for item in self.training_data]
        self.question_embeddings = self.intent_agent.sentence_model.encode(questions)
        
        print(f"Loaded {len(self.training_data)} training examples")
        print(f"Difficulty distribution: {Counter([item['difficulty'] for item in self.training_data])}")
    
    async def classify_question(self, question: str) -> Dict:
        """Classify a question using the multi-agent system"""
        
        initial_state = DifficultyAnalysisState(
            question=question,
            training_examples=[],
            intent_analysis=None,
            question_patterns=None,
            semantic_features=None,
            learned_mappings=None,
            pattern_confidence=None,
            similar_examples=[],
            difficulty_prediction=None,
            confidence_score=0.0,
            reasoning_chain=[],
            improvement_feedback=None
        )
        
        # Run through LangGraph workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        return {
            "question": question,
            "predicted_difficulty": final_state["difficulty_prediction"],
            "confidence": final_state["confidence_score"],
            "intent_analysis": final_state["intent_analysis"],
            "learned_patterns": final_state["learned_mappings"],
            "similar_examples": final_state["similar_examples"][:3],  # Top 3
            "reasoning": final_state["reasoning_chain"]
        }
    
    async def provide_feedback(self, question: str, predicted: str, actual: str, reasoning: str):
        """Provide feedback to improve the learning agent"""
        feedback = await self.learning_agent.learn_from_feedback(
            predicted, actual, question, reasoning
        )
        return feedback
    
    def save_system(self, filepath: str):
        """Save the trained system"""
        system_data = {
            "training_data": self.training_data,
            "question_embeddings": self.question_embeddings,
            "learned_patterns": dict(self.learning_agent.learned_patterns),
            "pattern_confidence": dict(self.learning_agent.pattern_confidence),
            "feedback_history": self.learning_agent.feedback_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(system_data, f)
        print(f"System saved to {filepath}")

# Main execution function
async def main():
    print("AI-Powered Difficulty Classification System")
    print("=" * 60)
    
    # DeepSeek API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    # Initialize system
    print("\nInitializing AI agents...")
    system = DifficultyClassificationSystem(api_key)
    
    # New dataset path and column names
    dataset_path = r"C:\Users\eshaa\OneDrive\Desktop\Y4S1\Capstone\leetcode dataset\predictiveDataset.csv"
    question_col = "description"
    difficulty_col = "difficulty"
    
    try:
        # Load and verify dataset
        df_check = pd.read_csv(dataset_path)
        print(f"\nDataset columns: {df_check.columns.tolist()}")
        print(f"Dataset shape: {df_check.shape}")
        print(f"First few rows:")
        print(df_check.head())
        
        # Load training data
        system.load_training_data(dataset_path, question_col, difficulty_col)
        
        # Save the system
        system.save_system("ai_difficulty_system.pkl")
        
        print("\n" + "="*60)
        print("Testing AI Classification System")
        print("="*60)
        
        # Test the system
        while True:
            test_question = input("\nEnter a question to classify (or 'quit' to exit): ").strip()
            if test_question.lower() == 'quit':
                break
            
            if test_question:
                print("\nAI agents analyzing...")
                result = await system.classify_question(test_question)
                
                print(f"\nResults:")
                print(f"Predicted Difficulty: {result['predicted_difficulty']}")
                print(f"Confidence: {result['confidence']:.3f}")
                
                print(f"\nIntent Analysis:")
                intent = result['intent_analysis']
                print(f"Question Type: {intent.get('question_type', 'N/A')}")
                print(f"Complexity Indicators: {intent.get('complexity_indicators', [])}")
                
                print(f"\nSimilar Examples:")
                for i, ex in enumerate(result['similar_examples'], 1):
                    print(f"{i}. [{ex['difficulty']}] {ex['question'][:80]}... (sim: {ex['similarity']:.3f})")
                
                print(f"\nAI Reasoning:")
                for step in result['reasoning']:
                    print(f"  • {step}")
                
                # Optional: Provide feedback
                feedback_choice = input("\nProvide feedback? (y/n): ").strip().lower()
                if feedback_choice == 'y':
                    actual_difficulty = input("What's the actual difficulty? (easy/medium/hard): ").strip()
                    if actual_difficulty:
                        feedback = await system.provide_feedback(
                            test_question, 
                            result['predicted_difficulty'], 
                            actual_difficulty,
                            " -> ".join(result['reasoning'])
                        )
                        print("Feedback provided - system learning improved!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())