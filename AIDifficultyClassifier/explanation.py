import pandas as pd
import pickle
import asyncio
from pathlib import Path
import json
from datetime import datetime
import sys
import os

# Import required components
from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

class SimpleClassifier:
    """Simple classifier that outputs difficulty, confidence, and explanation to new Excel file"""
    
    def __init__(self, model_path: str):
        """Load the trained system"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        
        print(f"Loading trained model from {model_path}...")
        with open(model_path, 'rb') as f:
            system_data = pickle.load(f)
        
        self.training_data = system_data["training_data"]
        self.question_embeddings = system_data["question_embeddings"]
        self.learned_patterns = system_data.get("learned_patterns", {})
        self.pattern_confidence = system_data.get("pattern_confidence", {})
        
        print(f"✓ Loaded training data: {len(self.training_data)} examples")
        
        # DeepSeek API configuration
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.1,
            max_tokens=1000  # Increase token limit for longer responses
        )
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ AI components initialized\n")
    
    async def classify_question(self, question: str) -> Optional[Dict]:
        """Classify a single question with detailed explanation"""
        try:
            # Find similar examples
            similar_examples = self._find_similar_examples(question)
            print(f"  Found {len(similar_examples)} similar examples")
            
            # Get AI analysis
            intent_analysis = await self._get_intent_analysis(question, similar_examples)
            if "error" in intent_analysis:
                print(f"  Intent analysis error: {intent_analysis}")
                return None
            
            print(f"  Intent analysis: {intent_analysis.get('intent_classification', 'Unknown')}")
            
            # Get learning analysis
            learning_result = await self._apply_learned_patterns(question, intent_analysis, similar_examples)
            if "error" in learning_result:
                print(f"  Learning analysis error: {learning_result}")
                return None
                
            print(f"  Learning analysis: {learning_result.get('predicted_difficulty', 'Unknown')}")
            
            # Make final prediction
            final_result = self._make_final_prediction(intent_analysis, learning_result, similar_examples)
            
            # Generate detailed explanation
            explanation = self._generate_explanation(
                question, intent_analysis, learning_result, similar_examples, final_result
            )
            
            return {
                "difficulty": final_result["difficulty"],
                "confidence": final_result["confidence"],
                "explanation": explanation
            }
            
        except Exception as e:
            print(f"  Classification exception: {str(e)}")
            return None
    
    def _find_similar_examples(self, question: str, top_k: int = 5) -> List[Dict]:
        """Find similar examples from training data"""
        question_embedding = self.sentence_model.encode([question])[0]
        similarities = cosine_similarity([question_embedding], self.question_embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        similar_examples = []
        
        for idx in top_indices:
            if similarities[idx] > 0.3:
                similar_examples.append({
                    "question": self.training_data[idx]["question"],
                    "difficulty": self.training_data[idx]["difficulty"],
                    "similarity": float(similarities[idx]),
                    "question_number": idx + 1  # Add question number (1-indexed)
                })
        
        return similar_examples
    
    async def _get_intent_analysis(self, question: str, similar_examples: List[Dict]) -> Dict:
        """Get AI intent analysis with detailed reasoning"""
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
    "reasoning": "brief explanation of why this difficulty was chosen based on the question content and context"
}}

Focus on identifying specific keywords, algorithmic concepts, data structures mentioned, and complexity indicators.
"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = self._parse_json_response(response.content)
            if "error" in result:
                print(f"  JSON parsing failed for intent: {response.content[:200]}")
                return {"error": "Failed to parse"}
            return result
        except Exception as e:
            print(f"  DeepSeek API error for intent: {str(e)}")
            return {"error": str(e)}
    
    async def _apply_learned_patterns(self, question: str, intent_analysis: Dict, similar_examples: List[Dict]) -> Dict:
        """Apply learned patterns with detailed reasoning"""
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
    "key_patterns": ["pattern1", "pattern2", "pattern3"],
    "matched_concepts": ["concept1", "concept2"],
    "reasoning": "explanation of which patterns from training data influenced this classification"
}}

Identify specific patterns, concepts, or problem-solving techniques that match the training data.
"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = self._parse_json_response(response.content)
            if "error" in result:
                print(f"  JSON parsing failed for learning: {response.content[:200]}")
                return {"error": "Failed to parse"}
            return result
        except Exception as e:
            print(f"  DeepSeek API error for learning: {str(e)}")
            return {"error": str(e)}
    
    def _make_final_prediction(self, intent_analysis: Dict, learning_result: Dict, similar_examples: List[Dict]) -> Dict:
        """Make final prediction by combining all analyses"""
        intent_pred = intent_analysis.get("intent_classification", "Medium")
        learning_pred = learning_result.get("predicted_difficulty", "Medium")
        
        intent_conf = intent_analysis.get("confidence", 0.5)
        learning_conf = learning_result.get("learning_confidence", 0.5)
        
        # Similarity voting
        similar_difficulties = [ex["difficulty"] for ex in similar_examples[:3]]
        similarity_pred = max(set(similar_difficulties), key=similar_difficulties.count) if similar_difficulties else "Medium"
        
        # Final voting
        predictions = [intent_pred, learning_pred, similarity_pred]
        final_pred = max(set(predictions), key=predictions.count)
        
        # Final confidence (Agreement Score)
        agreement_score = predictions.count(final_pred) / len(predictions)
        avg_confidence = (intent_conf + learning_conf) / 2
        final_confidence = avg_confidence * agreement_score
        
        print(f"  Final: {final_pred} (confidence: {final_confidence:.3f})")
        
        return {
            "difficulty": final_pred,
            "confidence": final_confidence,
            "intent_pred": intent_pred,
            "learning_pred": learning_pred,
            "similarity_pred": similarity_pred
        }
    
    def _generate_explanation(self, question: str, intent_analysis: Dict, 
                             learning_result: Dict, similar_examples: List[Dict],
                             final_result: Dict) -> str:
        """Generate comprehensive human-readable explanation for the classification"""
        
        explanation_parts = []
        
        # 1. Final classification decision
        explanation_parts.append(f"CLASSIFICATION: {final_result['difficulty']}")
        
        # 2. Confidence level with interpretation
        confidence = final_result['confidence']
        conf_level = "HIGH" if confidence > 0.7 else "MODERATE" if confidence > 0.5 else "LOW"
        explanation_parts.append(f"CONFIDENCE: {confidence:.2f} ({conf_level})")
        
        # 3. Analysis agreement
        intent_pred = final_result.get('intent_pred', 'N/A')
        learning_pred = final_result.get('learning_pred', 'N/A')
        similarity_pred = final_result.get('similarity_pred', 'N/A')
        
        if intent_pred == learning_pred == similarity_pred:
            explanation_parts.append(f"AGREEMENT: All methods agree")
        else:
            explanation_parts.append(
                f"VOTES: Intent={intent_pred}, Patterns={learning_pred}, Similar={similarity_pred}"
            )
        
        # 4. Most similar question from training data (question number)
        if similar_examples and len(similar_examples) > 0:
            top_example = similar_examples[0]
            q_num = top_example.get('question_number', 'Unknown')
            sim_score = top_example['similarity']
            sim_difficulty = top_example['difficulty']
            explanation_parts.append(f"MOST_SIMILAR: Question #{q_num} (Difficulty: {sim_difficulty}, Similarity: {sim_score:.2f})")
        
        # 5. Key indicators from intent analysis (all of them, not limited)
        if "key_indicators" in intent_analysis and intent_analysis["key_indicators"]:
            indicators = ", ".join(str(x) for x in intent_analysis["key_indicators"])
            if indicators:
                explanation_parts.append(f"KEYWORDS: {indicators}")
        
        # 6. Question type
        if "question_type" in intent_analysis and intent_analysis["question_type"]:
            explanation_parts.append(f"TYPE: {intent_analysis['question_type']}")
        
        # 7. Complexity signals (all of them)
        if "complexity_signals" in intent_analysis and intent_analysis["complexity_signals"]:
            signals = ", ".join(str(x) for x in intent_analysis["complexity_signals"])
            if signals:
                explanation_parts.append(f"COMPLEXITY: {signals}")
        
        # 8. Matched patterns (all of them)
        if "key_patterns" in learning_result and learning_result["key_patterns"]:
            patterns = ", ".join(str(x) for x in learning_result["key_patterns"])
            if patterns:
                explanation_parts.append(f"PATTERNS: {patterns}")
        
        # 9. Matched concepts (all of them)
        if "matched_concepts" in learning_result and learning_result["matched_concepts"]:
            concepts = ", ".join(str(x) for x in learning_result["matched_concepts"])
            if concepts:
                explanation_parts.append(f"CONCEPTS: {concepts}")
        
        # 10. Similar examples difficulties
        if similar_examples:
            similar_diffs = [ex["difficulty"] for ex in similar_examples[:3]]
            explanation_parts.append(f"SIMILAR_DIFFICULTIES: {', '.join(similar_diffs)}")
        
        # 11. FULL reasoning from intent analysis (NO TRUNCATION)
        if "reasoning" in intent_analysis and intent_analysis["reasoning"]:
            reasoning = str(intent_analysis["reasoning"]).replace('\n', ' ').replace('\r', ' ').strip()
            if reasoning:
                explanation_parts.append(f"INTENT_REASONING: {reasoning}")
        
        # 12. FULL reasoning from pattern learning (NO TRUNCATION)
        if "reasoning" in learning_result and learning_result["reasoning"]:
            reasoning = str(learning_result["reasoning"]).replace('\n', ' ').replace('\r', ' ').strip()
            if reasoning:
                explanation_parts.append(f"PATTERN_REASONING: {reasoning}")
        
        # Join all parts with separator
        full_explanation = " | ".join(explanation_parts)
        
        # Excel cell limit is 32,767 characters - should be fine now with increased tokens
        if len(full_explanation) > 32000:
            full_explanation = full_explanation[:31997] + "..."
        
        return full_explanation
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception as e:
            print(f"  JSON parsing error: {e}")
            pass
        return {"error": "Failed to parse response"}
    
    async def process_test_data(self, input_file: str = "HackerrankTest.csv"):
        """Process test data and create new Excel file with results including explanations"""
        
        # Read input file (CSV or Excel)
        try:
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
            else:
                df = pd.read_excel(input_file)
        except Exception as e:
            print(f"Error reading file: {e}")
            return
        
        print(f"✓ Loaded {len(df)} questions from {input_file}")
        print(f"✓ Columns in file: {df.columns.tolist()}\n")
        
        # Check if problem_statement column exists
        if 'problem_statement' not in df.columns:
            print(f"⚠ 'problem_statement' column not found!")
            print(f"Available columns: {df.columns.tolist()}")
            
            # Try to find the correct column
            possible_columns = [col for col in df.columns if 'problem' in col.lower() or 'question' in col.lower() or 'statement' in col.lower()]
            if possible_columns:
                column_name = possible_columns[0]
                print(f"✓ Using column '{column_name}' instead\n")
            else:
                # Use the first column
                column_name = df.columns[0]
                print(f"✓ Using first column '{column_name}' as questions\n")
        else:
            column_name = 'problem_statement'
        
        # Get questions
        questions = df[column_name].dropna().astype(str).tolist()
        print(f"Processing {len(questions)} questions...\n")
        print("=" * 80)
        
        # Process questions with detailed output
        results = []
        for i, question in enumerate(questions):
            question_preview = question[:70].replace('\r', ' ').replace('\n', ' ')
            print(f"\n[Question {i+1}/{len(questions)}]")
            print(f"Text: {question_preview}...")
            
            result = await self.classify_question(question)
            
            if result is not None:
                print(f"✓ Result: {result['difficulty']} (confidence: {result['confidence']:.3f})")
                results.append({
                    column_name: question,
                    'difficulty': result['difficulty'],
                    'confidence_score': round(result['confidence'], 3),
                    'explanation': result['explanation']
                })
            else:
                print(f"✗ CLASSIFICATION FAILED")
                results.append({
                    column_name: question,
                    'difficulty': 'FAILED',
                    'confidence_score': 0.0,
                    'explanation': 'Classification failed - unable to analyze question'
                })
        
        print("\n" + "=" * 80)
        
        # Create new DataFrame
        results_df = pd.DataFrame(results)
        
        # Save to new Excel file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"classified_results_{timestamp}.xlsx"
        
        print(f"\nSaving results...")
        
        try:
            # Save with explanation column
            results_df.to_excel(output_file, index=False, engine='openpyxl')
            print(f"✓ Results saved to: {output_file}")
            
            # Calculate and show statistics
            successful_results = [r for r in results if r['difficulty'] != 'FAILED']
            
            if successful_results:
                print(f"\n" + "=" * 80)
                print("SUMMARY STATISTICS")
                print("=" * 80)
                
                # Difficulty distribution
                diff_counts = results_df[results_df['difficulty'] != 'FAILED']['difficulty'].value_counts()
                print(f"\nDifficulty Distribution:")
                for diff, count in diff_counts.items():
                    percentage = (count / len(successful_results)) * 100
                    print(f"  {diff}: {count} questions ({percentage:.1f}%)")
                
                # Confidence distribution
                successful_scores = [r['confidence_score'] for r in successful_results]
                high_conf = sum(1 for c in successful_scores if c > 0.7)
                med_conf = sum(1 for c in successful_scores if 0.5 <= c <= 0.7)
                low_conf = sum(1 for c in successful_scores if c < 0.5)
                
                print(f"\nConfidence Distribution:")
                print(f"  High confidence (>0.7): {high_conf} questions ({high_conf/len(successful_results)*100:.1f}%)")
                print(f"  Medium confidence (0.5-0.7): {med_conf} questions ({med_conf/len(successful_results)*100:.1f}%)")
                print(f"  Low confidence (<0.5): {low_conf} questions ({low_conf/len(successful_results)*100:.1f}%)")
                
                avg_confidence = sum(successful_scores) / len(successful_scores)
                print(f"\n  Average confidence: {avg_confidence:.3f}")
                
                # Failed classifications
                failed_count = len(results) - len(successful_results)
                if failed_count > 0:
                    print(f"\n⚠ Failed classifications: {failed_count} questions")
            else:
                print("\n✗ All classifications failed!")
            
            print("\n" + "=" * 80)
            print("✓ Processing complete!")
            print("=" * 80)
            
        except Exception as e:
            print(f"✗ Error saving Excel file: {e}")
            print(f"Trying to save as CSV instead...")
            try:
                csv_file = f"classified_results_{timestamp}.csv"
                results_df.to_csv(csv_file, index=False)
                print(f"✓ Results saved to CSV: {csv_file}")
            except Exception as csv_error:
                print(f"✗ CSV save also failed: {csv_error}")
                print("\nResults DataFrame preview:")
                print(results_df.head())

async def main():
    print("=" * 80)
    print("AI QUESTION DIFFICULTY CLASSIFIER WITH EXPLANATIONS")
    print("=" * 80)
    print()
    
    # Check files
    model_path = "ai_difficulty_system.pkl"
    test_file = "HackerrankTest.csv"
    
    print("Checking required files...")
    if not Path(model_path).exists():
        print(f"✗ Error: {model_path} not found")
        print("  Please ensure the trained model file is in the current directory")
        return
    print(f"✓ Found trained model: {model_path}")
    
    if not Path(test_file).exists():
        print(f"✗ Error: {test_file} not found")
        print("  Please ensure the test data file is in the current directory")
        return
    print(f"✓ Found test data: {test_file}")
    print()
    
    # Process
    try:
        classifier = SimpleClassifier(model_path)
        await classifier.process_test_data(test_file)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())