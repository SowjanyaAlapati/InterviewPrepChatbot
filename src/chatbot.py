import pandas as pd
import random
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob

class InterviewChatbot:
    def __init__(self, csv_path="../data/interview_questions.csv"):
        self.df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(self.df)} questions.")
        
        # Load semantic similarity model
        print("⏳ Loading AI model for answer evaluation...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model loaded.")

    def get_random_question(self, category=None):
        if category:
            subset = self.df[self.df['Category'].str.lower() == category.lower()]
            if subset.empty:
                return "No questions found for that category.", "", ""
            row = subset.sample(n=1).iloc[0]
        else:
            row = self.df.sample(n=1).iloc[0]
        return row['Question'], row['IdealAnswer'], row['Keywords']

    def evaluate_answer(self, user_answer, ideal_answer, keywords):
        # Semantic similarity
        score = util.cos_sim(self.model.encode(user_answer), self.model.encode(ideal_answer)).item()
        similarity_score = round(score * 10, 2)  # scale 0-10
        
        # Keyword checking
        missing_keywords = []
        for kw in keywords.split(','):
            if kw.lower().strip() not in user_answer.lower():
                missing_keywords.append(kw.strip())
        
        # Sentiment analysis
        sentiment = TextBlob(user_answer).sentiment.polarity
        sentiment_feedback = "Positive tone" if sentiment > 0 else "Neutral/Negative tone"
        
        feedback = f"Similarity Score: {similarity_score}/10\n" \
                   f"Missing keywords: {missing_keywords}\n" \
                   f"Sentiment: {sentiment_feedback}"
        
        return similarity_score, feedback


if __name__ == "__main__":
    bot = InterviewChatbot()
    
    print("\nWelcome to the AI Interview Simulator!")
    print("Available categories:", ", ".join(bot.df['Category'].unique()))
    
    category = input("Choose a category (or press Enter for random): ").strip()
    try:
        num_questions = int(input("How many questions do you want to attempt? "))
    except:
        num_questions = 5  # default

    total_score = 0
    review_list = []  # <- to store Q/A pairs for final review

    for i in range(num_questions):
        print(f"\n--- Question {i+1} ---")
        q, ideal, kw = bot.get_random_question(category if category else None)
        print("💬", q)
        user_answer = input("Your Answer: ")

        score, fb = bot.evaluate_answer(user_answer, ideal, kw)
        total_score += score

        # Save info for final review
        review_list.append({
            "question": q,
            "your_answer": user_answer,
            "ideal_answer": ideal,
            "score": score,
            "feedback": fb
        })

        print("\n📊 Evaluation:\n", fb)

    avg_score = round(total_score / num_questions, 2)
    print("\n🎯 Interview Complete!")
    print(f"Your Average Score: {avg_score}/10")

    # ===== Final Review Section =====
    print("\n📝 Review & Ideal Answers:")
    for idx, item in enumerate(review_list, 1):
        print(f"\nQ{idx}: {item['question']}")
        print(f"Your Answer: {item['your_answer']}")
        print(f"Ideal Answer: {item['ideal_answer']}")
        print(f"Score: {round(item['score'],2)}/10")
        print(f"Feedback: {item['feedback']}")


