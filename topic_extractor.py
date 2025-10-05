#!/usr/bin/env python3
"""
Topic Extractor: Experimental semantic topic analysis for YouTube channels

This script experiments with different approaches to extract the main topic
from channel descriptions using various NLP techniques.
"""

import json
import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional
import sys

# Try to import optional dependencies for enhanced analysis
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available - using basic stopwords")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("SentenceTransformers not available - using TF-IDF only")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available - using basic clustering")

class TopicExtractor:
    """Experimental topic extraction using multiple approaches."""

    def __init__(self):
        self.channels = []
        self.stopwords = self._load_stopwords()
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None

        # Enhanced topic keywords for better classification
        self.topic_keywords = {
            'music': ['music', 'song', 'artist', 'band', 'album', 'concert', 'singer', 'guitar', 'piano', 'drums', 'jazz', 'rock', 'pop', 'classical', 'hip hop', 'rapper', 'musician', 'producer', 'audio', 'soundtrack', 'melody', 'harmony', 'rhythm', 'composition', 'performance', 'live music', 'recording', 'studio'],
            'technology': ['technology', 'tech', 'software', 'programming', 'computer', 'coding', 'developer', 'app', 'application', 'digital', 'innovation', 'ai', 'artificial intelligence', 'machine learning', 'data', 'algorithm', 'automation', 'robotics', 'cybersecurity', 'blockchain', 'cloud', 'api', 'framework', 'platform'],
            'gaming': ['gaming', 'game', 'gamer', 'video game', 'esports', 'streaming', 'twitch', 'playstation', 'xbox', 'nintendo', 'pc gaming', 'mobile game', 'console', 'tournament', 'league', 'competitive gaming', 'speedrun', 'walkthrough', 'gameplay', 'review'],
            'education': ['education', 'learning', 'tutorial', 'course', 'lesson', 'teaching', 'school', 'university', 'student', 'teacher', 'academic', 'study', 'knowledge', 'skill', 'training', 'workshop', 'seminar', 'lecture', 'degree', 'certification'],
            'business': ['business', 'entrepreneur', 'startup', 'company', 'corporate', 'finance', 'marketing', 'sales', 'management', 'leadership', 'strategy', 'investment', 'economics', 'industry', 'commerce', 'entrepreneurship', 'consulting', 'agency'],
            'news': ['news', 'journalism', 'reporter', 'media', 'broadcast', 'current events', 'politics', 'government', 'policy', 'election', 'campaign', 'debate', 'opinion', 'analysis', 'commentary', 'breaking news', 'headline'],
            'entertainment': ['entertainment', 'celebrity', 'hollywood', 'movie', 'film', 'television', 'tv show', 'actor', 'actress', 'director', 'producer', 'comedy', 'drama', 'reality tv', 'talk show', 'interview', 'podcast'],
            'sports': ['sports', 'athletics', 'football', 'basketball', 'baseball', 'soccer', 'tennis', 'golf', 'swimming', 'running', 'fitness', 'training', 'coach', 'team', 'league', 'championship', 'tournament', 'olympics'],
            'health': ['health', 'medical', 'fitness', 'wellness', 'nutrition', 'diet', 'exercise', 'doctor', 'physician', 'hospital', 'treatment', 'therapy', 'mental health', 'medicine', 'pharmacy', 'diagnosis', 'patient'],
            'science': ['science', 'research', 'discovery', 'experiment', 'theory', 'physics', 'chemistry', 'biology', 'astronomy', 'geology', 'mathematics', 'engineering', 'innovation', 'study', 'analysis', 'data'],
            'travel': ['travel', 'adventure', 'vacation', 'trip', 'destination', 'tourism', 'hotel', 'flight', 'cruise', 'backpacking', 'road trip', 'culture', 'exploration', 'sightseeing', 'landmark', 'journey'],
            'food': ['food', 'cooking', 'recipe', 'restaurant', 'chef', 'cuisine', 'baking', 'grilling', 'nutrition', 'diet', 'ingredient', 'meal', 'dinner', 'breakfast', 'lunch', 'culinary', 'kitchen'],
            'fashion': ['fashion', 'style', 'clothing', 'beauty', 'makeup', 'skincare', 'hair', 'accessories', 'designer', 'model', 'runway', 'boutique', 'shopping', 'trend', 'cosmetics', 'hairstyle'],
            'automotive': ['car', 'vehicle', 'automotive', 'truck', 'motorcycle', 'engine', 'repair', 'maintenance', 'racing', 'classic car', 'motorcycle', 'garage', 'mechanic', 'auto parts', 'driving'],
            'home': ['home', 'house', 'garden', 'diy', 'renovation', 'decor', 'furniture', 'appliance', 'cleaning', 'organization', 'interior design', 'landscaping', 'plants', 'flowers'],
            'finance': ['finance', 'money', 'investment', 'stock', 'trading', 'cryptocurrency', 'banking', 'wealth', 'financial planning', 'retirement', 'insurance', 'tax', 'budget', 'savings', 'loan']
        }

    def _load_stopwords(self) -> Set[str]:
        """Load comprehensive stopwords."""
        basic_stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
            'have', 'had', 'has', 'do', 'does', 'did', 'but', 'or', 'not', 'no', 'yes',
            'this', 'these', 'those', 'they', 'them', 'their', 'there', 'here', 'where',
            'when', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose', 'i', 'you',
            'we', 'us', 'me', 'my', 'your', 'our', 'ours', 'yours', 'mine', 'hers', 'his',
            'channel', 'video', 'videos', 'subscribe', 'like', 'comment', 'share',
            'follow', 'instagram', 'twitter', 'facebook', 'youtube', 'social media',
            'content', 'new', 'watch', 'check', 'thanks', 'thank', 'please', 'welcome'
        }

        if NLTK_AVAILABLE:
            try:
                nltk_stopwords = set(stopwords.words('english'))
                return basic_stopwords | nltk_stopwords
            except:
                pass

        return basic_stopwords

    def load_channels(self, filepath: str):
        """Load channels from JSON file."""
        print(f"Loading channels from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.channels = []
        for channel in data.get('channels', []):
            channel_data = {
                'id': channel['id'],
                'title': channel.get('title', ''),
                'description': channel.get('description', ''),
            }
            self.channels.append(channel_data)

        print(f"Loaded {len(self.channels)} channels")

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing."""
        if not text:
            return ""

        # Remove URLs and emails
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)

        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract meaningful keywords using multiple techniques."""
        if not text:
            return []

        # Method 1: Topic keyword matching
        words = text.lower().split()
        topic_scores = defaultdict(int)

        for word in words:
            if word in self.stopwords or len(word) < 3:
                continue

            # Score based on topic keywords
            for topic, keywords in self.topic_keywords.items():
                if word in keywords:
                    topic_scores[topic] += 1

        # Get top topics
        top_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        if top_topics:
            return [top_topics[0][0]]

        # Method 2: TF-IDF keyword extraction
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        all_words = []
        for sentence in sentences:
            # Weight first sentence more heavily
            weight = 2.0 if sentence == sentences[0] else 1.0
            sentence_words = [word for word in sentence.split() if len(word) > 3 and word not in self.stopwords]

            # Apply weighting
            if weight > 1.0:
                sentence_words *= int(weight)

            all_words.extend(sentence_words)

        if not all_words:
            return []

        # Get most frequent meaningful words
        word_freq = Counter(all_words)
        keywords = [word for word, freq in word_freq.most_common(max_keywords) if freq >= 2]

        return keywords if keywords else [word_freq.most_common(1)[0][0]]

    def extract_topic_simple(self, channel: Dict) -> str:
        """Simple keyword-based topic extraction."""
        title = channel.get('title', '')
        description = channel.get('description', '')

        # Combine and preprocess
        combined_text = f"{title} {description}"
        clean_text = self.preprocess_text(combined_text)

        if not clean_text:
            return "unknown"

        # Extract keywords
        keywords = self.extract_keywords(clean_text)

        return keywords[0] if keywords else "unknown"

    def extract_topic_semantic(self, channel: Dict) -> str:
        """Semantic topic extraction using sentence transformers."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return self.extract_topic_simple(channel)

        title = channel.get('title', '')
        description = channel.get('description', '')

        # Combine texts
        texts = []
        if title.strip():
            texts.append(title.strip())
        if description.strip():
            texts.append(description.strip())

        if not texts:
            return "unknown"

        combined_text = " ".join(texts)

        # Use sentence transformer to get embeddings
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
            embedding = model.encode([combined_text])

            # Compare with topic descriptions
            topic_descriptions = {
                'music': 'music songs artists bands albums concerts performances singing guitar piano drums jazz rock pop classical hip hop',
                'technology': 'technology software programming computers coding development artificial intelligence machine learning data science robotics',
                'gaming': 'video games gaming esports streaming gameplay reviews walkthroughs tournaments competitive gaming',
                'education': 'learning teaching courses tutorials lessons school university study knowledge skills training',
                'business': 'business entrepreneurship startup company finance marketing sales management strategy investment',
                'news': 'news journalism current events politics government policy elections breaking news headlines',
                'entertainment': 'entertainment movies films television shows celebrities comedy drama reality tv',
                'sports': 'sports athletics football basketball baseball soccer tennis golf fitness training competition',
                'health': 'health fitness wellness medical nutrition exercise doctor treatment therapy medicine',
                'science': 'science research discovery experiments physics chemistry biology astronomy geology mathematics',
                'travel': 'travel vacation adventure destinations tourism hotels flights cruises exploration',
                'food': 'food cooking recipes restaurants chefs cuisine baking grilling nutrition ingredients',
                'fashion': 'fashion clothing style beauty makeup accessories designers models trends shopping',
                'automotive': 'cars vehicles automotive trucks motorcycles engines repair racing maintenance',
                'home': 'home improvement diy renovation gardening decorating furniture appliances cleaning'
            }

            # Get similarities
            topic_embeddings = model.encode(list(topic_descriptions.values()))
            similarities = {}

            for (topic, desc), topic_emb in zip(topic_descriptions.items(), topic_embeddings):
                sim = self._cosine_similarity(embedding[0], topic_emb)
                similarities[topic] = sim

            # Return best matching topic
            best_topic = max(similarities.items(), key=lambda x: x[1])
            return best_topic[0] if best_topic[1] > 0.1 else "unknown"

        except Exception as e:
            print(f"Error in semantic analysis: {e}")
            return self.extract_topic_simple(channel)

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def run_experiment(self, input_file: str, output_file: str = 'topic_analysis.json'):
        """Run topic extraction experiment."""
        print("Starting topic extraction experiment...")

        # Load channels
        self.load_channels(input_file)

        results = []

        for i, channel in enumerate(self.channels):
            if i % 100 == 0:
                print(f"Processing channel {i+1}/{len(self.channels)}")

            # Try semantic extraction first, fall back to simple
            topic = self.extract_topic_semantic(channel)

            result = {
                'id': channel['id'],
                'title': channel['title'],
                'description': channel['description'],
                'extracted_topic': topic,
                'confidence': 'high' if topic != 'unknown' else 'low'
            }

            results.append(result)

        # Save results as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_channels': len(self.channels),
                'results': results
            }, f, indent=2, ensure_ascii=False)

        # Also save as CSV for easier review
        csv_file = output_file.replace('.json', '.csv')
        self.save_csv_results(results, csv_file)

        print(f"Topic extraction complete! Results saved to {output_file} and {csv_file}")
        return results

    def save_csv_results(self, results: List[Dict], csv_file: str):
        """Save results as CSV with channel_name, topic_extracted, descriptions columns."""
        import csv

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(['channel_name', 'topic_extracted', 'descriptions'])

            # Write data rows
            for result in results:
                writer.writerow([
                    result.get('title', ''),
                    result.get('extracted_topic', ''),
                    result.get('description', '')
                ])

        print(f"CSV results saved to {csv_file}")

    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze the topic extraction results."""
        topic_counts = Counter(result['extracted_topic'] for result in results)

        analysis = {
            'total_channels': len(results),
            'topic_distribution': dict(topic_counts.most_common()),
            'unknown_count': topic_counts.get('unknown', 0),
            'coverage_rate': (len(results) - topic_counts.get('unknown', 0)) / len(results) if results else 0
        }

        return analysis

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python topic_extractor.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'topic_analysis.json'

    extractor = TopicExtractor()
    results = extractor.run_experiment(input_file, output_file)

    # Analyze and print results
    analysis = extractor.analyze_results(results)
    print("\n" + "="*50)
    print("TOPIC EXTRACTION ANALYSIS")
    print("="*50)
    print(f"Total channels processed: {analysis['total_channels']}")
    print(f"Coverage rate: {analysis['coverage_rate']:.1%}")
    print(f"Unknown classifications: {analysis['unknown_count']}")
    print("\nTopic Distribution:")
    for topic, count in analysis['topic_distribution'].items():
        percentage = (count / analysis['total_channels']) * 100
        print(f"  {topic}: {count} ({percentage:.1f}%)")

if __name__ == '__main__':
    main()
