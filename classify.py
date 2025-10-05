#!/usr/bin/env python3
"""
YT-Classify: Portable YouTube Channel Classification System

A data-driven classifier that groups YouTube channels into human-sensible topics
using only title and description text. No external APIs, no hand-tuned rules.

Usage: python classify.py [input_file] [output_dir]
"""

import json
import math
import re
import sys
import os
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional

# Configuration
CONFIG = {
    'TITLE_WEIGHT': 2.0,           # Title duplication factor
    'DESC_WEIGHT': 1.8,            # Description weight for umbrella assignment
    'SIM_MIN': 0.20,               # Minimum similarity for centroid reassignment
    'GAP_MIN': 0.05,               # Minimum gap for centroid reassignment
    'MIN_CLUSTER_SIZE': 5,          # Minimum channels per umbrella for centroid
    'MIN_SUB_CLUSTER_SIZE': 6,     # Minimum size before merging sub-clusters
    'MAX_SUB_CLUSTERS': 6,         # Maximum sub-clusters per umbrella
    'RANDOM_SEED': 42,             # For reproducible results
    'TOP_TERMS_COUNT': 3,          # Terms to show in explanations
}

# Global stopwords (minimal set for portability)
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'have', 'had', 'has', 'do', 'does', 'did', 'but', 'or', 'not', 'no', 'yes',
    'this', 'these', 'those', 'they', 'them', 'their', 'there', 'here', 'where',
    'when', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose', 'i', 'you',
    'we', 'us', 'me', 'my', 'your', 'our', 'ours', 'yours', 'mine', 'hers', 'his'
}

# Fixed umbrella taxonomy (24 categories)
UMBRELLAS = [
    "News & Commentary", "Science & Technology", "Business & Finance",
    "Music & Musicians", "Film & TV", "Gaming", "Sports", "Health & Wellness",
    "Food & Cooking", "Travel & Places", "Education & Learning", "Arts & Design",
    "DIY & Making", "Home & Garden", "Auto & Vehicles", "Aviation & Flight",
    "Weather & Climate", "History & Culture", "Reading & Literature",
    "Spirituality & Philosophy", "Fashion & Beauty", "Pets & Animals",
    "Comedy & Entertainment", "Podcasts & Interviews"
]

class TextProcessor:
    """Handles text tokenization and TF-IDF computation."""

    def __init__(self):
        self.vocab = set()
        self.doc_freq = Counter()
        self.documents = []

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric tokens."""
        if not text:
            return []
        # Remove URLs, punctuation, and convert to lowercase
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [word for word in text.split() if len(word) > 2 and word not in STOPWORDS]
        return tokens

    def add_document(self, tokens: List[str]):
        """Add document tokens to vocabulary and document frequency."""
        self.documents.append(tokens)
        for token in set(tokens):
            self.vocab.add(token)
            self.doc_freq[token] += 1

    def compute_tfidf(self, tokens: List[str], weight: float = 1.0) -> Dict[str, float]:
        """Compute weighted TF-IDF vector for tokens."""
        if not tokens:
            return {}

        doc_len = len(tokens)
        term_freq = Counter(tokens)

        # Compute TF-IDF
        tfidf = {}
        for term, freq in term_freq.items():
            if term in self.vocab:
                tf = freq / doc_len
                idf = math.log(len(self.documents) / (1 + self.doc_freq[term]))
                tfidf[term] = (tf * idf) * weight

        # L2 normalize
        norm = math.sqrt(sum(score**2 for score in tfidf.values()))
        if norm > 0:
            tfidf = {term: score/norm for term, score in tfidf.items()}

        return tfidf

class YTClassifier:
    """Main classification system."""

    def __init__(self, config: Dict = None):
        self.config = {**CONFIG, **(config or {})}
        random.seed(self.config['RANDOM_SEED'])
        self.processor = TextProcessor()
        self.channels = []
        self.umbrella_vectors = {}
        self.cluster_assignments = {}

    def load_channels(self, filepath: str):
        """Load channels from JSON file."""
        print(f"Loading channels from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.channels = []
        for channel in data.get('channels', []):
            # Extract video count from totalItemCount
            video_count = channel.get('contentDetails', {}).get('totalItemCount', 0)

            channel_data = {
                'id': channel['id'],
                'title': channel.get('title', ''),
                'description': channel.get('description', ''),
                'videoCount': video_count
            }
            self.channels.append(channel_data)

        print(f"Loaded {len(self.channels)} channels")

    def prepare_texts(self):
        """Prepare and tokenize all channel texts."""
        print("Preparing texts and building vocabulary...")

        for channel in self.channels:
            # Combine title (weighted) + description
            title_tokens = self.processor.tokenize(channel['title'])
            desc_tokens = self.processor.tokenize(channel['description'])

            # Duplicate title tokens for weighting
            weighted_tokens = title_tokens * int(self.config['TITLE_WEIGHT']) + desc_tokens
            self.processor.add_document(weighted_tokens)

        print(f"Built vocabulary with {len(self.processor.vocab)} unique terms")

    def compute_umbrella_vectors(self) -> Dict[str, Dict[str, float]]:
        """Compute prototype vectors for each umbrella using generic seed texts."""
        print("Computing umbrella prototype vectors...")

        # Enhanced seed texts for each umbrella (more comprehensive and distinctive)
        umbrella_seeds = {
            "News & Commentary": "news politics opinion analysis commentary current events breaking journalism reporter media broadcast editorial debate policy government election campaign democracy",
            "Science & Technology": "science technology research innovation discovery engineering computer software programming artificial intelligence machine learning data robotics physics chemistry biology laboratory experiment hypothesis theory scientific method",
            "Business & Finance": "business finance money investment economy market stock trade corporation company entrepreneur startup venture capital banking financial analysis budget revenue profit growth strategy management leadership",
            "Music & Musicians": "music song artist musician band album concert performance singer songwriter guitar piano drums jazz rock classical pop hip hop orchestra symphony melody harmony rhythm composition",
            "Film & TV": "movie film television cinema series show documentary entertainment director producer actor actress hollywood blockbuster streaming netflix disney cinema theater screenplay cinematography",
            "Gaming": "game gaming video game player console online multiplayer esports tournament league competitive play strategy adventure action role playing shooter racing sports simulation virtual reality",
            "Sports": "sport athletics game competition team player tournament league championship coach training fitness exercise olympics basketball football soccer baseball tennis golf swimming",
            "Health & Wellness": "health wellness fitness nutrition medical exercise lifestyle doctor physician hospital treatment therapy mental health wellbeing meditation yoga mindfulness diet nutritionist physical activity",
            "Food & Cooking": "food cooking recipe cuisine kitchen meal ingredient nutrition restaurant chef baking grilling culinary technique flavor taste preparation ingredient combination traditional authentic",
            "Travel & Places": "travel destination vacation trip place location adventure explore tourism tourist attraction landmark culture experience journey expedition sightseeing vacation holiday getaway",
            "Education & Learning": "education learning study course school university knowledge teaching professor student classroom lecture curriculum degree program academic research library textbook homework assignment",
            "Arts & Design": "art design creative drawing painting craft visual artistic gallery museum sculpture architecture graphic design illustration portfolio exhibition aesthetic composition color theory",
            "DIY & Making": "diy project build make craft tutorial repair woodworking metalworking fabrication construction renovation improvement handmade homemade workshop tools technique craftsmanship",
            "Home & Garden": "home garden house interior decor plant landscape outdoor lawn yard patio furniture decoration renovation remodeling maintenance landscaping horticulture botanical greenhouse",
            "Auto & Vehicles": "car vehicle automobile truck motorcycle engine repair maintenance mechanic garage automotive industry transportation driving road highway traffic safety performance modification",
            "Aviation & Flight": "airplane aircraft flight aviation pilot airport space drone helicopter airline commercial military flight simulator aerial navigation cockpit takeoff landing airspace",
            "Weather & Climate": "weather climate forecast storm temperature meteorology environment atmosphere precipitation rainfall snow hurricane tornado climate change global warming meteorologist radar satellite",
            "History & Culture": "history culture tradition heritage society civilization ancient medieval modern archaeology anthropology sociology customs folklore language ethnicity diversity civilization evolution",
            "Reading & Literature": "book reading literature novel story author writing fiction non fiction biography memoir poetry prose library bookstore publisher literary criticism analysis interpretation narrative",
            "Spirituality & Philosophy": "spirituality philosophy religion faith belief meditation mindfulness consciousness wisdom ethics morality existence meaning purpose soul spirit divine sacred contemplation",
            "Fashion & Beauty": "fashion beauty style clothing makeup cosmetic trend design model runway boutique glamour aesthetic skincare hairstyle beauty salon cosmetic surgery fashion show designer",
            "Pets & Animals": "pet animal dog cat bird fish wildlife nature care veterinary medicine adoption training behavior habitat ecosystem conservation zoo aquarium wildlife sanctuary domestic animal",
            "Comedy & Entertainment": "comedy humor funny entertainment joke laugh comedy show standup comedian sketch satire parody hilarious witty amusing entertaining performance audience laughter",
            "Podcasts & Interviews": "podcast interview discussion conversation talk show host guest audio radio broadcast dialogue debate panel roundtable monologue storytelling narrative journalism documentary"
        }

        umbrella_vectors = {}
        for umbrella in UMBRELLAS:
            seed_text = umbrella_seeds.get(umbrella, "")
            tokens = self.processor.tokenize(seed_text)
            vector = self.processor.compute_tfidf(tokens, weight=self.config['DESC_WEIGHT'])
            umbrella_vectors[umbrella] = vector

        return umbrella_vectors

    def assign_to_umbrellas(self) -> Tuple[Dict[str, List], List]:
        """Assign channels to umbrellas using prototype vectors."""
        print("Assigning channels to umbrellas...")

        assignments = defaultdict(list)
        unclassified = []

        for channel in self.channels:
            # Prepare channel text (title ×2 + description)
            title_tokens = self.processor.tokenize(channel['title'])
            desc_tokens = self.processor.tokenize(channel['description'])
            channel_tokens = title_tokens * 2 + desc_tokens

            if not channel_tokens:
                unclassified.append(channel)
                continue

            channel_vector = self.processor.compute_tfidf(channel_tokens)

            # Find best matching umbrella
            best_umbrella = None
            best_score = 0
            second_best_score = 0

            for umbrella, umbrella_vector in self.umbrella_vectors.items():
                if not umbrella_vector:
                    continue

                # Compute cosine similarity
                score = self._cosine_similarity(channel_vector, umbrella_vector)

                if score > best_score:
                    second_best_score = best_score
                    best_score = score
                    best_umbrella = umbrella
                elif score > second_best_score:
                    second_best_score = score

            # Apply more lenient thresholds for better coverage
            margin = best_score - second_best_score if second_best_score > 0 else best_score

            # Lower similarity threshold and make margin requirement less strict
            if best_score >= 0.08 and margin >= 0.01:  # More lenient thresholds
                assignments[best_umbrella].append({
                    **channel,
                    'similarity': best_score,
                    'margin': margin
                })
            else:
                unclassified.append(channel)

        print(f"Assigned {sum(len(chs) for chs in assignments.values())} channels to umbrellas")
        print(f"Left {len(unclassified)} channels unclassified")

        return assignments, unclassified

    def compute_centroid_reassignment(self, assignments: Dict[str, List], unclassified: List) -> Tuple[Dict[str, List], List]:
        """Reassign unclassified channels using computed centroids."""
        print("Computing centroids for reassignment...")

        # Compute centroids from assigned channels
        centroids = {}
        for umbrella, channels in assignments.items():
            if len(channels) >= self.config['MIN_CLUSTER_SIZE']:
                centroid = self._compute_centroid(channels)
                if centroid:
                    centroids[umbrella] = centroid

        # Reassign unclassified channels
        new_assignments = assignments.copy()
        still_unclassified = []

        for channel in unclassified:
            title_tokens = self.processor.tokenize(channel['title'])
            desc_tokens = self.processor.tokenize(channel['description'])
            channel_tokens = title_tokens * 2 + desc_tokens

            if not channel_tokens:
                still_unclassified.append(channel)
                continue

            channel_vector = self.processor.compute_tfidf(channel_tokens)

            # Find best centroid
            best_umbrella = None
            best_score = 0
            second_best_score = 0

            for umbrella, centroid in centroids.items():
                score = self._cosine_similarity(channel_vector, centroid)

                if score > best_score:
                    second_best_score = best_score
                    best_score = score
                    best_umbrella = umbrella
                elif score > second_best_score:
                    second_best_score = score

            # Apply more lenient centroid thresholds for better coverage
            margin = best_score - second_best_score if second_best_score > 0 else best_score

            # Lower centroid reassignment thresholds
            if best_score >= 0.15 and margin >= 0.02:  # More lenient for reassignment
                new_assignments[best_umbrella].append({
                    **channel,
                    'similarity': best_score,
                    'margin': margin
                })
            else:
                still_unclassified.append(channel)

        print(f"Reassigned {len(unclassified) - len(still_unclassified)} channels using centroids")
        print(f"Final unclassified: {len(still_unclassified)}")

        return new_assignments, still_unclassified

    def create_sub_clusters(self, assignments: Dict[str, List]) -> Dict[str, List]:
        """Create sub-clusters within each umbrella."""
        print("Creating sub-clusters...")

        umbrella_clusters = {}

        for umbrella, channels in assignments.items():
            if len(channels) < self.config['MIN_CLUSTER_SIZE']:
                # Keep as single "Other" cluster
                umbrella_clusters[umbrella] = [{
                    'label': f'Other ({umbrella})',
                    'channels': channels,
                    'top_terms': []
                }]
                continue

            # Determine number of clusters (1-6 based on size)
            num_clusters = min(self.config['MAX_SUB_CLUSTERS'],
                             max(1, len(channels) // 20))

            clusters = self._kmeans_cluster(channels, num_clusters)

            # Merge tiny clusters
            merged_clusters = []
            other_channels = []

            for cluster in clusters:
                if len(cluster['channels']) >= self.config['MIN_SUB_CLUSTER_SIZE']:
                    merged_clusters.append(cluster)
                else:
                    other_channels.extend(cluster['channels'])

            # Add "Other" cluster if needed
            if other_channels or len(merged_clusters) == 0:
                merged_clusters.append({
                    'label': f'Other ({umbrella})',
                    'channels': other_channels,
                    'top_terms': []
                })

            umbrella_clusters[umbrella] = merged_clusters

        return umbrella_clusters

    def _kmeans_cluster(self, channels: List, k: int) -> List[Dict]:
        """Improved term-seeded clustering with better initialization and labeling."""
        if k <= 1 or len(channels) < 3:
            # For small groups, create a single meaningful cluster
            centroid = self._compute_centroid(channels)
            top_terms = self._get_top_terms(centroid, 3)
            label = self._generate_cluster_label(channels, top_terms)
            return [{
                'label': label,
                'channels': channels,
                'top_terms': top_terms,
                'centroid': centroid
            }]

        # Use more sophisticated seeding based on channel content diversity
        clusters = self._initialize_clusters(channels, k)

        # Assign channels using multiple iterations for better convergence
        max_iterations = 10
        for iteration in range(max_iterations):
            # Clear cluster assignments
            for cluster in clusters:
                cluster['channels'] = []

            # Assign each channel to best cluster
            for channel in channels:
                best_cluster_idx = self._assign_channel_to_cluster(channel, clusters)
                clusters[best_cluster_idx]['channels'].append(channel)

            # Update centroids
            updated = False
            for cluster in clusters:
                if cluster['channels']:
                    new_centroid = self._compute_centroid(cluster['channels'])
                    if cluster['centroid']:
                        # Check if centroid changed significantly
                        similarity = self._cosine_similarity(cluster['centroid'], new_centroid)
                        if similarity < 0.95:  # Significant change
                            updated = True
                    cluster['centroid'] = new_centroid

            # If no significant changes, we're done
            if not updated and iteration > 2:
                break

        # Update cluster labels and filter out empty clusters
        final_clusters = []
        for cluster in clusters:
            if cluster['channels']:
                # Update top terms and label
                cluster['top_terms'] = self._get_top_terms(cluster['centroid'], 3)
                cluster['label'] = self._generate_cluster_label(cluster['channels'], cluster['top_terms'])

                # Only keep clusters with meaningful size
                if len(cluster['channels']) >= max(2, len(channels) // (k * 2)):
                    final_clusters.append(cluster)

        # If we ended up with too few clusters, merge small ones
        if len(final_clusters) < max(1, k // 2):
            return self._merge_small_clusters(final_clusters, channels)

        return final_clusters if final_clusters else [{
            'label': 'General',
            'channels': channels,
            'top_terms': self._get_top_terms(self._compute_centroid(channels), 3)
        }]

    def _initialize_clusters(self, channels: List, k: int) -> List[Dict]:
        """Better cluster initialization using diverse seeding."""
        clusters = []

        # Strategy 1: Use most frequent terms across all channels
        all_terms = []
        for channel in channels:
            title_tokens = self.processor.tokenize(channel['title'])
            desc_tokens = self.processor.tokenize(channel['description'])
            channel_tokens = title_tokens * 2 + desc_tokens
            channel_vector = self.processor.compute_tfidf(channel_tokens)
            top_terms = sorted(channel_vector.items(), key=lambda x: x[1], reverse=True)[:3]
            all_terms.extend([term for term, _ in top_terms])

        term_freq = Counter(all_terms)

        # Get top k most frequent terms as seeds
        seed_terms = [term for term, _ in term_freq.most_common(k)]

        # Create initial clusters
        for i, seed_term in enumerate(seed_terms):
            clusters.append({
                'seed_term': seed_term,
                'channels': [],
                'centroid': {seed_term: 1.0},  # Simple unit vector for seed
                'top_terms': [(seed_term, 1.0)]
            })

        return clusters

    def _assign_channel_to_cluster(self, channel: Dict, clusters: List[Dict]) -> int:
        """Assign channel to best matching cluster."""
        title_tokens = self.processor.tokenize(channel['title'])
        desc_tokens = self.processor.tokenize(channel['description'])
        channel_tokens = title_tokens * 2 + desc_tokens
        channel_vector = self.processor.compute_tfidf(channel_tokens)

        best_cluster_idx = 0
        best_score = -1

        for i, cluster in enumerate(clusters):
            if not cluster['centroid']:
                continue

            score = self._cosine_similarity(channel_vector, cluster['centroid'])

            if score > best_score:
                best_score = score
                best_cluster_idx = i

        return best_cluster_idx

    def _generate_cluster_label(self, channels: List, top_terms: List) -> str:
        """Generate meaningful cluster labels from top terms and channel content."""
        if not top_terms:
            # Fallback: use most common words in titles
            all_titles = ' '.join(channel['title'] for channel in channels)
            title_tokens = self.processor.tokenize(all_titles)
            if title_tokens:
                common_words = [word for word, count in Counter(title_tokens).most_common(2)]
                return ' '.join(common_words).title() if common_words else 'General'
            return 'General'

        # Use top terms to create label
        top_words = [term for term, _ in top_terms[:2]]

        # If we have meaningful terms, use them
        if top_words and len(top_words[0]) > 3:
            return ' '.join(top_words).title()

        # Otherwise, analyze channel titles for patterns
        titles = [channel['title'].lower() for channel in channels]

        # Look for common patterns in titles
        common_patterns = []
        for channel in channels[:5]:  # Check first few channels
            title_words = channel['title'].lower().split()
            for word in title_words[:3]:  # First few words often indicative
                if len(word) > 4 and word not in ['channel', 'official', 'video', 'music']:
                    common_patterns.append(word)

        if common_patterns:
            pattern_freq = Counter(common_patterns)
            top_pattern = pattern_freq.most_common(1)
            if top_pattern and top_pattern[0][1] >= 2:
                return top_pattern[0][0].title()

        # Final fallback
        return 'General'

    def _merge_small_clusters(self, clusters: List[Dict], all_channels: List) -> List[Dict]:
        """Merge small clusters back into main clusters."""
        if not clusters:
            return []

        # Keep the largest cluster and merge others into it
        clusters.sort(key=lambda x: len(x['channels']), reverse=True)
        main_cluster = clusters[0]
        other_channels = []

        for cluster in clusters[1:]:
            other_channels.extend(cluster['channels'])

        if other_channels:
            # Reassign other channels to main cluster
            main_cluster['channels'].extend(other_channels)
            main_cluster['centroid'] = self._compute_centroid(main_cluster['channels'])
            main_cluster['top_terms'] = self._get_top_terms(main_cluster['centroid'], 3)
            main_cluster['label'] = self._generate_cluster_label(main_cluster['channels'], main_cluster['top_terms'])

        return [main_cluster]

    def _compute_centroid(self, channels: List) -> Dict[str, float]:
        """Compute centroid vector from channels."""
        if not channels:
            return {}

        all_vectors = []
        for channel in channels:
            title_tokens = self.processor.tokenize(channel['title'])
            desc_tokens = self.processor.tokenize(channel['description'])
            channel_tokens = title_tokens * 2 + desc_tokens
            vector = self.processor.compute_tfidf(channel_tokens)
            all_vectors.append(vector)

        if not all_vectors:
            return {}

        # Average all vectors
        centroid = defaultdict(float)
        for vector in all_vectors:
            for term, score in vector.items():
                centroid[term] += score

        # Normalize
        total = sum(centroid.values())
        if total > 0:
            centroid = {term: score/total for term, score in centroid.items()}

        return dict(centroid)

    def _get_top_terms(self, vector: Dict[str, float], n: int = 3) -> List[Tuple[str, float]]:
        """Get top N terms from vector."""
        return sorted(vector.items(), key=lambda x: x[1], reverse=True)[:n]

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0

        # Dot product
        dot = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in set(vec1) | set(vec2))

        # Magnitudes
        mag1 = math.sqrt(sum(score**2 for score in vec1.values()))
        mag2 = math.sqrt(sum(score**2 for score in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1 * mag2)

    def generate_output(self, umbrella_clusters: Dict[str, List], unclassified: List) -> Dict:
        """Generate final JSON output."""
        print("Generating output...")

        output = {
            'umbrellas': [],
            'unclassified': []
        }

        # Process umbrellas
        for umbrella_name, clusters in umbrella_clusters.items():
            umbrella_data = {
                'name': umbrella_name,
                'clusters': []
            }

            for cluster in clusters:
                cluster_data = {
                    'label': cluster['label'],
                    'explain': {
                        'topTerms': [term for term, _ in cluster['top_terms']]
                    },
                    'items': []
                }

                for channel in cluster['channels']:
                    item = {
                        'id': channel['id'],
                        'title': channel['title'],
                        'description': channel['description'],
                        'videoCount': channel['videoCount'],
                        'score': {
                            'sim': channel.get('similarity', 0),
                            'margin': channel.get('margin', 0)
                        }
                    }
                    cluster_data['items'].append(item)

                umbrella_data['clusters'].append(cluster_data)

            output['umbrellas'].append(umbrella_data)

        # Add unclassified
        for channel in unclassified:
            output['unclassified'].append({
                'id': channel['id'],
                'title': channel['title'],
                'description': channel['description'],
                'videoCount': channel['videoCount']
            })

        return output

    def generate_csv(self, output: Dict, filepath: str):
        """Generate CSV output for UI integration."""
        print(f"Generating CSV at {filepath}...")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('cluster_id,cluster_label,id,title,description,sim,margin,videoCount\n')

            for umbrella in output['umbrellas']:
                umbrella_slug = umbrella['name'].lower().replace(' ', '-').replace('&', 'and')

                for cluster in umbrella['clusters']:
                    cluster_slug = cluster['label'].lower().replace(' ', '-').replace('(', '').replace(')', '')
                    cluster_id = f"{umbrella_slug}::{cluster_slug}"

                    for item in cluster['items']:
                        # Escape quotes and handle newlines
                        title = item['title'].replace('"', '""')
                        description = item['description'].replace('"', '""').replace('\n', ' ')

                        f.write(f'"{cluster_id}","{cluster['label']}","{item["id"]}","{title}","{description}",{item["score"]["sim"]},{item["score"]["margin"]},{item["videoCount"]}\n')

    def generate_report(self, output: Dict, filepath: str):
        """Generate human-readable report."""
        print(f"Generating report at {filepath}...")

        lines = []
        lines.append("# YT-Classify Report")
        lines.append("")
        lines.append(f"Total channels: {len(self.channels)}")

        # Umbrella summary
        total_assigned = sum(
            sum(len(cluster['items']) for cluster in umbrella['clusters'])
            for umbrella in output['umbrellas']
        )
        coverage = total_assigned / len(self.channels) if self.channels else 0

        lines.append(f"Assigned: {total_assigned} ({coverage:.1%})")
        lines.append(f"Unclassified: {len(output['unclassified'])} ({1-coverage:.1%})")
        lines.append("")

        # Per-umbrella stats
        lines.append("## Umbrella Summary")
        lines.append("")
        lines.append("| Umbrella | Channels | Clusters | Coverage |")
        lines.append("|----------|----------|----------|----------|")

        for umbrella in output['umbrellas']:
            total_channels = sum(len(cluster['items']) for cluster in umbrella['clusters'])
            num_clusters = len(umbrella['clusters'])

            lines.append(f"| {umbrella['name']} | {total_channels} | {num_clusters} | {total_channels/len(self.channels):.1%}" if self.channels else "0 |")

        lines.append("")

        # Top clusters
        lines.append("## Largest Clusters")
        lines.append("")
        all_clusters = []
        for umbrella in output['umbrellas']:
            for cluster in umbrella['clusters']:
                all_clusters.append((cluster['label'], len(cluster['items']), umbrella['name']))

        all_clusters.sort(key=lambda x: x[1], reverse=True)

        for label, size, umbrella in all_clusters[:20]:
            lines.append(f"- **{label}** ({umbrella}): {size} channels")

        # Write report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def run(self, input_file: str, output_dir: str = 'out'):
        """Run complete classification pipeline."""
        print("Starting YT-Classify pipeline...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load and prepare data
        self.load_channels(input_file)
        self.prepare_texts()

        # Compute umbrella vectors
        self.umbrella_vectors = self.compute_umbrella_vectors()

        # Initial assignment
        assignments, unclassified = self.assign_to_umbrellas()

        # Centroid reassignment
        assignments, unclassified = self.compute_centroid_reassignment(assignments, unclassified)

        # Create sub-clusters
        umbrella_clusters = self.create_sub_clusters(assignments)

        # Generate outputs
        output = self.generate_output(umbrella_clusters, unclassified)

        json_path = os.path.join(output_dir, 'clusters.json')
        csv_path = os.path.join(output_dir, 'clusters.csv')
        report_path = os.path.join(output_dir, 'report.md')

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self.generate_csv(output, csv_path)
        self.generate_report(output, report_path)

        print("Pipeline complete!")
        print(f"Output files: {json_path}, {csv_path}, {report_path}")

        return output

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python classify.py <input_file> [output_dir]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'out'

    classifier = YTClassifier()
    classifier.run(input_file, output_dir)

if __name__ == '__main__':
    main()
