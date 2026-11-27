"""
Linguistic Feature Extraction & Explainability Engine
Title: A Multimodal and Linguistic Framework for AI-Generated Content Detection
"""

import math
import re
import numpy as np
from typing import Dict, List, Tuple, Any

# Universal POS tagger fallback if NLTK tagger is not downloaded
COMMON_NOUN_SUFFIXES = ('tion', 'sion', 'ment', 'ence', 'ance', 'ity', 'ship', 'ness', 'ism', 'ist', 'er', 'or')
COMMON_VERB_SUFFIXES = ('ate', 'ify', 'ise', 'ize', 'ing', 'ed')
COMMON_ADJ_SUFFIXES = ('able', 'ible', 'al', 'ful', 'ic', 'ive', 'less', 'ous', 'ish')

FUNCTION_WORDS = {
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
    'if', 'because', 'as', 'what', 'when', 'where', 'who', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'can', 'will', 'just', 'should', 'now', 'into', 'in', 'to', 'for', 'with',
    'about', 'against', 'between', 'through', 'during', 'before', 'after',
    'above', 'below', 'from', 'up', 'down', 'by', 'off', 'over', 'under',
    'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their'
}

# Paraphrase & Thesaurus markers specific to QuillBot, StealthGPT, and Adversarial Rewriters
HUMANIZER_SIGNATURE_LEXICON = {
    'undeniably', 'essentially', 'substantially', 'inherently', 'manifestly',
    'proficient', 'assorted', 'unrivaled', 'predicament', 'multilateral',
    'proactively', 'apparatuses', 'transfigured', 'pedagogical', 'mechanisms',
    'facet', 'unquestionably', 'paramount', 'pivotal',
    'in essence', 'it is worth noting', 'a testament to',
    'pedagogical pursuit', 'intellectual operations', 'testing baselines',
    'yields substantial', 'proactively deploy', 'an assortment of',
    'unrivaled outcomes', 'an indispensable facet', 'of prime significance',
    'multifaceted', 'delineate', 'furnish', 'ameliorate', 'paradigm',
    'trajectory', 'subservient', 'concomitant', 'plethora', 'myriad',
    'epitome', 'quintessential', 'metamorphosis', 'catalyst', 'explicate',
    'elucidate', 'salient', 'paucity', 'scintillating', 'germane', 'copious',
    'indispensable apparatus', 'metamorphosed', 'interconnected mechanisms',
    'progressively', 'innovative', 'streamline', 'utilize', 'utilizing',
    'incorporate', 'incorporating', 'facilitate', 'facilitating', 'crucial',
    'vital', 'indispensable', 'comprehensive', 'intricate', 'nuanced',
    'holistic', 'mundane', 'profound', 'profoundly', 'notably',
    'consequently', 'subsequently', 'predominantly', 'intrinsically',
    'synergy', 'dynamic', 'meticulous', 'rigorous', 'seamless', 'robust',
    'unwavering', 'foundational', 'cornerstone', 'linchpin', 'bedrock',
    'keystone', 'overarching', 'interconnected', 'intricacies', 'complexities',
    'dynamics', 'nuances', 'realms', 'facets', 'dimensions', 'trajectories',
    'landscapes', 'paradigms', 'it is widely acknowledged',
    'it is generally acknowledged', 'it is evident that', 'stands as a',
    'plays a crucial role', 'serves as a', 'beacon of'
}

# Discourse markers common in raw LLMs (ChatGPT / Claude / Gemini)
PURE_AI_DISCOURSE_MARKERS = {
    'delve', 'tapestry', 'beacon', 'realm', 'harness', 'crucial', 'notably',
    'furthermore', 'moreover', 'in conclusion', 'it is important to note',
    'a testament to', 'navigating', 'landscape', 'evolving landscape', 'paradigm shift'
}

class LinguisticAnalyzer:
    """
    Computes 15 core linguistic, stylometric, and syntactic indicators.
    """
    def __init__(self):
        self.nltk_available = False
        try:
            import nltk
            from nltk.tokenize import sent_tokenize, word_tokenize
            from nltk import pos_tag
            self.sent_tokenize = sent_tokenize
            self.word_tokenize = word_tokenize
            self.pos_tag = pos_tag
            self.nltk_available = True
        except Exception:
            self.nltk_available = False

    def tokenize_sentences(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if self.nltk_available:
            try:
                return self.sent_tokenize(text)
            except Exception:
                pass
        # Robust regex sentence boundary splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_words(self, text: str) -> List[str]:
        text = text.strip().lower()
        if not text:
            return []
        if self.nltk_available:
            try:
                words = self.word_tokenize(text)
                return [w for w in words if re.match(r'^[a-zA-Z0-9\'-]+$', w)]
            except Exception:
                pass
        return re.findall(r'\b[a-zA-Z0-9\'-]+\b', text)

    def count_syllables(self, word: str) -> int:
        word = word.lower()
        count = len(re.findall(r'[aeiouy]+', word))
        if word.endswith('e') and not word.endswith('le') and len(word) > 2:
            count = max(1, count - 1)
        return max(1, count)

    def extract_features(self, text: str) -> Dict[str, float]:
        sentences = self.tokenize_sentences(text)
        words = self.tokenize_words(text)
        
        num_sentences = max(len(sentences), 1)
        num_words = max(len(words), 1)
        unique_words = set(words)
        num_unique = len(unique_words)

        # 1. Lexical Diversity (Type-Token Ratio - TTR)
        ttr = num_unique / num_words
        
        # 2. Root TTR (Guiraud's Index)
        rttr = num_unique / math.sqrt(num_words)

        # 3. Hapax Legomena Ratio (words occurring exactly once)
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        hapax_count = sum(1 for c in word_counts.values() if c == 1)
        hapax_ratio = hapax_count / num_words

        # 4 & 5. Sentence length statistics & Burstiness
        sentence_lengths = [len(self.tokenize_words(s)) for s in sentences]
        avg_sentence_len = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
        sentence_len_std = float(np.std(sentence_lengths)) if sentence_lengths else 0.0
        var_sentence_len = float(np.var(sentence_lengths)) if sentence_lengths else 0.0
        # Burstiness Coefficient of Variation: CV = sigma / mu
        sentence_burstiness = (sentence_len_std / avg_sentence_len) if avg_sentence_len > 0 else 0.0

        # 6. Average word length in characters
        avg_word_len = sum(len(w) for w in words) / num_words

        # 7. Punctuation Density
        punct_count = len(re.findall(r'[,.;:!?\"\'\(\)\[\]\-]', text))
        punct_density = punct_count / max(len(text), 1)

        # 8, 9, 10. POS Density (Nouns, Verbs, Adjectives/Adverbs)
        noun_count, verb_count, adj_adv_count = 0, 0, 0
        if self.nltk_available:
            try:
                tagged = self.pos_tag(words)
                for _, tag in tagged:
                    if tag.startswith('NN'):
                        noun_count += 1
                    elif tag.startswith('VB'):
                        verb_count += 1
                    elif tag.startswith('JJ') or tag.startswith('RB'):
                        adj_adv_count += 1
            except Exception:
                self.nltk_available = False

        if not self.nltk_available:
            # Rule-based morphological fallback
            for w in words:
                if w.endswith(COMMON_NOUN_SUFFIXES):
                    noun_count += 1
                elif w.endswith(COMMON_VERB_SUFFIXES):
                    verb_count += 1
                elif w.endswith(COMMON_ADJ_SUFFIXES) or w.endswith('ly'):
                    adj_adv_count += 1

        noun_ratio = noun_count / num_words
        verb_ratio = verb_count / num_words
        adj_adv_ratio = adj_adv_count / num_words

        # 11. Function Word Ratio
        func_count = sum(1 for w in words if w in FUNCTION_WORDS)
        func_word_ratio = func_count / num_words

        # 12. Flesch Reading Ease (Clamped to 0 - 100)
        total_syllables = sum(self.count_syllables(w) for w in words)
        syllables_per_word = total_syllables / num_words
        raw_fre = 206.835 - (1.015 * avg_sentence_len) - (84.6 * syllables_per_word)
        flesch_reading_ease = max(0.0, min(100.0, raw_fre))

        # 13. Shannon Entropy of Word Distribution
        word_probs = [count / num_words for count in word_counts.values()]
        shannon_entropy = -sum(p * math.log2(p) for p in word_probs if p > 0)

        # 14. Syntactic Uniformity Index
        syntactic_uniformity = 1.0 / (1.0 + var_sentence_len)

        # 15. Formality Score (Heylighen & Dewaele metric proxy)
        # Formality = (Noun% + Adj% + Prep% - Pronoun% - Verb% - Adv%) + 50
        formality = (noun_ratio + adj_adv_ratio - verb_ratio - (1.0 - func_word_ratio)) * 50.0 + 50.0
        formality = max(0.0, min(100.0, formality))

        return {
            "lexical_diversity_ttr": float(ttr),
            "root_ttr": float(rttr),
            "hapax_legomena_ratio": float(hapax_ratio),
            "avg_sentence_len": float(avg_sentence_len),
            "sentence_len_std": float(sentence_len_std),
            "sentence_burstiness_cv": float(sentence_burstiness),
            "avg_word_len": float(avg_word_len),
            "punct_density": float(punct_density),
            "noun_ratio": float(noun_ratio),
            "verb_ratio": float(verb_ratio),
            "adj_adv_ratio": float(adj_adv_ratio),
            "function_word_ratio": float(func_word_ratio),
            "flesch_reading_ease": float(flesch_reading_ease),
            "shannon_entropy": float(shannon_entropy),
            "syntactic_uniformity": float(syntactic_uniformity),
            "formality_score": float(formality)
        }

    def get_feature_vector(self, text: str) -> np.ndarray:
        feats = self.extract_features(text)
        return np.array(list(feats.values()), dtype=np.float32)

    def extract_humanizer_signatures(self, text: str, ling_feats: Dict[str, float]) -> Dict[str, Any]:
        """
        Specialized detection of adversarial humanizers (QuillBot, StealthGPT, Undetectable AI).
        Evaluates lexical-burstiness decoupling, Latinate paraphrasing markers, and syntactic rigidity.
        """
        words = self.tokenize_words(text)
        N = len(words)
        ttr = ling_feats.get("lexical_diversity_ttr", 0.5)
        burstiness_cv = ling_feats.get("sentence_burstiness_cv", 0.3)
        syntactic_uniformity = ling_feats.get("syntactic_uniformity", 0.2)

        # 1. Lexical-Burstiness Decoupling Metric
        expected_ttr = 0.85 if N < 60 else (0.72 if N < 120 else 0.62)
        ttr_elevation = max(0.0, (ttr - expected_ttr) / 0.20)
        burstiness_deficit = max(0.0, (0.24 - burstiness_cv) / 0.24)
        decouple_score = ttr_elevation * burstiness_deficit * 3.5

        # 2. Paraphraser Lexical Markers
        marker_hits = []
        text_lower = text.lower()
        for phrase in HUMANIZER_SIGNATURE_LEXICON:
            if ' ' in phrase:
                if phrase in text_lower:
                    marker_hits.append(phrase)
            else:
                if phrase in words:
                    marker_hits.append(phrase)
        
        marker_density = len(marker_hits) / (N + 1e-7)
        marker_score = min(1.0, marker_density * 35.0)

        # 3. Syntactic Rigidity
        rigidity_score = min(1.0, syntactic_uniformity * 2.0) if burstiness_cv < 0.16 else 0.0

        # Combined Humanizer Score
        humanizer_score = (0.45 * decouple_score) + (0.40 * marker_score) + (0.15 * rigidity_score)
        humanizer_score = max(0.0, min(1.0, humanizer_score))

        has_markers = (len(marker_hits) >= 2) or (len(marker_hits) >= 1 and (decouple_score > 0.25 or marker_density >= 0.012))
        is_humanized = has_markers or ((decouple_score >= 0.45) and (burstiness_cv < 0.14))

        if len(marker_hits) >= 2 or marker_density >= 0.015:
            tool_signature = "QuillBot / Thesaurus Paraphraser"
        elif burstiness_cv < 0.14 and N > 30 and is_humanized:
            tool_signature = "StealthGPT / Perplexity Smoothing Bypass"
        elif is_humanized:
            tool_signature = "Adversarial Paraphrasing Bypass Engine"
        else:
            tool_signature = "Natural Human or Raw AI"

        return {
            "is_humanized": is_humanized,
            "humanizer_score": round(humanizer_score, 3),
            "tool_signature": tool_signature,
            "marker_hits": marker_hits,
            "marker_density": round(marker_density, 4),
            "decouple_score": round(decouple_score, 3)
        }


class PerplexityBurstinessEstimator:
    """
    Sentence-level perplexity & burstiness evaluator with XAI color highlights.
    Uses Causal LM (DistilGPT-2 / GPT-2) if available, with mathematical n-gram fallback.
    """
    def __init__(self, use_neural: bool = True, model_name: str = "distilgpt2", device: str = None):
        self.use_neural = use_neural
        self.model = None
        self.tokenizer = None
        self.device = device
        
        if self.use_neural:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                if self.device is None:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
                self.model.eval()
            except Exception:
                self.use_neural = False

    def compute_sentence_perplexity(self, sentence: str) -> float:
        if not sentence.strip():
            return 45.0

        if self.use_neural and self.model is not None and self.tokenizer is not None:
            import torch
            try:
                inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                if inputs["input_ids"].size(1) < 2:
                    return 40.0
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                    ppl = math.exp(min(loss, 10.0))
                    return float(ppl)
            except Exception:
                pass

        # Statistical Perplexity Fallback:
        # Measures token unexpectedness and n-gram repetition
        words = re.findall(r'\b[a-zA-Z0-9\'-]+\b', sentence.lower())
        if not words:
            return 40.0
        
        unique_ratio = len(set(words)) / len(words)
        avg_word_len = sum(len(w) for w in words) / len(words)
        # LLMs generate text with very uniform word length and predictable patterns
        statistical_surprisal = (avg_word_len * 4.5) * (unique_ratio ** 1.2)
        return float(max(8.0, min(140.0, statistical_surprisal * 1.8)))

    def analyze_text(self, text: str) -> Dict[str, Any]:
        analyzer = LinguisticAnalyzer()
        sentences = analyzer.tokenize_sentences(text)
        if not sentences:
            return {
                "overall_perplexity": 45.0,
                "perplexity_std": 0.0,
                "burstiness_score": 0.0,
                "sentence_details": [],
                "ai_likelihood_linguistic": 0.5,
                "explanation_summary": "Input text was empty."
            }

        sentence_ppls = []
        sentence_details = []

        for idx, sent in enumerate(sentences):
            ppl = self.compute_sentence_perplexity(sent)
            sentence_ppls.append(ppl)
            
            # Sentence Classification Thresholds:
            # PPL < 22: Very predictable token choice (Typical of greedy/top-p LLM autoregressive decoding)
            # PPL 22-38: Moderately predictable
            # PPL 38-58: Natural variation
            # PPL > 58: High entropy / creative burst (Typical human writing)
            if ppl < 22.0:
                flag = "High AI Signature (Very Low Perplexity / Highly Predictable)"
                color = "#ff4d4f" # Red
                bg_color = "rgba(255, 77, 79, 0.18)"
            elif ppl < 38.0:
                flag = "Moderate AI Signature (Standard LLM Structure)"
                color = "#fa8c16" # Orange
                bg_color = "rgba(250, 140, 22, 0.16)"
            elif ppl < 55.0:
                flag = "Neutral / Hybrid Characteristics"
                color = "#faad14" # Yellow
                bg_color = "rgba(250, 173, 20, 0.14)"
            else:
                flag = "Human Signature (High Perplexity & High Entropy Variation)"
                color = "#52c41a" # Green
                bg_color = "rgba(82, 196, 26, 0.18)"

            sentence_details.append({
                "sentence_index": idx + 1,
                "text": sent,
                "perplexity": round(ppl, 2),
                "assessment": flag,
                "color_code": color,
                "bg_color": bg_color
            })

        mean_ppl = float(np.mean(sentence_ppls))
        std_ppl = float(np.std(sentence_ppls))
        # Burstiness: B = std / mean
        burstiness = float(std_ppl / mean_ppl) if mean_ppl > 0 else 0.0

        # Linguistic AI Likelihood Scoring
        # Low Perplexity + Low Burstiness = Strong AI Indicator
        ai_score = 0.50
        if mean_ppl < 22.0:
            ai_score += 0.28
        elif mean_ppl < 35.0:
            ai_score += 0.15
        elif mean_ppl > 55.0:
            ai_score -= 0.25

        if burstiness < 0.20:
            ai_score += 0.20
        elif burstiness < 0.32:
            ai_score += 0.10
        elif burstiness > 0.45:
            ai_score -= 0.20

        ai_score = max(0.05, min(0.96, ai_score))

        # Explainability diagnosis
        reasons = []
        if mean_ppl < 28.0:
            reasons.append(f"Low overall perplexity ({mean_ppl:.1f}), showing high word predictability characteristic of autoregressive LLMs.")
        else:
            reasons.append(f"Perplexity ({mean_ppl:.1f}) shows natural lexical variance.")

        if burstiness < 0.25:
            reasons.append(f"Low burstiness coefficient ({burstiness:.2f}), reflecting unnatural sentence length and complexity uniformity.")
        else:
            reasons.append(f"High burstiness coefficient ({burstiness:.2f}), reflecting human-like alternating sentence dynamics.")

        explanation = " ".join(reasons)

        return {
            "overall_perplexity": round(mean_ppl, 2),
            "perplexity_std": round(std_ppl, 2),
            "burstiness_score": round(burstiness, 3),
            "sentence_details": sentence_details,
            "ai_likelihood_linguistic": round(ai_score, 3),
            "explanation_summary": explanation
        }
