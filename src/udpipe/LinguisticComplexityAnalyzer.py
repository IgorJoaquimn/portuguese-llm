import numpy as np
from conllu import parse

class LinguisticComplexityAnalyzer:
    def __init__(self):
        pass

    def count_func(self, sentence, func):
        return len([token for token in sentence if(func(token))])
    
    def profundidade_maxima(self, no):
        # Se nó não tem filhos, ele é folha
        if(not no.children):
            return 0
        
        filho_mais_fundo = max([self.profundidade_maxima(child) for child in no.children])
        return 1 + filho_mais_fundo
    
    def is_clause(self, token):
        for deprel in ["csubj", "ccomp", "xcomp", "advcl", "acl"]:
            if deprel in token["deprel"]:
                return True
        return False
    
    def is_dependent_clause(self, token):
        for deprel in ["advcl", "acl"]:
            if deprel in token["deprel"]:
                return True
        return False
    
    def is_Coordination(self, token):
        for deprel in ["conj", "cc"]:
            if deprel in token["deprel"]:
                return True
        return False
    
    def count_token(self, sentence):
        return len(sentence)
    
    def is_lexical_words(self, token):
        if token["upos"] in ["NOUN", "ADJ", "VERB"]:
            return True
        if "advmod" in token["deprel"]:
            return True
        return False

    def is_adjective(self, token):
        if token["upos"] == "ADJ":
            return True
        return False
    
    def is_substantive(self, token):
        if token["upos"] == "NOUN":
            return True
        return False
    
    def analyze_sentences(self, sentences):
        
        total_clauses = sum(self.count_func(sentence, self.is_clause) for sentence in sentences)
        total_dependent_clauses = sum(self.count_func(sentence, self.is_dependent_clause) for sentence in sentences)
        total_coordinated_phrases = sum(self.count_func(sentence, self.is_Coordination) for sentence in sentences)
        
        # Count total tokens and total sentences
        total_tokens = sum(self.count_token(sentence) for sentence in sentences)
        total_sentences = len(sentences)
        
        # Calculate average tree depth and maximum tree depth
        depths = [self.profundidade_maxima(sentence.to_tree()) for sentence in sentences]
        profundidade_media = np.mean(depths)
        profundidade_max = np.max(depths)
        
        # Calculate Type-Token Ratio (TTR)
        tokens = [token for sentence in sentences for token in sentence if token["upos"] != "PUNCT"]
        types = set(token["form"] for sentence in sentences for token in sentence)
        ttr = len(types) / len(tokens) if tokens else 0
        
        # Calculate lexical density (number of lexical words / total tokens)
        lexical_density = sum(self.count_func(sentence, self.is_lexical_words) for sentence in sentences) / total_tokens if total_tokens > 0 else 0
        
        # Calculate Measures of Linguistic Complexity (MLC) and Sentence Complexity (MLS)
        MLC = total_tokens / total_clauses if total_clauses > 0 else 0
        MLS = total_tokens / total_sentences if total_sentences > 0 else 0
        
        # Calculate Dependent Clauses per Clause (DCC) and Coordination per Clause (CPC)
        DCC = total_dependent_clauses / total_clauses if total_clauses > 0 else 0
        CPC = total_coordinated_phrases / total_clauses if total_clauses > 0 else 0

        # Counter the adjectives and substantives
        adjective_list = [token for sentence in sentences for token in sentence if self.is_adjective(token)]
        substantive_list = [token for sentence in sentences for token in sentence if self.is_substantive(token)]

        return {
            "MLC": MLC,
            "MLS": MLS,
            "DCC": DCC,
            "CPC": CPC,
            "profundidade_media": profundidade_media,
            "profundidade_max": profundidade_max,
            "ttr": ttr,
            "lexical_density": lexical_density,
            "token_quantity": total_tokens,
            "adjective_list": adjective_list,
            "substantive_list": substantive_list,
        }
    
    def generate_statistics(self, text):
        sentences = parse(text)
        return self.analyze_sentences(sentences)