import os
import json
import pickle
import torch
import torch.nn.functional as F
import math
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from datasets import Dataset
from tqdm import tqdm
import random
import numpy as np
from scipy.spatial.distance import cosine
import re
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
import optuna



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")


# ==================== 标准实现 ====================

class BleuScore:
    """标准BLEU分数计算器"""

    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

    def compute_blue_score(self, real, predicted):
        score = []
        for sent1, sent2 in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2, weights=(self.w1, self.w2, self.w3, self.w4)))
        return score


class Channels:
    """标准信道模拟器"""

    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig


def PowerNormalize(x):
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    return x


def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std


# ==================== 数据加载和处理函数 ====================

def load_pkl_dataset(data_dir):
    train_file = os.path.join(data_dir, "train_data.pkl")
    test_file = os.path.join(data_dir, "test_data.pkl")
    try:
        print(f"📂 尝试加载PKL数据集...")
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"训练集文件不存在: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"测试集文件不存在: {test_file}")
        with open(train_file, 'rb') as f:
            train_data = pickle.load(f)
        print(f"✅ 成功加载训练集，包含 {len(train_data)} 条数据")
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
        print(f"✅ 成功加载测试集，包含 {len(test_data)} 条数据")
        return train_data, test_data
    except Exception as e:
        print(f"❌ 加载PKL数据集失败: {e}")
        raise


def load_json_vocabulary(data_dir):
    json_file = os.path.join(data_dir, "vocab.json")
    try:
        print(f"📂 尝试加载vocab.json...")
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON文件不存在: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        print(f"✅ 成功加载vocab.json")
        if isinstance(json_data, dict) and 'token_to_idx' in json_data:
            vocab_dict = {
                'token_to_idx': json_data['token_to_idx'],
                'idx_to_token': {v: k for k, v in json_data['token_to_idx'].items()}
            }
        else:
            vocab_dict = {
                'token_to_idx': {str(item): idx for idx, item in enumerate(json_data)},
                'idx_to_token': {idx: str(item) for idx, item in enumerate(json_data)}
            }
        print(f"✅ 提取到 {len(vocab_dict['token_to_idx'])} 个词汇")
        return vocab_dict
    except Exception as e:
        print(f"❌ 加载vocab.json失败: {e}")
        raise


def extract_sentences_from_data(data, vocab_dict=None, min_length=4, max_length=30):
    sentences = []
    idx_to_token = vocab_dict.get('idx_to_token', {}) if vocab_dict else {}
    print(f"开始提取句子，输入数据类型: {type(data)}, 数据量: {len(data) if hasattr(data, '__len__') else '未知'}")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                sentences.append(item)
            elif isinstance(item, (list, tuple)) and all(isinstance(i, (int, float)) for i in item):
                sentence = ' '.join(idx_to_token.get(int(i), '') for i in item if int(i) in idx_to_token)
                if sentence.strip():
                    sentences.append(sentence)
            elif isinstance(item, dict):
                text_fields = ['text', 'sentence', 'source_text', 'content', 'input', 'source']
                for field in text_fields:
                    if field in item and isinstance(item[field], str):
                        sentences.append(item[field])
                        break
                else:
                    for value in item.values():
                        if isinstance(value, str) and len(value.strip()) > 0:
                            sentences.append(value)
                            break
                        elif isinstance(value, (list, tuple)) and all(isinstance(i, (int, float)) for i in value):
                            sentence = ' '.join(idx_to_token.get(int(i), '') for i in value if int(i) in idx_to_token)
                            if sentence.strip():
                                sentences.append(sentence)
    print(f"初步提取 {len(sentences)} 条句子")
    filtered_sentences = []
    token_to_idx = vocab_dict.get('token_to_idx', {}) if vocab_dict else {}
    for s in sentences:
        if not isinstance(s, str) or not s.strip():
            continue
        words = re.findall(r'\b\w+\b', s.lower())
        tokens = [token_to_idx.get(word, -1) for word in words if word in token_to_idx]
        if len(tokens) < min_length or len(tokens) > max_length:
            continue
        if token_to_idx and not any(t != -1 for t in tokens):
            continue
        filtered_sentences.append(s.strip())
    filtered_sentences = list(dict.fromkeys(filtered_sentences))
    print(f"过滤结果：总句子 {len(sentences)}，保留 {filtered_sentences}")
    return filtered_sentences


class CrosswordEvaluator:
    def __init__(self, model_path='bert-base-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path).to(device)
        self.model.eval()

    def compute_semantic_loss(self, original_sentences, recovered_sentences):
        if len(original_sentences) != len(recovered_sentences):
            raise ValueError("原始句子和恢复句子的数量不匹配")
        total_semantic_loss = 0.0
        sentence_losses = []
        for orig, recovered in zip(original_sentences, recovered_sentences):
            orig_embedding = self._get_bert_embedding(orig)
            recovered_embedding = self._get_bert_embedding(recovered)
            cosine_sim = np.dot(orig_embedding, recovered_embedding) / (
                    np.linalg.norm(orig_embedding) * np.linalg.norm(recovered_embedding) + 1e-10
            )
            semantic_loss = 1 - cosine_sim
            alpha_i = len(orig.split())
            weighted_loss = alpha_i * semantic_loss
            total_semantic_loss += weighted_loss
            sentence_losses.append(semantic_loss)
        average_semantic_loss = total_semantic_loss / sum(len(s.split()) for s in original_sentences)
        return {
            'total_semantic_loss': total_semantic_loss,
            'average_semantic_loss': average_semantic_loss,
            'sentence_losses': sentence_losses
        }

    def _get_bert_embedding(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[0, 0, :].cpu().numpy()

    def compute_compression_rate(self, original_text, compressed_text):
        original_bits = len(original_text.encode('utf-8')) * 8
        compressed_bits = len(compressed_text.encode('utf-8')) * 8
        original_words = len(original_text.split())
        if original_words == 0:
            return 0
        compression_rate = compressed_bits / original_words
        compression_ratio = compressed_bits / original_bits if original_bits > 0 else 1
        return {
            'bits_per_word': compression_rate,
            'compression_ratio': compression_ratio,
            'original_bits': original_bits,
            'compressed_bits': compressed_bits,
            'original_words': original_words
        }

    def evaluate_compression_performance(self, original_sentences, compressed_sentences, recovered_sentences):
        semantic_metrics = self.compute_semantic_loss(original_sentences, recovered_sentences)
        original_text = ' '.join(original_sentences)
        compressed_text = ' '.join(compressed_sentences)
        compression_metrics = self.compute_compression_rate(original_text, compressed_text)
        total_original_words = sum(len(s.split()) for s in original_sentences)
        total_compressed_words = sum(len(s.split()) for s in compressed_sentences)
        word_reduction_ratio = 1 - (total_compressed_words / total_original_words) if total_original_words > 0 else 0
        results = {
            'semantic_loss': semantic_metrics['average_semantic_loss'],
            'total_semantic_loss': semantic_metrics['total_semantic_loss'],
            'sentence_losses': semantic_metrics['sentence_losses'],
            'bits_per_word': compression_metrics['bits_per_word'],
            'compression_ratio': compression_metrics['compression_ratio'],
            'word_reduction_ratio': word_reduction_ratio,
            'original_words': total_original_words,
            'compressed_words': total_compressed_words,
            'num_sentences': len(original_sentences)
        }
        return results


class NoiseSimulator:
    def __init__(self, snr_db=DEFAULT_SNR, channel_type=DEFAULT_CHANNEL):
        self.snr_db = snr_db
        self.channel_type = channel_type
        self.snr_linear = 10 ** (snr_db / 10)
        self.noise_std = SNR_to_noise(snr_db)
        self.channels = Channels()

    def set_snr(self, snr_db):
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10)
        self.noise_std = SNR_to_noise(snr_db)

    def set_channel(self, channel_type):
        if channel_type in CHANNEL_TYPES:
            self.channel_type = channel_type
        else:
            raise ValueError(f"不支持的信道类型: {channel_type}")

    def add_channel_noise(self, signal):
        normalized_signal = PowerNormalize(signal)
        if self.channel_type == 'AWGN':
            return self.channels.AWGN(normalized_signal, self.noise_std)
        elif self.channel_type == 'Rayleigh':
            return self.channels.Rayleigh(normalized_signal, self.noise_std)
        else:
            return normalized_signal

    def add_token_noise(self, token_ids, tokenizer, noise_prob=None):
        if noise_prob is None:
            base_prob = 0.2 / self.snr_linear
            if self.channel_type == 'Rayleigh':
                base_prob *= 1.5
            noise_prob = max(0.01, min(0.3, base_prob))
        noisy_tokens = token_ids.clone()
        vocab_size = tokenizer.vocab_size
        for i in range(len(token_ids)):
            if random.random() < noise_prob:
                noise_type = random.choice(['replace', 'delete', 'insert'])
                if noise_type == 'replace':
                    noisy_tokens[i] = random.randint(0, vocab_size - 1)
        return noisy_tokens


class SentenceEmbedder:
    def __init__(self, model_path, device=device, snr_db=DEFAULT_SNR, channel_type=DEFAULT_CHANNEL):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path).to(device)
        self.device = device
        self.model.eval()
        self.noise_sim = NoiseSimulator(snr_db, channel_type)

    def encode(self, sentence, add_noise=False):
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        if add_noise:
            token_embeddings = self.noise_sim.add_channel_noise(token_embeddings)
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def set_snr(self, snr_db):
        self.noise_sim.set_snr(snr_db)

    def set_channel(self, channel_type):
        self.noise_sim.set_channel(channel_type)



class KeywordCompressor:
    def __init__(self, model_path, device=device, snr_db=DEFAULT_SNR, channel_type=DEFAULT_CHANNEL):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path, output_attentions=True).to(device)
        self.mlm = BertForMaskedLM.from_pretrained(model_path).to(device)
        self.device = device
        self.model.eval()
        self.mlm.eval()
        self.noise_sim = NoiseSimulator(snr_db, channel_type)
        self.embedder = SentenceEmbedder(model_path, device, snr_db, channel_type)
    def extract_keywords(self, sentence, top_k=5, add_noise=False):
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True).to(self.device)
        if add_noise:
            inputs['input_ids'] = self.noise_sim.add_token_noise(inputs['input_ids'][0], self.tokenizer).unsqueeze(0)
        outputs = self.model(**inputs)
        attentions = outputs.attentions[-1]
        if add_noise:
            attentions = self.noise_sim.add_channel_noise(attentions)
        cls_attn = attentions[0, :, 0, :].mean(dim=0)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        valid_tokens_scores = [
            (token, score.item()) for token, score in zip(tokens, cls_attn)
            if token not in ['[CLS]', '[SEP]', '[PAD]']
        ]
        sorted_tokens = sorted(valid_tokens_scores, key=lambda x: x[1], reverse=True)
        keywords = [token.replace("##", "") for token, _ in sorted_tokens[:top_k]]
        return list(dict.fromkeys(keywords))

    def compress_sentence(self, sentence, top_k=5, add_noise=False, compression_ratio=0.5):
        tokens = self.tokenizer.tokenize(sentence)
        if not tokens:
            return sentence
        original_length = len(tokens)
        target_length = max(2, int(original_length * compression_ratio))
        top_k = min(top_k, target_length, original_length)
        keywords = self.extract_keywords(sentence, top_k=top_k, add_noise=add_noise)
        if len(keywords) < 2:
            words = sentence.split()
            return " ".join(words[:min(3, len(words))])
        original_words = sentence.lower().split()
        ordered_keywords = []
        used_keywords = set()
        for word in original_words:
            for keyword in keywords:
                if (
                        keyword.lower() in word.lower() or word.lower() in keyword.lower()
                ) and keyword not in used_keywords:
                    ordered_keywords.append(keyword)
                    used_keywords.add(keyword)
                    break
        for keyword in keywords:
            if keyword not in used_keywords:
                ordered_keywords.append(keyword)
        return " ".join(ordered_keywords[:top_k])

    def compute_entropy(self, sentence, add_noise=False):
        if not sentence.strip():
            return 0.0
        embedding = self.embedder.encode(sentence, add_noise=add_noise)  # [1, 768]
        embedding = embedding.squeeze(0)  # [768]
        probs = F.softmax(embedding, dim=-1)  
     H = -∑ p_i * log_2(p_i)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
        return entropy

    def set_snr(self, snr_db):
        self.noise_sim.set_snr(snr_db)
        self.embedder.set_snr(snr_db)

    def set_channel(self, channel_type):
        self.noise_sim.set_channel(channel_type)
        self.embedder.set_channel(channel_type)


class AdvancedDecompressor:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForMaskedLM.from_pretrained(model_path).to(device)
        self.model.eval()
        self.function_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'when', 'where', 'how', 'why',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'up', 'about',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between'
        ]

    def recover_sentence(self, compressed_sentence, target_length=None):
        keywords = compressed_sentence.split()
        if not keywords:
            return compressed_sentence
        if len(keywords) >= 6:
            return self._light_expansion(keywords)
        return self._full_reconstruction(keywords, target_length)

    def _light_expansion(self, keywords):
        expanded = []
        for i, word in enumerate(keywords):
            expanded.append(word)
            if i < len(keywords) - 1:
                next_word = keywords[i + 1]
                if self._is_likely_noun(next_word) and word not in ['the', 'a', 'an']:
                    if random.random() < 0.3:
                        expanded.append('the')
                elif self._is_likely_verb(word) and random.random() < 0.2:
                    if next_word not in self.function_words:
                        expanded.append(random.choice(['to', 'in', 'for', 'with']))
        return ' '.join(expanded)

    def _full_reconstruction(self, keywords, target_length=None):
        if not target_length:
            target_length = len(keywords) * 2
        reconstructed = []
        if keywords and not keywords[0].lower() in ['the', 'a', 'an', 'this', 'that']:
            if self._is_likely_noun(keywords[0]):
                reconstructed.append('the')
        for i, keyword in enumerate(keywords):
            reconstructed.append(keyword)
            if i < len(keywords) - 1 and len(reconstructed) < target_length:
                next_keyword = keywords[i + 1]
                connector = self._choose_connector(keyword, next_keyword)
                if connector and len(reconstructed) + 1 < target_length:
                    reconstructed.append(connector)
        return self._post_process(reconstructed)

    def _is_likely_noun(self, word):
        noun_endings = ['tion', 'sion', 'ment', 'ness', 'ity', 'ty', 'er', 'or', 'ist']
        return any(word.lower().endswith(ending) for ending in noun_endings) or word.istitle()

    def _is_likely_verb(self, word):
        verb_endings = ['ed', 'ing', 'ize', 'ise', 'ate']
        verb_starts = ['re', 'un', 'pre', 'dis']
        return any(word.lower().endswith(ending) for ending in verb_endings) or any(
            word.lower().startswith(start) for start in verb_starts)

    def _choose_connector(self, word1, word2):
        connectors = {
            'noun_noun': ['and', 'of', 'in'],
            'verb_noun': ['the', 'a', 'to'],
            'noun_verb': ['is', 'was', 'can', 'will'],
            'default': ['and', 'the', 'to', 'in', 'of']
        }
        if self._is_likely_noun(word1) and self._is_likely_noun(word2):
            return random.choice(connectors['noun_noun'])
        elif self._is_likely_verb(word1) and self._is_likely_noun(word2):
            return random.choice(connectors['verb_noun'])
        elif self._is_likely_noun(word1) and self._is_likely_verb(word2):
            return random.choice(connectors['noun_verb'])
        else:
            return random.choice(connectors['default']) if random.random() < 0.4 else None

    def _post_process(self, words):
        if not words:
            return ""
        sentence = ' '.join(words)
        if sentence:
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
        words = sentence.split()
        cleaned_words = []
        prev_word = None
        for word in words:
            if word.lower() != prev_word or word.lower() not in self.function_words:
                cleaned_words.append(word)
            prev_word = word.lower()
        return ' '.join(cleaned_words)


def evaluate_crossword_metrics(original_sentences, compressed_sentences, recovered_sentences, evaluator):
       results = []
    for orig, comp, rec in zip(original_sentences, compressed_sentences, recovered_sentences):
        result = evaluator.evaluate_compression_performance([orig], [comp], [rec])
        results.append(result)
    if results:
        avg_semantic_loss = np.mean([r['semantic_loss'] for r in results])
        avg_bits_per_word = np.mean([r['bits_per_word'] for r in results])
        avg_compression_ratio = np.mean([r['compression_ratio'] for r in results])
        avg_word_reduction = np.mean([r['word_reduction_ratio'] for r in results])
        total_original_words = sum(r['original_words'] for r in results)
        total_compressed_words = sum(r['compressed_words'] for r in results)
        summary = {
            'average_semantic_loss': avg_semantic_loss,
            'average_bits_per_word': avg_bits_per_word,
            'average_compression_ratio': avg_compression_ratio,
            'average_word_reduction_ratio': avg_word_reduction,
            'total_original_words': total_original_words,
            'total_compressed_words': total_compressed_words,
            'num_sentences': len(results),
            'individual_results': results
        }
        return summary
    return None



def evaluate_multiple_conditions_crossword(test_sentences, embedder, compressor, decompressor, snr_levels,
                                           channel_types, compression_ratio, top_k):
    results = {}
    evaluator = CrosswordEvaluator()
    for channel_type in channel_types:
        results[channel_type] = {}
        print(f"\n{'=' * 60}")
        print(f"📡 评估信道类型: {channel_type}")
        print(f"{'=' * 60}")
        embedder.set_channel(channel_type)
        compressor.set_channel(channel_type)
        for snr_db in snr_levels:
            print(f"\n{'=' * 50}")
            print(f"🔊 评估 {channel_type} 信道, SNR = {snr_db}dB")
            print(f"{'=' * 50}")
            embedder.set_snr(snr_db)
            compressor.set_snr(snr_db)
            add_noise = (snr_db < 100)
            original_sentences = []
            compressed_sentences = []
            recovered_sentences = []
            for sent in tqdm(test_sentences, desc=f"评估 {channel_type} SNR={snr_db}dB"):
                if len(compressor.tokenizer.tokenize(sent)) > MAX_TOKEN_LENGTH:
                    continue
                try:
                    compressed = compressor.compress_sentence(sent, top_k=top_k, add_noise=add_noise,
                                                              compression_ratio=compression_ratio)
                    target_length = len(sent.split())
                    recovered = decompressor.recover_sentence(compressed, target_length)
                    original_sentences.append(sent)
                    compressed_sentences.append(compressed)
                    recovered_sentences.append(recovered)
                except Exception as e:
                    print(f"处理句子时出错: {e}")
                    continue
            if original_sentences:
                crossword_results = evaluate_crossword_metrics(original_sentences, compressed_sentences,
                                                               recovered_sentences, evaluator)
                if crossword_results:
                    performance_score = print_crossword_results(crossword_results, snr_db, channel_type)
                    results[channel_type][snr_db] = {
                        'crossword_metrics': crossword_results,
                        'performance_score': performance_score
                    }
                else:
                    print(f"⚠️ {channel_type} 信道 SNR={snr_db}dB 下评估失败")
            else:
                print(f"⚠️ {channel_type} 信道 SNR={snr_db}dB 下没有有效的评估样本")
    return results



def is_compression_bad_crossword(semantic_loss, compression_ratio, word_reduction_ratio, entropy, semantic_thresh=0.2,
                                 compression_thresh=0.8, reduction_thresh=0.1, entropy_thresh=6.0, log_count=[0]):
       # if not all(isinstance(x, (int, float)) for x in [semantic_loss, compression_ratio, word_reduction_ratio, entropy]):
    #     print(f"⚠️ 无效参数: 语义损失={semantic_loss}, 压缩比={compression_ratio}, 词汇减少率={word_reduction_ratio}, 语义熵={entropy}")
    #     return False
    # if entropy == 0.0:
    #     print(f"⚠️ 语义熵为0，可能是空句子，标记为失败样本")
    #     return True
    # if log_count[0] < 100:  # 仅打印前 100 次
    #     print(
    #         f"📏 评估失败样本: 语义损失={semantic_loss:.4f}, 压缩比={compression_ratio:.4f}, 词汇减少率={word_reduction_ratio:.4f}, 语义熵={entropy:.4f}")
    #     log_count[0] += 1
    return (
            semantic_loss > semantic_thresh or
            compression_ratio > compression_thresh or
            word_reduction_ratio < reduction_thresh or
            entropy > entropy_thresh
    )


def compute_reward(evaluator, compressor, original_sentence, compressed_sentence, recovered_sentence, top_k,
                   compression_ratio):
      result = evaluator.evaluate_compression_performance([original_sentence], [compressed_sentence],
                                                        [recovered_sentence])
    semantic_loss = result['semantic_loss']
    bits_per_word = result['bits_per_word']
    word_reduction_ratio = result['word_reduction_ratio']

    reward = (1 - semantic_loss) * 0.7 + word_reduction_ratio * 0.3
    entropy = compressor.compute_entropy(compressed_sentence)
    if is_compression_bad_crossword(semantic_loss, result['compression_ratio'], word_reduction_ratio, entropy):
        reward -= 0.5
    return reward


def generate_compressed_sentence(model, tokenizer, compressor, decompressor, sentence, top_k, compression_ratio,
                                 device):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    input_ids = inputs['input_ids']

      mask_prob = 0.15  # Same as MLM probability
    mask_indices = torch.rand(input_ids.shape, device=device) < mask_prob
    mask_indices[:, 0] = False  # Don't mask [CLS]
    mask_indices[:, -1] = False  # Don't mask [SEP]
    masked_input_ids = input_ids.clone()
    masked_input_ids[mask_indices] = tokenizer.mask_token_id

    outputs = model(masked_input_ids, attention_mask=inputs['attention_mask'])
    logits = outputs.logits

    probs = F.softmax(logits, dim=-1)
    sampled_ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size()[:-1])
    sampled_input_ids = masked_input_ids.clone()
    sampled_input_ids[mask_indices] = sampled_ids[mask_indices]

    compressed_tokens = tokenizer.convert_ids_to_tokens(sampled_input_ids[0], skip_special_tokens=True)
    compressed_sentence = compressor.compress_sentence(
        tokenizer.decode(sampled_input_ids[0], skip_special_tokens=True),
        top_k=top_k,
        compression_ratio=compression_ratio
    )
    recovered_sentence = decompressor.recover_sentence(compressed_sentence, len(sentence.split()))

    log_probs = F.log_softmax(logits, dim=-1)
    log_prob_action = torch.sum(
        log_probs[mask_indices] * F.one_hot(sampled_ids[mask_indices], num_classes=tokenizer.vocab_size).float())

    return compressed_sentence, recovered_sentence, log_prob_action


def fine_tune_bert(samples, model, tokenizer, compressor, learning_rate=2e-4, mlm_probability=0.15,
                   num_train_epochs=20):
    if not samples:
        print("⚠️ 没有样本用于微调")
        return
    print(f"\n🔧 使用 {len(samples)} 个失败样本进行强化学习微调，训练轮数={num_train_epochs}")

    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        decompressor = AdvancedDecompressor(model_path="bert-base-cased")
        evaluator = CrosswordEvaluator()

        train_texts = [sample["input"] for sample in samples]
        train_targets = [sample["target"] for sample in samples]

        model.train()
        for epoch in range(num_train_epochs):
            total_loss = 0.0
            for sentence, target in tqdm(zip(train_texts, train_targets), total=len(train_texts),
                                         desc=f"Epoch {epoch + 1}"):
                try:
                    # Generate action (compressed sentence) and get log probability
                    compressed, recovered, log_prob_action = generate_compressed_sentence(
                        model, tokenizer, compressor, decompressor, target, top_k=5, compression_ratio=0.5,
                        device=device
                    )

                    # Compute reward
                    reward = compute_reward(evaluator, compressor, target, compressed, recovered, top_k=5,
                                            compression_ratio=0.5)

                    # Compute policy gradient loss (negative for gradient descent)
                    loss = -log_prob_action * reward
                    total_loss += loss.item()

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    # Apply gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                except Exception as e:
                    print(f"处理样本时出错: {e}")
                    continue

            print(f"Epoch {epoch + 1}/{num_train_epochs}, Average Loss: {total_loss / len(train_texts):.4f}")

        print("✅ 强化学习微调完成")
        model.eval()

    except Exception as e:
        print(f"❌ 强化学习微调失败: {e}")


def process_and_evaluate_crossword(model_path, data_dir, data_ratio=DATA_RATIO, evaluate_snr=True, channel_type='both',
                                   compression_ratio=0.5, top_k=5, learning_rate=2e-4, mlm_probability=0.15,
                                   num_train_epochs=20):
       try:
        embedder = SentenceEmbedder(model_path, device, DEFAULT_SNR,
                                    channel_type if channel_type != 'both' else DEFAULT_CHANNEL)
        compressor = KeywordCompressor(model_path, device, DEFAULT_SNR,
                                       channel_type if channel_type != 'both' else DEFAULT_CHANNEL)
        decompressor = AdvancedDecompressor(model_path)
    except Exception as e:
        print(f"❌ 初始化模型失败: {e}")
        return None
    try:
        train_sentences, test_sentences = load_and_combine_datasets(data_dir)
        if not train_sentences and not test_sentences:
            print("❌ 无法提取有效句子")
            return None
              if data_ratio < 1.0:
            train_sentences = train_sentences[:int(len(train_sentences) * data_ratio)]
            test_sentences = test_sentences[:int(len(test_sentences) * data_ratio)]
            print(f"  应用数据比例 {data_ratio * 100}% 后 - 训练: {len(train_sentences)}, 测试: {len(test_sentences)}")
    except Exception as e:
        print(f"❌ 处理数据集失败: {e}")
        return None
    if test_sentences:
        evaluator = CrosswordEvaluator()
        demonstrate_compression_example(compressor, decompressor, evaluator, test_sentences, top_k, compression_ratio)
    print("\n🔧 处理训练样本以识别需要微调的样本...")
    training_samples = []
    evaluator = CrosswordEvaluator()
    entropies = []
    for sent in tqdm(train_sentences, desc="处理训练样本"):
        if len(compressor.tokenizer.tokenize(sent)) > MAX_TOKEN_LENGTH:
            continue
        try:
            compressed = compressor.compress_sentence(sent, top_k=top_k, compression_ratio=compression_ratio)
            if not compressed.strip():
                print(f"⚠️ 压缩句子为空: {sent}")
                training_samples.append({"input": compressed, "target": sent})
                continue
            recovered = decompressor.recover_sentence(compressed, len(sent.split()))
            result = evaluator.evaluate_compression_performance([sent], [compressed], [recovered])
            entropy = compressor.compute_entropy(compressed)
            entropy_thresh = 5.0 + 0.2 * len(compressor.tokenizer.tokenize(sent))  # 动态阈值适应语义熵
            if is_compression_bad_crossword(
                    result['semantic_loss'],
                    result['compression_ratio'],
                    result['word_reduction_ratio'],
                    entropy,
                    entropy_thresh=entropy_thresh
            ):
                training_samples.append({"input": compressed, "target": sent})
        except Exception as e:
            print(f"处理训练样本时出错: {e}")
            continue
    if entropies:
        print(
            f"语义熵分布统计: 平均={np.mean(entropies):.4f}, 标准差={np.std(entropies):.4f}, 最大={max(entropies):.4f}, 最小={min(entropies):.4f}")
    print(f"识别到 {len(training_samples)} 个失败样本用于微调")
    if training_samples:
        fine_tune_bert(training_samples, compressor.mlm, compressor.tokenizer, compressor, learning_rate,
                       mlm_probability,
                       num_train_epochs)
    else:
        print("没有失败样本用于微调")
    print("\n微调完成，开始评估测试集:")
    if evaluate_snr:
        if channel_type == 'both':
            channels_to_test = CHANNEL_TYPES
        else:
            channels_to_test = [channel_type]
        crossword_results = evaluate_multiple_conditions_crossword(
            test_sentences, embedder, compressor, decompressor, SNR_LEVELS, channels_to_test, compression_ratio, top_k
        )
        print_crossword_performance_summary(crossword_results)
        return crossword_results
    else:
        original_sentences = []
        compressed_sentences = []
        recovered_sentences = []
        for sent in tqdm(test_sentences, desc="评估测试集"):
            if len(compressor.tokenizer.tokenize(sent)) > MAX_TOKEN_LENGTH:
                continue
            try:
                compressed = compressor.compress_sentence(sent, top_k=top_k, compression_ratio=compression_ratio)
                recovered = decompressor.recover_sentence(compressed, len(sent.split()))
                original_sentences.append(sent)
                compressed_sentences.append(compressed)
                recovered_sentences.append(recovered)
            except Exception as e:
                print(f"评估测试样本时出错: {e}")
                continue
        if original_sentences:
            crossword_results = evaluate_crossword_metrics(original_sentences, compressed_sentences,
                                                           recovered_sentences, evaluator)
            if crossword_results:
                performance_score = print_crossword_results(crossword_results, DEFAULT_SNR, channel_type)
                return {
                    'crossword_metrics': crossword_results,
                    'performance_score': performance_score
                }
    return None



def main():
    model_path = "bert-base-cased"
    data_dir = "/home/data/"
    
    # 加载数据集
    try:
        train_sentences, test_sentences = load_and_combine_datasets(data_dir)
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return

    # 超参数优化
    print("\n🔍 开始超参数优化...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, model_path, data_dir, train_sentences, test_sentences),
        n_trials=20,  # 运行 20 次试验
        n_jobs=1,  # 单线程运行以避免 GPU 冲突
    )

    # 打印最佳超参数
    print(f"\n🎉 超参数优化完成！")
    print(f"最佳超参数: {study.best_params}")
    print(f"最佳性能分数: {study.best_value:.4f}")

    # 保存最佳超参数
    with open("best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f, indent=4)

    # 使用最佳超参数重新运行评估
    print("\n🚀 使用最佳超参数进行最终评估...")
    results = process_and_evaluate_crossword(
        model_path=model_path,
        data_dir=data_dir,
        data_ratio=DATA_RATIO,
        evaluate_snr=True,
        channel_type='both',
        compression_ratio=study.best_params['compression_ratio'],
        top_k=study.best_params['top_k'],
        learning_rate=study.best_params['learning_rate'],
        mlm_probability=study.best_params['mlm_probability'],
        num_train_epochs=study.best_params['num_train_epochs']
    )

    if results:
            for snr_db in SNR_LEVELS:
            print(f"\nSNR = {snr_db}dB:")
            for channel_type in CHANNEL_TYPES:
                if (channel_type in results and
                        snr_db in results[channel_type] and
                        'performance_score' in results[channel_type][snr_db]):
                    score = results[channel_type][snr_db]['performance_score']
                    metrics = results[channel_type][snr_db]['crossword_metrics']
                    print(f"  {channel_type:<12}: 性能分数 = {score:.4f}, "
                          f"语义损失 = {metrics['average_semantic_loss']:.4f}, "
                          f"压缩率 = {metrics['average_bits_per_word']:.2f} bits/word")
          else:
        print("❌ 评估失败，没有生成结果")


if __name__ == "__main__":

    main()
