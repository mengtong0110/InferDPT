import json
import string
import tiktoken
import random
from sklearn.metrics.pairwise import euclidean_distances
import tqdm
from decimal import getcontext
import numpy as np
import json
from transformers import GPT2Tokenizer
import os 

getcontext().prec = 100
#os.environ["http_proxy"] = "http://127.0.0.1:10809"
#os.environ["https_proxy"] = "http://127.0.0.1:10809"


def get_first_50_tokens(text):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.tokenize(text)
    first_50_tokens = tokens[:50]
    tokenized_string = tokenizer.convert_tokens_to_string(first_50_tokens)
    return tokenized_string

def get_first_100_tokens(text):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.tokenize(text)
    first_100_tokens = tokens[:100]
    tokenized_string = tokenizer.convert_tokens_to_string(first_100_tokens)
    return tokenized_string

def calculate_distance(i, j, vector_matrix, pb):
    distance = euclidean_distances(vector_matrix[i].reshape(1, -1).astype(np.longdouble), 
                                   vector_matrix[j].reshape(1, -1).astype(np.longdouble))
    pb.update(1)
    return i, j, distance[0, 0]

punctuation_string = string.punctuation
punctuation_list = list(punctuation_string)

def generate_tasks(n_vectors):
    for i in range(n_vectors):
        for j in range(i + 1, n_vectors):
            yield (i, j)
def add_laplace_noise_to_vector(vector, epsilon, delta_f_new=None):
    vector = np.asarray(vector, dtype=np.longdouble)
    if not os.path.exists(f'./data/sorted_cl100_embeddings.json'):
        with open("./data/cl100_embeddings.json",'r') as f:
            data_t=json.load(f)
            data = {k: data_t[k] for k in list(data_t.keys())}
            data_t=None
        word_list = list(data.keys())
        vector_matrix = np.array(list(data.values()))
        data=None
        n_vectors = len(word_list)
        distance_matrix = np.zeros((n_vectors, n_vectors))
        total_tasks = (n_vectors * (n_vectors - 1)) // 2
        results = [None] * total_tasks
        if not os.path.exists(f'./data/temp_distance_json_path.json'):
            with tqdm.tqdm(total=int(n_vectors * (n_vectors - 1) / 2)) as pb:
                pb.set_description('Inference process')
                tasks = list(generate_tasks(n_vectors))
                for index, task in enumerate(tasks):
                    try:
                        results[index] = calculate_distance(task[0], task[1], vector_matrix, pb)
                    except Exception as e:
                        print(f"Task at index {index} failed with exception {e}")
                for i, j, distance in results:
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance  
            temp_distance_matrix =distance_matrix
            temp_distance_dict_matrix = {}
            for i, word1 in enumerate(word_list):
                for j, word2 in enumerate(word_list):
                    pair = tuple(sorted([word1, word2]))
                    if pair in temp_distance_dict_matrix:
                        continue
                    temp_distance_dict_matrix[str(pair)] = float(temp_distance_matrix[i, j])
            with open('./data/temp_distance_json_path.json', 'w') as f:
                json.dump(temp_distance_dict_matrix, f)
        if os.path.exists(f'./data/temp_distance_json_path.json'):
            with open('./data/temp_distance_json_path.json', 'r') as f:
                temp_distance_dict_matrix = json.load(f)
            word_to_index = {}
            with tqdm.tqdm(total=len(word_list)) as pbwi:
                pbwi.set_description('word_to_index process')
                for idx, word in enumerate(word_list):
                    word_to_index[word] = idx
                    pbwi.update(1)
            n = len(word_list)
            temp_distance_matrix = np.zeros((n, n))
            with tqdm.tqdm(total=len(temp_distance_dict_matrix)) as pbm:
                pbm.set_description('')
                for key, value in temp_distance_dict_matrix.items():
                    word1, word2 = tuple(key.strip("()").split(", "))
                    i = word_to_index[word1.strip("'")]
                    j = word_to_index[word2.strip("'")]
                    temp_distance_matrix[i, j] = value
                    temp_distance_matrix[j, i] = value  
                    pbm.update(1)
            sorted_distance_dict_matrix = {}
            with tqdm.tqdm(total=n) as pbm:
                pbm.set_description('Sorted process')
                for i, word in enumerate(word_list):
                    sorted_indices = np.argsort(temp_distance_matrix[i])
                    sorted_words = [(word_list[j], temp_distance_matrix[i, j]) for j in sorted_indices]
                    sorted_distance_dict_matrix[word] = sorted_words
                    pbm.update(1)

        with open('./data/sorted_cl100_embeddings.json', 'w') as f:
            json.dump(sorted_distance_dict_matrix, f)
    if not os.path.exists(f'./data/sensitivity_of_embeddings.json'):
        json_path = "./data/cl100_embeddings.json"
        with open(json_path, 'r') as f:
            vector_data_json = json.load(f)
        word_list = list(vector_data_json.keys())
        vector_matrix = np.array(list(vector_data_json.values()))
        n_dimensions = vector_matrix.shape[1]
        delta_f_new = np.zeros(n_dimensions)
        for dim in tqdm.trange(n_dimensions):
            dim_data = vector_matrix[:, dim]
            sorted_dim_data = np.sort(dim_data)
            differences =sorted_dim_data[-1]-sorted_dim_data[0]
            delta_f_new[dim] = differences   
        delta_f_new_json_path = './data/sensitivity_of_embeddings.json'
        with open(delta_f_new_json_path, 'w') as f:
            json.dump(delta_f_new.tolist(), f)
    else:
        if delta_f_new is None:
            with open('./data/sensitivity_of_embeddings.json', 'r') as f:
                delta_f_new = np.array(json.load(f))
    tt=0
    if (epsilon*19.064721649556482-38.1294334077209)>0:
        tt=0.01658160142016071*np.log(epsilon*19.064721649556482-38.1294334077209)+9.311083811697406
    if epsilon <2:
        beta_values = delta_f_new/epsilon
    else:
        beta_values = delta_f_new/tt
    beta_values = beta_values.astype(np.longdouble)  
    noisy_vector = np.zeros_like(vector, dtype=np.longdouble)  
    for dim in range(len(vector)):
        noise = np.random.laplace(0, beta_values[dim])
        noisy_vector[dim] = vector[dim] + noise
    return noisy_vector.astype(float) 


def perturb_sentence(sent, epsilon, model, token_to_vector_dict,sorted_distance_data,delta_f_new):
    enc = tiktoken.encoding_for_model(model)
    tokens_b=enc.encode(sent)
    tokens=[(enc.decode_single_token_bytes(t)).decode('Latin-1') for t in tokens_b]
    new_tokens=[]
    Delta_u = 1.0  
    exp_factor = epsilon / (2 * Delta_u)
    for origin_token in tokens:
        if(origin_token.isnumeric()):
            new_tokens.append(str(random.randint(1, 1000)))
            continue
        if(origin_token[0]==' '):
            origin_token=origin_token[1:]
        origin_embed = token_to_vector_dict.get(origin_token, None)
        if origin_embed is None:
            continue
        noise_embed = add_laplace_noise_to_vector(origin_embed, epsilon,delta_f_new)
        distance = np.linalg.norm(origin_embed - noise_embed)
        sorted_distances_for_token = sorted_distance_data.get(origin_token, None)
        if sorted_distances_for_token is None:
            continue
        distances_only = np.array([item[1] for item in sorted_distances_for_token])
        index = np.searchsorted(distances_only, distance)
        close_tokens = [item[0] for item in sorted_distances_for_token[:index] ]
        close_distances = np.array([item[1] for item in sorted_distances_for_token[:index]])
        if not close_tokens:
            continue
        unnormalized_probabilities = np.exp(exp_factor * ((distance-close_distances)/distance))
        total_unnormalized_prob = np.sum(unnormalized_probabilities)
        probabilities = unnormalized_probabilities / total_unnormalized_prob
        selected_token = np.random.choice(close_tokens, p=probabilities)
        new_tokens.append(selected_token)      
    sanitized_sent = ' '.join(new_tokens)
    return sanitized_sent

def init_func(epsilon,token_to_vector_dict):  
    origin_embed = token_to_vector_dict.get('he', None)
    add_laplace_noise_to_vector(origin_embed,epsilon)
   