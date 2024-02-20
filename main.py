from func import *
from args import *
import openai

def text_generaton_with_black_box_LLMs(prompt,tem):
        res=openai.ChatCompletion.create(model='gpt-4',
                                messages=[
                                {'role': 'user', 'content': prompt}],
                                max_tokens=150,
                                temperature=tem,
                                        )
        return res

#===============Configure network proxy===============
#os.environ["http_proxy"] = "http://127.0.0.1:10809"
#os.environ["https_proxy"] = "http://127.0.0.1:10809"


#==================Api_key of openai==================
openai.api_key = "You need to put your api_key here."
parser = get_parser()
args = parser.parse_args()


#================Load token embeddings================
with open("./data/cl100_embeddings.json", 'r') as f:
        cl100_emb=json.load(f)
        vector_data_json = {k: cl100_emb[k] for k in list(cl100_emb.keys())[:11000]}
        cl100_emb=None
        token_to_vector_dict = {token: np.array(vector) for token, vector in vector_data_json.items()}
if not os.path.exists(f'./data/sorted_cl100_embeddings.json'):
        init_func(1.0,token_to_vector_dict)
with open('./data/sorted_cl100_embeddings.json', 'r') as f1:
        sorted_cl100_emb = json.load(f1)
with open('./data/sensitivity_of_embeddings.json', 'r') as f:
        sen_emb = np.array(json.load(f))


#=============Input your document to perturb=============
raw_document='You need to put your private raw text here'

raw_tokens = get_first_50_tokens(raw_document)
perturbed_tokens=perturb_sentence(raw_tokens,args.eps,args.model,token_to_vector_dict,sorted_cl100_emb,sen_emb)
print(perturbed_tokens)
prompt="""Your task is to extend Prefix Text. Provide only your Extension. 
- Prefix Text:"""+perturbed_tokens+"""
\n- Extension:"""

response = text_generaton_with_black_box_LLMs(prompt,0.5)
response_text = get_first_100_tokens(response['choices'][0]['message']['content'])
print(response_text)





#python main.py --eps 6.0 --model gpt-4

