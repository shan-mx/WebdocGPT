from helper import *

os.environ["OPENAI_API_KEY"] = ''


def ask_bot(doc_name):
    index = GPTSimpleVectorIndex.load_from_disk(f'{doc_name}.json')
    while True:
        query = input('What do you want to ask the bot?   \n')
        response = index.query(query, response_mode="compact")
        print("\nBot says: \n\n" + response.response + "\n\n\n")


root_url = "https://gpt-index.readthedocs.io/en/latest/"
doc_name = "gpt-index"
construct_index(root_url, doc_name)
ask_bot(doc_name)
