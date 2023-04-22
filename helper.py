from langchain.chat_models import ChatOpenAI
from llama_index import PromptHelper
from llama_index import LLMPredictor, ServiceContext
from llama_index import GPTSimpleVectorIndex, download_loader
import os
from bs4 import BeautifulSoup
import requests


def get_docs_links(url, doc_name):
    docs_links = []
    visited_links = set()

    visit_links([url], visited_links, docs_links, doc_name)
    print('Fetched pages:', docs_links)
    return docs_links


def visit_links(links, visited_links, docs_links, doc_name):
    if not links:
        return
    link = links.pop()

    if link in visited_links:
        return
    try:
        response = requests.get(link)
    except:
        return

    visited_links.add(link)

    if not ('text/html' in response.headers['Content-Type']):
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    page_links = set()
    for a in soup.find_all('a', href=True):
        link = a['href']
        if not link.startswith('http'):
            link = f"{''.join(response.url.split('/')[:-1])+'/' if 'html' in response.url else response.url}{link}"
        page_links.add(link)

    for link in page_links:
        if 'docs' in link and '#' not in link and doc_name in link:
            docs_links.append(link)
    visit_links(list(page_links - visited_links), visited_links, docs_links, doc_name)


def construct_index(root_url, doc_name):
    if not os.path.exists(f'{doc_name}.json'):
        print(f'Constructing index for {root_url}...')
        # set maximum input size
        max_input_size = 4096
        # set number of output tokens
        num_outputs = 1024
        # set maximum chunk overlap
        max_chunk_overlap = 20
        # set chunk size limit
        chunk_size_limit = 600

        prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

        # define LLM
        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

        BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")

        loader = BeautifulSoupWebReader()
        documents = loader.load_data(urls=get_docs_links(root_url, doc_name))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

        index.save_to_disk(f'{doc_name}.json')
