from langchain.chat_models import ChatOpenAI
from llama_index import PromptHelper
from llama_index import LLMPredictor, ServiceContext
from llama_index import GPTSimpleVectorIndex, download_loader
import os
from bs4 import BeautifulSoup
import requests

from urllib.parse import urlparse, urlunparse, urljoin


def get_docs_links(root_url, doc_name):
    docs_links = set()
    visited_links = set()
    visit_links([root_url], visited_links, docs_links, doc_name, root_url)
    print('Fetched pages:', docs_links)
    print('Total num:', len(list(docs_links)))
    return list(docs_links)


def visit_links(links, visited_links, docs_links, doc_name, root_url):
    if not links:
        return
    link = links.pop()
    try:
        response = requests.get(link)
    except:
        return
    visited_links.add(link)
    if not ('text/html' in response.headers['Content-Type']):
        return
    soup = BeautifulSoup(response.content, 'html.parser')
    page_links = set()
    for a in soup.find_all('a', href=True, class_='reference internal'):
        link = a['href']
        if not link.startswith('http'):
            if 'html' in response.url:
                link = '/'.join(response.url.split('/')[:-1]) + '/' + link
            else:
                link = root_url + '/' + link
        if '..' in link:
            parsed_url = urlparse(link)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            abs_path = urljoin(base_url, parsed_url.path)
            link = urlunparse(['', '', abs_path, parsed_url.params, parsed_url.query,
                               parsed_url.fragment])
        if link not in visited_links:
            if doc_name in link:
                if '#' in link:
                    link = link.split('#')[0]
                page_links.add(link)
    for link in page_links:
        docs_links.add(link)
    visit_links(list(page_links - visited_links), visited_links, docs_links, doc_name, root_url)


def construct_index(root_url, doc_name):
    if not os.path.exists(f'{doc_name}.json'):
        print(f'Fetching pages in {root_url}')
        max_input_size = 4096
        num_outputs = 1024
        max_chunk_overlap = 20
        chunk_size_limit = 600
        prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
        loader = download_loader("BeautifulSoupWebReader")()
        urls = get_docs_links(root_url, doc_name)
        print(f'Constructing index for {root_url}...')
        documents = loader.load_data(urls=urls)
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
        index.save_to_disk(f'{doc_name}.json')
    else:
        print("Index already exists.")
