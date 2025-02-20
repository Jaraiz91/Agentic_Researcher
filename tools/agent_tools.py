import os
from dotenv import load_dotenv,find_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
import asyncio
from crawl4ai import AsyncWebCrawler

_ = load_dotenv(find_dotenv())
tavily_api_key= os.environ['TAVILY_API_KEY']

def tool_read_document(path):
    """Herramienta para poder leer archivos. En este caso se trata del estudio que se está creando.\
    Args: 
        path: ruta al archivo para leer"""
    try:
        with open(path, 'r') as archivo:  # Abre el archivo en modo lectura
            contenido = archivo.read()  # Lee todo el contenido del archivo
            return contenido  # Devuelve el contenido leído
    except FileNotFoundError:
        return "El archivo no fue encontrado."
    except Exception as e:
        return f"Ocurrió un error: {e}"
    
def tool_write_document(content, path):
    """Herramienta para poder escribir el contentido en un archivo. Si el archivo ya existe se pretende sobreescribirlo entero.
    Args:
        content: El contenido completo a escribir en el archivo
        path: ruta del archivo"""
    try:
        with open(path, 'w') as archivo:  # Abre el archivo en modo escritura
            archivo.write(content)  # Escribe el contenido en el archivo
            return "Contenido escrito exitosamente."
    except Exception as e:
        return f"Ocurrió un error: {e}"


def tool_search_results(query):
    """Herramienta para que el agente realice busquedas en internet y recopilar información. 
    args:
        query: pregunta para realziar la busqueda en internet"""
    searcher = TavilySearchResults(max_results=5)
    results = searcher.invoke({'query': query})
    urls = [x['url'] for x in results]
    return urls


async def simple_crawl(url):
    beginning = "\n\n source: {url}\n"
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
        )
        return beginning + result.markdown
    
def tool_get_search_info(urls):
    """Herramienta para recopilar el contenido de los resultados de la búsqueda en internet.
        Args:
            urls: Lista de urls obtenida en el resultado de la busqueda"""
    document  = ""
    for u in urls:
        asyncio.run(simple_crawl(u))
    return document
    
