import os
from dotenv import load_dotenv,find_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
import asyncio
from contextlib import asynccontextmanager
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
    


class CrawlerManager:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(1)  # Allow only one crawler at a time
        self._lock = asyncio.Lock()
        self.active_crawler = None

    @asynccontextmanager
    async def get_crawler(self, verbose=True):
        async with self.semaphore:
            async with AsyncWebCrawler(verbose=verbose) as crawler:
                async with self._lock:
                    self.active_crawler = crawler
                try:
                    yield crawler
                finally:
                    async with self._lock:
                        self.active_crawler = None

# Create a global instance of the manager
crawler_manager = CrawlerManager()

async def tool_urls_crawler(urls):
    """Herramienta para recopilar el contenido de los resultados de la búsqueda en internet.
    Args:
        urls: Lista de urls obtenida en el resultado de la busqueda
    """
    try:
        async with crawler_manager.get_crawler(verbose=True) as crawler:
            # Set up crawling parameters
            word_count_threshold = 100

            # Run the crawling process for multiple URLs
            results = await crawler.arun_many(
                urls=urls,
                word_count_threshold=word_count_threshold,
                bypass_cache=True,
                verbose=True,
            )

            # Process the results
            md_results = ""
            for result in results:
                if result.success:
                    print(f"Successfully crawled: {result.url}")
                    print(f"Title: {result.metadata.get('title', 'N/A')}")
                    print(f"Word count: {len(result.markdown.split())}")
                    print(
                        f"Number of links: {len(result.links.get('internal', [])) + len(result.links.get('external', []))}"
                    )
                    print(f"Number of images: {len(result.media.get('images', []))}")
                    print("---")
                else:
                    print(f"Failed to crawl: {result.url}")
                    print(f"Error: {result.error_message}")
                    print("---")
                title = f"\n\nTitle: {result.metadata.get('title', 'N/A')}"
                content = f"\nContent: {result.markdown}"
                md_results += title
                md_results += content

            return md_results
    except Exception as e:
        print(f"Error during crawling: {str(e)}")
        return f"Error during crawling: {str(e)}"

def tool_sync_url_crawler(urls):
    """Herramienta para recopilar el contenido de los resultados de la búsqueda en internet.
    Args:
        urls: Lista de urls obtenida en el resultado de la busqueda
    """
    try:
        return asyncio.run(tool_urls_crawler(urls=urls))
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(tool_urls_crawler(urls=urls))
        raise