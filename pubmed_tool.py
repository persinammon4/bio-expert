import requests
from bs4 import BeautifulSoup

class PubMedTool:
    name = "pubmed_search"
    description = "Use this to search PubMed for relevant biology papers. Input: a search query. Output: a list of article titles with URLs."

    @staticmethod
    def run(query: str) -> str:
        base_url = "https://pubmed.ncbi.nlm.nih.gov/"
        params = {"term": query}
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            return "Failed to fetch PubMed results."

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("article", class_="full-docsum")
        results = []
        for a in articles[:5]:  # return top 5 results
            title_tag = a.find("a", class_="docsum-title")
            if title_tag:
                title = title_tag.get_text(strip=True)
                link = base_url.rstrip("/") + title_tag['href']
                results.append(f"{title} - {link}")
        return "\n".join(results) if results else "No results found."
