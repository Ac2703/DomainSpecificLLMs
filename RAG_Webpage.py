import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re


class WebScrape: 
    def __init__(self):
        self.webpage_url = "https://tinman.cs.gsu.edu/~raj/past2.html"
        self.keyword = input("Enter desired link keyword: ")
        # Get the main page
        response = requests.get(self.webpage_url, verify=False)
        self.soup = BeautifulSoup(response.text, "html.parser")

    def examine_link(self):
    # Go through each link on the main page
        for link in self.soup.find_all("a", href=True):
            first_url = urljoin(self.webpage_url, link["href"])
            print(f"\nVisiting: {first_url}")
            
            try:
                # Visit the first link
                page = requests.get(first_url)
                linked_soup = BeautifulSoup(page.text, "html.parser")

                # Search for keyword-matching links
                for sublink in linked_soup.find_all("a", string=re.compile(self.keyword, re.IGNORECASE)):
                    if not first_url.endswith('/'):
                        first_url += '/'
                    sub_url = urljoin(first_url, sublink["href"])
                    print(f"  Found match: {sublink.string} → {sub_url}")

                    sublink_href = str(sublink['href'])
                    if sublink_href.endswith('.pdf'):
                        print("  PDF link Found:", sublink_href)
                        
                    else:
                        # Visit the matched link and extract head
                        subpage = requests.get(sub_url)
                        sub_soup = BeautifulSoup(subpage.text, "html.parser")
                        print("  Title of Page: " + sub_soup.title.string)

            except Exception as e:
                print("  Error:", e)
