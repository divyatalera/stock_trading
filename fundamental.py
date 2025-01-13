
from bs4 import BeautifulSoup
import requests
import pandas as pd

page = requests.get('https://economictimes.indiatimes.com/indices/nifty_50_companies')
soup = BeautifulSoup(page.text, 'html.parser')

links = []

stocks = soup.find(class_='dataContainer')
stockLinks = stocks.find_all(class_='flt w120')

for stock_links in stockLinks:
    link = stock_links.find('a').get('href')
    link = 'https://economictimes.indiatimes.com'+link
    link = link.replace("stocks","stocksupdate")
    links.append(link)


# get symbols
symbols = []
stockSymbols = stocks.find_all(class_='dataList')

for stock in stockSymbols:
    symbol = stock.find(class_='w80')
    print(symbol)
    symbol = symbol.find('a').get('href')
    start = symbol.find('symbol=')
    start = start + 7
    end = symbol.find('&exchange')
    end = end -2
    symbols.append(symbol[start:end])


zipped = list(zip(symbols,links))   #for parallel iteration of lists
df = pd.DataFrame(zipped, columns=['Symbols','Links'])
df.to_csv('links.csv', index=False, encoding='utf-8')





